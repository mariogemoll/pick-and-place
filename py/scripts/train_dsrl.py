#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Steer a pretrained Diffusion Policy with latent-noise reinforcement learning.

Loads the checkpoint named on the command line, freezes it, and trains a small
SAC agent over the noise it denoises from. The diffusion policy's weights are
never written -- what this produces is a `state_*.pt` holding the latent actor
and critics, which `check_dppo_rl_env.py --dsrl-actor` then scores against the
same paired oracle every other policy in this project is measured with.

    python scripts/train_dsrl.py \
        --config ../config/diffusion_policy/dsrl_so101.yaml \
        --checkpoint <pretrained state_500.pt> \
        --normalization <artifact>/normalization.npz \
        --output-dir <run dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from pick_and_place.spec.action_encoding import read_action_encoding


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="the DSRL YAML")
    parser.add_argument(
        "--checkpoint", type=Path, required=True, help="pretrained diffusion state_*.pt"
    )
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-gl", default=os.environ.get("MUJOCO_GL", "egl"))
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--total-iterations", type=int, default=None)
    parser.add_argument("--warmup-iterations", type=int, default=None)
    parser.add_argument("--gradient-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--action-magnitude", type=float, default=None)
    parser.add_argument(
        "--init-temperature",
        type=float,
        default=None,
        help=(
            "SAC's initial entropy weight. Not cosmetic on a 96-dimensional "
            "latent: the target adds alpha * entropy to every bootstrap, and at "
            "alpha 1 with entropy ~65 nats that is ~65 per step against a task "
            "reward of at most 8, so the soft value is ~97 percent entropy bonus. "
            "The auto-tuner does bring it down toward target_entropy, but "
            "measured at ~0.835 per 600 gradient steps, which is hundreds of "
            "iterations of fitting a value the task barely enters."
        ),
    )
    parser.add_argument("--buffer-capacity", type=int, default=None)
    parser.add_argument(
        "--scene-seed-base",
        type=int,
        default=None,
        help="override the training scene stream; must stay off the evaluation bases",
    )
    parser.add_argument(
        "--observable-critic",
        action="store_true",
        help=(
            "give the critic the same features the actor sees instead of privileged "
            "simulator state. Slower to learn, but the whole learner then transfers "
            "to hardware unchanged."
        ),
    )
    parser.add_argument(
        "--expect-action-encoding",
        choices=["absolute", "delta"],
        default=None,
        help="fail unless the bounds declare this encoding",
    )
    parser.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    return parser.parse_args()


def _override(value, fallback):
    return fallback if value is None else value


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    OmegaConf.register_new_resolver(
        "now", lambda pattern: datetime.now(UTC).strftime(pattern), replace=True
    )

    from pick_and_place.dppo_rl.env import EnvConfig
    from pick_and_place.dppo_rl.vector_env import DppoVectorEnv
    from pick_and_place.dsrl.noise_policy import latent_shape
    from pick_and_place.dsrl.sac import SacConfig
    from pick_and_place.dsrl.trainer import DsrlTrainer, TrainConfig
    from pick_and_place.sim.scene_appearance import parse_appearance

    with np.load(args.normalization) as bounds:
        action_encoding = read_action_encoding(bounds)
    if (
        args.expect_action_encoding is not None
        and action_encoding.value != args.expect_action_encoding
    ):
        raise SystemExit(
            f"{args.normalization} declares {action_encoding.value} actions, but "
            f"{args.expect_action_encoding} was expected. The checkpoint and the "
            "bounds it was fitted against have to move together."
        )

    config = OmegaConf.load(args.config)
    config.base_policy_path = str(args.checkpoint)
    config.normalization_path = str(args.normalization)
    config.device = args.device
    for name in ("DPPO_LOG_DIR", "DPPO_DATA_DIR", "DPPO_BASE_POLICY"):
        os.environ.setdefault(name, "unused-by-this-script")
    OmegaConf.resolve(config)

    n_envs = _override(args.n_envs, int(config.env.n_envs))
    seed = _override(args.seed, int(config.seed))
    privileged = bool(config.dsrl.privileged_critic) and not args.observable_critic

    env_config = EnvConfig(
        normalization_path=args.normalization,
        image_hw=(
            int(config.shape_meta.obs.rgb.shape[1]),
            int(config.shape_meta.obs.rgb.shape[2]),
        ),
        render_hw=tuple(int(value) for value in config.env.render_hw),
        cond_steps=int(config.cond_steps),
        act_steps=int(config.act_steps),
        control_hz=float(config.env.control_hz),
        max_steps=int(config.env.max_episode_steps),
        seed_base=_override(args.scene_seed_base, int(config.env.scene_seed_base)),
        scene_appearance=parse_appearance(str(config.env.scene_appearance))[1],
        dense_success_reward=bool(config.env.dense_success_reward),
        shaping_weight=float(config.env.shaping_weight),
        privileged_obs=privileged,
    )

    torch.manual_seed(seed)
    device = torch.device(args.device)
    model = hydra.utils.instantiate(config.model)
    model.eval()
    # Nothing here trains the diffusion policy. Saying so with requires_grad
    # rather than only by omission means an accidental backward pass through the
    # denoiser raises instead of quietly costing memory.
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    horizon_steps, action_dim = latent_shape(model)
    train_config = TrainConfig(
        total_iterations=_override(args.total_iterations, int(config.dsrl.total_iterations)),
        warmup_iterations=_override(
            args.warmup_iterations, int(config.dsrl.warmup_iterations)
        ),
        gradient_steps_per_iteration=_override(
            args.gradient_steps, int(config.dsrl.gradient_steps_per_iteration)
        ),
        batch_size=_override(args.batch_size, int(config.dsrl.batch_size)),
        buffer_capacity=_override(args.buffer_capacity, int(config.dsrl.buffer_capacity)),
        save_freq=int(config.dsrl.save_freq),
        log_freq=int(config.dsrl.log_freq),
        seed=seed,
        privileged_critic=privileged,
    )

    def sac_config(actor_feature_dim: int, critic_feature_dim: int) -> SacConfig:
        return SacConfig(
            latent_dim=horizon_steps * action_dim,
            actor_feature_dim=actor_feature_dim,
            critic_feature_dim=critic_feature_dim,
            action_magnitude=_override(
                args.action_magnitude, float(config.dsrl.action_magnitude)
            ),
            hidden_dim=int(config.dsrl.hidden_dim),
            n_layers=int(config.dsrl.n_layers),
            actor_lr=float(config.dsrl.actor_lr),
            critic_lr=float(config.dsrl.critic_lr),
            temperature_lr=float(config.dsrl.temperature_lr),
            gamma=float(config.dsrl.gamma),
            tau=float(config.dsrl.tau),
            target_entropy=float(config.dsrl.target_entropy),
            init_temperature=_override(
                args.init_temperature, float(config.dsrl.init_temperature)
            ),
            n_critics=int(config.dsrl.n_critics),
            activation=str(config.dsrl.activation),
            device=args.device,
            seed=seed,
        )

    on_log = None
    if args.wandb:
        import wandb

        wandb.init(
            entity=config.wandb.entity,
            project=str(config.wandb.project),
            name=str(config.wandb.run),
            config=OmegaConf.to_container(config, resolve=True),
        )
        on_log = wandb.log

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.yaml").write_text(OmegaConf.to_yaml(config))

    venv = DppoVectorEnv(env_config, n_envs, mujoco_gl=args.mujoco_gl)
    venv.seed([seed + index for index in range(n_envs)])
    trainer = DsrlTrainer(
        model=model,
        venv=venv,
        act_steps=env_config.act_steps,
        sac_config_factory=sac_config,
        train_config=train_config,
        output_dir=args.output_dir,
        device=device,
        on_log=on_log,
    )
    try:
        result = trainer.run()
    finally:
        venv.close()

    (args.output_dir / "history.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["final"], indent=2))


if __name__ == "__main__":
    sys.exit(main())
