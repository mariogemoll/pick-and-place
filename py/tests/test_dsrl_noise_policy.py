# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""The noise wrapper against the real vendored sampler.

These tests need ``third_party/dppo`` on the path and are skipped without it,
the way the rest of the suite is skipped without its heavy dependencies. They
are the ones that matter most: everything DSRL learns is defined against the
claim that ``denoise(model, cond, w)`` is exactly what the checkpoint would have
done had it drawn ``w`` itself, and that claim is about upstream's code rather
than about anything written here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_DPPO = Path(__file__).resolve().parents[2] / "third_party" / "dppo"
if not (_DPPO / "model" / "diffusion" / "diffusion.py").exists():
    pytest.skip("third_party/dppo is not checked out", allow_module_level=True)
if str(_DPPO) not in sys.path:
    sys.path.insert(0, str(_DPPO))

from model.diffusion.diffusion import DiffusionModel  # noqa: E402

from pick_and_place.dsrl.noise_policy import denoise, latent_shape  # noqa: E402

HORIZON_STEPS = 3
ACTION_DIM = 2
OBS_DIM = 4
BATCH = 5


class _StubNetwork(torch.nn.Module):
    """A denoiser standing in for ``VisionUnet1D``.

    Predicts epsilon as a fixed linear function of the sample, the timestep and
    the conditioning, so the sampling loop exercises every term while staying
    reproducible on CPU.
    """

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.linspace(0.1, 0.5, ACTION_DIM))

    def forward(self, x, time, cond=None, **kwargs):
        del kwargs
        conditioning = cond["state"].reshape(len(x), -1).mean(dim=-1)
        offset = (time.float() / 100.0 + conditioning).view(-1, 1, 1)
        return torch.tanh(x * self.scale + offset)


class _TestableDiffusion(DiffusionModel):
    """``DiffusionModel`` whose ``p_mean_var`` takes the kwarg ``forward`` passes.

    Upstream's base class declares ``p_mean_var`` without ``deterministic`` but
    its own ``forward`` passes it, so the base class cannot be sampled from as
    shipped -- only the VPG/PPO subclasses that widen the signature can. This
    adds exactly that kwarg and nothing else, which is what makes the comparison
    below one against upstream's real loop.
    """

    def p_mean_var(self, x, t, cond, index=None, deterministic=False, **kwargs):
        del deterministic
        return super().p_mean_var(x, t, cond, index=index, **kwargs)


def _model(**overrides):
    defaults = {
        "network": _StubNetwork(),
        "horizon_steps": HORIZON_STEPS,
        "obs_dim": OBS_DIM,
        "action_dim": ACTION_DIM,
        "denoising_steps": 20,
        "use_ddim": True,
        "ddim_steps": 4,
        "device": "cpu",
    }
    model = _TestableDiffusion(**{**defaults, **overrides})
    model.eval()
    return model


def _cond():
    generator = torch.Generator().manual_seed(7)
    return {"state": torch.randn(BATCH, 2, OBS_DIM, generator=generator)}


def test_denoise_reproduces_the_checkpoints_own_sampler():
    """Same starting noise, same chunk -- to the bit.

    ``DiffusionModel.forward`` draws ``x_T`` with the very first call to
    ``torch.randn``, so seeding identically and drawing that tensor by hand
    gives the noise it would have used.
    """
    model = _model()
    cond = _cond()

    torch.manual_seed(0)
    expected = model(cond=cond, deterministic=True).trajectories

    torch.manual_seed(0)
    noise = torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM))
    actual = denoise(model, cond, noise)

    assert torch.allclose(expected, actual, atol=0, rtol=0)


def test_the_map_from_noise_to_action_is_deterministic():
    """The premise of the whole method: w alone decides the chunk."""
    model = _model()
    cond = _cond()
    noise = torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM))
    first = denoise(model, cond, noise)
    second = denoise(model, cond, noise)
    assert torch.equal(first, second)


def test_different_noise_gives_different_actions():
    """The other half: a policy that ignored w would make DSRL a no-op.

    This is what ``measure_dsrl_steerability.py`` measures on the real
    checkpoint; here it only pins that the wrapper propagates the noise at all.
    """
    model = _model()
    cond = _cond()
    generator = torch.Generator().manual_seed(1)
    first = denoise(model, cond, torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM), generator=generator))
    second = denoise(model, cond, torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM), generator=generator))
    assert not torch.allclose(first, second)


def test_ddpm_sampling_is_refused():
    """DDPM injects fresh noise per step, so x_T would not determine the action."""
    model = _model(use_ddim=False, ddim_steps=None)
    with pytest.raises(ValueError, match="deterministic map"):
        denoise(model, _cond(), torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM)))


def test_wrong_noise_shape_is_refused():
    model = _model()
    with pytest.raises(ValueError, match="noise must be"):
        denoise(model, _cond(), torch.randn((BATCH, HORIZON_STEPS + 1, ACTION_DIM)))


def test_latent_shape_is_the_whole_predicted_horizon():
    """Not the executed prefix: one x_T produces every predicted step."""
    assert latent_shape(_model()) == (HORIZON_STEPS, ACTION_DIM)


def test_final_action_clip_is_applied():
    model = _model(final_action_clip_value=0.01)
    actions = denoise(model, _cond(), torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM)))
    assert actions.abs().max().item() <= 0.01


# -- the subclass production actually instantiates ----------------------------


class _VpgShapedDiffusion(_TestableDiffusion):
    """A model with ``DiffusionVPG``'s p_mean_var contract, not the base class's.

    Two differences, both of which broke the first pod run. It returns
    ``(mu, logvar, etas)`` rather than a pair, and it takes ``deterministic`` --
    which under DDIM decides whether ``etas`` is zero or ``self.eta(cond)``, and
    ``etas`` enters the *mean*. Passing the wrong one does not raise; it traces
    a different trajectory than the scorer does.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.deterministic_calls: list[bool] = []

    def p_mean_var(self, x, t, cond, index=None, use_base_policy=False, deterministic=False):
        del use_base_policy
        self.deterministic_calls.append(deterministic)
        mean, logvar = super().p_mean_var(x, t, cond, index=index)
        # Stand in for the eta term: nonzero only when not deterministic, and
        # large enough that a wrong call is visible rather than a rounding tick.
        if not deterministic:
            mean = mean + 0.5
        return mean, logvar, torch.zeros_like(mean)


def _vpg_model():
    model = _VpgShapedDiffusion(
        network=_StubNetwork(),
        horizon_steps=HORIZON_STEPS,
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        denoising_steps=20,
        use_ddim=True,
        ddim_steps=4,
        device="cpu",
    )
    model.eval()
    return model


def test_denoise_handles_the_three_value_return():
    """The base class returns a pair; VPG and PPO return (mu, logvar, etas)."""
    model = _vpg_model()
    actions = denoise(model, _cond(), torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM)))
    assert actions.shape == (BATCH, HORIZON_STEPS, ACTION_DIM)


def test_denoise_asks_for_the_deterministic_branch():
    """Omitting it silently traces a different trajectory than the scorer.

    `check_dppo_rl_env.py` evaluates with deterministic=True, so the latent
    space DSRL learns over has to be defined against that same branch.
    """
    model = _vpg_model()
    denoise(model, _cond(), torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM)))
    assert model.deterministic_calls
    assert all(model.deterministic_calls)


def test_the_deterministic_branch_actually_changes_the_result():
    """Guards the test above from passing on a model where the flag is inert."""
    model = _vpg_model()
    cond = _cond()
    noise = torch.randn((BATCH, HORIZON_STEPS, ACTION_DIM))
    steered = denoise(model, cond, noise)

    stochastic = _vpg_model()
    original = stochastic.p_mean_var
    stochastic.p_mean_var = lambda **kwargs: original(**{**kwargs, "deterministic": False})
    assert not torch.allclose(steered, denoise(stochastic, cond, noise))


# -- the features the latent actor reads --------------------------------------


def _vision_network():
    """The two-camera encoder this project's checkpoints are pretrained with."""
    pytest.importorskip("hydra")
    pytest.importorskip("einops")
    from types import SimpleNamespace

    from model.common.vit import VitEncoder
    from model.diffusion.unet import VisionUnet1D

    cfg = SimpleNamespace(
        patch_size=8, depth=1, embed_dim=128, num_heads=4, embed_style="embed2", embed_norm=0
    )
    backbone = VitEncoder(
        obs_shape=[6, 96, 96], num_channel=6, img_h=96, img_w=96, cfg=cfg
    )
    return VisionUnet1D(
        backbone=backbone,
        action_dim=6,
        img_cond_steps=2,
        cond_dim=12,
        diffusion_step_embed_dim=32,
        dim=64,
        dim_mults=[1, 2, 4],
        kernel_size=5,
        n_groups=8,
        smaller_encoder=False,
        cond_predict_scale=True,
        groupnorm_eps=1e-4,
        spatial_emb=128,
        num_img=2,
        augment=False,
    )


def _vision_model():
    from types import SimpleNamespace

    return SimpleNamespace(actor=_vision_network(), obs_dim=6)


def _vision_cond(batch: int = 4):
    generator = torch.Generator().manual_seed(3)
    return {
        "state": torch.randn(batch, 2, 6, generator=generator),
        "rgb": torch.randn(batch, 2, 6, 96, 96, generator=generator),
    }


def test_feature_dim_predicts_the_actual_feature_width():
    """The SAC networks are sized from this before a single observation arrives.

    ``VisionUnet1D`` stores neither ``cond_dim`` nor the projection width
    directly, so both are read off ``SpatialEmb``. If that reading is wrong the
    actor is built with the wrong input width and fails much later, on a shape
    error inside a training run.
    """
    from pick_and_place.dsrl.noise_policy import feature_dim, visual_features

    model = _vision_model()
    features = visual_features(model, _vision_cond())
    assert features.shape[1] == feature_dim(model)
    # 128 per camera, two cameras, plus the six joints at two history steps.
    assert feature_dim(model) == 268


def test_visual_features_are_a_fixed_function_of_the_observation():
    """What licenses caching them in the replay buffer instead of the pixels."""
    from pick_and_place.dsrl.noise_policy import visual_features

    model = _vision_model()
    cond = _vision_cond()
    assert torch.equal(visual_features(model, cond), visual_features(model, cond))


def test_visual_features_refuse_an_unexpected_encoder():
    """A single-camera or non-spatial network needs its own feature path."""
    from types import SimpleNamespace

    from pick_and_place.dsrl.noise_policy import visual_features

    model = SimpleNamespace(actor=SimpleNamespace(num_img=1), obs_dim=6)
    with pytest.raises(ValueError, match="two-camera"):
        visual_features(model, _vision_cond())
