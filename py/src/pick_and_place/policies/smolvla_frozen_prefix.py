# SPDX-FileCopyrightText: 2026 Mario Gemoll
# SPDX-License-Identifier: 0BSD

"""Stop backpropagating the part of a SmolVLA prefix that cannot train.

A cached step is dominated by its prefix: 141 tokens of images, language and
state against 50 action tokens, and sweeping the tokens per camera puts **69% of
the step** in the image tokens alone. Nothing in that path trains --
``freeze_vision_encoder`` and ``train_expert_only`` are both set, and the prefix
attention mask keeps image and language tokens from ever seeing ``state_proj``,
the one trainable thing in the prefix, at any depth.

Autograd cannot use that argument. Every layer concatenates the prefix and the
suffix into one tensor before attending, and the suffix requires grad, so from
the first layer on the prefix half does too and its backward runs. Freezing
``state_proj`` outright changes the step by 0.5%, which is the measurement that
says the concatenation, not the requires_grad flags, is what keeps it alive.

So this splits the stack by hand. The frozen 140 tokens run under ``no_grad``,
the state token and the action tokens run with grad against the keys and values
they produced, and the arithmetic is otherwise the one
``SmolVLMWithExpertModel.forward`` does -- same projections, same rotary
positions, same mask slices, same order of keys. What changes is only which
tensors autograd keeps.

The model's own inference path already has this shape (prefill a prefix, then
decode against its cache), but its decode branch assumes the VLM stream is
finished, and here the state token still has to travel with gradients. Hence a
fork of the layer loop rather than a call into it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def patch_policy_for_frozen_prefix(policy: SmolVLAPolicy) -> Callable[[], None]:
    """Run one policy's training forward with the frozen prefix out of the graph.

    Returns a callable that puts the original back, so one process can measure
    both arms against one GPU. Only the training forward is replaced; sampling
    actions still goes through lerobot's own code.

    ``compile_model`` is honoured rather than lost. `SmolVLAFlowMatching.__init__`
    implements it as ``self.forward = torch.compile(self.forward)``, so replacing
    ``forward`` afterwards would silently drop the compiled artifact and hand back
    an eager step -- which is how this was first measured, at exactly the eager
    seconds.
    """
    model = policy.model
    _require_supported(policy.config, model.vlm_with_expert)
    original = model.forward

    def forward(  # noqa: ANN202
        images,  # noqa: ANN001
        img_masks,  # noqa: ANN001
        lang_tokens,  # noqa: ANN001
        lang_masks,  # noqa: ANN001
        state,  # noqa: ANN001
        actions,  # noqa: ANN001
        noise=None,  # noqa: ANN001
        time=None,  # noqa: ANN001
    ):
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        if noise is None:
            noise = model.sample_noise(actions.shape, actions.device)
        if time is None:
            time = model.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = model.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        frozen_length = _frozen_prefix_length(prefix_att_masks, att_2d_masks)
        suffix_out = _split_forward(
            model.vlm_with_expert,
            att_2d_masks,
            position_ids,
            prefix_embs,
            suffix_embs,
            frozen_length,
        )
        suffix_out = suffix_out[:, -model.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)
        return torch.nn.functional.mse_loss(u_t, v_t, reduction="none")

    if policy.config.compile_model:
        forward = torch.compile(forward, mode=policy.config.compile_mode)
    model.forward = forward

    def restore() -> None:
        model.forward = original

    return restore


def _require_supported(config, vlm) -> None:  # noqa: ANN001
    """Refuse a configuration this split would train differently.

    The first two are the ones that matter: if anything the frozen tokens pass
    through could train, detaching them would drop its gradient silently and the
    run would look fine while learning less than it should.
    """
    if not config.freeze_vision_encoder:
        raise ValueError("the vision encoder trains here, so the prefix is not frozen")
    if not config.train_expert_only:
        raise ValueError("the VLM trains here, so the prefix is not frozen")
    if "cross" not in vlm.attention_mode:
        raise ValueError(f"attention_mode {vlm.attention_mode!r} is not the cross-attention loop")
    if vlm.self_attn_every_n_layers <= 0:
        raise ValueError("self_attn_every_n_layers must be positive")
    if vlm.num_expert_layers != vlm.num_vlm_layers:
        raise ValueError(
            f"{vlm.num_expert_layers} expert layers against {vlm.num_vlm_layers} VLM layers; "
            "this fork pairs them one to one"
        )


def _frozen_prefix_length(prefix_att_masks: torch.Tensor, att_2d_masks: torch.Tensor) -> int:
    """How many leading prefix tokens can be taken out of the graph.

    They are the ones whose `att_masks` entry is 0 -- images and language -- and
    the check that matters is not that count but that nothing in it attends to
    anything after it, which is what makes their representations independent of
    `state_proj`. That is asserted against the mask the model will actually use.
    """
    masks = prefix_att_masks[0].bool()
    trainable = masks.nonzero()
    if trainable.numel() == 0:
        raise ValueError("this prefix has no state token; nothing marks the frozen boundary")
    frozen_length = int(trainable[0])
    if not bool(masks[frozen_length:].all()):
        raise ValueError("the prefix mixes frozen and attending tokens; the split would be wrong")
    if bool(att_2d_masks[:, :frozen_length, frozen_length:].any()):
        raise ValueError("frozen prefix tokens attend to later ones, so they are not frozen")
    return frozen_length


def _project(
    layer, hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, ...]:  # noqa: ANN001
    """One layer's queries, keys and values for one stream, rotary positions applied."""
    from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

    hidden_states = layer.input_layernorm(hidden_states)
    shape = (*hidden_states.shape[:-1], -1, layer.self_attn.head_dim)
    hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
    query = layer.self_attn.q_proj(hidden_states).view(shape)
    key = layer.self_attn.k_proj(hidden_states).view(shape)
    value = layer.self_attn.v_proj(hidden_states).view(shape)
    return apply_rope(query, position_ids), apply_rope(key, position_ids), value


def _residual(layer, hidden_states: torch.Tensor, att_output: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
    """The part of a layer after attention: projection, both residuals, the MLP."""
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
    out = layer.self_attn.o_proj(att_output)
    out = out + hidden_states
    after_first_residual = out
    out = layer.post_attention_layernorm(out)
    out = layer.mlp(out)
    return out + after_first_residual


def _split_forward(  # noqa: ANN001
    vlm,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    prefix_embs: torch.Tensor,
    suffix_embs: torch.Tensor,
    frozen_length: int,
) -> torch.Tensor:
    """`SmolVLMWithExpertModel.forward`, with the frozen prefix under `no_grad`."""
    from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope

    models = [vlm.get_vlm_model().text_model, vlm.lm_expert]
    layers = vlm.get_model_layers(models)
    attention = vlm.get_attention_interface()
    head_dim = vlm.vlm.config.text_config.head_dim
    batch_size, prefix_length = prefix_embs.shape[:2]

    frozen = prefix_embs[:, :frozen_length].detach()
    state = prefix_embs[:, frozen_length:]
    suffix = suffix_embs

    frozen_positions = position_ids[:, :frozen_length]
    state_positions = position_ids[:, frozen_length:prefix_length]
    suffix_positions = position_ids[:, prefix_length:]
    # The expert's own rotary positions start at zero, as lerobot's cross-attention
    # layer sets them; the self-attention layer uses the sequence's own.
    expert_positions = suffix_positions - suffix_positions.min(dim=1, keepdim=True).values

    frozen_mask = attention_mask[:, :frozen_length, :frozen_length]
    state_mask = attention_mask[:, frozen_length:prefix_length, :prefix_length]
    suffix_mask = attention_mask[:, prefix_length:, :]
    suffix_prefix_mask = attention_mask[:, prefix_length:, :prefix_length]

    for layer_idx in range(vlm.num_vlm_layers):
        vlm_layer, expert_layer = layers[0][layer_idx], layers[1][layer_idx]
        joint = layer_idx % vlm.self_attn_every_n_layers == 0

        with torch.no_grad():
            frozen_query, frozen_key, frozen_value = _project(vlm_layer, frozen, frozen_positions)
            frozen_att = attention(
                frozen_mask, batch_size, head_dim, frozen_query, frozen_key, frozen_value
            )
        state_query, state_key, state_value = _project(vlm_layer, state, state_positions)
        prefix_key = torch.cat([frozen_key, state_key], dim=1)
        prefix_value = torch.cat([frozen_value, state_value], dim=1)
        state_att = attention(
            state_mask, batch_size, head_dim, state_query, prefix_key, prefix_value
        )

        if joint:
            expert_query, expert_key, expert_value = _project(
                expert_layer, suffix, suffix_positions
            )
            suffix_att = attention(
                suffix_mask,
                batch_size,
                head_dim,
                expert_query,
                torch.cat([prefix_key, expert_key], dim=1),
                torch.cat([prefix_value, expert_value], dim=1),
            )
        else:
            # The expert reads the VLM's keys and values through its own
            # projections, which take them already rotated, and brings its own
            # queries rotated from zero.
            expert_hidden = expert_layer.input_layernorm(suffix)
            expert_hidden = expert_hidden.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query = expert_layer.self_attn.q_proj(expert_hidden).view(
                *expert_hidden.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            flat_key = prefix_key.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *prefix_key.shape[:2], -1
            )
            flat_value = prefix_value.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *prefix_value.shape[:2], -1
            )
            expert_key = expert_layer.self_attn.k_proj(flat_key).view(
                *flat_key.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            expert_value = expert_layer.self_attn.v_proj(flat_value).view(
                *flat_value.shape[:-1], -1, expert_layer.self_attn.head_dim
            )
            suffix_att = attention(
                suffix_prefix_mask,
                batch_size,
                head_dim,
                apply_rope(expert_query, expert_positions),
                expert_key,
                expert_value,
            )

        with torch.no_grad():
            frozen = _residual(vlm_layer, frozen, frozen_att)
        state = _residual(vlm_layer, state, state_att)
        suffix = _residual(expert_layer, suffix, suffix_att)

    return models[1].norm(suffix)
