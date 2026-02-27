"""Adversarial patch attack against world-model planning.

Learns a localized image patch that, when composited into observations,
causes the Dreamer agent's imagination/planning to select worse actions,
lowering true episode return.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class AdversarialPatch(nn.Module):
    """Localized adversarial patch for RSSM-based world-model agents.

    Parameters
    ----------
    image_hw : tuple of int
        Observation image size (H, W). Default (64, 64).
    patch_hw : tuple of int
        Patch size (ph, pw). Default (16, 16).
    placement : str
        One of "fixed", "bottom_center", "top_center", "random".
    x0, y0 : int or None
        Top-left coordinates for "fixed" placement.
    eps : float or None
        If set, uses base + delta parameterization with l_inf clamp.
    base_color : tuple of float
        RGB base color for base+delta mode. Default (0.5, 0.5, 0.5).
    eot_translations : int
        Number of random translation samples for EoT. 0 = disabled.
    eot_max_shift : int
        Maximum pixel shift for EoT translations.
    eot_brightness : float
        Maximum brightness perturbation for EoT. 0 = disabled.
    floor_mask_path : str or None
        Path to a precomputed floor mask tensor (.pt) of shape (H, W).
    """

    def __init__(
        self,
        image_hw=(64, 64),
        patch_hw=(16, 16),
        placement="bottom_center",
        x0=None,
        y0=None,
        eps=None,
        base_color=(0.5, 0.5, 0.5),
        eot_translations=0,
        eot_max_shift=2,
        eot_brightness=0.0,
        floor_mask_path=None,
    ):
        super().__init__()
        self.H, self.W = image_hw
        self.ph, self.pw = patch_hw
        self.placement = placement
        self.eps = eps
        self.eot_translations = eot_translations
        self.eot_max_shift = eot_max_shift
        self.eot_brightness = eot_brightness

        assert self.ph <= self.H and self.pw <= self.W, (
            f"Patch {patch_hw} must fit inside image {image_hw}"
        )

        # Determine patch location
        self._x0, self._y0 = self._compute_placement(x0, y0)

        # Patch parameterization
        if eps is not None:
            # base + delta mode
            base = torch.full((1, 1, self.ph, self.pw, 3), 0.0)
            base[..., 0] = base_color[0]
            base[..., 1] = base_color[1]
            base[..., 2] = base_color[2]
            self.register_buffer("base", base)
            self.delta = nn.Parameter(torch.zeros(1, 1, self.ph, self.pw, 3))
        else:
            # Direct parameterization in [0, 1]
            self.delta = nn.Parameter(torch.rand(1, 1, self.ph, self.pw, 3) * 0.5 + 0.25)

        # Floor mask (optional)
        if floor_mask_path is not None:
            floor_mask = torch.load(floor_mask_path, weights_only=True)
            assert floor_mask.shape == (self.H, self.W), (
                f"Floor mask shape {floor_mask.shape} != image size {(self.H, self.W)}"
            )
            self.register_buffer("floor_mask", floor_mask.float())
        else:
            self.floor_mask = None

        # Precompute rectangle mask indices
        self.register_buffer(
            "_rect_mask", self._build_rect_mask(), persistent=False
        )

    def _compute_placement(self, x0, y0):
        """Determine top-left corner (x0, y0) of the patch."""
        if self.placement == "fixed":
            assert x0 is not None and y0 is not None
            return int(x0), int(y0)
        elif self.placement == "bottom_center":
            x0 = (self.W - self.pw) // 2
            y0 = self.H - self.ph
            return x0, y0
        elif self.placement == "top_center":
            x0 = (self.W - self.pw) // 2
            y0 = 0
            return x0, y0
        elif self.placement == "center":
            x0 = (self.W - self.pw) // 2
            y0 = (self.H - self.ph) // 2
            return x0, y0
        elif self.placement == "random":
            # Will be randomized per forward pass
            return 0, 0
        else:
            raise ValueError(f"Unknown placement: {self.placement}")

    def _build_rect_mask(self):
        """Build a binary mask (H, W, 1) for the patch rectangle."""
        mask = torch.zeros(self.H, self.W, 1)
        mask[self._y0 : self._y0 + self.ph, self._x0 : self._x0 + self.pw, :] = 1.0
        return mask

    def get_patch(self):
        """Return the patch tensor in [0, 1], shape (1, 1, ph, pw, 3)."""
        if self.eps is not None:
            clamped_delta = torch.clamp(self.delta, -self.eps, self.eps)
            patch = torch.clamp(self.base + clamped_delta, 0.0, 1.0)
        else:
            patch = torch.clamp(self.delta, 0.0, 1.0)
        return patch

    def get_mask(self, device):
        """Return the compositing mask (H, W, 1), optionally intersected with floor mask."""
        mask = self._rect_mask.to(device)
        if self.floor_mask is not None:
            fm = self.floor_mask.to(device).unsqueeze(-1)  # (H, W, 1)
            mask = mask * fm
        return mask

    def apply(self, images, randomize_location=False):
        """Composite the patch onto float images in [0, 1].

        Parameters
        ----------
        images : Tensor
            Float images in [0, 1] with shape (..., H, W, C).
            Typically (B, T, H, W, C) or (B, 1, H, W, C).
        randomize_location : bool
            If True and placement=="random", sample a new location.

        Returns
        -------
        Tensor
            Patched images, same shape as input.
        """
        leading_shape = images.shape[:-3]
        H, W, C = images.shape[-3:]
        assert H == self.H and W == self.W and C == 3, (
            f"Expected (..., {self.H}, {self.W}, 3), got {images.shape}"
        )

        device = images.device
        patch = self.get_patch().to(device)  # (1, 1, ph, pw, 3)

        # Handle EoT transforms
        if self.training and self.eot_translations > 0:
            return self._apply_eot(images, patch)

        # Standard application
        if randomize_location or self.placement == "random":
            y0 = torch.randint(0, self.H - self.ph + 1, (1,)).item()
            x0 = torch.randint(0, self.W - self.pw + 1, (1,)).item()
        else:
            y0, x0 = self._y0, self._x0

        return self._composite(images, patch, y0, x0, device)

    def _composite(self, images, patch, y0, x0, device):
        """Alpha-composite the patch at (x0, y0)."""
        # Build mask for this position
        mask = torch.zeros(self.H, self.W, 1, device=device)
        mask[y0 : y0 + self.ph, x0 : x0 + self.pw, :] = 1.0

        if self.floor_mask is not None:
            fm = self.floor_mask.to(device).unsqueeze(-1)
            mask = mask * fm

        # Place patch into full image canvas, matching the rank of images
        n_leading = images.ndim - 3
        leading_ones = (1,) * n_leading
        full_patch = torch.zeros(*leading_ones, self.H, self.W, 3, device=device)
        full_patch[..., y0 : y0 + self.ph, x0 : x0 + self.pw, :] = patch.view(*leading_ones, self.ph, self.pw, 3)

        # Composite: patched = mask * patch + (1 - mask) * original
        patched = mask * full_patch + (1.0 - mask) * images
        return patched

    def _apply_eot(self, images, patch):
        """Apply Expectation over Transformations: average over random shifts + brightness."""
        device = images.device
        n = self.eot_translations
        max_s = self.eot_max_shift

        accumulated = torch.zeros_like(images)

        for _ in range(n):
            # Random translation
            dy = torch.randint(-max_s, max_s + 1, (1,)).item()
            dx = torch.randint(-max_s, max_s + 1, (1,)).item()

            y0 = max(0, min(self.H - self.ph, self._y0 + dy))
            x0 = max(0, min(self.W - self.pw, self._x0 + dx))

            # Optional brightness perturbation
            cur_patch = patch
            if self.eot_brightness > 0:
                b = (torch.rand(1, device=device) * 2 - 1) * self.eot_brightness
                cur_patch = torch.clamp(cur_patch + b, 0.0, 1.0)

            accumulated = accumulated + self._composite(images, cur_patch, y0, x0, device)

        return accumulated / n

    def tv_loss(self):
        """Total variation regularization on the patch."""
        patch = self.get_patch()  # (1, 1, ph, pw, 3)
        # Differences along height
        dh = patch[:, :, 1:, :, :] - patch[:, :, :-1, :, :]
        # Differences along width
        dw = patch[:, :, :, 1:, :] - patch[:, :, :, :-1, :]
        return dh.abs().mean() + dw.abs().mean()

    def l2_loss(self):
        """L2 regularization on the delta parameter."""
        if self.eps is not None:
            return self.delta.pow(2).mean()
        else:
            return (self.delta - 0.5).pow(2).mean()

    def project(self):
        """Project patch parameters to valid range (call after optimizer step)."""
        with torch.no_grad():
            if self.eps is not None:
                self.delta.clamp_(-self.eps, self.eps)
            else:
                self.delta.clamp_(0.0, 1.0)

    def save(self, path):
        """Save patch state dict to file."""
        state = {
            "delta": self.delta.data,
            "image_hw": (self.H, self.W),
            "patch_hw": (self.ph, self.pw),
            "placement": self.placement,
            "x0": self._x0,
            "y0": self._y0,
            "eps": self.eps,
        }
        if self.eps is not None:
            state["base"] = self.base.data
        torch.save(state, path)

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a saved patch."""
        state = torch.load(path, map_location=device, weights_only=True)
        patch = cls(
            image_hw=state["image_hw"],
            patch_hw=state["patch_hw"],
            placement="fixed",
            x0=state["x0"],
            y0=state["y0"],
            eps=state["eps"],
        )
        patch.delta.data.copy_(state["delta"])
        if state["eps"] is not None and "base" in state:
            patch.base.data.copy_(state["base"])
        return patch.to(device)


def compute_saliency_placement(agent, data_batch, patch_hw, image_hw=(64, 64),
                                allowed_region=None):
    """Compute saliency-based patch placement.

    Finds the image region where gradients of imagined return w.r.t. pixel
    values are highest, then places the patch there.

    Parameters
    ----------
    agent : Dreamer
        The trained agent (parameters frozen for attack).
    data_batch : dict
        A batch with keys "image" (B, T, H, W, C) float [0,1],
        "action" (B, T, A), "is_first" (B, T).
    patch_hw : tuple of int
        Patch size (ph, pw).
    image_hw : tuple of int
        Image dimensions (H, W).
    allowed_region : tuple or None
        (y_min, y_max, x_min, x_max) restricting placement search area.

    Returns
    -------
    (y0, x0) : tuple of int
        Optimal placement coordinates.
    saliency_map : Tensor
        The aggregated saliency map (H, W).
    """
    H, W = image_hw
    ph, pw = patch_hw
    device = next(agent.parameters()).device

    images = data_batch["image"].clone().detach().requires_grad_(True)
    obs = dict(data_batch)
    obs["image"] = images

    # Forward through world model
    embed = agent.encoder(obs)
    B, T = embed.shape[:2]
    initial = agent.rssm.initial(B)
    post_stoch, post_deter, _ = agent.rssm.observe(
        embed, obs["action"], initial, obs["is_first"]
    )

    # Imagination from last timestep
    stoch0 = post_stoch[:, -1]
    deter0 = post_deter[:, -1]

    disc = 1.0 - 1.0 / agent.horizon
    total_return = torch.tensor(0.0, device=device)
    stoch, deter = stoch0, deter0
    gamma_prod = 1.0

    for _ in range(min(15, agent.imag_horizon)):
        feat = agent.rssm.get_feat(stoch, deter)
        action = agent.actor(feat).mode
        reward = agent.reward(feat).mode()
        cont = agent.cont(feat).mean
        total_return = total_return + gamma_prod * reward.mean()
        gamma_prod = gamma_prod * disc * cont.mean()
        stoch, deter = agent.rssm.img_step(stoch, deter, action)

    total_return.backward()
    # Aggregate saliency
    saliency = images.grad.abs().sum(dim=(0, 1, -1))  # (H, W)

    # Convolve with box filter of patch size to find best placement
    sal = saliency.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    kernel = torch.ones(1, 1, ph, pw, device=device)
    conv_sal = F.conv2d(sal, kernel, padding=0)  # (1, 1, H-ph+1, W-pw+1)
    conv_sal = conv_sal.squeeze()

    # Restrict to allowed region
    if allowed_region is not None:
        y_min, y_max, x_min, x_max = allowed_region
        mask = torch.zeros_like(conv_sal)
        mask[
            max(0, y_min) : min(conv_sal.shape[0], y_max),
            max(0, x_min) : min(conv_sal.shape[1], x_max),
        ] = 1.0
        conv_sal = conv_sal * mask

    best_idx = conv_sal.argmax()
    y0 = (best_idx // conv_sal.shape[1]).item()
    x0 = (best_idx % conv_sal.shape[1]).item()

    return (y0, x0), saliency.detach()


class PlanningAttackLoss(nn.Module):
    """Differentiable loss through world-model imagination.

    Computes:
    L = w_return * imagined_return
      + w_kl * KL(posterior_patched || posterior_clean)
      + w_action * action_shift
      + w_tv * total_variation
      + w_l2 * l2_regularization

    We minimize this, i.e., minimize imagined return under the patch.
    """

    def __init__(
        self,
        w_return=1.0,
        w_kl=0.0,
        w_action=0.0,
        w_tv=0.01,
        w_l2=0.001,
        imag_horizon=15,
        context_len=16,
        use_mode_actions=True,
    ):
        super().__init__()
        self.w_return = w_return
        self.w_kl = w_kl
        self.w_action = w_action
        self.w_tv = w_tv
        self.w_l2 = w_l2
        self.imag_horizon = imag_horizon
        self.context_len = context_len
        self.use_mode_actions = use_mode_actions

    def forward(self, agent, patch_module, images_clean, actions, is_first, initial):
        """Compute attack loss.

        Parameters
        ----------
        agent : Dreamer
            Agent with frozen parameters.
        patch_module : AdversarialPatch
            The patch being optimized.
        images_clean : Tensor
            Clean images (B, T, H, W, C) float in [0, 1].
        actions : Tensor
            Actions (B, T, A).
        is_first : Tensor
            Episode start flags (B, T).
        initial : tuple
            (stoch_init, deter_init) for RSSM.

        Returns
        -------
        loss : Tensor
            Scalar attack loss.
        metrics : dict
            Breakdown of loss components.
        """
        device = images_clean.device
        B, T = images_clean.shape[:2]

        # Apply patch (images stay in [0, 1]; encoder handles centering internally)
        images_patched = patch_module.apply(images_clean)

        # Build observation dicts (encoder expects [0, 1] and subtracts 0.5 internally)
        obs_patched = {"image": images_patched}
        obs_clean_dict = {"image": images_clean}

        # Forward patched through encoder + RSSM posterior
        embed_patched = agent.encoder(obs_patched)
        post_stoch_p, post_deter_p, post_logit_p = agent.rssm.observe(
            embed_patched, actions, initial, is_first
        )

        # Start imagination from last context step
        stoch0 = post_stoch_p[:, -1]
        deter0 = post_deter_p[:, -1]

        disc = 1.0 - 1.0 / agent.horizon

        # Differentiable imagination rollout
        rewards = []
        conts = []
        stoch, deter = stoch0, deter0
        for _ in range(self.imag_horizon):
            feat = agent.rssm.get_feat(stoch, deter)
            if self.use_mode_actions:
                action = agent.actor(feat).mode
            else:
                action = agent.actor(feat).rsample()
            reward = agent.reward(feat).mode()  # (B, 1)
            cont = agent.cont(feat).mean  # (B, 1)
            rewards.append(reward)
            conts.append(cont)
            stoch, deter = agent.rssm.img_step(stoch, deter, action)

        # Compute discounted return
        rewards = torch.stack(rewards, dim=1)  # (B, H, 1)
        conts = torch.stack(conts, dim=1)  # (B, H, 1)
        gamma = disc * conts
        # Cumulative discount: gamma_0=1, gamma_1=disc*cont_0, ...
        weights = torch.cumprod(
            torch.cat([torch.ones_like(gamma[:, :1]), gamma[:, :-1]], dim=1),
            dim=1,
        )
        discounted_return = (weights * rewards).sum(dim=1).mean()  # scalar

        metrics = {"imagined_return": discounted_return.item()}
        loss = self.w_return * discounted_return

        # Auxiliary: KL between patched and clean posteriors
        if self.w_kl > 0:
            with torch.no_grad():
                embed_clean = agent.encoder(obs_clean_dict)
                _, _, post_logit_c = agent.rssm.observe(
                    embed_clean, actions, initial, is_first
                )
            # Negative KL: we want posteriors to DIFFER, so we minimize -KL
            # KL(patched || clean)
            from distributions import kl as kl_fn
            kl_val = kl_fn(post_logit_p, post_logit_c.detach()).sum(-1).mean()
            loss = loss - self.w_kl * kl_val  # minimize return AND maximize KL divergence
            metrics["kl_post"] = kl_val.item()

        # Auxiliary: action shift
        if self.w_action > 0:
            with torch.no_grad():
                embed_clean = agent.encoder(obs_clean_dict)
                post_stoch_c, post_deter_c, _ = agent.rssm.observe(
                    embed_clean, actions, initial, is_first
                )
                feat_clean = agent.rssm.get_feat(post_stoch_c[:, -1], post_deter_c[:, -1])
                action_clean = agent.actor(feat_clean).mode

            feat_patched = agent.rssm.get_feat(stoch0, deter0)
            action_patched = agent.actor(feat_patched).mode
            action_diff = (action_patched - action_clean.detach()).pow(2).mean()
            loss = loss - self.w_action * action_diff  # maximize action shift
            metrics["action_shift"] = action_diff.item()

        # Regularization
        if self.w_tv > 0:
            tv = patch_module.tv_loss()
            loss = loss + self.w_tv * tv
            metrics["tv_loss"] = tv.item()

        if self.w_l2 > 0:
            l2 = patch_module.l2_loss()
            loss = loss + self.w_l2 * l2
            metrics["l2_loss"] = l2.item()

        metrics["total_loss"] = loss.item()
        return loss, metrics
