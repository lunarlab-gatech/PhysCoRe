"""
The MfM network: per-particle material and confidence from observed motion.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-2], dims[1:-1]):
        layers += [nn.Linear(a, b), nn.GELU()]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    return nn.Sequential(*layers)


def gather(x: Tensor, idx: Tensor) -> Tensor:
    b, n, k = idx.shape
    return x[torch.arange(b, device=x.device).view(b, 1, 1).expand(b, n, k), idx]


def fourier(x: Tensor, bands: int) -> Tensor:
    if bands <= 0:
        return x
    freq = (2.0 ** torch.arange(bands, device=x.device, dtype=x.dtype)).view(*([1] * (x.ndim - 1)), bands, 1)
    y = x.unsqueeze(-2) * freq * torch.pi
    return torch.cat([torch.sin(y), torch.cos(y)], dim=-2).flatten(-2)


def observed_evidence_from_x(
    x: Tensor,
    tracked_mask_index: int,
    control_vec_start: int,
    control_mask_index: int,
    tracked_weight: float,
    control_weight: float,
    control_radius: float,
) -> tuple[Tensor, Tensor]:
    if x.shape[-1] <= max(tracked_mask_index, control_vec_start + 2, control_mask_index):
        return x.new_ones(*x.shape[:2]), torch.zeros(*x.shape[:2], device=x.device, dtype=torch.bool)
    tracked = x[..., tracked_mask_index].clamp(0, 1)
    control_vec = x[..., control_vec_start:control_vec_start + 3]
    control_mask = x[..., control_mask_index].clamp(0, 1)
    dist = control_vec.norm(dim=-1)
    near = torch.exp(-dist / max(float(control_radius), 1.0e-6)) * control_mask
    evidence = 1.0 + float(tracked_weight) * tracked + float(control_weight) * near
    return evidence, tracked > 0.5

def weighted_neighbor_mean(msg: Tensor, idx: Tensor, evidence: Optional[Tensor], protected: Optional[Tensor]) -> Tensor:
    if evidence is None and protected is None:
        return msg.mean(dim=2)
    w = msg.new_ones(*msg.shape[:3], 1) if evidence is None else gather(evidence[..., None], idx).clamp_min(1.0e-6)
    if protected is not None:
        src = gather(protected.to(msg.dtype)[..., None], idx)
        dst = protected[:, :, None, None]
        w = w * torch.where(dst, src, torch.ones_like(src))
    return (msg * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0e-6)


def aggregation_mult(mode: str) -> int:
    return {"mean": 1, "mean_max": 2}.get(str(mode), 1)


def multi_aggregate(msg: Tensor, idx: Tensor, evidence: Optional[Tensor], protected: Optional[Tensor], mode: str = "mean") -> Tensor:
    """Aggregate per-neighbor messages over the K-neighbor dim.

    msg shape: (B, N, K, D). Returns (B, N, D * aggregation_mult(mode)).
      mean     -> evidence/protected weighted mean (existing behavior)
      mean_max -> concat([weighted_mean, max], dim=-1)
    """
    mode = str(mode)
    if mode == "mean":
        return weighted_neighbor_mean(msg, idx, evidence, protected)
    if mode == "mean_max":
        m = weighted_neighbor_mean(msg, idx, evidence, protected)
        mx = msg.max(dim=2).values
        return torch.cat([m, mx], dim=-1)
    raise ValueError(f"unknown aggregation mode: {mode}")


def pool_evidence_to_coarse(evidence: Optional[Tensor], protected: Optional[Tensor], up: Dict[str, Tensor], coarse_n: int) -> tuple[Optional[Tensor], Optional[Tensor]]:
    if evidence is None or protected is None:
        return None, None
    idx = up["idx"][..., 0].expand(evidence.shape[0], -1)
    coarse_e = evidence.new_zeros(evidence.shape[0], coarse_n)
    coarse_p = evidence.new_zeros(evidence.shape[0], coarse_n)
    coarse_e.scatter_reduce_(1, idx, evidence, reduce="amax", include_self=True)
    coarse_p.scatter_reduce_(1, idx, protected.to(evidence.dtype), reduce="amax", include_self=True)
    return coarse_e.clamp_min(1.0), coarse_p > 0.5

class FeatureWindowAggregator(nn.Module):
    def __init__(self, dim, k):
        super().__init__()
        self.dim = int(dim)
        self.k = max(int(k), 1)
        self.output_dim = self.dim * self.k

    def forward(self, x):  # B,N,T,F
        if x.shape[2] > self.k:
            x = x[:, :, -self.k:]
        if x.shape[2] < self.k:
            pad = x.new_zeros(*x.shape[:2], self.k - x.shape[2], self.dim)
            x = torch.cat([pad, x], dim=2)
        return x.reshape(x.shape[0], x.shape[1], -1)

def sample_points(x: Tensor, m: int, mode: str = "stride") -> Tensor:
    b, n, _ = x.shape
    m = max(1, min(int(m), n))
    if mode != "fps":
        return torch.linspace(0, n - 1, m, device=x.device).round().long()[None].expand(b, -1)
    out = torch.zeros(b, m, dtype=torch.long, device=x.device)
    for bi in range(b):
        dist = x.new_full((n,), float("inf"))
        far = torch.zeros((), dtype=torch.long, device=x.device)
        for i in range(m):
            out[bi, i] = far
            d = (x[bi] - x[bi, far]).square().sum(-1)
            dist = torch.minimum(dist, d)
            far = dist.argmax()
    return out


def batched_index(x: Tensor, idx: Tensor) -> Tensor:
    return x[torch.arange(x.shape[0], device=x.device)[:, None], idx]


class GraphUNetSpatial(nn.Module):
    def __init__(self, node_dim: int, hidden: int, cfg) -> None:
        super().__init__()
        self.k = int(cfg.graph_unet_k)
        self.ratios = tuple(float(v) for v in cfg.get("graph_unet_ratios", (1.0, 0.5, 0.25)))
        self.pool = str(cfg.get("graph_unet_pool", "stride"))
        self.edge_feature = str(cfg.get("edge_feature", "canonical_length_only"))
        self.edge_dir = bool(cfg.get("edge_direction", False))
        self.evidence_aggregation = bool(cfg.get("evidence_aggregation", False))
        self.tracked_weight = float(cfg.get("evidence_tracked_weight", 8.0))
        self.control_weight = float(cfg.get("evidence_control_weight", 4.0))
        self.control_radius = float(cfg.get("evidence_control_radius", 0.04))
        self.aggregation = str(cfg.get("aggregation", "mean"))
        can_dim = 6 * int(cfg.get("canonical_bands", 0)) if int(cfg.get("canonical_bands", 0)) > 0 else 3
        td_dim = 6 * int(cfg.get("tracked_bands", 0)) if int(cfg.get("tracked_bands", 0)) > 0 else 3
        self.tracked_mask_index = can_dim + td_dim
        self.control_vec_start = self.tracked_mask_index + 1
        self.control_mask_index = self.control_vec_start + 6
        agg_mult = aggregation_mult(self.aggregation)
        self.nlevels = len(self.ratios)
        self.layers = 2 * self.nlevels - 1
        edge_dim = ({"none": 0, "canonical_length_only": 1}.get(self.edge_feature, 2)) + (3 if self.edge_dir else 0)
        self.node = mlp([node_dim, hidden, hidden])
        self.down_msg = nn.ModuleList([mlp([2 * hidden + edge_dim, hidden, hidden]) for _ in range(self.nlevels)])
        self.down_upd = nn.ModuleList([mlp([(1 + agg_mult) * hidden, hidden, hidden]) for _ in range(self.nlevels)])
        self.up_fuse = nn.ModuleList([mlp([2 * hidden, hidden, hidden]) for _ in range(self.nlevels - 1)])
        self.up_msg = nn.ModuleList([mlp([2 * hidden + edge_dim, hidden, hidden]) for _ in range(self.nlevels - 1)])
        self.up_upd = nn.ModuleList([mlp([(1 + agg_mult) * hidden, hidden, hidden]) for _ in range(self.nlevels - 1)])
        self.norm = nn.LayerNorm(hidden)

    @torch.no_grad()
    def cache(self, canonical: Tensor) -> Dict[str, object]:
        levels, down = [canonical], []
        for ratio in self.ratios[1:]:
            idx = sample_points(levels[-1], round(levels[-1].shape[1] * ratio), self.pool)
            down.append(idx)
            levels.append(batched_index(levels[-1], idx))
        graphs, ups = [], []
        for pts in levels:
            k = min(self.k, pts.shape[1] - 1)
            dist = torch.cdist(pts, pts)
            eye = torch.eye(pts.shape[1], device=pts.device, dtype=torch.bool)[None]
            idx = dist.masked_fill(eye, torch.finfo(dist.dtype).max).topk(k, largest=False).indices
            graphs.append({"idx": idx, "len0": torch.gather(dist, -1, idx).clamp_min(1e-6)})
        for fine, coarse in zip(levels[:-1], levels[1:]):
            d = torch.cdist(fine, coarse).clamp_min(1e-8)
            kk = min(3, coarse.shape[1])
            idx = d.topk(kk, largest=False).indices
            w = 1.0 / torch.gather(d, -1, idx)
            ups.append({"idx": idx, "w": w / w.sum(-1, keepdim=True).clamp_min(1e-8)})
        return {"levels": levels, "down": down, "graphs": graphs, "ups": ups}

    def edge(self, cur: Tensor, prev: Tensor, graph: Dict[str, Tensor]) -> Tensor:
        idx, len0 = graph["idx"].expand(cur.shape[0], -1, -1), graph["len0"].expand(cur.shape[0], -1, -1)
        e = gather(cur, idx) - cur[:, :, None]
        ep = gather(prev, idx) - prev[:, :, None]
        length = e.norm(dim=-1).clamp_min(1e-6)
        if self.edge_feature == "none":
            out = []
        elif self.edge_feature == "canonical_length_only":
            scale = len0.mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            out = [(len0 / scale - 1.0)[..., None]]
        else:
            out = [v[..., None] for v in [length / len0 - 1.0, (length - ep.norm(dim=-1)) / len0]]
        if self.edge_dir:
            out.append(e / length[..., None])
        return cur.new_zeros(cur.shape[0], cur.shape[1], idx.shape[-1], 0) if not out else torch.cat(out, -1)

    def message(self, h: Tensor, edge: Tensor, graph: Dict[str, Tensor], level: int, up: bool = False) -> Tensor:
        idx = graph["idx"].expand(h.shape[0], -1, -1)
        nb = gather(h, idx)
        msg_net = self.up_msg[level] if up else self.down_msg[level]
        upd_net = self.up_upd[level] if up else self.down_upd[level]
        raw = msg_net(torch.cat([h[:, :, None].expand_as(nb), nb, edge], -1))
        msg = multi_aggregate(raw, idx, graph.get("evidence", None), graph.get("protected", None), self.aggregation)
        return h + upd_net(torch.cat([h, msg], -1))

    def begin(self, x: Tensor, cur: Tensor, prev: Tensor, cache: Dict[str, object]) -> Dict[str, object]:
        curs, prevs = [cur], [prev]
        if self.evidence_aggregation:
            evidence0, protected0 = observed_evidence_from_x(
                x,
                self.tracked_mask_index,
                self.control_vec_start,
                self.control_mask_index,
                self.tracked_weight,
                self.control_weight,
                self.control_radius,
            )
            evidences, protected = [evidence0], [protected0]
        else:
            evidences, protected = [None], [None]
        for idx, up in zip(cache["down"], cache["ups"]):
            curs.append(batched_index(curs[-1], idx))
            prevs.append(batched_index(prevs[-1], idx))
            coarse_e, coarse_p = pool_evidence_to_coarse(evidences[-1], protected[-1], up, curs[-1].shape[1])
            evidences.append(coarse_e)
            protected.append(coarse_p)
        graphs = []
        for graph, evidence, protect in zip(cache["graphs"], evidences, protected):
            graphs.append({**graph, "evidence": evidence, "protected": protect})
        return {"hs": [self.node(x)] + [None] * (self.nlevels - 1), "curs": curs, "prevs": prevs, **cache, "graphs": graphs}

    def step(self, ctx: Dict[str, object], layer: int) -> Tensor:
        if layer < self.nlevels:
            level = layer
            if level > 0 and ctx["hs"][level] is None:
                ctx["hs"][level] = batched_index(ctx["hs"][level - 1], ctx["down"][level - 1])
            h = self.message(ctx["hs"][level], self.edge(ctx["curs"][level], ctx["prevs"][level], ctx["graphs"][level]), ctx["graphs"][level], level)
        else:
            target = self.layers - layer - 1
            up = ctx["ups"][target]
            coarse = ctx["hs"][target + 1]
            interp = (gather(coarse, up["idx"].expand(coarse.shape[0], -1, -1)) * up["w"].expand(coarse.shape[0], -1, -1)[..., None]).sum(2)
            h = ctx["hs"][target] + self.up_fuse[target](torch.cat([ctx["hs"][target], interp], -1))
            h = self.message(h, self.edge(ctx["curs"][target], ctx["prevs"][target], ctx["graphs"][target]), ctx["graphs"][target], target, up=True)
        ctx["active_level"] = level if layer < self.nlevels else target
        return h

    def set(self, ctx: Dict[str, object], layer: int, h: Tensor) -> None:
        ctx["hs"][int(ctx["active_level"])] = h

    def finish(self, ctx: Dict[str, object]) -> Tensor:
        return self.norm(ctx["hs"][0])


class Temporal(nn.Module):
    def __init__(self, layers: int, hidden: int, window: int, conv: int, attention_norm: bool = True) -> None:
        super().__init__()
        self.layers, self.hidden, self.window, self.conv = int(layers), int(hidden), int(window), int(conv)
        self.attention_norm = bool(attention_norm)
        kernel = max(1, self.conv)
        self.tconv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, groups=hidden),
                nn.GELU(),
                nn.Conv1d(hidden, hidden, 1),
            )
            for _ in range(layers)
        ])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)]) if self.attention_norm else None
        self.gru = nn.ModuleList([nn.GRUCell(hidden, hidden) for _ in range(layers)])
        self.gru_start = layers // 2

    def detach(self, state: Optional[object]) -> Optional[object]:
        if state is None:
            return None
        if torch.is_tensor(state):
            return state.detach()
        if isinstance(state, list):
            return [v.detach() if torch.is_tensor(v) else v for v in state]
        return {
            k: [None if t is None else t.detach() for t in v] if isinstance(v, list)
            else (v.detach() if torch.is_tensor(v) else v)
            for k, v in state.items()
        }

    def step(self, x: Tensor, state: Optional[object], layer: int) -> tuple[Tensor, Dict[str, Tensor]]:
        hist_len = max(0, self.window - 1)
        fill = min(0 if state is None else int(state["fill"]), hist_len)
        hist = x.new_zeros(x.shape[0], x.shape[1], hist_len, x.shape[2]) if state is None else state["hist"][layer][:, :, -hist_len:]
        tokens = torch.cat([hist[:, :, hist_len - fill:], x[:, :, None]], dim=2)
        flat = tokens.flatten(0, 1)
        z = F.pad(flat.transpose(1, 2), (max(1, self.conv) - 1, 0))
        conv_current = self.tconv[layer](z).transpose(1, 2)[:, -1].view_as(x)
        y = x + conv_current
        gru_state = None
        if layer >= self.gru_start:
            prev = None if state is None else state.get("gru", [None] * self.layers)[layer]
            gru_in = conv_current.flatten(0, 1)
            if prev is None:
                gru_state = self.gru[layer](gru_in).view_as(x)
            else:
                gru_state = self.gru[layer](gru_in, prev.flatten(0, 1)).view_as(x)
                y = y + gru_state
        if self.norm is not None:
            y = self.norm[layer](y)
        return y, {"hist": torch.cat([hist[:, :, 1:], x[:, :, None]], dim=2) if hist_len else hist, "gru": gru_state}


class InterleavedModel(nn.Module):
    def __init__(self, node_dim: int, cfg) -> None:
        super().__init__()
        h = int(cfg.hidden)
        self.spatial = GraphUNetSpatial(node_dim, h, cfg)
        self.temporal = Temporal(
            self.spatial.layers, h, cfg.attention_window, cfg.temporal_conv,
            bool(cfg.get("temporal_attention_norm", True)),
        )

    def cache(self, canonical: Tensor) -> Dict[str, object]:
        cache = self.spatial.cache(canonical)
        cache["canonical"] = canonical
        return cache

    def detach(self, state: Optional[object]) -> Optional[object]:
        return self.temporal.detach(state)

    def forward(self, x: Tensor, cur: Tensor, prev: Tensor, cache: Dict[str, object], state: Optional[object]) -> Dict[str, object]:
        ctx = self.spatial.begin(x, cur, prev, cache)
        layer_states = []
        for i in range(self.spatial.layers):
            h = self.spatial.step(ctx, i)
            h, layer_state = self.temporal.step(h, state, i)
            if hasattr(self.spatial, "set"):
                self.spatial.set(ctx, i, h)
            else:
                ctx["h"] = h
            layer_states.append(layer_state)
        fill = 0 if state is None else int(state["fill"])
        hists = [s["hist"] for s in layer_states]
        grus = [s["gru"] for s in layer_states]
        same_hist = all(t.shape == hists[0].shape for t in hists)
        new_state = {
            "hist": torch.stack(hists) if same_hist else hists,
            "gru": grus,
            "fill": min(fill + 1, max(0, self.temporal.window - 1)),
        }
        latent = self.spatial.finish(ctx)
        return {"latent": latent, "state": new_state}


class Refiner(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.log_E_min, self.log_E_max = float(cfg.log_E_min), float(cfg.log_E_max)
        self.nu_min, self.nu_max = float(cfg.nu_min), float(cfg.nu_max)
        self.max_controls = int(cfg.get("max_controls", 0))
        self.tracked_disp_scale = float(cfg.get("tracked_disp_scale", 1.0))
        self.tracked_mask_scale = float(cfg.get("tracked_mask_scale", 1.0))
        self.control_vec_scale = float(cfg.get("control_vec_scale", 1.0))
        self.control_disp_scale = float(cfg.get("control_disp_scale", 1.0))
        self.canonical_bands = int(cfg.get("canonical_bands", 0))
        self.tracked_bands = int(cfg.get("tracked_bands", 0))
        can_dim = 6 * self.canonical_bands if self.canonical_bands > 0 else 3
        td_dim = 6 * self.tracked_bands if self.tracked_bands > 0 else 3
        ctrl_dim = 7  # weighted mean vec(3) + disp(3) + weight(1)
        node_dim = can_dim + td_dim + 1 + ctrl_dim
        self.agg = FeatureWindowAggregator(node_dim, int(cfg.get("graph_aggregation_chunks", 1)))
        node_dim = self.agg.output_dim
        self.net = InterleavedModel(node_dim, cfg)
        self.dec = mlp([int(cfg.hidden), int(cfg.hidden), int(cfg.hidden), 4])
        nn.init.zeros_(self.dec[-1].weight[2:4])
        nn.init.constant_(self.dec[-1].bias[2:4], float(cfg.get("initial_material_confidence_bias", 0.0)))
        h = int(cfg.hidden)
        self.plasticity_head = nn.Sequential(
            nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1),
        )

    def material_norm(self, material: Tensor) -> Tensor:
        return torch.stack([
            (material[..., 0] - self.log_E_min) / (self.log_E_max - self.log_E_min),
            (material[..., 1] - self.nu_min) / (self.nu_max - self.nu_min),
        ], dim=-1).clamp(0, 1)

    def build_features(self, cur: Tensor, prev: Tensor, canonical: Tensor, material: Tensor, velocity: Tensor, Fm: Tensor,
                       correction: Tensor, mask: Tensor, observed: Optional[Dict[str, Tensor]] = None) -> Tensor:
        if observed is None:
            raise ValueError("observed_control input requires observed features")
        centered = canonical - canonical.mean(dim=-2, keepdim=True)
        td = observed["tracked_disp"] * self.tracked_disp_scale
        parts = [
            fourier(centered, self.canonical_bands) if self.canonical_bands > 0 else centered,
            fourier(td, self.tracked_bands) if self.tracked_bands > 0 else td,
            observed["tracked_mask"][..., None].to(cur.dtype) * self.tracked_mask_scale,
            observed["control_vecs"] * self.control_vec_scale,
            observed["control_disp"] * self.control_disp_scale,
            observed["control_mask"],
        ]
        return torch.cat(parts, dim=-1)

    def cache(self, canonical: Tensor) -> Dict[str, Tensor]:
        return self.net.cache(canonical)

    def detach(self, state: Optional[object]) -> Optional[object]:
        return self.net.detach(state)

    def decode(self, out: Dict[str, object]) -> Dict[str, object]:
        raw = self.dec(out["latent"])
        s = torch.sigmoid(raw)
        mat = torch.stack([
            self.log_E_min + (self.log_E_max - self.log_E_min) * s[..., 0],
            self.nu_min + (self.nu_max - self.nu_min) * s[..., 1],
        ], dim=-1)
        confidence = 1.0 + raw[..., 2:4].exp()
        pooled = out["latent"].max(dim=1).values
        plasticity = torch.sigmoid(self.plasticity_head(pooled))
        return {"material": mat, "material_confidence": confidence, "plasticity": plasticity, "latent": out["latent"], "state": out["state"]}

    def forward_features(
        self,
        x: Tensor,
        cur: Tensor,
        prev: Tensor,
        cache: Dict[str, Tensor],
        state: Optional[object],
    ) -> Dict[str, object]:
        if x.ndim == 4:
            x = self.agg(x)
        else:
            x = self.agg(x[:, :, None])
        return self.decode(self.net(x, cur, prev, cache, state))

    def forward(self, cur: Tensor, prev: Tensor, canonical: Tensor, material: Tensor, velocity: Tensor, Fm: Tensor,
                correction: Tensor, mask: Tensor, cache: Dict[str, Tensor], state: Optional[object],
                observed: Optional[Dict[str, Tensor]] = None) -> Dict[str, object]:
        x = self.build_features(cur, prev, canonical, material, velocity, Fm, correction, mask, observed)
        return self.forward_features(x, cur, prev, cache, state)
