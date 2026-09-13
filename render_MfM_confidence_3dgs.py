"""
Render the MfM confidence field as a dynamic 3DGS video for one episode.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from physcore.model_MfM import Refiner
from physcore.particle_flow import episode_runtime as ert
from physcore.particle_flow.dataset import ParticleFlowEpisodeDataset
from physcore.particle_flow.mfm_artifacts import load_checkpoint, model_state_from_checkpoint
from physcore.particle_flow.mfm_training import defaults, material_guess, observed_features

SH_C0 = 0.28209479177387814  # rendered color = SH_C0 * dc + 0.5


# --- Small helpers ---
def _bbox_center(pts: np.ndarray) -> np.ndarray:
    return 0.5 * (pts.min(axis=0) + pts.max(axis=0))


def _focal2fov(focal: float, pixels: int) -> float:
    return 2 * float(np.arctan(pixels / (2 * focal)))


def load_refiner(checkpoint: str, user_cfg, device):
    """Build a frozen MfM Refiner. The config's `model` block rebuilds the architecture
    and wins over a checkpoint's own `cfg` (same precedence as validate_MfM.py)."""
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    cfg = defaults(OmegaConf.merge(OmegaConf.create(ckpt.get("cfg", {})), user_cfg))
    cfg.model.input_mode = "observed_control"
    model = Refiner(cfg.model).to(device)
    missing, unexpected = model.load_state_dict(model_state_from_checkpoint(ckpt), strict=False)
    if missing or unexpected:
        print(f"  [mfm] missing={list(missing)[:3]} unexpected={list(unexpected)[:3]}", flush=True)
    model.eval()
    return model, cfg


@torch.no_grad()
def extract_confidence_field(model, ep, cfg, predicted_frames, *, full_episode=True, seed=0,
                             warmup_frames=0):
    """Replay validate_MfM.py's MfM schedule (no MPM) -> (confidence (Tp, N, C),
    channels, mid_frame) aligned to predicted_frames.

    full_episode=True refreshes every K-window (refresh-tail); False freezes at the
    pure-tail midpoint. warmup_frames>0 first holds frame start+1 that many steps to
    settle the recurrent state, whose cold transient otherwise brightens the opening
    windows."""
    coords, vel, Fm = ep["coords"], ep["particle_v"], ep["particle_F"]
    device = coords.device
    K = max(int(cfg.train.update_every), 1)
    T, N, _ = coords.shape
    start = 0
    end_frame = start + ((T - 1 - start) // K) * K
    mid_frame = start + ((end_frame - start) // 2 // K) * K
    last = end_frame if full_episode else mid_frame

    canonical = coords[start:start + 1]
    cache = model.cache(canonical)
    state = None
    material = material_guess(ep, cfg, start, seed)
    correction = torch.zeros_like(canonical)
    mask = torch.zeros(canonical.shape[:2], device=device, dtype=torch.bool)

    def observe(frame, cur):
        return observed_features(
            ep, frame, canonical[0], cur[0],
            int(cfg.model.get("max_controls", 0)),
            bool(cfg.model.get("fixed_tracked_mask", False)),
            str(cfg.model.get("tracked_disp_mode", "incremental")),
            str(cfg.model.get("control_disp_mode", "absolute")),
            bool(cfg.model.get("use_persistent_tracks", True)),
            bool(cfg.model.get("persistent_track_use_motion_valid", False)),
            aggregate_controls=True,   # matches every MfM train/validate call site
        )

    # Same window loop pinned to frame start+1; confidence discarded, state carried.
    if int(warmup_frames) > 0:
        cur, prev = coords[start + 1:start + 2], coords[start:start + 1]
        ww = []
        for step in range(1, int(warmup_frames) + 1):
            ww.append(model.build_features(cur, prev, canonical, material,
                                           vel[start + 1:start + 2], Fm[start + 1:start + 2],
                                           correction, mask, observe(start + 1, cur)))
            if step % K:
                continue
            pred = model.forward_features(torch.stack(ww, dim=2), cur, prev, cache, state)
            ww.clear()
            state = pred["state"]
            material = pred["material"].detach()

    fw, conf_windows = [], []
    for frame in range(start + 1, last + 1):
        cur, prev = coords[frame:frame + 1], coords[frame - 1:frame]
        fw.append(model.build_features(cur, prev, canonical, material,
                                       vel[frame:frame + 1], Fm[frame:frame + 1],
                                       correction, mask, observe(frame, cur)))
        if (frame - start) % K:
            continue
        pred = model.forward_features(torch.stack(fw, dim=2), cur, prev, cache, state)
        fw.clear()
        state = pred["state"]
        material = pred["material"].detach()
        conf = pred.get("material_confidence", None)
        if conf is None:
            conf = material.new_ones(material.shape[0], material.shape[1], 1)
        conf_windows.append(conf.detach().squeeze(0).float().cpu())

    if not conf_windows:
        raise RuntimeError("episode too short for the K-window split")
    n_win, Cc = len(conf_windows), conf_windows[0].shape[-1]
    out = torch.empty(len(predicted_frames), N, Cc, dtype=torch.float32)
    for fi, f in enumerate(predicted_frames):
        f = int(f)
        out[fi] = conf_windows[0 if f <= 0 else min((f - 1) // K, n_win - 1)]
    channels = ["log_E", "nu"] if Cc == 2 else [f"conf_{c}" for c in range(Cc)]
    return out, channels, int(mid_frame)


def episode_source_metadata(episode_root: str) -> dict:
    """`source_metadata.json` from convert_to_episode.py (config.yaml as fallback).
    episode_data.pt has no `source_coordinate_transform`, so flip_z comes from here."""
    root = Path(episode_root)
    meta_path = root / "source_metadata.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text())
    cfg_path = root / "config.yaml"
    if cfg_path.exists():
        return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True) or {}
    return {}


def load_trajectory(traj_path: str, episode_root: str, domain_center, flip_z: bool):
    """Load the MPM trajectory and undo the dataset domain-shift + z-flip so it lands
    in the SAME world frame as the Gaussians."""
    t = torch.load(str(traj_path), weights_only=False, map_location="cpu")
    pred = t["predicted"].numpy()
    pred_frames = t["predicted_frames"].numpy()
    ep = torch.load(str(Path(episode_root) / "episode_data.pt"), weights_only=False, map_location="cpu")
    if bool(ep.get("is_real_world", False)):
        shift = (np.asarray(domain_center, np.float32) - _bbox_center(ep["particle_coords"][0].numpy())).astype(np.float32)
    else:
        shift = np.zeros(3, np.float32)
    pred_world = pred - shift
    if flip_z:
        pred_world = pred_world.copy()
        pred_world[..., 2] = -pred_world[..., 2]
    return pred_world, pred_frames


def load_gaussians(gs_dir, iteration, opacity_threshold, GaussianModel, sh_degree=3):
    ply = Path(gs_dir) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not ply.exists():
        raise FileNotFoundError(f"Gaussian PLY not found: {ply}")
    g = GaussianModel(sh_degree)
    g.load_ply(str(ply))
    if g._scaling.shape[-1] == 1:
        g.isotropic = True
    opa = g.get_opacity.squeeze(-1)
    keep = opa > opacity_threshold
    if keep.sum().item() < opa.numel():
        for a in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
            setattr(g, a, getattr(g, a)[keep])
    return g


def filter_background(gaussians, anchors, dist_threshold):
    if dist_threshold is None or dist_threshold <= 0:
        return
    g_xyz = gaussians._xyz.detach()
    keep = torch.zeros(g_xyz.shape[0], dtype=torch.bool, device=g_xyz.device)
    for s in range(0, g_xyz.shape[0], 16384):
        e = min(s + 16384, g_xyz.shape[0])
        keep[s:e] = torch.cdist(g_xyz[s:e], anchors).min(dim=-1).values < dist_threshold
    if int(keep.sum()) == g_xyz.shape[0]:
        return
    print(f"  background filter (dist<{dist_threshold}m): keep {int(keep.sum())}/{g_xyz.shape[0]}", flush=True)
    for a in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation"):
        setattr(gaussians, a, getattr(gaussians, a)[keep])


def camera_from_entry(entry, Camera, K_real=None, device="cuda"):
    position = np.array(entry["position"], dtype=np.float64)
    rotation = np.array(entry["rotation"], dtype=np.float64)
    W, H = int(entry["width"]), int(entry["height"])
    fx, fy = float(entry["fx"]), float(entry["fy"])
    C2W = np.eye(4, dtype=np.float64); C2W[:3, :3] = rotation; C2W[:3, 3] = position
    W2C = np.linalg.inv(C2W)
    return Camera(resolution=(W, H), colmap_id=int(entry["id"]),
                  R=W2C[:3, :3].T, T=W2C[:3, 3],
                  FoVx=_focal2fov(fx, W), FoVy=_focal2fov(fy, H),
                  depth_params=None, image=None, invdepthmap=None,
                  image_name=entry.get("img_name", f"cam{entry['id']}"),
                  uid=int(entry["id"]), data_device=device,
                  K=torch.as_tensor(K_real, dtype=torch.float32) if K_real is not None else None)


def find_key(obj, key: str):
    """First value for `key` anywhere in a nested dict (source_metadata.json keeps
    source_coordinate_transform at the top level, config.yaml nests it)."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for value in obj.values():
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def resolve_rgb_root(meta: dict, explicit) -> Path | None:
    """Root of the original recording, laid out as color/<view>/<frame>.png."""
    if explicit:
        return Path(str(explicit)).expanduser()
    src = meta.get("source_dir", None)
    return Path(str(src)) if src else None


def load_rgb_frames(rgb_root: Path, view: int, n_frames: int) -> np.ndarray | None:
    """(n_frames, H, W, 3) uint8, or None when the view's frames aren't there."""
    view_dir = rgb_root / "color" / str(view)
    paths = [view_dir / f"{f}.png" for f in range(n_frames)]
    if not all(p.exists() for p in paths):
        return None
    return np.stack([imageio.imread(str(p))[..., :3] for p in paths])


def warp_frame(canonical_xyz, canonical_quat, mpm_canon, mpm_t, relations, knn_idx,
               interpolate_motions, chunk=4096):
    n = canonical_xyz.shape[0]
    xyz_t, quat_t = canonical_xyz.clone(), canonical_quat.clone()
    motions = mpm_t - mpm_canon
    n_bones = mpm_canon.shape[0]
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        idx = knn_idx[s:e]
        cx, cq = canonical_xyz[s:e], canonical_quat[s:e]
        w = torch.zeros(e - s, n_bones, device=mpm_canon.device, dtype=torch.float32)
        dist = (cx[:, None, :] - mpm_canon[idx]).norm(dim=-1)
        iw = 1.0 / (dist + 1e-6); iw = iw / iw.sum(dim=-1, keepdim=True)
        w.scatter_(1, idx, iw)
        with contextlib.redirect_stdout(io.StringIO()):
            cxt, cqt, _ = interpolate_motions(bones=mpm_canon, motions=motions, relations=relations,
                                              xyz=cx, quat=cq, weights=w, device=str(mpm_canon.device))
        xyz_t[s:e], quat_t[s:e] = cxt, cqt
    return xyz_t, quat_t


# --- Driver ---
@torch.no_grad()
def main():
    p = argparse.ArgumentParser(
        description="Render the MfM confidence field as a dynamic 3DGS video (and an "
                    "overlay on the original RGB) for ONE episode. Everything needed "
                    "comes from the config; the flags below only override it ad hoc.",
    )
    p.add_argument("--config", default="configs/render_MfM_confidence_3dgs.yaml")
    # Default None → fall back to the config's `render:` block.
    p.add_argument("--checkpoint", default=None, help="override render.checkpoint")
    p.add_argument("--episode-root", default=None, help="override render.episode_root")
    p.add_argument("--gs-dir", default=None, help="override render.gs_dir")
    p.add_argument("--traj-path", default=None, help="override render.traj_path")
    p.add_argument("--out-dir", default=None, help="override render.out_dir")
    p.add_argument("--name", default=None, help="override render.name")
    p.add_argument("--device", default=None, help="override render.device")
    args, overrides = p.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        p.error(f"unrecognized argument(s): {bad}")

    # The config is the source of truth; a CLI flag only wins when passed.
    user_cfg = OmegaConf.load(args.config)
    if overrides:
        user_cfg = OmegaConf.merge(user_cfg, OmegaConf.from_dotlist(overrides))
    r = user_cfg.get("render", {}) or {}

    checkpoint = args.checkpoint or r.get("checkpoint", None)
    episode_root = args.episode_root or r.get("episode_root", None)
    gs_dir = args.gs_dir or r.get("gs_dir", None)
    traj_path = args.traj_path or r.get("traj_path", None)
    out_dir_arg = args.out_dir or r.get("out_dir", None)
    for label, value in (("checkpoint", checkpoint), ("episode_root", episode_root),
                         ("gs_dir", gs_dir), ("traj_path", traj_path), ("out_dir", out_dir_arg)):
        if not value:
            p.error(f"no {label}: set `render.{label}` in the config or pass --{label.replace('_', '-')}")
    device_name = args.device or r.get("device", "cuda")
    mem_fraction = float(r.get("mem_fraction", 0.0))
    iteration = int(r.get("iteration", 10000))
    opacity_threshold = float(r.get("opacity_threshold", 0.01))
    knn_k = int(r.get("knn_k", 16))
    bg_dist_threshold = float(r.get("bg_dist_threshold", 0.03))
    norm_q = float(r.get("norm_q", 0.99))
    norm_hi = r.get("norm_hi", None)
    warp_chunk = int(r.get("warp_chunk", 4096))
    cmap_name = str(r.get("cmap", "viridis"))
    fps = int(r.get("fps", 30))
    video_frames = r.get("video_frames", None)
    midpoint_only = bool(r.get("midpoint_only", False))
    running_max = bool(r.get("running_max", True))
    warmup_frames = int(r.get("warmup_frames", 0))
    save_pngs = bool(r.get("save_pngs", True))
    overlay_on_rgb = bool(r.get("overlay_on_rgb", True))
    overlay_alpha_gain = float(r.get("overlay_alpha_gain", 1.0))
    overlay_bg_fade = float(r.get("overlay_bg_fade", 0.0))
    overlay_front_opacity = float(r.get("overlay_front_opacity", 1.0))
    want_channels = r.get("channels", None)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if mem_fraction > 0 and device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(mem_fraction, device.index or 0)

    # Vendored gaussian_splatting lives at the repo root.
    from gaussian_splatting.gaussian_renderer import GaussianModel, render
    from gaussian_splatting.dynamic_utils import interpolate_motions, get_topk_indices
    from gaussian_splatting.scene.cameras import Camera

    name = args.name or r.get("name", None) or Path(episode_root).parent.name
    meta = episode_source_metadata(episode_root)
    flip_z = r.get("flip_z", None)
    flip_z = bool(find_key(meta, "flip_z_to_z_up")) if flip_z is None else bool(flip_z)

    # --- 1) MfM + episode -> per-frame confidence aligned to the trajectory ---
    model, cfg = load_refiner(checkpoint, user_cfg, device)
    domain_center = cfg.dataset.get("real_world_domain_center", [0.5, 0.5, 0.2])
    pred_world, pred_frames = load_trajectory(traj_path, episode_root, domain_center, flip_z)
    predicted_frames = [int(f) for f in pred_frames]
    ds = ParticleFlowEpisodeDataset(
        [str(episode_root)], cache_size=1,
        real_world_domain_center=domain_center,
        observation_views=cfg.dataset.get("observation_views", "all"),
        observation_view_index=int(cfg.dataset.get("observation_view_index", 0)))
    ep = ert._load_episode_tensors(ds[0], device)
    confidence, channels, mid = extract_confidence_field(
        model, ep, cfg, predicted_frames, full_episode=not midpoint_only,
        seed=int(cfg.train.get("seed", 0)), warmup_frames=warmup_frames)
    if running_max:
        # Per-particle high-water mark, so a later dip (occlusion, a dropped track)
        # never un-accumulates the color. Before normalization, so the printed span
        # describes what is drawn.
        confidence = confidence.cummax(dim=0).values
    print(f"[{name}] flip_z={flip_z} running_max={running_max} warmup_frames={warmup_frames} "
          f"confidence {tuple(confidence.shape)} "
          f"channels={channels} range=[{float(confidence.min()):.3f},{float(confidence.max()):.3f}] "
          f"mid={mid}", flush=True)

    # --- 2) gaussians + canonical kNN (warp + confidence interpolation) ---
    mpm = torch.from_numpy(np.ascontiguousarray(pred_world)).to(device=device, dtype=torch.float32)
    if confidence.shape[0] != mpm.shape[0] or confidence.shape[1] != mpm.shape[1]:
        raise RuntimeError(f"confidence {tuple(confidence.shape)} vs trajectory {tuple(mpm.shape)} mismatch "
                           "(episode / trajectory particle count differ)")
    gaussians = load_gaussians(gs_dir, iteration, opacity_threshold, GaussianModel)
    filter_background(gaussians, mpm[0], bg_dist_threshold)
    canonical_xyz = gaussians.get_xyz.detach().clone()
    canonical_quat = gaussians.get_rotation.detach().clone()
    n_g = canonical_xyz.shape[0]
    knn_idx = torch.empty((n_g, knn_k), dtype=torch.long, device=device)
    knn_w = torch.empty((n_g, knn_k), dtype=torch.float32, device=device)
    for s in range(0, n_g, 8192):
        e = min(s + 8192, n_g)
        dk, ik = torch.cdist(canonical_xyz[s:e], mpm[0]).topk(knn_k, dim=-1, largest=False)
        knn_idx[s:e] = ik
        w = 1.0 / (dk + 1e-6); knn_w[s:e] = w / w.sum(dim=-1, keepdim=True)
    relations = get_topk_indices(mpm[0], K=knn_k)

    # --- cameras (train views from cameras.json; real K optional) ---
    # cameras.json has no principal point, so without real K gsplat centers cx/cy and the
    # render shifts vs the RGB. Prefer the episode's camera_intrinsics (same order as
    # cam_entries); render.metadata_json overrides.
    cam_entries = [c for c in json.loads((Path(gs_dir) / "cameras.json").read_text())
                   if not c["img_name"].startswith("test_")]
    real_K = None
    metadata_json = r.get("metadata_json", None)
    if metadata_json and Path(str(metadata_json)).exists():
        Ks = json.loads(Path(str(metadata_json)).read_text()).get("intrinsics", [])
        real_K = [np.array(K, np.float32) for K in Ks]
        if len(real_K) != len(cam_entries):
            real_K = None
    if real_K is None:
        ep_raw = torch.load(str(Path(episode_root) / "episode_data.pt"),
                            weights_only=False, map_location="cpu")
        Ks = ep_raw.get("camera_intrinsics")
        if Ks is not None:
            Ks = np.asarray(Ks, dtype=np.float32)
            if Ks.ndim == 3 and Ks.shape[0] == len(cam_entries):
                real_K = [Ks[i] for i in range(len(cam_entries))]
                print(f"[{name}] using real intrinsics from episode camera_intrinsics "
                      f"(cx,cy per view: {', '.join(f'({K[0,2]:.1f},{K[1,2]:.1f})' for K in real_K)})",
                      flush=True)
    if real_K is None:
        print(f"[{name}] WARNING: no real intrinsics found — rendering with CENTERED "
              "principal point; overlay may be shifted vs the RGB.", flush=True)
    cameras = [camera_from_entry(e, Camera, K_real=(real_K[i] if real_K else None), device=device_name)
               for i, e in enumerate(cam_entries)]
    cam_names = [e["img_name"] for e in cam_entries]

    # --- original RGB frames for the overlay (one stack per camera, or None) ---
    rgb_stacks = [None] * len(cameras)
    if overlay_on_rgb:
        rgb_root = resolve_rgb_root(meta, r.get("rgb_root", None))
        if rgb_root is None or not rgb_root.exists():
            print(f"[{name}] WARNING: overlay_on_rgb set but no RGB root ({rgb_root}) — "
                  "writing the confidence render only.", flush=True)
        else:
            n_src = int(meta.get("num_frames", 0) or 0)
            if n_src and n_src != int(ep["coords"].shape[0]):
                print(f"[{name}] WARNING: source has {n_src} frames but the episode has "
                      f"{int(ep['coords'].shape[0])} — overlay frame indices may be off.", flush=True)
            for ci in range(len(cameras)):
                rgb_stacks[ci] = load_rgb_frames(rgb_root, ci, int(ep["coords"].shape[0]))
                if rgb_stacks[ci] is None:
                    print(f"[{name}] WARNING: no complete RGB stack for view {ci} under "
                          f"{rgb_root / 'color' / str(ci)} — that view gets the render only.", flush=True)
            found = sum(s is not None for s in rgb_stacks)
            print(f"[{name}] overlay RGB from {rgb_root} ({found}/{len(cameras)} views)", flush=True)

    # --- color LUT + per-channel normalization ([1.0, pinned or quantile ceiling]) ---
    want = [str(c).strip() for c in want_channels] if want_channels else channels
    chan_idx = [channels.index(c) for c in want if c in channels]
    if not chan_idx:
        raise ValueError(f"render.channels={want} matches none of {channels}")
    lut = torch.tensor(matplotlib.colormaps[cmap_name](np.linspace(0, 1, 256))[:, :3], dtype=torch.float32, device=device)
    conf_dev = confidence.to(device=device, dtype=torch.float32)
    # `lo` is the decoder's exact floor (confidence = 1 + exp(.)), not a data statistic.
    # `hi` is this run's norm_q quantile unless render.norm_hi pins it; pin it to compare
    # runs, since the quantile moves with the checkpoint.
    norm = {}
    print(f"[{name}] confidence over all {conf_dev.shape[0]} frames "
          f"x {conf_dev.shape[1]} particles:", flush=True)
    for c in chan_idx:
        flat = conf_dev[..., c].reshape(-1)
        q = float(torch.quantile(flat, norm_q))
        lo = 1.0
        if norm_hi is not None:
            hi = float(norm_hi)
            if hi <= lo:
                raise ValueError(f"render.norm_hi={hi} must exceed the confidence floor {lo}")
            source = "pinned"
        else:
            hi = q if q > lo else float(flat.max()) + 1e-3
            source = f"q{norm_q:g}"
        if hi <= lo:
            raise ValueError(f"degenerate color range for {channels[c]}: [{lo}, {hi}]")
        norm[c] = (lo, hi)
        clipped = float((flat > hi).float().mean()) * 100.0
        print(f"  {channels[c]:<6} min={float(flat.min()):.6g}  max={float(flat.max()):.6g}  "
              f"q{norm_q:g}={q:.6g}   ->  color [{lo:.6g},{hi:.6g}] ({source}), "
              f"{clipped:.2f}% saturated", flush=True)
    print(f"[{name}] cmap={cmap_name} norm={{ {', '.join(f'{channels[c]}:[{norm[c][0]:.6g},{norm[c][1]:.6g}]' for c in chan_idx)} }}", flush=True)

    # --- 3) render ---
    # render.out_dir IS the run dir: videos directly in it, PNGs under
    # conf_<channel>/<cam index>/<frame>.png, overlays under overlay/ and with an
    # `_overlay` suffix. `conf_` = the decoder's precision head, not the material.
    out_dir = Path(out_dir_arg); out_dir.mkdir(parents=True, exist_ok=True)
    png_dirs, overlay_png_dirs = {}, {}
    for c in chan_idx:
        for ci in range(len(cameras)):
            if save_pngs:
                d = out_dir / f"conf_{channels[c]}" / str(ci)
                d.mkdir(parents=True, exist_ok=True)
                png_dirs[(ci, c)] = d
                if rgb_stacks[ci] is not None:
                    od = out_dir / "overlay" / f"conf_{channels[c]}" / str(ci)
                    od.mkdir(parents=True, exist_ok=True)
                    overlay_png_dirs[(ci, c)] = od
    bg = torch.zeros(3, dtype=torch.float32, device=device)
    pipe = type("P", (), dict(debug=False, compute_cov3D_python=False, convert_SHs_python=False, antialiasing=False))()
    cxyz_buf, cquat_buf = gaussians._xyz.detach().clone(), gaussians._rotation.detach().clone()
    zero_rest = torch.zeros_like(gaussians._features_rest)

    def writer(path):
        return imageio.get_writer(str(path), fps=fps, codec="libx264", quality=7, macro_block_size=1)

    # video_frames caps the whole render, mp4s and PNGs alike; null renders every frame.
    n_total = len(predicted_frames)
    n_render = n_total if video_frames is None else max(min(int(video_frames), n_total), 1)
    writers = {(ci, c): writer(out_dir / f"{name}_{cam_names[ci]}_conf_{channels[c]}.mp4")
               for ci in range(len(cameras)) for c in chan_idx}
    overlay_writers = {(ci, c): writer(out_dir / f"{name}_{cam_names[ci]}_conf_{channels[c]}_overlay.mp4")
                       for ci in range(len(cameras)) for c in chan_idx if rgb_stacks[ci] is not None}
    capped = "" if n_render == n_total else f" (capped from {n_total})"
    print(f"[{name}] rendering {n_render} frames × {len(cameras)} cams × "
          f"{len(chan_idx)} channels -> {out_dir}{capped}", flush=True)
    try:
        for fi, src_frame in enumerate(predicted_frames[:n_render]):
            if fi == 0:
                gaussians._xyz, gaussians._rotation = cxyz_buf, cquat_buf
            else:
                xyz_t, quat_t = warp_frame(canonical_xyz, canonical_quat, mpm[0], mpm[fi],
                                           relations, knn_idx, interpolate_motions, chunk=warp_chunk)
                gaussians._xyz = xyz_t
                gaussians._rotation = torch.nn.functional.normalize(quat_t, dim=-1)
            g_conf = (conf_dev[fi][knn_idx] * knn_w.unsqueeze(-1)).sum(dim=1)   # (n_g, C)
            for c in chan_idx:
                lo, hi = norm[c]
                normed = ((g_conf[:, c] - lo) / (hi - lo + 1e-8)).clamp(0, 1)
                gaussians._features_dc = ((lut[(normed * 255).long().clamp(0, 255)] - 0.5) / SH_C0).unsqueeze(1)
                gaussians._features_rest = zero_rest
                for ci, camera in enumerate(cameras):
                    # gsplat returns RGBA; channel 3 is the coverage mask for compositing.
                    rgba = render(camera, gaussians, pipe, bg, use_gsplat=True)["render"].clamp(0, 1).cpu().numpy()
                    rgb = np.transpose(rgba[:3], (1, 2, 0))
                    img = (rgb * 255).astype(np.uint8)
                    h, w = img.shape[:2]
                    ch, cw = (h // 2) * 2, (w // 2) * 2   # libx264 needs even dimensions
                    frame = img[:ch, :cw]
                    writers[(ci, c)].append_data(frame)
                    if save_pngs:
                        imageio.imwrite(str(png_dirs[(ci, c)] / f"{fi}.png"), frame)
                    if rgb_stacks[ci] is None:
                        continue
                    alpha = (np.clip(rgba[3] * overlay_alpha_gain, 0.0, 1.0))[..., None]
                    real = rgb_stacks[ci][int(src_frame)].astype(np.float32) / 255.0
                    # bg_fade whitens the background where the splat is absent;
                    # front_opacity makes the splat see-through. Both default to off.
                    backdrop = real
                    if overlay_bg_fade > 0.0:
                        fade = overlay_bg_fade * (1.0 - alpha)
                        backdrop = real * (1.0 - fade) + fade
                    a = alpha * overlay_front_opacity
                    over = (a * rgb + (1.0 - a) * backdrop)
                    over = (np.clip(over, 0, 1) * 255).astype(np.uint8)[:ch, :cw]
                    overlay_writers[(ci, c)].append_data(over)
                    if save_pngs:
                        imageio.imwrite(str(overlay_png_dirs[(ci, c)] / f"{fi}.png"), over)
    finally:
        gaussians._xyz, gaussians._rotation = cxyz_buf, cquat_buf
        for wr in list(writers.values()) + list(overlay_writers.values()):
            wr.close()
    for ci in range(len(cameras)):
        for c in chan_idx:
            suffix = " (+_overlay.mp4)" if rgb_stacks[ci] is not None else ""
            pngs = f" + {n_render} pngs" if save_pngs else ""
            print(f"  wrote {out_dir}/{name}_{cam_names[ci]}_conf_{channels[c]}.mp4{suffix} "
                  f"({n_render} frames{pngs})", flush=True)


if __name__ == "__main__":
    main()
