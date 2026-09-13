"""
Drive smooth-field augmentation from one self-contained YAML config.

Output layout:
  <output_root>/<source_name>_<group_name>/episode_NNNN/episode_data.pt
  <viz_root>/<source_name>_<group_name>/episode_NNNN.mp4
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


PY = sys.executable


def discover_sources(sources_root: Path) -> List[Path]:
    """Return every '<source_name>/episode_0000/' with an episode_data.pt."""
    if not sources_root.exists():
        return []
    out = []
    for sub in sorted(sources_root.iterdir()):
        if not sub.is_dir() or sub.name.startswith(("_", ".")):
            continue
        ep0 = sub / "episode_0000"
        if (ep0 / "episode_data.pt").is_file():
            out.append(ep0)
    return out


def _require_keys(section: Dict, keys: List[str], label: str) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(missing)}")


def validate_recipe(recipe: Dict, config_path: Path) -> None:
    _require_keys(recipe, ["sources_root", "output_root", "viz_root", "groups"], str(config_path))
    if not isinstance(recipe["groups"], list) or not recipe["groups"]:
        raise ValueError(f"{config_path}: groups must be a non-empty list")
    for idx, group in enumerate(recipe["groups"]):
        label = f"group #{idx} ({group.get('name', '<unnamed>')})"
        _require_keys(group, ["name", "mode", "num_episodes", "seed_base", "dataset", "rollout", "material"], label)
        _require_keys(group["material"], ["log_E", "nu", "noise"], f"{label}.material")
        for prop in ("log_E", "nu"):
            _require_keys(group["material"][prop], ["mean_range", "std_range", "clamp"], f"{label}.material.{prop}")
        _require_keys(
            group["material"]["noise"],
            ["freq_range", "octaves", "persistence", "lacunarity"],
            f"{label}.material.noise",
        )


def run_augmentation(group: Dict, source_ep: Path, aug_dir: Path, device: str, log_fh, bar) -> bool:
    n = int(group["num_episodes"])
    pending = [
        i
        for i in range(n)
        if not (aug_dir / f"episode_{i:04d}" / "episode_data.pt").is_file()
    ]
    bar.update(n - len(pending))  # count already-present episodes toward the global bar
    if not pending:
        bar.write(f"  SKIP-AUG (all {n} present)")
        return True
    if len(pending) != n:
        tail = pending[-3:] if len(pending) > 6 else ""
        bar.write(f"  RESUME-AUG ({len(pending)} of {n} missing: {pending[:3]}...{tail})")
    try:
        from physcore.particle_flow.augmentation import generate_smoothfield_episodes

        # bar lives in main() (built before any redirect) so it stays on the
        # console while the per-episode logs go to log_fh.
        with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
            generate_smoothfield_episodes(source_ep, aug_dir, group, pending, device, progress=bar.update)
    except Exception:
        traceback.print_exc(file=log_fh)
        bar.write("  AUG-FAIL - see log")
        return False
    return True


def run_viz(aug_dir: Path, vid_dir: Path, num_episodes: int, log_fh) -> bool:
    vid_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        f"episode_{i:04d}"
        for i in range(num_episodes)
        if (aug_dir / f"episode_{i:04d}" / "episode_data.pt").is_file()
        and not (vid_dir / f"episode_{i:04d}.mp4").is_file()
    ]
    if not pending:
        return True
    cmd = [
        PY,
        "datagen/augment/viz_episodes.py",
        "--root",
        str(aug_dir),
        "--episodes",
        *pending,
        "--out_dir",
        str(vid_dir),
    ]
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=log_fh, stderr=log_fh)
    if r.returncode != 0:
        print(f"  VIZ-FAIL (exit {r.returncode}) - see log", flush=True)
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/augment_recipe.yaml")
    ap.add_argument("--groups", nargs="*", default=None, help="optional whitelist of group names; default = all")
    ap.add_argument("--sources", nargs="*", default=None, help="optional whitelist of source names; default = all")
    ap.add_argument("--no-viz", action="store_true", help="skip viz_episodes even if config sets viz: true")
    ap.add_argument("--dry-run", action="store_true", help="validate config and print planned source/group work")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    config_path = (ROOT / args.config) if not Path(args.config).is_absolute() else Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        recipe = yaml.safe_load(f)
    validate_recipe(recipe, config_path)

    sources_root = (ROOT / recipe["sources_root"]).resolve()
    output_root = (ROOT / recipe["output_root"]).resolve()
    viz_root = (ROOT / recipe["viz_root"]).resolve()
    device = str(recipe.get("device", "cuda"))
    do_viz = (not args.no_viz) and bool(recipe.get("viz", False))

    sources = discover_sources(sources_root)
    if args.sources:
        wanted_sources = set(args.sources)
        sources = [s for s in sources if s.parent.name in wanted_sources]
    if not sources and not args.dry_run:
        raise SystemExit(f"no source episodes under {sources_root}")

    groups = recipe["groups"]
    if args.groups:
        wanted_groups = set(args.groups)
        groups = [g for g in groups if g["name"] in wanted_groups]
    if not groups:
        raise SystemExit(f"no matching groups in {config_path}")

    print(f"config: {config_path}")
    print(f"sources ({len(sources)}): {[s.parent.name for s in sources]}")
    print(f"groups ({len(groups)}): {[g['name'] for g in groups]}")
    print(f"output: {output_root}")
    print(f"device: {device}")
    print()

    if args.dry_run:
        total = len(sources) * len(groups)
        print(f"dry-run: planned source/group jobs={total}, viz={do_viz}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "_recipe_run.log"
    log_fh = log_path.open("w", encoding="utf-8")
    print(f"log: {log_path}")
    print()

    from tqdm.auto import tqdm

    total = 0
    aug_ok = 0
    aug_fail = 0
    viz_ok = 0
    viz_fail = 0
    t0 = time.perf_counter()
    # One global bar over every episode across all source x group jobs.
    total_episodes = len(sources) * sum(int(g["num_episodes"]) for g in groups)
    bar = tqdm(total=total_episodes, desc="augment", unit="ep", dynamic_ncols=True)
    try:
        for source_ep in sources:
            source_name = source_ep.parent.name
            for group in groups:
                total += 1
                tag = f"{source_name} x {group['name']}"
                bar.set_description(tag)
                log_fh.write(f"\n\n=== {tag} ===\n")
                log_fh.flush()
                aug_dir = output_root / f"{source_name}_{group['name']}"
                ok = run_augmentation(group, source_ep, aug_dir, device, log_fh, bar)
                if ok:
                    aug_ok += 1
                    if do_viz:
                        vid_dir = viz_root / f"{source_name}_{group['name']}"
                        if run_viz(aug_dir, vid_dir, int(group["num_episodes"]), log_fh):
                            viz_ok += 1
                        else:
                            viz_fail += 1
                else:
                    aug_fail += 1
    finally:
        bar.close()
        log_fh.close()

    print(
        f"\ndone - total={total} aug_ok={aug_ok} aug_fail={aug_fail} "
        f"viz_ok={viz_ok} viz_fail={viz_fail} wall={time.perf_counter() - t0:.1f}s"
    )


if __name__ == "__main__":
    main()
