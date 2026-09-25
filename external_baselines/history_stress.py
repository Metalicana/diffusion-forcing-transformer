"""Prespecified DFoT history pressure, gap, and natural-pose-revisit stress tests."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from external_baselines import native_history_transfer as pilot
from external_baselines.common import digest
from external_baselines.history_selection import POLICIES, camera_to_c2w, load_scorer

REGIMES = ("continuation_h16", "continuation_h64", "gap32_h64", "natural_revisit_h64")
REVISIT_RULE = dict(history=64, history_stride=2, cutoff_to_target_frames=16,
    minimum_match_age_frames=64, maximum_rotation_deg=5.0,
    maximum_position_fraction=0.05, excursion_rotation_deg=20.0,
    excursion_position_fraction=0.2, target_frames=4, target_stride=2,
    scan_stride=2, position_scale="diagonal of history camera-center bounding box")


def fixed_plan(regime, start):
    if regime not in REGIMES[:3]:
        raise ValueError("Unknown fixed stress regime")
    count = 16 if regime == "continuation_h16" else 64
    cutoff = int(start) + 126
    gap = 32 if regime == "gap32_h64" else 2
    history = list(range(cutoff - 2 * (count - 1), cutoff + 1, 2))
    targets = list(range(cutoff + gap, cutoff + gap + 8, 2))
    return dict(history=history, targets=targets, cutoff_to_target_frames=gap)


def pose_differences(a, b):
    position = np.linalg.norm(a[:, None, :3, 3] - b[None, :, :3, 3], axis=-1)
    cosine = (np.einsum("aij,bij->ab", a[:, :3, :3], b[:, :3, :3]) - 1) / 2
    rotation = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return position, rotation


def find_revisit(conditions):
    """Earliest eligible return; inspect only cameras, never generated or target RGB."""
    poses = camera_to_c2w(conditions)
    for query in range(142, len(poses) - 6, 2):
        history = list(range(query - 142, query - 14, 2))
        targets = list(range(query, query + 8, 2))
        old = [i for i in history if i <= query - 64]
        scale = float(np.linalg.norm(np.ptp(poses[history, :3, 3], axis=0)))
        distance, angle = pose_differences(poses[targets], poses[old])
        valid = (distance <= max(scale * 0.05, 1e-8)) & (angle <= 5.0)
        if not valid.any(axis=1).all():
            continue
        cost = np.where(valid, distance / max(scale, 1e-8) + angle / 180, np.inf)
        matches = np.argmin(cost, axis=1)
        first_match = old[int(matches[0])]
        away = [i for i in history if i > first_match]
        away_distance, away_angle = pose_differences(poses[[first_match]], poses[away])
        if not ((away_distance > max(scale * 0.2, 1e-8)) | (away_angle > 20)).any():
            continue
        return dict(history=history, targets=targets, cutoff_to_target_frames=16,
            matched_history=[old[int(i)] for i in matches], position_scale=scale,
            match_position=[float(distance[j, i]) for j, i in enumerate(matches)],
            match_rotation_deg=[float(angle[j, i]) for j, i in enumerate(matches)],
            excursion_position=float(away_distance.max()), excursion_rotation_deg=float(away_angle.max()))
    return None


def cache_case(dataset, metadata, plan, output, regime, seed_index):
    history, targets = plan["history"], plan["targets"]
    indices = history + targets
    if (len(targets) != 4 or len(history) not in (16, 64)
            or any(not isinstance(i, int) for i in indices) or indices[0] < 0
            or sorted(set(indices)) != indices or indices[-1] >= dataset.video_length(metadata)):
        raise ValueError("Invalid or non-causal stress-test frame mapping")
    scene = Path(metadata["video_paths"]).stem
    source = dataset.video_path_to_preprocessed_path(Path(metadata["video_paths"]))
    poses = dataset.save_dir / "test_poses" / f"{scene}.pt"
    case = dict(index=seed_index, scene=f"{scene}__{regime}", source_scene=scene,
        regime=regime, history_count=len(history), frame_indices=indices, plan=plan,
        source=str(source.resolve()), source_sha256=digest(source),
        poses=str(poses.resolve()), poses_sha256=digest(poses))
    directory = output / "cases" / case["scene"]
    directory.mkdir(parents=True, exist_ok=True)
    pilot.freeze(directory / "input.json", case)
    payload, receipt = directory / "observations.npz", directory / "observations.json"
    if receipt.exists():
        if not payload.exists() or json.loads(receipt.read_text())["sha256"] != digest(payload):
            raise ValueError(f"Changed prepared observations: {payload}")
    else:
        start, end = indices[0], indices[-1] + 1
        video, raw = dataset.load_video_and_cond(metadata, start, end)
        if len(video) != end - start or len(raw) != len(video):
            raise ValueError(f"Native decoder/pose length mismatch: {scene}")
        relative = np.asarray(indices) - start
        images = dataset.transform(video[relative]).cpu().numpy().astype(np.float32)
        conditions = dataset._process_external_cond(raw, frame_skip=1)[relative].cpu().numpy().astype(np.float32)
        if (images.shape != (len(indices), 3, 256, 256) or not np.isfinite(images).all()
                or images.min() < 0 or images.max() > 1):
            raise ValueError(f"Invalid native images: {scene}")
        camera_to_c2w(conditions)
        np.savez_compressed(payload, images=images, conditions=conditions)
        pilot.save(receipt, dict(sha256=digest(payload)))
    case.update(directory=str(directory), observations_sha256=digest(payload))
    print(f"PREPARED {scene} {regime}: {len(history)} history / 4 targets", flush=True)
    return case


def prepare_stress_cases(dataset, args):
    if len(dataset) != args.scenes:
        raise ValueError("Insufficient fixed-cohort videos")
    previous = json.loads((args.pilot_root / "cohort.json").read_text())
    fixed, cases = [], []
    for i in range(args.scenes):
        video_index, start = dataset.get_clip_location(i)
        metadata = dataset.metadata[video_index]
        scene = Path(metadata["video_paths"]).stem
        if i < len(previous) and (previous[i]["scene"] != scene or previous[i]["start_frame"] != start):
            raise ValueError("Native cohort no longer matches the completed pilot prefix")
        fixed.append((i, metadata, int(start)))
    # Finish complete matched regimes early if the allocation runs out.
    for regime in REGIMES[:3]:
        for i, metadata, start in fixed:
            cases.append(cache_case(dataset, metadata, fixed_plan(regime, start), args.output, regime, i))

    audit, eligible = [], []
    metadata_order = sorted(dataset.metadata, key=lambda m: hashlib.sha256(
        f"{args.seed}:{Path(m['video_paths']).stem}".encode()).hexdigest())
    for metadata in metadata_order:
        scene = Path(metadata["video_paths"]).stem
        path = dataset.save_dir / "test_poses" / f"{scene}.pt"
        row = dict(scene=scene, pose_sha256="", frames=dataset.video_length(metadata),
                   status="", reason="", first_target="", selected=False)
        try:
            row["pose_sha256"] = digest(path)
            raw = dataset.load_cond(metadata, 0, row["frames"])
            if len(raw) != row["frames"]:
                raise ValueError("Pose/video length mismatch")
            conditions = dataset._process_external_cond(raw, frame_skip=1).cpu().numpy()
            plan = find_revisit(conditions)
            row["status"] = "eligible" if plan is not None else "no_pose_revisit"
            if plan is not None:
                row["first_target"] = plan["targets"][0]
                if len(eligible) < args.revisit_scenes:
                    row["selected"] = True
                    eligible.append((metadata, plan))
        except (ValueError, OSError, RuntimeError) as exc:
            row.update(status="invalid_input", reason=str(exc))
        audit.append(row)
        if len(audit) % 50 == 0:
            print(f"POSE AUDIT {len(audit)}/{len(metadata_order)}: "
                  f"{sum(r['status'] == 'eligible' for r in audit)} eligible clips", flush=True)
    pilot.freeze(args.output / "revisit_audit.json", audit)
    pilot.write_csv(args.output / "revisit_audit.csv", audit)
    coverage = dict(fixed_clips=args.scenes, requested_revisit_clips=args.revisit_scenes,
        scanned_clips=len(audit), eligible_revisit_clips=sum(r["status"] == "eligible" for r in audit),
        selected_revisit_clips=len(eligible), invalid_clips=sum(r["status"] == "invalid_input" for r in audit),
        rule=REVISIT_RULE, selection="first eligible by SHA256(seed:scene); earliest qualified return per clip")
    pilot.freeze(args.output / "coverage.json", coverage)
    print(f"POSE AUDIT: {coverage}", flush=True)
    if coverage["invalid_clips"]:
        raise ValueError("Pose audit found invalid inputs; inspect revisit_audit.csv before inference")
    for i, (metadata, plan) in enumerate(eligible):
        cases.append(cache_case(dataset, metadata, plan, args.output, REGIMES[3], 1000 + i))
    pilot.freeze(args.output / "cohort.json", cases)
    return cases


def export_regimes(output, cases, results, complete):
    scores, coverage = [], []
    for regime in REGIMES:
        group = [c for c in cases if c["regime"] == regime]
        done = sum((c["scene"], p) in results for c in group for p in POLICIES)
        ready = bool(group) and done == len(group) * len(POLICIES)
        coverage.append(dict(regime=regime, videos=len(group), completed_cells=done,
                             planned_cells=len(group) * 3, status="complete" if ready else "incomplete" if group else "unavailable"))
        if not group:
            continue
        directory = output / "regimes" / regime
        directory.mkdir(parents=True, exist_ok=True)
        if ready:
            announce = not (directory / "scores.csv").exists()
            if announce:
                print(f"REGIME COMPLETE: {regime}", flush=True)
            pilot.export(directory, group, results, True, announce=announce)
            with (directory / "scores.csv").open() as handle:
                scores.extend(dict(regime=regime, **row) for row in csv.DictReader(handle))
        else:
            pilot.export(directory, group, results, False)
    pilot.write_csv(output / "regime_status.csv", coverage)
    if complete and any(row["status"] == "incomplete" for row in coverage):
        raise ValueError("Missing paired cells in stress results")
    pilot.write_csv(output / ("scores.csv" if complete else "completed_regime_scores.csv"), scores)


def execute(args, status):
    from omegaconf import OmegaConf
    from datasets.video.realestate10k_mini import RealEstate10KMiniAdvancedVideoDataset
    if not (args.data / "metadata/test.pt").exists():
        raise FileNotFoundError("Run the Mini pilot first; stress testing reuses its data, no installation needed")
    cfg = pilot.configuration(args)
    scorer, scorer_path = load_scorer(args.memcam_root)
    code = pilot.code_identity(scorer_path)
    config = dict(protocol="observed_history_stress_v1", scenes=args.scenes, seed=args.seed,
        revisit_scenes=args.revisit_scenes, revisit_rule=REVISIT_RULE, regimes=list(REGIMES),
        policies=list(POLICIES), budget=4, future=4, history_stride=2,
        fixed_rule="original upstream seed-0 cohort; cutoff=start+126; gap2 or gap32",
        pilot_cohort_sha256=digest(args.pilot_root / "cohort.json"),
        interpretation="observed-history conditioning only; real revisits are pose-selected, not output-selected",
        code=code, config=OmegaConf.to_container(cfg, resolve=True))
    pilot.freeze(args.output / "config.json", config)
    pilot.freeze(args.output / "encoder.json", json.loads((args.pilot_root / "encoder.json").read_text()))
    dataset = RealEstate10KMiniAdvancedVideoDataset(cfg.dataset, split="validation")
    cases = prepare_stress_cases(dataset, args)
    del dataset
    status["planned_cells"] = 3 * len(cases)
    pilot.save(args.output / "status.json", status)
    if args.prepare_only:
        status.update(status="prepared", cases=len(cases))
        return
    pilot.evaluate_cases(args, status, cfg, cases, config, scorer, scorer_path, code, exporter=export_regimes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=pilot.ROOT / "data/real-estate-10k-mini")
    parser.add_argument("--memcam-root", type=Path, default=pilot.ROOT.parent / "MemCam")
    parser.add_argument("--pilot-root", type=Path, default=Path.home() / "memcam_results/dfot_history_transfer_mini_n5")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scenes", type=int, default=10)
    parser.add_argument("--revisit-scenes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (not 5 <= args.scenes <= 500 or not 0 <= args.revisit_scenes <= 500
            or args.max_seconds < 1 or not 0 <= args.seed < 2**32 - 1500):
        parser.error("Require 5..500 clips, 0..500 revisit clips, a positive time budget and valid seed")
    for name in ("output", "data", "memcam_root", "pilot_root"):
        setattr(args, name, getattr(args, name).resolve())
    args.started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(status="running", job=os.environ.get("SLURM_JOB_ID"))
        pilot.save(args.output / "status.json", status)
        try:
            execute(args, status)
        except BaseException as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            status["elapsed_seconds"] = time.monotonic() - args.started
            pilot.save(args.output / "status.json", status)
        print(f"{status['status'].upper()}: {args.output}", flush=True)
        if status["status"] == "incomplete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
