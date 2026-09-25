"""Frozen DFoT / RealEstate10K Mini observed-history transfer, not long-rollout FVD."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from external_baselines.common import digest
from external_baselines.history_selection import (
    POLICIES, choose_history, conditioning_indices, load_scorer,
)

ROOT = Path(__file__).resolve().parents[1]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def identity(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def freeze(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError(f"Frozen inputs/config changed: {path}. Use a new output directory.")
    else:
        save(path, data)


def write_csv(path, rows):
    if not rows:
        return
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def configuration(args):
    from hydra import compose, initialize_config_dir
    overrides = ["dataset=realestate10k_mini", "algorithm=dfot_video_pose",
                 "dataset.context_length=4", "dataset.frame_skip=10", "dataset.n_frames=20",
                 f"dataset.num_eval_videos={args.scenes}", f"dataset.save_dir={args.data}",
                 "algorithm.compile=false", "algorithm.logging.metrics=[]",
                 "algorithm.diffusion.is_continuous=true", "++algorithm.diffusion.precond_scale=0.125",
                 "++algorithm.backbone.use_fourier_noise_embedding=true",
                 "algorithm.tasks.prediction.keyframe_density=1.0",
                 "algorithm.tasks.prediction.history_guidance.name=vanilla",
                 "+algorithm.tasks.prediction.history_guidance.guidance_scale=2.0",
                 "+algorithm.tasks.prediction.history_guidance.visualize=false",
                 "algorithm.tasks.interpolation.enabled=false"]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "configurations")):
        return compose(config_name="config", overrides=overrides)


def code_identity(scorer_path):
    paths = set()
    for directory in ("algorithms", "utils", "configurations", "datasets", "external_baselines"):
        for suffix in ("*.py", "*.yaml"):
            paths.update((ROOT / directory).rglob(suffix))
    result = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    result["memcam_scorer"] = digest(scorer_path)
    return result


def prepare_cases(dataset, output, count):
    """Use upstream deterministic test clips; never rank/filter by output quality."""
    if len(dataset) != count:
        raise ValueError(f"Requested {count} complete test clips, found {len(dataset)}")
    cases, scenes = [], set()
    for index in range(count):
        video_index, start = dataset.get_clip_location(index)
        metadata = dataset.metadata[video_index]
        scene = Path(metadata["video_paths"]).stem
        if scene in scenes:
            raise ValueError("Duplicate source clip identity")
        scenes.add(scene)
        source = dataset.video_path_to_preprocessed_path(Path(metadata["video_paths"]))
        poses = dataset.save_dir / "test_poses" / f"{scene}.pt"
        case = dict(index=index, scene=scene, start_frame=int(start), frame_skip=10,
                    frame_indices=list(range(int(start), int(start) + 191, 10)),
                    source=str(source.resolve()), source_sha256=digest(source),
                    poses=str(poses.resolve()), poses_sha256=digest(poses))
        directory = output / "cases" / f"{index:03d}_{scene}"
        directory.mkdir(parents=True, exist_ok=True)
        freeze(directory / "input.json", case)
        payload, receipt = directory / "observations.npz", directory / "observations.json"
        if receipt.exists():
            if not payload.exists() or json.loads(receipt.read_text())["sha256"] != digest(payload):
                raise ValueError(f"Cached input changed: {payload}")
        else:
            sample = dataset[index]
            images = sample["videos"].cpu().numpy().astype(np.float32)
            conditions = sample["conds"].cpu().numpy().astype(np.float32)
            if (images.shape != (20, 3, 256, 256) or conditions.shape != (20, 16)
                    or not sample["nonterminal"].all().item() or not np.isfinite(images).all()
                    or not np.isfinite(conditions).all() or images.min() < 0 or images.max() > 1):
                raise ValueError(f"Incomplete or invalid native sample: {scene}")
            # Targets stay on disk/CPU for metrics; only the first 16 frames are selectable.
            np.savez_compressed(payload, images=images, conditions=conditions)
            save(receipt, {"sha256": digest(payload)})
        case.update(directory=str(directory), observations_sha256=digest(payload))
        cases.append(case)
        print(f"Prepared {index + 1}/{count}: {scene}, 16 observed + 4 target views", flush=True)
    freeze(output / "cohort.json", cases)
    return cases


def prepare_features(cases, scorer, output, device):
    import torch
    from PIL import Image
    from huggingface_hub import HfApi
    from transformers import AutoImageProcessor, AutoModel
    encoder_file = output / "encoder.json"
    if encoder_file.exists():
        revision = json.loads(encoder_file.read_text())["revision"]
    else:
        revision = HfApi().model_info("facebook/dinov2-base").sha
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", revision=revision)
    encoder = None
    info = dict(name="facebook/dinov2-base", revision=revision, preprocessing=processor.to_dict(),
                pooling="pooler_output else CLS; float32 L2", source="observed historical RGB only")
    # Processor config may contain tuples; compare its JSON representation on resume.
    freeze(encoder_file, json.loads(json.dumps(info)))
    for case in cases:
        directory = Path(case["directory"])
        archive = np.load(directory / "observations.npz")
        feature_file, receipt = directory / "history_dino.npy", directory / "history_dino.json"
        signature = identity([case["observations_sha256"], info])
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["signature"] != signature or saved["sha256"] != digest(feature_file):
                raise ValueError("Stale DINO history cache")
            features = np.load(feature_file)
        else:
            if encoder is None:
                print("Loading DINOv2 for observed history features", flush=True)
                encoder = AutoModel.from_pretrained(info["name"], revision=revision).eval().to(device)
            pixels = archive["images"][:16]
            images = [Image.fromarray(np.rint(x.transpose(1, 2, 0) * 255).astype(np.uint8)) for x in pixels]
            batches = []
            with torch.inference_mode():
                for first in range(0, 16, 4):
                    inputs = processor(images=images[first:first + 4], return_tensors="pt").to(device)
                    result = encoder(**inputs)
                    pooled = getattr(result, "pooler_output", None)
                    if pooled is None:
                        pooled = result.last_hidden_state[:, 0]
                    batches.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
            features = np.concatenate(batches)
            np.save(feature_file, features)
            save(receipt, dict(signature=signature, sha256=digest(feature_file)))
        selections = {}
        for policy in POLICIES:
            selected, updates = choose_history(policy, archive["conditions"][:16], features, 4, scorer)
            selections[policy] = dict(selected=selected,
                source_indices=[case["frame_indices"][i] for i in selected], updates=updates)
        freeze(directory / "selection.json", json.loads(json.dumps(selections)))
        print(f"Selected history for {case['scene']}: " + str({k: v['selected'] for k, v in selections.items()}), flush=True)
    del encoder
    if device == "cuda":
        torch.cuda.empty_cache()


def predict(model, history_images, history_conditions, target_conditions, selected, seed):
    """The sampling API deliberately has no target-pixels argument."""
    import torch
    if history_images.shape != (16, 3, 256, 256) or target_conditions.shape != (4, 16):
        raise ValueError("Invalid history/target camera contract")
    indices = conditioning_indices(selected, 16, 4, 4)
    device = model.device
    inputs = torch.zeros((1, 8, 3, 256, 256), device=device)
    inputs[0, :4] = torch.as_tensor(history_images[selected], device=device)
    cond = np.concatenate([history_conditions, target_conditions])[indices]
    conditions = torch.as_tensor(cond[None], device=device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        generated = model._unnormalize_x(model._predict_videos(model._normalize_x(inputs), conditions))
    return generated[0, 4:].clamp(0, 1).float().cpu().numpy()


def quality(prediction, target, lpips_model, device):
    import torch
    from skimage.metrics import structural_similarity
    if prediction.shape != target.shape or not np.isfinite(prediction).all():
        raise ValueError("Invalid predictions")
    with torch.inference_mode():
        perceptual = lpips_model(torch.as_tensor(prediction * 2 - 1, device=device),
                                 torch.as_tensor(target * 2 - 1, device=device)).flatten().cpu().numpy()
    rows = []
    for index, (image, gt) in enumerate(zip(prediction, target)):
        mse = float(np.mean((image.astype(np.float64) - gt) ** 2))
        rows.append(dict(target_slot=index, mse=mse,
            psnr_db=float(-10 * np.log10(max(mse, 1e-12))),
            ssim=float(structural_similarity(gt.transpose(1, 2, 0), image.transpose(1, 2, 0),
                data_range=1.0, channel_axis=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)),
            lpips=float(perceptual[index])))
    return rows


def verify_receipt(directory, name, signature):
    path = directory / f"{name}.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value["signature"] != signature:
        raise ValueError(f"Stale {name} config: {directory}")
    for file, sha in value["artifacts"].items():
        if not (directory / file).exists() or digest(directory / file) != sha:
            raise ValueError(f"Changed {name} artifact: {directory / file}")
    return value


def export(output, cases, results, complete):
    rows = []
    for case in cases:
        for policy in POLICIES:
            data = results.get((case["scene"], policy))
            if data is None:
                continue
            rows.append(dict(scene=case["scene"], policy=policy, budget=4, observed_history=16,
                predicted_frames=4, **{metric: float(np.mean([r[metric] for r in data]))
                for metric in ("psnr_db", "ssim", "lpips")}))
    write_csv(output / ("per_video.csv" if complete else "partial_per_video.csv"), rows)
    if complete:
        summary = []
        for policy in POLICIES:
            policy_rows = [r for r in rows if r["policy"] == policy]
            if len(policy_rows) != len(cases):
                raise ValueError("Cannot export a complete table with missing paired cells")
            summary.append(dict(system="DFoT", policy=policy, budget=4, videos=len(cases),
                **{metric: float(np.mean([r[metric] for r in policy_rows])) for metric in ("psnr_db", "ssim", "lpips")}))
        write_csv(output / "scores.csv", summary)
        differences = []
        for case in cases:
            paired = {r["policy"]: r for r in rows if r["scene"] == case["scene"]}
            for reference in ("fifo", "uniform"):
                differences.append(dict(scene=case["scene"], contrast=f"keepsake_minus_{reference}",
                    **{m: paired["keepsake"][m] - paired[reference][m] for m in ("psnr_db", "ssim", "lpips")}))
        write_csv(output / "paired_differences.csv", differences)
        for row in summary:
            print(f"{row['policy']:10s} N={row['videos']} PSNR={row['psnr_db']:.4f} SSIM={row['ssim']:.4f} LPIPS={row['lpips']:.4f}", flush=True)


def execute(args, status):
    from omegaconf import OmegaConf
    from datasets.video.realestate10k_mini import RealEstate10KMiniAdvancedVideoDataset
    cfg = configuration(args)
    scorer, scorer_path = load_scorer(args.memcam_root)
    code = code_identity(scorer_path)
    config = dict(protocol="observed_history_transfer_v1", scenes=args.scenes, seed=args.seed,
        policies=list(POLICIES), history=16, budget=4, future=4, stride=10,
        retention="KEEPSAKE streams past observations; newest protected, no permanent initial anchor",
        uniform="offline uniform context subset; not a bounded online archive implementation",
        evaluation="native Mini test clips; not the paper's 32-case OOD benchmark or closed-loop generation",
        score_source=str(scorer_path), code=code,
        config=OmegaConf.to_container(cfg, resolve=True))
    freeze(args.output / "config.json", config)
    print("Preparing official Mini test data; never downloading the full training dataset", flush=True)
    dataset = RealEstate10KMiniAdvancedVideoDataset(cfg.dataset, split="validation")
    cases = prepare_cases(dataset, args.output, args.scenes)
    del dataset
    if args.prepare_only:
        status.update(status="prepared", prepared_cases=len(cases))
        return
    import torch
    print("Checking real CUDA allocation/kernel", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; allocation/GPU mask must work before generation")
    test = torch.ones(1, device="cuda") * 2
    torch.cuda.synchronize()
    assert test.item() == 2
    del test
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    environment = dict(python=sys.version, torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), host=platform.node(),
        packages={})
    from importlib.metadata import version
    for name in ("numpy", "torchvision", "transformers", "diffusers", "lpips", "scikit-image"):
        environment["packages"][name] = version(name)
    # Hardware may change between allocations; numerical software may not.
    freeze(args.output / "software.json", {k: v for k, v in environment.items() if k not in ("gpu", "host")})
    save(args.output / "environment.json", dict(environment, job=os.environ.get("SLURM_JOB_ID"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")))
    prepare_features(cases, scorer, args.output, "cuda")
    from utils.ckpt_utils import download_pretrained
    checkpoint = args.checkpoint or Path(download_pretrained("pretrained:DFoT_RE10K.ckpt"))
    checkpoint_info = dict(path=str(checkpoint.resolve()), sha256=digest(checkpoint))
    freeze(args.output / "checkpoint.json", checkpoint_info)
    cfg.dataset.n_frames = 8
    from algorithms.dfot.dfot_video_pose import DFoTVideoPose
    from external_baselines.worker import load_dfot_checkpoint
    print("Loading frozen DFoT_RE10K checkpoint (strict weights)", flush=True)
    model = DFoTVideoPose(cfg.algorithm)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_dfot_checkpoint(model, state)
    del state
    model.eval().to("cuda")
    if model.is_latent_diffusion or model.max_tokens != 8 or model.n_context_tokens != 4:
        raise ValueError("Expected official pixel-space RE10K eight-slot / four-context checkpoint")
    import lpips
    from PIL import Image
    print("Loading LPIPS AlexNet evaluator", flush=True)
    metric = lpips.LPIPS(net="alex").eval().to("cuda")
    weights_hash = hashlib.sha256()
    for key, value in sorted(metric.state_dict().items()):
        weights_hash.update(key.encode())
        weights_hash.update(value.detach().cpu().numpy().tobytes())
    metric_info = dict(lpips="AlexNet, [-1,1], unquantized predictions and native loader GT",
        lpips_weights_sha256=weights_hash.hexdigest(), psnr="mean per-target PSNR, MSE floor 1e-12",
        ssim="skimage Gaussian sigma1.5 population covariance, RGB, data_range1",
        aggregation="mean four targets per clip, then equal clip means; no inferred CI")
    freeze(args.output / "metrics.json", metric_info)
    results = {}
    for case in cases:
        directory = Path(case["directory"])
        sample = np.load(directory / "observations.npz")
        images, conditions = sample["images"], sample["conditions"]
        selections = json.loads((directory / "selection.json").read_text())
        for policy in POLICIES:
            cell = directory / policy
            cell.mkdir(exist_ok=True)
            selection = selections[policy]["selected"]
            signature = identity([config, checkpoint_info, case, selection, args.seed + case["index"],
                                  digest(directory / "history_dino.json"), metric_info])
            receipt = verify_receipt(cell, "generation", signature)
            if receipt is None:
                if time.monotonic() - args.started > args.max_seconds:
                    status.update(status="incomplete", reason="Time budget reached between cells")
                    export(args.output, cases, results, False)
                    return
                print(f"GENERATE [{case['index'] + 1}/{len(cases)}] {case['scene']} {policy} context={selection}", flush=True)
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                begin = time.monotonic()
                prediction = predict(model, images[:16], conditions[:16], conditions[16:], selection,
                                     args.seed + case["index"])
                torch.cuda.synchronize()
                elapsed = time.monotonic() - begin
                if prediction.shape != (4, 3, 256, 256) or not np.isfinite(prediction).all():
                    raise ValueError("Incomplete prediction")
                np.save(cell / "prediction.npy", prediction)
                for name, frames in (("context", images[selection]), ("target_gt", images[16:]), ("predicted", prediction)):
                    for number, frame in enumerate(frames):
                        Image.fromarray(np.rint(frame.transpose(1, 2, 0) * 255).astype(np.uint8)).save(cell / f"{name}_{number}.png")
                artifacts = {p.name: digest(p) for p in cell.iterdir() if p.suffix in (".png", ".npy")}
                save(cell / "generation.json", dict(signature=signature, artifacts=artifacts,
                    generation_seconds=elapsed, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    selection=selection, source_indices=selections[policy]["source_indices"]))
            metric_receipt = verify_receipt(cell, "quality", signature)
            if metric_receipt is None:
                frame_rows = quality(np.load(cell / "prediction.npy"), images[16:], metric, "cuda")
                write_csv(cell / "frame_metrics.csv", frame_rows)
                save(cell / "quality.json", dict(signature=signature,
                    artifacts={"frame_metrics.csv": digest(cell / "frame_metrics.csv")}, rows=frame_rows))
            else:
                frame_rows = metric_receipt["rows"]
            results[case["scene"], policy] = frame_rows
            export(args.output, cases, results, False)
            save(args.output / "progress.json", dict(completed_cells=len(results), planned_cells=len(cases) * 3,
                last_scene=case["scene"], last_policy=policy, elapsed_seconds=time.monotonic() - args.started))
            print(f"COMPLETE CELL {len(results)}/{len(cases) * 3}", flush=True)
    if code_identity(scorer_path) != code:
        raise ValueError("Source code changed during the pilot")
    export(args.output, cases, results, True)
    status.update(status="complete", completed_cells=len(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/real-estate-10k-mini")
    parser.add_argument("--memcam-root", type=Path, default=ROOT.parent / "MemCam")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--scenes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-seconds", type=int, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.scenes < 1 or args.max_seconds < 1 or not 0 <= args.seed < 2**32 - args.scenes:
        parser.error("Invalid scene count, seed or wall-time budget")
    args.output, args.data, args.memcam_root = args.output.resolve(), args.data.resolve(), args.memcam_root.resolve()
    args.started = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = dict(status="running", job=os.environ.get("SLURM_JOB_ID"), planned_cells=3 * args.scenes)
        save(args.output / "status.json", status)
        try:
            execute(args, status)
        except BaseException as exc:
            status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            status["elapsed_seconds"] = time.monotonic() - args.started
            save(args.output / "status.json", status)
        print(f"{status['status'].upper()}: {args.output}", flush=True)
        if status["status"] == "incomplete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
