# DFoT History Stress Suite

Run after the five-clip observed-history pilot. Keep that pilot's results,
including FIFO's lead, intact. This suite does not tune the score formula,
select clips by a policy's generated quality, or change checkpoint/guidance.
It is a follow-up observed-history test, not a closed-loop rollout experiment.

## Run in the Existing H100 Allocation

```bash
cd "$HOME/diffusion-forcing-transformer"
bash external_baselines/run_history_stress.sh
```

No installation, full-dataset download, new allocation or fine-tuning. Uses the
working `dfot` environment and cached Mini data/model. The DINO encoder revision
and processor configuration are copied from the completed pilot. Slurm's GPU
mask is preserved. CUDA/import tests run before starting the suite.

Default output: `~/memcam_results/dfot_history_stress_mini_n10`.
Default runner budget: 7,200 seconds, including preparation; checked between
generation cells. Set `DFOT_STRESS_SECONDS` to leave time within the existing
allocation. It is not a hard timeout or a completion estimate. Repeating the
same command resumes verified cells. Do not overwrite the old pilot directory.
The code update deliberately changes provenance hashes; do not rerun the old
completed pilot into its original output directory with this updated source.

## Prespecified Conditions

FIFO, uniform and KEEPSAKE all supply four context images to the same frozen
eight-slot DFoT checkpoint, generating the same four target views. Every method
receives the same available history and seed within a case. KEEPSAKE uses the
unchanged production scorer, streaming B4, newest observation protected. Uniform
is an offline subset baseline, not an online bounded archive.

The first three conditions use ten deterministic upstream Mini clips, including
the original five as a checked prefix. No output-quality filtering is applied.
Let s be the upstream-selected starting index in each source video:

| Condition | Observed history indices | Four target indices |
| --- | --- | --- |
| continuation_h16 | s+96,98,...,126 | s+128,130,132,134 |
| continuation_h64 | s+0,2,...,126 | s+128,130,132,134 |
| gap32_h64 | s+0,2,...,126 | s+158,160,162,164 |

Thus the first pair changes available history without changing targets or the
latest observation. The third changes the last-observed-to-target gap from 2
to 32 source frames, holding the historical bank input fixed. Intermediate gap
frames are unobserved by all methods. At the loader's 10 processed FPS these
gaps are 0.2 and 3.2 seconds, not long-video generation durations. The original
pilot sampled stride 10; these conditions use stride 2 and are reported separately.

## Natural Pose-Revisit Condition

Scan camera metadata for every available Mini clip (normally 500). All four
future target poses must return close to historical poses at least 64 source
frames older than the first target. Require rotation <=5 degrees and position
distance <=5% of the historical camera-center bounding-box diagonal. Require
an intervening observed excursion of >20 degrees or >20% of that diagonal from
the first matched historical pose. Numerical positional floor: 1e-8 source units.
This rejects a stationary camera masquerading as a revisit; rotation-only
excursions are supported. All thresholds are fixed before any stress inference.

The 64 historical frames are spaced two source frames apart, ending 16 frames
before the first target. Targets are q,q+2,q+4,q+6. Search candidate q values
from 142 with stride 2. Use the earliest qualifying return in each clip, then
take up to ten eligible clips ordered by SHA256("2026:scene"). Do not require
FIFO to fail or KEEPSAKE to retain the matched observation. Cameras alone define
eligibility; no pixels or policy output scores enter the selection criterion.

Export all scanned identities, pose hashes and eligibility statuses to
`revisit_audit.csv` and its frozen JSON counterpart. Input errors stop the run
after exporting the audit. Fewer than ten qualifying returns are reported with
the actual N. Zero returns produces an explicit unavailable condition, not an
invented or relaxed revisit. Pose proximity is not an occlusion/visibility test
or proof that an object is unchanged; inspect generated and GT frames afterward.

## Results

Maximum default workload: 90 fixed-cohort cells + 30 pose-revisit cells. Each
cell generates four frames. It does not generate 120 full-length videos.
Seed = 2026 + fixed clip index, or 2026 + 1000 + revisit clip index, identical
across policies. Fixed conditions share clip seeds. Source/target frame mappings,
original file hashes, selected IDs, update scores, RGB and camera arrays,
predictions, GT/context PNGs and per-frame metrics are recorded.

- `coverage.json`, `revisit_audit.csv`: geometric eligibility and audit.
- `regime_status.csv`: expected and completed matched cells per condition.
- `progress.json`, `status.json`: run progress and completion/failure.
- `regimes/<condition>/scores.csv`: written once that entire paired cohort is
  complete; includes all three policies, not a partial-cohort average.
- `completed_regime_scores.csv`: completed conditions while the rest is running.
- `scores.csv`: final separate rows for each available condition/policy.
- Per-condition `per_video.csv`, `paired_differences.csv`: inspect per-clip variation.

PSNR, SSIM and AlexNet LPIPS use the pilot's definitions: four-target mean within
clip, then equal clip weighting. No pooled overall winner across different
conditions, no FVD/VBench on this tiny sample, and no automatic confidence claims.
Keep unfavorable and null outcomes. Camera-matched revisits and ordinary future
continuation answer different questions; neither replaces the original pilot.
