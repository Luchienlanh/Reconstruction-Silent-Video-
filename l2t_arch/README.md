# l2t_arch

Lip-to-text experiments for silent video recognition using frozen VTP visual
features and lightweight Transformer/PLIF decoders.

## What This Package Does

- Builds train/val/test manifests from an existing VTP cache.
- Trains visual-only and VTP-assisted lip-to-text models.
- Evaluates checkpoints with WER, CER, and exact match.
- Serves a demo API for uploading a video and returning transcript text.

The package does not store raw LRS2/LRS3 videos, VTP checkpoints, or generated
feature caches. Those artifacts should remain outside Git.

## Main Architectures

- `visual_ctc_transformer`: visual-only CTC baseline.
- `visual_plif_seq2seq`: single-branch PLIF visual seq2seq model.
- `dual_plif_monotonic`: visual-only Dual-Memory PLIF model with monotonic
  decoder bias.
- `dual_path`: VTP visual features plus VTP text hypothesis refiner.
- `text_only`: VTP text-only refiner baseline.

## Prepare Dataset

```powershell
python -m l2t_arch.prepare_dataset `
  --cache-manifest cache_l2s_itw_vtp_raw_lrs2_stage\manifest.jsonl `
  --output-dir datasets_l2t_arch_50k `
  --min-conf 4
```

## Train

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_plif_monotonic.json `
  --set data.train_manifest=datasets_l2t_arch_50k\train.jsonl `
  --set data.val_manifest=datasets_l2t_arch_50k\val.jsonl `
  --set training.output_dir=checkpoints_l2t_arch_dual_plif_monotonic `
  --set training.device=cuda `
  --set training.amp=true
```

Resume:

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_plif_monotonic.json `
  --set training.resume_from=checkpoints_l2t_arch_dual_plif_monotonic\latest_model.pth
```

## Evaluate

```powershell
python -m l2t_arch.evaluate `
  --checkpoint checkpoints_l2t_arch_dual_plif_monotonic\best_model.pth `
  --manifest datasets_l2t_arch_50k\test.jsonl `
  --output-dir reports_l2t_arch_dual_plif_monotonic
```

Outputs:

- `summary.json`
- `predictions.jsonl`

## Demo API

The demo API depends on `l2t_vtp` and an external VTP checkout in `external/vtp`.

```powershell
python -m l2t_arch.demo_api
```

Optional environment variables:

```powershell
$env:L2T_CHECKPOINT="checkpoints_l2t_arch_full_dual_path\best_model.pth"
$env:L2T_DEVICE="cuda"
```

Health endpoint:

```text
GET /api/health
```

Transcription endpoint:

```text
POST /api/transcribe
```

Multipart field:

```text
video
```
