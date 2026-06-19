# l2t_vtp

Standalone lip-to-text pipeline for the assistive-captioning direction.

This package is intentionally separate from `l2s_itw`. It reuses existing
VTP cache/manifests but does not import the lip-to-speech code.

## Evaluate Cached VTP Text

```powershell
python -m l2t_vtp.evaluate_cache `
  --pred-manifest cache_l2s_itw_vtp_raw_lrs2_stage\manifest.jsonl `
  --gt-manifest data_l2s_itw_raw_lrs2_mels_stage\manifest.jsonl `
  --output-dir reports_l2t_vtp_50k
```

For longer clips only:

```powershell
python -m l2t_vtp.evaluate_cache `
  --pred-manifest cache_l2s_itw_vtp_raw_lrs2_stage\manifest.jsonl `
  --gt-manifest data_l2s_itw_raw_lrs2_mels_stage\manifest.jsonl `
  --output-dir reports_l2t_vtp_50k_long `
  --min-seconds 3
```

## Infer One Video

```powershell
python -m l2t_vtp.infer_video `
  --video path\to\video.mp4 `
  --repo-dir external\vtp `
  --ckpt-path pretrained_models\vtp\ft_lrs2.pth `
  --cnn-ckpt-path pretrained_models\vtp\feature_extractor.pth `
  --device cuda
```

