# srcV10

`srcV10` is a practical LRS2 speech-reconstruction branch built on the strongest local pieces:

- V6 R2INR memory alignment from `mel_times` into video/landmark memory tokens.
- Optional cached AV-HuBERT visual features from `srcV8` as pretrained semantic conditioning.
- V4 TFiLM-Conformer decoder plus SIREN residual mel refinement.
- Auxiliary speech-unit, prosody, and optional rectified-flow losses.

`--amp` resolves to BF16 automatically when the CUDA device supports it. FP16 is intentionally avoided by default because it can make this model produce non-finite losses early in training.

## Smoke Train

```powershell
python -m srcV10.training.train `
  --data-dir Processed_Data_R2INR_LRS2_10k `
  --epochs 1 `
  --limit-files 8 `
  --batch-size 1 `
  --dim 128 `
  --decoder-layers 2 `
  --no-use-avhubert-features `
  --device cpu
```

## Recommended First Run

Start without AV-HuBERT features to verify the reconstruction path:

```powershell
python -m srcV10.training.train `
  --data-dir Processed_Data_R2INR_LRS2_10k `
  --output-dir checkpoints_srcV10_lrs2_base `
  --epochs 80 `
  --batch-size 4 `
  --max-frames 125 `
  --freeze-visual-epochs 8 `
  --no-use-avhubert-features `
  --amp
```

## Cache AV-HuBERT Features

After cloning `facebookresearch/av_hubert` and downloading an AV-HuBERT checkpoint:

```powershell
python -m srcV8.training.cache_avhubert_features `
  --data-dir Processed_Data_R2INR_LRS2_10k `
  --output-dir Processed_Data_AVHubertFeatures_LRS2_10k `
  --avhubert-dir path\to\av_hubert `
  --checkpoint path\to\avhubert_checkpoint.pt `
  --batch-size 1 `
  --device cuda `
  --amp
```

Then train V10 with the pretrained feature branch:

```powershell
python -m srcV10.training.train `
  --data-dir Processed_Data_R2INR_LRS2_10k `
  --av-feature-dir Processed_Data_AVHubertFeatures_LRS2_10k `
  --output-dir checkpoints_srcV10_lrs2_avhubert `
  --epochs 80 `
  --batch-size 4 `
  --max-frames 125 `
  --freeze-visual-epochs 8 `
  --amp
```

## Optional Flow Auxiliary

Use this only after the base run is stable:

```powershell
python -m srcV10.training.train `
  --data-dir Processed_Data_R2INR_LRS2_10k `
  --av-feature-dir Processed_Data_AVHubertFeatures_LRS2_10k `
  --output-dir checkpoints_srcV10_lrs2_flow `
  --use-flow-refiner `
  --flow-loss-weight 0.05 `
  --epochs 80 `
  --batch-size 4 `
  --amp
```

## Inference

```powershell
python -m srcV10.inference.infer_cache `
  --data-dir Processed_Data_R2INR_LRS2_10k `
  --checkpoint checkpoints_srcV10_lrs2_avhubert\best_model.pth `
  --av-feature-dir Processed_Data_AVHubertFeatures_LRS2_10k `
  --output-dir infer_srcV10
```
