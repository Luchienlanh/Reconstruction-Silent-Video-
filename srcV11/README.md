# srcV11

`srcV11` is the lip-to-text branch for LRS2. It predicts English text from cached AV-HuBERT visual features using CTC loss.

The main goal is accessibility-friendly text/subtitle output. Audio should be added later as a fixed-voice TTS wrapper:

```text
silent video -> srcV11 text -> fixed MC/native-speaker TTS -> wav
```

## 1. Cache AV-HuBERT Features

```bash
python -m srcV8.training.cache_avhubert_features \
  --data-dir Processed_Data_R2INR_LRS2_10k \
  --output-dir Processed_Data_AVHubertFeatures_LRS2_10k \
  --avhubert-dir path/to/av_hubert \
  --checkpoint path/to/avhubert_checkpoint.pt \
  --batch-size 1 \
  --device cuda \
  --amp
```

## 2. Smoke Train

```bash
python -m srcV11.training.train_ctc \
  --feature-dir Processed_Data_AVHubertFeatures_LRS2_10k \
  --output-dir tmp_srcV11_smoke \
  --epochs 1 \
  --limit-files 8 \
  --batch-size 2 \
  --device cuda
```

## 3. Train

```bash
python -m srcV11.training.train_ctc \
  --feature-dir Processed_Data_AVHubertFeatures_LRS2_10k \
  --output-dir checkpoints_srcV11_lrs2_char_ctc \
  --epochs 80 \
  --batch-size 8 \
  --val-ratio 0.1 \
  --multi-gpu
```

Use `best_model.pth`; it is selected by lowest validation CER.

## 4. Inference

```bash
python -m srcV11.inference.infer_ctc \
  --feature-dir Processed_Data_AVHubertFeatures_LRS2_10k \
  --checkpoint checkpoints_srcV11_lrs2_char_ctc/best_model.pth \
  --output-dir infer_srcV11 \
  --split val \
  --sample-index 0
```

Outputs:

```text
*_predicted_text.txt
*_confidence.json
```
