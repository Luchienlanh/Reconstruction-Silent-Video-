# srcV11

`srcV11` is the lip-to-text branch for LRS2. It predicts English text from cached AV-HuBERT visual features using CTC loss.

The main goal is accessibility-friendly text/subtitle output. Audio should be added later as a fixed-voice TTS wrapper:

```text
silent video -> srcV11 text -> fixed MC/native-speaker TTS -> wav
```

## 1A. Cache Native Features, No AV-HuBERT

```bash
python -m srcV11.training.cache_native_features \
  --data-dir Processed_Data_R2INR_LRS2_10k \
  --output-dir Processed_Data_NativeFeatures_LRS2_10k \
  --batch-size 4 \
  --device cuda \
  --amp
```

This fallback does not need `av_hubert`, `fairseq`, or Python 3.8. It is weaker than AV-HuBERT but works on Kaggle's current Python runtime.

## 1B. Optional: Cache AV-HuBERT Features

Use this later on a Python/fairseq environment that supports the official AV-HuBERT stack.

On Kaggle, if the AV-HuBERT repo is at `/kaggle/working/av_hubert` and the Python 3.8 env is at
`/kaggle/working/envs/avhubert38`, include the inner `avhubert` directory in `PYTHONPATH`:

```bash
PY=/kaggle/working/envs/avhubert38/bin/python
REPO=/kaggle/working/Reconstruction-Silent-Video-
AVH=/kaggle/working/av_hubert
CKPT=/kaggle/working/pretrained/avhubert/base_vox_iter5.pt

PYTHONPATH=$AVH/avhubert:$AVH:$AVH/fairseq:$REPO \
$PY -m srcV8.training.cache_avhubert_features \
  --data-dir /kaggle/input/datasets/ludocute/antirs/Processed_Data_R2INR_LRS2_10k \
  --output-dir /kaggle/working/Processed_Data_AVHubertFeatures_LRS2_1k \
  --avhubert-dir $AVH \
  --checkpoint $CKPT \
  --limit-files 1000 \
  --batch-size 1 \
  --device cuda \
  --amp
```

## 2. Smoke Train

```bash
python -m srcV11.training.train_ctc \
  --feature-dir Processed_Data_NativeFeatures_LRS2_10k \
  --output-dir tmp_srcV11_smoke \
  --epochs 1 \
  --limit-files 8 \
  --batch-size 2 \
  --device cuda
```

## 3. Train

After AV-HuBERT feature caching, train CTC with the normal Kaggle Python runtime.
The Python 3.8 AV-HuBERT env is only needed for feature extraction.

```bash
python -m srcV11.training.train_ctc \
  --feature-dir Processed_Data_NativeFeatures_LRS2_10k \
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
  --feature-dir Processed_Data_NativeFeatures_LRS2_10k \
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
