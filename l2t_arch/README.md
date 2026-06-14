# l2t_arch

`l2t_arch` contains lip-to-text experiments for LRS2 using frozen VTP visual
features. The package trains second-stage models that map lip-motion features
to text, or refine a noisy VTP text hypothesis into the ground-truth transcript.

This package does not download LRS2 and does not crop or encode raw videos from
scratch. Its main input is an existing VTP cache manifest with:

- `visual_feature_path`: a `.pt` file containing `visual_features` as `Tensor[T, 512]`.
- `text`: the noisy VTP hypothesis, used as `vtp_text`.
- `source_text_path`: the original LRS2 `.txt` file, used to read the ground-truth transcript.

## 1. Environment Setup

Python 3.10 or 3.11 is recommended. If you only train `l2t_arch` from an
existing cache, the minimum libraries are `torch`, `numpy`, and `tqdm`. If you
also need to build the cache from videos, install the video/audio stack as well:
`opencv`, `librosa`, `soundfile`, `ffmpeg`, and the dependencies required by
the VTP pipeline.

Run from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch build that matches your CUDA/server setup. Example:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Then install the remaining Python dependencies:

```powershell
pip install -r requirements.txt
```

If your cache-building pipeline needs to read video/audio, install `ffmpeg` at
the operating-system level:

```powershell
winget install Gyan.FFmpeg
```

Quick checks:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m l2t_arch.train --help
python -m l2t_arch.prepare_dataset --help
```

If `torch.cuda.is_available()` returns `False`, you can still run a small CPU
smoke test, but real training should use a GPU.

## 2. Requesting and Downloading LRS2

LRS2 is an Oxford VGG/BBC dataset for non-commercial academic research. The
official LRS2 page states that access to the video and metadata package requires
a Data Sharing Agreement with BBC Research & Development. After approval, you
receive a password that allows downloading the dataset from Oxford VGG.

Official links:

- LRS2 page: <https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html>
- Lip Reading datasets index: <https://www.robots.ox.ac.uk/~vgg/data/lip_reading/>

Suggested access flow:

1. Open the official LRS2 page above.
2. Download the Data Sharing Agreement from the BBC link on that page.
3. Fill in your institution/user details, research purpose, and data-use commitments.
4. Submit the agreement following the BBC/VGG instructions.
5. After approval, use the provided password to download the LRS2 dataset and filelists.

After downloading, extract the dataset outside the repository if possible, for
example:

```text
D:\datasets\lrs2_v1\mvlrs_v1\
```

Common directory layout:

```text
mvlrs_v1\
  main\
    <speaker_or_video_id>\
      00001.mp4
      00001.txt
  pretrain\
    ...
```

Each LRS2 `.txt` file usually contains a `Text:` line and may contain a `Conf:`
line. `prepare_dataset.py` reads `Text:` as the ground-truth transcript and can
filter samples by `Conf:`.

## 3. Preparing a VTP Feature Cache

`l2t_arch` trains from cached features, not directly from `.mp4` files. The
cache manifest must be JSONL, where each line is one sample:

```json
{
  "id": "main_5535415699068794046_00001",
  "source_text_path": "D:\\datasets\\lrs2_v1\\mvlrs_v1\\main\\5535415699068794046\\00001.txt",
  "source_video_path": "D:\\datasets\\lrs2_v1\\mvlrs_v1\\main\\5535415699068794046\\00001.mp4",
  "visual_feature_path": "visual_features\\main_5535415699068794046_00001.visual.pt",
  "text": "when you look at chips at home",
  "split": "main",
  "mel_frames": 89,
  "hop_length": 256,
  "sample_rate": 16000
}
```

Notes:

- `visual_feature_path` can be absolute, or relative to the directory containing the cache manifest.
- The `.visual.pt` file can be a tensor directly or a dict with a `visual_features` key.
- Prefer absolute `source_text_path` values to avoid transcript loading errors.
- `text` is the VTP prediction and may be wrong; `gt_text` is read from `source_text_path`.

Example cache layout used by this repository:

```text
cache_l2s_itw_vtp_raw_lrs2_stage\manifest.jsonl
cache_l2s_itw_vtp_raw_lrs2_stage\visual_features\*.visual.pt
```

To create features for a single video with VTP:

```powershell
python -m l2t_vtp.infer_video `
  --video D:\datasets\lrs2_v1\mvlrs_v1\main\<id>\00001.mp4 `
  --repo-dir external\vtp `
  --ckpt-path pretrained_models\vtp\ft_lrs2.pth `
  --cnn-ckpt-path pretrained_models\vtp\feature_extractor.pth `
  --save-visual-features tmp\00001.visual.pt `
  --device cuda
```

For full training, build a large cache manifest first, then pass it to
`l2t_arch.prepare_dataset`.

## 4. Building l2t_arch Manifests

Create `train.jsonl`, `val.jsonl`, and `test.jsonl` from the VTP cache manifest:

```powershell
python -m l2t_arch.prepare_dataset `
  --cache-manifest cache_l2s_itw_vtp_raw_lrs2_stage\manifest.jsonl `
  --output-dir datasets_l2t_arch_50k `
  --min-conf 4 `
  --val-ratio 0.1 `
  --test-ratio 0.05 `
  --seed 1234
```

Useful arguments:

- `--limit 50000`: use only the first 50k rows from the cache manifest.
- `--min-conf 4`: remove samples with low transcript confidence.
- `--min-seconds 2.0`: remove very short clips when duration metadata exists.
- `--val-ratio`, `--test-ratio`: validation and test split ratios.

Output:

```text
datasets_l2t_arch_50k\
  train.jsonl
  val.jsonl
  test.jsonl
  summary.json
```

Check the prepared sample counts:

```powershell
Get-Content datasets_l2t_arch_50k\summary.json
```

## 5. Choosing a Training Config

Available configs:

```text
l2t_arch/configs/text_only.json
l2t_arch/configs/visual_ctc_transformer.json
l2t_arch/configs/visual_plif_seq2seq.json
l2t_arch/configs/dual_plif_monotonic.json
l2t_arch/configs/dual_plif_monotonic_ctc_aux.json
l2t_arch/configs/dual_path.json
```

Suggested use:

- `text_only.json`: baseline that refines VTP text only, without visual features.
- `visual_ctc_transformer.json`: visual-only CTC baseline.
- `visual_plif_seq2seq.json`: visual PLIF encoder with a seq2seq decoder.
- `dual_plif_monotonic.json`: visual-only dual PLIF model with monotonic-biased decoding.
- `dual_plif_monotonic_ctc_aux.json`: dual PLIF plus auxiliary CTC loss.
- `dual_path.json`: uses both frozen visual features and noisy VTP text; this is a good first config to try.

By default, the configs point to:

```json
"train_manifest": "datasets_l2t_arch_50k/train.jsonl",
"val_manifest": "datasets_l2t_arch_50k/val.jsonl"
```

If your manifests are elsewhere, override paths with `--set`.

## 6. Quick Smoke Test

Before running a long job, run one small epoch:

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_path.json `
  --set training.output_dir=checkpoints_l2t_arch_smoke `
  --set training.limit_samples=128 `
  --set training.epochs=1 `
  --set training.batch_size=4 `
  --set training.num_workers=0
```

If the run works, the terminal prints `train_loss`, `val_loss`, and `best_val`,
and the checkpoint directory contains:

```text
checkpoints_l2t_arch_smoke\
  config.json
  latest_model.pth
  best_model.pth
```

## 7. Full Training

Train `dual_path`:

```powershell
python -m l2t_arch.train --config l2t_arch/configs/dual_path.json
```

Train with explicit manifest and output overrides:

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_path.json `
  --set data.train_manifest=datasets_l2t_arch_50k\train.jsonl `
  --set data.val_manifest=datasets_l2t_arch_50k\val.jsonl `
  --set training.output_dir=checkpoints_l2t_arch_dual_path `
  --set training.batch_size=16 `
  --set training.epochs=25 `
  --set training.lr=0.0001 `
  --set training.amp=true
```

Train the PLIF + auxiliary CTC variant:

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_plif_monotonic_ctc_aux.json `
  --set training.output_dir=checkpoints_l2t_arch_dual_plif_monotonic_ctc_aux
```

Resume from a checkpoint:

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_path.json `
  --set training.resume_from=checkpoints_l2t_arch_dual_path\latest_model.pth
```

If CUDA runs out of memory, reduce the batch size:

```powershell
python -m l2t_arch.train `
  --config l2t_arch/configs/dual_path.json `
  --set training.batch_size=8 `
  --set training.num_workers=0
```

## 8. Evaluation

Evaluate the best checkpoint on the test set:

```powershell
python -m l2t_arch.evaluate `
  --checkpoint checkpoints_l2t_arch_dual_path\best_model.pth `
  --manifest datasets_l2t_arch_50k\test.jsonl `
  --output-dir reports_l2t_arch_dual_path `
  --batch-size 8 `
  --device cuda
```

Output:

```text
reports_l2t_arch_dual_path\
  summary.json
  predictions.jsonl
```

`summary.json` includes:

- `wer`: word error rate.
- `cer`: character error rate.
- `exact_match`: percentage of predictions exactly matching the ground truth.
- optional diagnostics such as `spike_rate`, `beta_mean`, or `fusion_gate_mean`.

Quickly inspect results:

```powershell
Get-Content reports_l2t_arch_dual_path\summary.json
Get-Content reports_l2t_arch_dual_path\predictions.jsonl | Select-Object -First 5
```

## 9. Prepared Manifest Format

`ArchDataset` reads each prepared manifest row in this form:

```json
{
  "id": "sample_id",
  "visual_feature_path": "C:\\path\\to\\sample.visual.pt",
  "vtp_text": "noisy vtp prediction",
  "gt_text": "ground truth transcript",
  "conf": 5,
  "split": "main",
  "seconds": 1.42,
  "source_text_path": "C:\\path\\to\\sample.txt",
  "source_video_path": "C:\\path\\to\\sample.mp4"
}
```

The default tokenizer accepts:

```text
abcdefghijklmnopqrstuvwxyz '
```

Text is lowercased and normalized before encoding. Characters outside the
vocabulary become `<unk>`, so keep LRS2 transcripts in clean English text.

## 10. Common Issues

`ModuleNotFoundError: No module named 'l2t_arch'`

Run commands from the repository root, not from inside `l2t_arch`:

```powershell
cd "C:\path\to\your\repo"
python -m l2t_arch.train --config l2t_arch/configs/dual_path.json
```

`Empty manifest`

`train.jsonl` or `val.jsonl` is empty. Check whether `prepare_dataset` filtered
too aggressively with `--min-conf` or `--min-seconds`, or whether
`source_text_path` is wrong.

`FileNotFoundError` for `.visual.pt`

Check `visual_feature_path`. If it is relative, it is resolved relative to the
directory that contains the original cache manifest.

CUDA out-of-memory

Reduce:

```powershell
--set training.batch_size=8
--set training.batch_size=4
```

You can also reduce model size in the config, for example `hidden_dim`,
`visual_layers`, or `decoder_layers`.

Unicode path issues

Some external video tools may fail on non-ASCII paths. If that happens, put the
dataset/cache in an ASCII-only path:

```text
D:\datasets\lrs2\
D:\cache\cache_l2s_itw_vtp_raw_lrs2_stage\
```

## 11. Citation

If you use LRS2, cite the official paper:

```bibtex
@InProceedings{Afouras18c,
  author    = "Afouras, T. and Chung, J.~S. and Senior, A. and Vinyals, O. and Zisserman, A.",
  title     = "Deep Audio-Visual Speech Recognition",
  booktitle = "arXiv:1809.02108",
  year      = "2018"
}
```

Follow the Data Sharing Agreement terms: do not redistribute the dataset, do not
publish download links or passwords, and use the data only within the approved
scope.
