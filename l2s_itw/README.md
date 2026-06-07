# l2s_itw

Clean lip-to-speech pipeline inspired by `Towards Accurate Lip-to-Speech Synthesis in-the-Wild`.

This package is intentionally independent from every previous `srcV*` branch.

## Core Direction

```text
silent lip video
  -> lip-to-text / visual-feature provider
  -> noisy text units + frame visual embeddings
  -> Visual TTS alignment model
  -> mel-spectrogram
  -> vocoder
  -> waveform
```

The important design choice from the paper is preserved: speech is not generated from visual input alone. The model receives:

- `visual_features`: frame-level lip/video embeddings.
- `text`: noisy or oracle text from a lip-to-text system.
- `speaker_embedding`: target voice identity.
- `mel`: training target.

## Manifest Contract

Training uses JSONL manifests. Each line describes one sample:

```json
{
  "id": "sample_0001",
  "visual_feature_path": "path/to/sample.visual.pt",
  "mel_path": "path/to/sample.mel.pt",
  "speaker_embedding_path": "path/to/sample.speaker.pt",
  "text": "the predicted or ground truth sentence"
}
```

Tensor shapes:

- `visual_feature_path`: `Tensor[T_video, visual_dim]`
- `mel_path`: `Tensor[T_mel, n_mels]`
- `speaker_embedding_path`: `Tensor[speaker_dim]`

For LRS-style data at 25 FPS video and 10 ms mel hop, `T_mel ~= T_video * 4`.
If the mel hop is 256 at 16 kHz, use `model.mel_frames_per_video_frame=2.5`.

## Smoke Test

Create tiny synthetic data:

```powershell
python -m l2s_itw.tools.create_toy_data --output-dir tmp_l2s_itw_toy --num-train 12 --num-val 4
```

Train one small run:

```powershell
python -m l2s_itw.training.train_visual_tts `
  --config l2s_itw/configs/base.json `
  --set data.train_manifest=tmp_l2s_itw_toy/train.jsonl `
  --set data.val_manifest=tmp_l2s_itw_toy/val.jsonl `
  --set training.output_dir=tmp_l2s_itw_ckpt `
  --set training.device=cpu `
  --set training.epochs=1 `
  --set training.batch_size=2 `
  --set model.hidden_dim=64 `
  --set model.num_heads=4 `
  --set model.text_layers=1 `
  --set model.visual_layers=1 `
  --set model.decoder_layers=1 `
  --set model.ffn_dim=128
```

Synthesize mel from one manifest sample:

```powershell
python -m l2s_itw.inference.synthesize `
  --config tmp_l2s_itw_ckpt/config.json `
  --checkpoint tmp_l2s_itw_ckpt/best_model.pth `
  --sample tmp_l2s_itw_toy/val.jsonl `
  --sample-index 0 `
  --output-dir tmp_l2s_itw_infer
```

## Config Roadmap

We will configure these one by one:

1. `provider`: VTP/lip-to-text source for noisy text and visual features.
2. `text`: character units first, then phoneme/G2P if needed.
3. `speaker`: fixed speaker, speaker encoder, or target voice cloning.
4. `vocoder`: mel-only first, then BigVGAN/HiFi-GAN.
5. `losses`: L1 mel first, then duration/sync/perceptual losses.

## VTP-First Provider

First target configuration:

```text
video -> official VTP
  -> text
  -> VTP visual embeddings [T_video, 512]
```

The adapter is in `l2s_itw.providers.vtp`. It expects the official repository from:

```text
https://github.com/prajwalkr/vtp
```

Prepare an input JSONL manifest with at least `video_path`. Include `mel_path` and `speaker_embedding_path` if you are preparing a training manifest:

```json
{"id":"sample_0001","video_path":"clips/sample_0001.mp4","mel_path":"mels/sample_0001.pt","speaker_embedding_path":"speakers/sample_0001.pt"}
```

`video_path` should point to the preprocessed face-track video expected by VTP. The official README suggests extracting face tracks with `syncnet_python` before running VTP.

Check local readiness:

```powershell
python -m l2s_itw.providers.check_vtp `
  --repo-dir external\vtp `
  --ckpt-path pretrained_models\vtp\ft_lrs2.pth `
  --cnn-ckpt-path pretrained_models\vtp\feature_extractor.pth
```

Cache VTP text and visual features:

```powershell
python -m l2s_itw.providers.cache_vtp `
  --input-manifest path\to\input_videos.jsonl `
  --output-dir cache_l2s_itw_vtp `
  --repo-dir external\vtp `
  --ckpt-path pretrained_models\vtp\ft_lrs2_or_lrs3.pth `
  --cnn-ckpt-path pretrained_models\vtp\feature_extractor.pth `
  --device cuda `
  --beam-size 30 `
  --max-decode-len 35
```

The output `cache_l2s_itw_vtp/manifest.jsonl` can be passed directly to `train_visual_tts.py` as `data.train_manifest` or `data.val_manifest`.

If your starting point is an existing `.pt` sample cache with `video` and `mel` tensors, export it into VTP-ready mp4 + mel files first:

```powershell
python -m l2s_itw.data.export_cache_manifest `
  --cache-dir Processed_Data_R2INR_LRS2_10k `
  --output-dir data_l2s_itw_export `
  --limit 100 `
  --overwrite
```

Then run `cache_vtp` on `data_l2s_itw_export/manifest.jsonl`.

Split a cached VTP manifest for training:

```powershell
python -m l2s_itw.data.split_manifest `
  --manifest cache_l2s_itw_vtp\manifest.jsonl `
  --output-dir manifests_l2s_itw_vtp `
  --val-ratio 0.1
```

Official public checkpoint URLs from the VTP README:

- Public train data feature extractor: `https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/feature_extractor.pth`
- Public train data FT-LRS2: `https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/ft_lrs2.pth`
- Public train data FT-LRS3: `https://www.robots.ox.ac.uk/~vgg/research/vtp-for-lip-reading/checkpoints/public_train_data/ft_lrs3.pth`

For a resume-capable download of the stronger extended checkpoints:

```powershell
python -m l2s_itw.providers.download_vtp_checkpoints `
  --variant extended `
  --target feature_extractor `
  --target ft_lrs2 `
  --output-dir pretrained_models\vtp
```
