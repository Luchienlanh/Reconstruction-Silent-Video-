# l2t_arch

Frozen-VTP architecture experiments for lip-to-text.

This package is separate from `l2s_itw`. It trains only second-stage models on
cached VTP outputs:

- frozen visual features: `visual_features/*.visual.pt`
- noisy VTP hypothesis: `text`
- ground-truth transcript: LRS2 `source_text_path`

## Prepare Dataset

```powershell
python -m l2t_arch.prepare_dataset `
  --cache-manifest cache_l2s_itw_vtp_raw_lrs2_stage\manifest.jsonl `
  --output-dir datasets_l2t_arch_50k `
  --min-conf 4
```

## Train

```powershell
python -m l2t_arch.train --config l2t_arch/configs/dual_path.json
```

Other configs:

```text
l2t_arch/configs/text_only.json
l2t_arch/configs/visual_ctc_transformer.json
l2t_arch/configs/visual_plif_seq2seq.json
l2t_arch/configs/dual_plif_monotonic.json
l2t_arch/configs/dual_plif_monotonic_ctc_aux.json
l2t_arch/configs/dual_path.json
```

The PLIF config uses frozen VTP visual features, a learnable PLIF temporal block,
membrane-potential sequence features, and a causal Transformer decoder. Spike
activations are used only for reset/gating and optional regularization.

The dual PLIF monotonic config keeps the model visual-only: frozen VTP visual
features feed short-memory and long-memory PLIF branches, which are fused before
a monotonic-biased causal Transformer decoder.

The CTC auxiliary variant keeps the same decoder output but adds a visual CTC
head during training to encourage frame-to-text alignment.

## Evaluate

```powershell
python -m l2t_arch.evaluate `
  --checkpoint checkpoints_l2t_arch_dual_path\best_model.pth `
  --manifest datasets_l2t_arch_50k\test.jsonl `
  --output-dir reports_l2t_arch_dual_path
```
