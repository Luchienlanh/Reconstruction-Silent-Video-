from __future__ import annotations

import torch

from l2t_arch.text import CharTokenizer


def decode_ctc_batch(logits: torch.Tensor, tokenizer: CharTokenizer) -> list[str]:
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    return [tokenizer.decode_ctc(seq) for seq in ids]


@torch.no_grad()
def decode_seq2seq_batch(model, batch: dict, tokenizer: CharTokenizer, max_len: int = 120) -> list[str]:
    device = next(model.parameters()).device
    bsz = int(batch["vtp_tokens"].shape[0])
    generated = torch.full((bsz, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    kwargs = {
        "visuals": batch["visuals"].to(device),
        "visual_lengths": batch["visual_lengths"].to(device),
        "vtp_tokens": batch["vtp_tokens"].to(device),
    }
    for _ in range(max_len):
        out = model(gt_in=generated, **kwargs)
        next_id = out["logits"][:, -1].argmax(dim=-1)
        generated = torch.cat([generated, next_id.unsqueeze(1)], dim=1)
        finished |= next_id.eq(tokenizer.eos_id)
        if bool(finished.all()):
            break
    return [tokenizer.decode(seq) for seq in generated.detach().cpu().tolist()]

