import os
import gc
import glob
import math
import random
import warnings
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from spikingjelly.activation_based import neuron, surrogate, functional
try:
    from speechbrain.inference.vocoders import HIFIGAN
except ImportError:
    pass
import types

def find_checkpoint():
    candidates = []

    if CHECKPOINT_PATH is not None:
        candidates.append(CHECKPOINT_PATH)

    if "checkpoint_dir" in globals():
        candidates.append(os.path.join(checkpoint_dir, "best_model.pth"))

    enc = globals().get("ENCODER_TYPE", "non_snn")
    target = globals().get("TARGET_TYPE", "mel_hifigan")
    lm_tag = "lm" if globals().get("USE_LANDMARKS", True) else "visual"
    candidates.append(f"my_checkpoints_{enc}_{target}_{lm_tag}/best_model.pth")
    candidates.append(f"my_checkpoints_{enc}/best_model.pth")

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Khong tim thay checkpoint. Hay set CHECKPOINT_PATH = r'.../best_model.pth'"
    )

def load_hifigan_cpu():
    return HIFIGAN.from_hparams(
        source=HIFIGAN_SOURCE,
        savedir=HIFIGAN_SAVEDIR,
        local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        run_opts={"device": "cpu"},
    )

def smoke_test_encoder(encoder_type="non_snn", T=30, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_encoder = build_encoder(encoder_type).to(device).eval()
    test_decoder = TFiLMSIRENDecoder(out_dim=80).to(device).eval()
    dummy_video = torch.randn(2, 1, T, 112, 112, device=device)
    with torch.no_grad():
        z = test_encoder(dummy_video)
        audio_pred = test_decoder(z)
    print(f"{encoder_type} z:", z.shape, "finite:", torch.isfinite(z).all().item())
    print("audio_pred:", audio_pred.shape, "finite:", torch.isfinite(audio_pred).all().item())
    del test_encoder, test_decoder, dummy_video, z, audio_pred
    if device.type == 'cuda':
        torch.cuda.empty_cache()

