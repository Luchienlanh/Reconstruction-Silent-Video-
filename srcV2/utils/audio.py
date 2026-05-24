from __future__ import annotations

import math
from pathlib import Path

import torch


def load_waveform(path: str | Path, sample_rate: int) -> tuple[torch.Tensor, int]:
    try:
        import torchaudio

        wav, sr = torchaudio.load(str(path))
        wav = wav.float()
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != int(sample_rate):
            wav = torchaudio.functional.resample(wav, int(sr), int(sample_rate))
            sr = int(sample_rate)
        return wav.clamp(-1.0, 1.0), int(sr)
    except Exception:
        from scipy import signal
        from scipy.io import wavfile

        sr, data = wavfile.read(str(path))
        wav = torch.from_numpy(data.astype("float32"))
        if wav.ndim == 2:
            wav = wav.mean(dim=1)
        if data.dtype.name == "int16":
            wav = wav / 32768.0
        elif data.dtype.name == "int32":
            wav = wav / 2147483648.0
        wav = wav.view(1, -1)
        if int(sr) != int(sample_rate):
            gcd = math.gcd(int(sr), int(sample_rate))
            wav_np = signal.resample_poly(wav.numpy(), int(sample_rate) // gcd, int(sr) // gcd, axis=1)
            wav = torch.from_numpy(wav_np).float()
            sr = int(sample_rate)
        return wav.clamp(-1.0, 1.0), int(sr)


def log_mel_from_audio(
    path: str | Path,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
) -> torch.Tensor:
    wav, sr = load_waveform(path, sample_rate)
    try:
        import torchaudio

        mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            center=True,
        )
        mel = mel_fn(wav).squeeze(0).transpose(0, 1).contiguous()
        return torch.log(mel.clamp_min(1e-5))
    except Exception:
        import librosa

        y = wav.squeeze(0).cpu().numpy()
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=1.0,
            center=True,
        )
        return torch.from_numpy(mel.T).float().clamp_min(1e-5).log()
