from __future__ import annotations

import math
from pathlib import Path

import torch


def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def _torch_mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    n_freqs = n_fft // 2 + 1
    mel_min = _hz_to_mel(torch.tensor(0.0, dtype=dtype, device=device))
    mel_max = _hz_to_mel(torch.tensor(float(sample_rate) / 2.0, dtype=dtype, device=device))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=dtype, device=device)
    hz_points = _mel_to_hz(mel_points)
    bins = torch.floor((n_fft + 1) * hz_points / float(sample_rate)).long().clamp(0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs, dtype=dtype, device=device)
    for i in range(n_mels):
        left, center, right = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
        if center > left:
            fb[i, left:center] = torch.linspace(0.0, 1.0, center - left, dtype=dtype, device=device)
        if right > center:
            fb[i, center:right] = torch.linspace(1.0, 0.0, right - center, dtype=dtype, device=device)
    return fb


def _torch_log_mel(
    wav: torch.Tensor,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_mels: int,
) -> torch.Tensor:
    wav_1d = wav.squeeze(0).float()
    window = torch.hann_window(win_length, dtype=wav_1d.dtype, device=wav_1d.device)
    spec = torch.stft(
        wav_1d,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    ).abs()
    fb = _torch_mel_filterbank(sample_rate, n_fft, n_mels, spec.dtype, spec.device)
    mel = fb @ spec
    return mel.transpose(0, 1).contiguous().clamp_min(1e-5).log()


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
        try:
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
        except Exception:
            return _torch_log_mel(wav, sr, n_fft, hop_length, win_length, n_mels)
