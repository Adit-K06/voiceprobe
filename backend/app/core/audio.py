import math

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_wav(path) -> tuple[np.ndarray, int]:
    """Return (mono float32 samples in [-1, 1], sample_rate)."""
    sample_rate, data = wavfile.read(str(path))

    if data.dtype == np.int16:
        audio = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        audio = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        audio = (data.astype(np.float32) - 128.0) / 128.0
    else:  # already float
        audio = data.astype(np.float32)

    if audio.ndim == 2:          # stereo -> mono
        audio = audio.mean(axis=1)

    return audio, sample_rate


def save_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio in [-1, 1] as a 16-bit mono WAV."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    wavfile.write(str(path), sample_rate, pcm)


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Change the sample rate."""
    if sr_from == sr_to:
        return audio
    g = math.gcd(sr_from, sr_to)
    up, down = sr_to // g, sr_from // g
    return resample_poly(audio, up, down).astype(np.float32)


def add_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white noise so that signal-to-noise ratio equals snr_db."""
    signal_power = float(np.mean(audio ** 2))
    if signal_power == 0.0:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0.0, math.sqrt(noise_power), size=len(audio))
    return (audio + noise).astype(np.float32)


def mu_law_roundtrip(audio: np.ndarray, mu: int = 255) -> np.ndarray:
    """Compress with mu-law, quantize to 8 bits, expand back."""
    x = np.clip(audio.astype(np.float64), -1.0, 1.0)

    # compress
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)

    # quantize to 256 levels (8 bits)
    y_q = np.round((y + 1.0) / 2.0 * 255.0) / 255.0 * 2.0 - 1.0

    # expand
    out = np.sign(y_q) * ((1.0 + mu) ** np.abs(y_q) - 1.0) / mu
    return out.astype(np.float32)


def telephone_degrade(audio, sr, snr_db=None, seed=0):
    """Make clean audio sound like a phone call. Returns (audio, sample_rate)."""
    original_len = len(audio)

    x = resample(audio, sr, 8000)

    if snr_db is not None:
        x = add_noise(x, snr_db, np.random.default_rng(seed))

    x = mu_law_roundtrip(x)
    x = resample(x, 8000, sr)

    # resampling can be off by a sample or two; force the original length
    if len(x) > original_len:
        x = x[:original_len]
    elif len(x) < original_len:
        x = np.pad(x, (0, original_len - len(x)))

    return x.astype(np.float32), sr