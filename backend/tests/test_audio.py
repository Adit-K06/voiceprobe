import numpy as np

from app.core.audio import resample, add_noise, mu_law_roundtrip, telephone_degrade


def sine(sr=16000, secs=1.0, freq=440.0, amp=0.5):
    t = np.arange(int(sr * secs)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_resample_length():
    x = sine(16000)
    y = resample(x, 16000, 8000)
    assert len(y) == 8000


def test_noise_hits_target_snr():
    x = sine()
    noisy = add_noise(x, 10, np.random.default_rng(0))
    noise = noisy - x
    measured = 10 * np.log10(np.mean(x ** 2) / np.mean(noise ** 2))
    assert abs(measured - 10) < 0.5


def test_noise_is_reproducible():
    x = sine()
    a = add_noise(x, 10, np.random.default_rng(1))
    b = add_noise(x, 10, np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_mu_law_is_close():
    x = sine()
    y = mu_law_roundtrip(x)
    assert np.max(np.abs(x - y)) < 0.05


def test_degrade_keeps_length_and_rate():
    x = sine()
    y, sr = telephone_degrade(x, 16000, snr_db=15)
    assert sr == 16000
    assert len(y) == len(x)