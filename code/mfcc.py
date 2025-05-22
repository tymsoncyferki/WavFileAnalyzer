import numpy as np
from scipy.fftpack import dct

def pre_emphasis(signal, alpha=0.97):
    """
    applies pre-emphasis filter to boost high frequencies

    Args:
        signal (np.array): input raw audio signal
        alpha (float): coefficient (default is 0.97)

    Returns:
        np.array: pre-emphasized signal
    """
    return np.append(signal[0], signal[1:] - alpha * signal[:-1])

def framing_and_windowing(signal, sample_rate, frame_size=0.025, frame_stride=0.01):
    """
    splits signal into overlapping frames and applies Hamming window

    Args:
        signal (np.array): pre-emphasized audio signal
        sample_rate (int): sampling rate in Hz
        frame_size (float): frame length in seconds (default 0.025)
        frame_stride (float): stride between frames in seconds (default 0.01)

    Returns:
        np.array: windowed frames of shape (num_frames, frame_length)
    """
    frame_len = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_len = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_len - frame_len)) / frame_step)) + 1

    pad_len = num_frames * frame_step + frame_len
    pad_signal = np.append(signal, np.zeros((pad_len - signal_len)))

    indices = (np.tile(np.arange(0, frame_len), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T)
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    hamming = np.hamming(frame_len)
    return frames * hamming

def compute_power_spectrum(frames, n_fft=512):
    """
    computes the power spectrum of each frame using FFT

    Args:
        frames (np.array): windowed frames
        n_fft (int): number of FFT points (default 512)

    Returns:
        np.array: power spectrum of shape (num_frames, n_fft/2 + 1)
    """
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    power_spectrum = (1.0 / n_fft) * (mag_frames ** 2)
    return power_spectrum

def mel_filter_bank(n_filt, n_fft, sample_rate):
    """
    creates a Mel filter bank with triangular filters

    Args:
        n_filt (int): number of Mel filters
        n_fft (int): number of FFT points
        sample_rate (int): sampling rate in Hz

    Returns:
        np.array: filter bank matrix of shape (n_filt, n_fft/2 + 1).
    """
    low_mel = 0
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_mel, high_mel, n_filt + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((n_filt, int(n_fft / 2 + 1)))
    for m in range(1, n_filt + 1):
        f_m_minus = bin[m - 1]
        f_m = bin[m]
        f_m_plus = bin[m + 1]

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    return fbank

def apply_log_energy(power_spectrum, fbank):
    """
    applies mel filter bank to power spectrum and performs log compression

    Args:
        power_spectrum (np.array): power spectrum of frames
        fbank (np.array): mel filter bank

    Returns:
        np.array: logarithmic mel-filtered energies
    """
    filter_banks = np.dot(power_spectrum, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    return np.log(filter_banks)

def compute_dct(log_filter_banks, num_ceps=13):
    """
    applies discrete cosine transform (DCT) to log mel filterbank energies

    Args:
        log_filter_banks (np.array): log-scaled Mel filterbank energies
        num_ceps (int): number of MFCC coefficients to return (excluding c0)

    Returns:
        np.array: MFCCs of shape (num_frames, num_ceps)
    """
    mfcc = dct(log_filter_banks, type=2, axis=1, norm='ortho')
    return mfcc[:, 1:num_ceps+1]  # Exclude c0

def mfcc(signal, sample_rate, n_ceps=13, n_filt=26, n_fft=512, frame_size=0.025, frame_stride=0.01):
    """
    Calculates MFCC features from an audio signal

    Args:
        signal (np.array): raw audio signal
        sample_rate (int): sampling rate of the signal in Hz
        num_ceps (int): number of cepstral coefficients to return (default 13)
        nfilt (int): number of Mel filters to use (default 26)
        NFFT (int): number of FFT points (default 512)
        frame_size (float): frame duration in seconds (default 0.025)
        frame_stride (float): stride duration in seconds (default 0.01)

    Returns:
        np.array: MFCC feature matrix of shape (num_frames, num_ceps)
    """

    emphasized_signal = pre_emphasis(signal)

    frames = framing_and_windowing(
        emphasized_signal,
        sample_rate,
        frame_size=frame_size,
        frame_stride=frame_stride
    )

    power_spec = compute_power_spectrum(frames, n_fft)

    mel_filters = mel_filter_bank(n_filt, n_fft, sample_rate)

    log_mel_energies = apply_log_energy(power_spec, mel_filters)

    mfcc_features = compute_dct(log_mel_energies, n_ceps)

    return np.float32(mfcc_features)
