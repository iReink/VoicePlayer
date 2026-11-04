"""
Аудио-процессинг и извлечение Mel-спектрограмм для KWS моделей
Частота дискретизации: 16000 Гц, Окно анализа: 1.5 сек (24000 отсчётов)
Спектрограмма: 40 Mel-фильтров x 151 временной шаг
"""
import numpy as np
import scipy.signal

SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 24000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160  # 10 мс шаг
WIN_LENGTH = 400  # 25 мс окно

def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def get_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = 80.0, fmax: float = 7600.0) -> np.ndarray:
    """Создаёт банк треугольных Mel-фильтров"""
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = bins[i], bins[i+1], bins[i+2]
        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / (right - center)
    return fb

# Кэшируем банк фильтров в памяти
MEL_FILTERBANK = get_mel_filterbank()
HANNING_WINDOW = np.hanning(WIN_LENGTH).astype(np.float32)

def prepare_audio_window(audio: np.ndarray, target_samples: int = WINDOW_SAMPLES) -> np.ndarray:
    """
    Приводит аудио к ровно target_samples (1.5 сек):
    - Если короче: центрирует с заполнением нулями (паддинг)
    - Если длиннее: вырезает центральный участок максимальной энергии
    """
    audio = audio.flatten().astype(np.float32)
    
    # Нормализация амплитуды
    max_val = np.max(np.abs(audio))
    if max_val > 1e-4:
        audio = audio / max_val
        
    n = len(audio)
    if n == target_samples:
        return audio
    elif n < target_samples:
        pad_total = target_samples - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        return np.pad(audio, (pad_left, pad_right), mode='constant')
    else:
        # Ищем 1.5-секундный отрезок с наибольшей энергией
        hop = 1600  # 100 мс шаг поиска
        max_energy = -1.0
        best_start = (n - target_samples) // 2
        for start in range(0, n - target_samples + 1, hop):
            chunk = audio[start:start+target_samples]
            energy = np.sum(chunk ** 2)
            if energy > max_energy:
                max_energy = energy
                best_start = start
        return audio[best_start:best_start+target_samples]

def extract_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Превращает 1.5-сек аудио (16 кГц) в лог-мел спектрограмму (40, 151)
    """
    clean_audio = prepare_audio_window(audio, WINDOW_SAMPLES)
    
    _, _, z = scipy.signal.stft(
        clean_audio,
        fs=SAMPLE_RATE,
        window=HANNING_WINDOW,
        nperseg=WIN_LENGTH,
        noverlap=WIN_LENGTH - HOP_LENGTH,
        nfft=N_FFT,
        boundary=None,
        padded=False
    )
    
    # Мощность спектра
    mag = np.abs(z) ** 2
    # Проекция на Mel-шкалу
    mel_spec = np.dot(MEL_FILTERBANK, mag)
    # Логарифмирование
    log_mel = np.log(np.maximum(mel_spec, 1e-6))
    
    # Стандартизация (zero-mean, unit-variance)
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-5
    norm_mel = (log_mel - mean) / std
    
    # Приводим к фиксированной длине по времени (151 шаг)
    target_frames = 151
    current_frames = norm_mel.shape[1]
    if current_frames < target_frames:
        norm_mel = np.pad(norm_mel, ((0, 0), (0, target_frames - current_frames)), mode='constant')
    elif current_frames > target_frames:
        norm_mel = norm_mel[:, :target_frames]
        
    return norm_mel.astype(np.float32)
