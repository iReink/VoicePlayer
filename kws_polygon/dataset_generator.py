"""
Генератор синтетического обучающего датасета KWS на базе локального Silero TTS v5
с аугментациями (скорость, шум, сдвиги, разные спикеры).
Создаёт кэш признаков dataset_cache.npz для мгновенного обучения и дообучения.
"""
import os
import sys
import time
import pathlib
import numpy as np
import scipy.signal
import torch

from audio_features import extract_mel_spectrogram, SAMPLE_RATE, WINDOW_SAMPLES

SILERO_MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "silero" / "v5_ru.pt"
DATASET_CACHE_PATH = pathlib.Path(__file__).parent / "dataset_cache.npz"

SPEAKERS = ["aidar", "eugene", "xenia", "baya", "kseniya"]

TARGET_WORDS = {
    "morsik": [
        "Морсик", "Морсик!", "Морсик?", "мОрсик", "морсИк", "Морсик, слушай"
    ],
    "karlusha": [
        "Карлуша", "Карлуша!", "Карлуша?", "карлУша", "Карлуша, слушай"
    ],
    "hey_wisp": [
        "Эй, Висп", "Эй, Висп!", "Эй Висп", "Эй... Висп", "Эй! Висп", "эй висп"
    ]
}

NEGATIVE_PHRASES = [
    # Фонетические ловушки (confusers)
    "персик", "борщ", "мостик", "морж", "носик", "котик", "барсик",
    "карлсон", "клуша", "послушай", "карусель", "ворона", "каркуша",
    "свист", "всплеск", "виски", "вирус", "эй, ты", "эй, стой", "эй, подожди", "эй, привет",
    # Обычные команды
    "включи музыку", "поставь паузу", "стоп", "сделай громче", "тише",
    "следующий трек", "яндекс музыка", "моя волна", "любимый плейлист",
    "какая погода", "который час", "доброе утро", "спокойной ночи",
    "привет, как дела", "слушай, давай поболтаем", "спасибо, пока",
    "алиса", "маруся", "салют", "гугл", "сири"
]

def load_silero_tts():
    if not SILERO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Модель Silero TTS не найдена по пути: {SILERO_MODEL_PATH}")
    importer = torch.package.PackageImporter(str(SILERO_MODEL_PATH))
    model = importer.load_pickle("tts_models", "model")
    return model

def change_speed(audio: np.ndarray, speed_factor: float) -> np.ndarray:
    """Изменяет скорость воспроизведения через ресемплинг"""
    if abs(speed_factor - 1.0) < 0.02:
        return audio
    num_samples = int(len(audio) / speed_factor)
    return scipy.signal.resample(audio, num_samples).astype(np.float32)

def add_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Добавляет белый/розовый шум с заданным отношением сигнал/шум (SNR)"""
    sig_power = np.mean(audio ** 2)
    if sig_power < 1e-6:
        return audio
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return audio + noise

def generate_augmentations(base_audio_16k: np.ndarray) -> list:
    """Генерирует варианты темпа, громкости, шума и сдвигов во времени"""
    samples = []
    speeds = [0.88, 1.0, 1.14]
    snrs = [None, 24.0, 14.0]
    
    for spd in speeds:
        scaled = change_speed(base_audio_16k, spd)
        for snr in snrs:
            noisy = add_noise(scaled, snr) if snr is not None else scaled.copy()
            
            # Пробуем разное размещение в окне 1.5 сек (24000 отсчетов)
            dur = len(noisy)
            if dur < WINDOW_SAMPLES:
                max_offset = WINDOW_SAMPLES - dur
                offsets = [0, max_offset // 2, max_offset]
                for off in offsets:
                    buf = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
                    buf[off:off+dur] = noisy
                    samples.append(buf)
            else:
                samples.append(noisy[:WINDOW_SAMPLES])
                
    return samples

def build_dataset_cache(force_rebuild: bool = False):
    """Синтезирует датасет и сохраняет в dataset_cache.npz"""
    if DATASET_CACHE_PATH.exists() and not force_rebuild:
        print(f"[*] Датасет уже существует: {DATASET_CACHE_PATH}")
        return

    print("[*] Инициализация Silero TTS для синтеза датасета...")
    tts = load_silero_tts()

    pos_mels = {k: [] for k in TARGET_WORDS}
    neg_mels = []

    print("[1/3] Синтез положительных сэмплов для всех 3 триггеров...")
    for label, phrases in TARGET_WORDS.items():
        print(f"  -> Генерация для «{label}»...")
        for spk in SPEAKERS:
            for phrase in phrases:
                try:
                    # Silero генерирует 24 кГц
                    audio_24k = tts.apply_tts(text=phrase, speaker=spk, sample_rate=24000).numpy()
                    # Ресемплинг в 16 кГц
                    num_16k = int(len(audio_24k) * 16000 / 24000)
                    audio_16k = scipy.signal.resample(audio_24k, num_16k).astype(np.float32)
                    
                    aug_list = generate_augmentations(audio_16k)
                    for aug in aug_list:
                        mel = extract_mel_spectrogram(aug)
                        pos_mels[label].append(mel)
                except Exception as e:
                    print(f"     [!] Ошибка синтеза {phrase} ({spk}): {e}")

    print("[2/3] Синтез отрицательных сэмплов (ловушки, команды, шумы)...")
    for phrase in NEGATIVE_PHRASES:
        for spk in ["aidar", "xenia", "eugene"]:
            try:
                audio_24k = tts.apply_tts(text=phrase, speaker=spk, sample_rate=24000).numpy()
                num_16k = int(len(audio_24k) * 16000 / 24000)
                audio_16k = scipy.signal.resample(audio_24k, num_16k).astype(np.float32)
                
                # По 2 варианта на фразу
                for spd in [0.95, 1.05]:
                    scaled = change_speed(audio_16k, spd)
                    mel = extract_mel_spectrogram(scaled)
                    neg_mels.append(mel)
            except Exception:
                pass

    # Синтетические шумы и тишина
    print("[3/3] Добавление акустических шумов и тишины...")
    for _ in range(60):
        # Белый / розовый шум разной амплитуды
        noise_amp = np.random.uniform(0.005, 0.08)
        noise = np.random.normal(0, noise_amp, WINDOW_SAMPLES).astype(np.float32)
        neg_mels.append(extract_mel_spectrogram(noise))
        
    for _ in range(40):
        # Тишина с микрокликами
        silence = np.random.normal(0, 0.001, WINDOW_SAMPLES).astype(np.float32)
        neg_mels.append(extract_mel_spectrogram(silence))

    # Преобразуем в numpy массивы
    data_dict = {}
    for k in TARGET_WORDS:
        data_dict[f"pos_{k}"] = np.array(pos_mels[k], dtype=np.float32)
        print(f"  [OK] «{k}»: {len(pos_mels[k])} сэмплов")
    data_dict["negatives"] = np.array(neg_mels, dtype=np.float32)
    print(f"  [OK] Негативные примеры: {len(neg_mels)} сэмплов")

    np.savez_compressed(DATASET_CACHE_PATH, **data_dict)
    print(f"[✓] Датасет успешно скомпилирован и сохранён в: {DATASET_CACHE_PATH}")

if __name__ == "__main__":
    build_dataset_cache(force_rebuild=True)
