"""
KWS (Keyword Spotting) Engine для голосового ассистента VoicePlayer.
Автоматически сканирует и загружает все обученные ONNX модели из models/kws/,
вычисляет Mel-спектрограммы в реальном времени и возвращает срабатывания
с защитой от ложных тревог (порог 0.80 + кулдаун).
"""
import os
import sys
import time
import pathlib
import threading
from typing import Dict, Optional, Tuple, NamedTuple
import numpy as np
import scipy.signal
import onnxruntime as ort

from logger import log_info, log_warning, log_error, log_debug

BASE_DIR = pathlib.Path(__file__).parent.resolve()
DEFAULT_KWS_MODELS_DIR = BASE_DIR / "models" / "kws"
SOUNDS_DIR = BASE_DIR / "sounds"
WAKE_CHIME_FILE = SOUNDS_DIR / "wake_chime.wav"

# Аудио-параметры (строго согласованы с обучением)
SAMPLE_RATE = 16000
WINDOW_SECONDS = 1.5
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 24000
N_MELS = 40
N_FFT = 512
HOP_LENGTH = 160
WIN_LENGTH = 400

def hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def get_mel_filterbank(sr: int = SAMPLE_RATE, n_fft: int = N_FFT, n_mels: int = N_MELS, fmin: float = 80.0, fmax: float = 7600.0) -> np.ndarray:
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

MEL_FILTERBANK = get_mel_filterbank()
HANNING_WINDOW = np.hanning(WIN_LENGTH).astype(np.float32)

def extract_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Приводит аудио (16 кГц) к лог-мел спектрограмме (1, 1, 40, 151) для ONNX модели.
    """
    audio = audio.flatten().astype(np.float32)
    
    # Нормализация амплитуды
    max_val = np.max(np.abs(audio))
    if max_val > 1e-4:
        audio = audio / max_val
        
    n = len(audio)
    if n < WINDOW_SAMPLES:
        pad_total = WINDOW_SAMPLES - n
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        clean_audio = np.pad(audio, (pad_left, pad_right), mode='constant')
    elif n > WINDOW_SAMPLES:
        # Берём последние 1.5 секунды аудио (хвост речевого потока)
        clean_audio = audio[-WINDOW_SAMPLES:]
    else:
        clean_audio = audio

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
    
    mag = np.abs(z) ** 2
    mel_spec = np.dot(MEL_FILTERBANK, mag)
    log_mel = np.log(np.maximum(mel_spec, 1e-6))
    
    mean = np.mean(log_mel)
    std = np.std(log_mel) + 1e-5
    norm_mel = (log_mel - mean) / std
    
    # Фиксированная длина 151 фрейм
    target_frames = 151
    current_frames = norm_mel.shape[1]
    if current_frames < target_frames:
        norm_mel = np.pad(norm_mel, ((0, 0), (0, target_frames - current_frames)), mode='constant')
    elif current_frames > target_frames:
        norm_mel = norm_mel[:, :target_frames]
        
    # Добавляем размерности батча и канала: (1, 1, 40, 151)
    return np.expand_dims(np.expand_dims(norm_mel.astype(np.float32), 0), 0)

class KWSResult(NamedTuple):
    model_name: str
    probability: float
    all_scores: Dict[str, float]

WAKE_WORD_DISPLAY_NAMES = {
    "morsik": "Морсик",
    "karlusha": "Карлуша",
    "hey_wisp": "Эй Висп"
}

def play_wake_chime_async():
    """Асинхронно воспроизводит приятный звуковой сигнал (chime) подтверждения wake-word"""
    def _play():
        if not WAKE_CHIME_FILE.exists():
            return
        try:
            import pygame
            if pygame.mixer.get_init():
                sound = pygame.mixer.Sound(str(WAKE_CHIME_FILE))
                sound.set_volume(0.85)
                sound.play()
                return
        except Exception:
            pass

        # Fallback на winsound в Windows
        try:
            import winsound
            winsound.PlaySound(str(WAKE_CHIME_FILE), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception:
            pass

    threading.Thread(target=_play, daemon=True).start()

class KWSEngine:
    """
    Движок непрерывного обнаружения ключевых слов.
    Загружает все модели из директории models/kws/*.onnx.
    """
    def __init__(
        self,
        models_dir: Optional[pathlib.Path] = None,
        threshold: float = 0.85,
        cooldown_sec: float = 2.0
    ):
        self.models_dir = pathlib.Path(models_dir) if models_dir else DEFAULT_KWS_MODELS_DIR
        self.threshold = threshold
        self.cooldown_sec = cooldown_sec
        self.last_trigger_time = 0.0

        self.sessions: Dict[str, ort.InferenceSession] = {}
        self.input_names: Dict[str, str] = {}
        self.output_names: Dict[str, str] = {}
        
        self.reload_models()

    def reload_models(self):
        """Сканирует models_dir и загружает/перезагружает все *.onnx модели"""
        self.sessions.clear()
        self.input_names.clear()
        self.output_names.clear()

        if not self.models_dir.exists():
            log_warning(f"[KWS] Папка моделей не найдена: {self.models_dir}")
            return

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        for onnx_path in self.models_dir.glob("*.onnx"):
            name = onnx_path.stem
            try:
                sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.sessions[name] = sess
                self.input_names[name] = sess.get_inputs()[0].name
                self.output_names[name] = sess.get_outputs()[0].name
                log_info(f"[KWS] Загружена модель wake-word: «{name}» ({onnx_path.name})")
            except Exception as e:
                log_error(f"[KWS] Ошибка загрузки модели {onnx_path.name}: {e}")

        log_info(f"[KWS] Всего активно wake-word моделей: {len(self.sessions)}")

    @property
    def has_models(self) -> bool:
        return len(self.sessions) > 0

    @property
    def active_trigger_names(self) -> list:
        names = []
        for k in sorted(self.sessions.keys()):
            names.append(WAKE_WORD_DISPLAY_NAMES.get(k, k.capitalize()))
        return names

    def process_audio(self, audio_window_16k: np.ndarray) -> Optional[KWSResult]:
        """
        Проверяет 1.5-секундный отрезок аудио по всем загруженным моделям.
        Возвращает KWSResult, если хотя бы одна модель выдала вероятность >= threshold,
        иначе None.
        """
        if not self.sessions:
            return None

        now = time.time()
        if now - self.last_trigger_time < self.cooldown_sec:
            return None

        # Быстрый RMS чек: если уровень сигнала слишком низкий (< -42 dB), пропускаем инференс
        rms = np.sqrt(np.mean(audio_window_16k ** 2))
        if rms < 0.008:
            return None

        mel = extract_mel_spectrogram(audio_window_16k)

        all_scores = {}
        best_name = None
        best_prob = -1.0

        for name, sess in self.sessions.items():
            try:
                inp_name = self.input_names[name]
                out_name = self.output_names[name]
                out = sess.run([out_name], {inp_name: mel})
                prob = float(out[0][0, 0])
            except Exception as e:
                prob = 0.0

            all_scores[name] = round(prob, 4)
            if prob > best_prob:
                best_prob = prob
                best_name = name

        if best_prob >= self.threshold and best_name is not None:
            self.last_trigger_time = now
            log_info(f"[KWS] 🔔 СРАБАТЫВАНИЕ: «{best_name}» (уверенность: {best_prob*100:.1f}%, порог: {self.threshold*100:.0f}%) | все: {all_scores}")
            return KWSResult(model_name=best_name, probability=best_prob, all_scores=all_scores)

        return None
