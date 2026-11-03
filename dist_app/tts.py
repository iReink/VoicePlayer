"""
Модуль локального синтеза речи (TTS) на базе Silero TTS v5.
Поддерживает мультиспикерную модель (Ксения Pro, Айдар, Бая, Евгений),
G2P-нормализацию латиницы, Ducking громкости и плавные затухания/паузы.
"""
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import numpy as np
import sounddevice as sd
import torch

from text_normalizer import normalize_text_for_silero
from logger import log_info, log_warning, log_error, log_debug

MODEL_PATH = Path(__file__).parent / "models" / "silero" / "v5_ru.pt"

AVAILABLE_VOICES: Dict[str, Dict[str, str]] = {
    "xenia": {
        "id": "xenia",
        "name": "Ксения",
        "description": "живой разговорный женский голос (эталон)",
        "gender": "женский"
    },
    "aidar": {
        "id": "aidar",
        "name": "Айдар",
        "description": "уверенный глубокий мужской баритон",
        "gender": "мужской"
    },
    "baya": {
        "id": "baya",
        "name": "Бая",
        "description": "мягкий и тёплый женский тембр",
        "gender": "женский"
    },
    "kseniya": {
        "id": "kseniya",
        "name": "Ксения Classic",
        "description": "строгий классический дикторский голос",
        "gender": "женский"
    },
    "eugene": {
        "id": "eugene",
        "name": "Евгений",
        "description": "энергичный молодой мужской голос",
        "gender": "мужской"
    }
}

class TTSEngine:
    def __init__(
        self,
        config_manager,
        player=None,
        on_speech_state_change: Optional[Callable[[bool], None]] = None,
        device_mode: str = "auto"
    ):
        self.config = config_manager
        self.player = player
        self.on_speech_state_change = on_speech_state_change
        self.sample_rate = 48000
        self._lock = threading.Lock()
        
        self.device = self._resolve_device(device_mode)
        self.model = None
        self._is_speaking = False

        self._init_model()

    def _resolve_device(self, mode: str) -> torch.device:
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            log_warning("CUDA недоступна для PyTorch, используется CPU")
            return torch.device("cpu")

        # Режим 'auto': на CPU инференс Silero v5 занимает всего ~180-220 мс
        # и оставляет 100% ресурсов GPU для Whisper и внешних задач.
        if torch.cuda.is_available():
            try:
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                if free_mb >= 500:
                    return torch.device("cuda")
            except Exception:
                pass
        return torch.device("cpu")

    def _init_model(self):
        log_info(f"Загрузка Silero TTS v5 на устройстве {self.device}...")
        try:
            if MODEL_PATH.exists():
                importer = torch.package.PackageImporter(str(MODEL_PATH))
                model = importer.load_pickle("tts_models", "model")
            else:
                log_info("Локальный файл модели не найден, попытка загрузки через torch.hub...")
                model, _ = torch.hub.load(
                    repo_or_dir="snakers4/silero-models",
                    model="silero_tts",
                    language="ru",
                    speaker="v5_ru",
                    trust_repo=True
                )

            model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            self.model = model
            log_info(f"Silero TTS v5 успешно загружен. Активный голос: {self.current_voice}")
        except Exception as e:
            log_error(f"Ошибка загрузки модели Silero TTS: {e}")
            self.model = None

    @property
    def current_voice(self) -> str:
        v = self.config.voice.lower() if hasattr(self.config, "voice") else "xenia"
        return v if v in AVAILABLE_VOICES else "xenia"

    def set_voice(self, voice_id: str) -> bool:
        """Переключает активный голос в настройках"""
        v_low = voice_id.lower().strip()
        if v_low in AVAILABLE_VOICES:
            self.config.voice = v_low
            log_info(f"Голос TTS переключен на: {v_low} ({AVAILABLE_VOICES[v_low]['name']})")
            return True
        return False

    def warmup(self, announce_text: Optional[str] = None):
        """Прогрев нейросети (фоновый или с приветственной фразой)"""
        if self.model is None:
            return

        def _warm():
            try:
                log_debug("Запуск прогрева Silero TTS...")
                with self._lock:
                    with torch.no_grad():
                        _ = self.model.apply_tts(
                            text="Прогрев.",
                            speaker=self.current_voice,
                            sample_rate=self.sample_rate
                        )
                log_info("Silero TTS прогрет и готов к мгновенному ответу")
                if announce_text:
                    self.say(announce_text, pause_music=False)
            except Exception as e:
                log_warning(f"Ошибка во время прогрева TTS: {e}")

        threading.Thread(target=_warm, daemon=True).start()

    def synthesize(self, text: str, voice: Optional[str] = None) -> Optional[np.ndarray]:
        """Синтезирует аудиомассив float32 с нормализацией английских слов"""
        if self.model is None or not text:
            return None

        clean_text = normalize_text_for_silero(text)
        if not clean_text:
            return None

        spk = voice or self.current_voice
        if spk not in AVAILABLE_VOICES:
            spk = "xenia"

        with self._lock:
            with torch.no_grad():
                audio_tensor = self.model.apply_tts(
                    text=clean_text,
                    speaker=spk,
                    sample_rate=self.sample_rate
                )
        return audio_tensor.detach().cpu().numpy()

    def say(self, text: str, pause_music: bool = False, voice: Optional[str] = None):
        """
        Произносит текст через sounddevice:
        - При pause_music=True (длинные ответы, меню) плавно гасит музыку за 0.5с и возобновляет после ответа.
        - При pause_music=False (короткие ответы) применяет ducking (приглушение до 15%).
        """
        if not text or not text.strip():
            return
        if hasattr(self.config, "tts_enabled") and not self.config.tts_enabled:
            return

        was_playing = False
        try:
            audio_np = self.synthesize(text, voice=voice)
            if audio_np is None:
                return

            was_playing = self.player.is_playing if self.player else False

            # Переход звука перед воспроизведением ответа
            if pause_music and was_playing:
                self.player.fade_out_pause(duration=0.5)
            elif not pause_music and was_playing:
                self.player.duck(True)

            # Сообщаем микрофону, что говорит ассистент (чтобы микрофон не слышал сам себя)
            self._is_speaking = True
            if self.on_speech_state_change:
                try:
                    self.on_speech_state_change(True)
                except Exception:
                    pass

            # Воспроизведение звука
            sd.play(audio_np, samplerate=self.sample_rate)
            sd.wait()

        except Exception as e:
            log_error(f"Ошибка воспроизведения речи через TTS: {e}")
        finally:
            self._is_speaking = False
            if self.on_speech_state_change:
                try:
                    self.on_speech_state_change(False)
                except Exception:
                    pass

            # Возврат звука после воспроизведения ответа
            if self.player and was_playing:
                if pause_music:
                    self.player.fade_in_resume(duration=0.5)
                else:
                    self.player.duck(False)

    def close(self):
        """Освобождение ресурсов TTS при завершении работы"""
        pass
