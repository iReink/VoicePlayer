"""
Модуль локального акустического эхоподавления (AEC) на базе WebRTC APM (AEC3)
и захвата системного звука через Windows WASAPI Loopback.

Соответствует спецификации локального эхоподавления:
- Работает полностью локально (0 байт в облако).
- Непрерывный адаптивный фильтр вычитает играющую музыку и вокал из микрофона.
- Обеспечивает чистый поток для KWS (wake-word) и Whisper ASR.
- Поддерживает состояния WARMING_UP, ACTIVE, DEGRADED, BYPASS, FAILED.
- Включает встроенный инструмент диагностической записи (raw, reference, clean).
"""
import os
import sys
import time
import json
import threading
import collections
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np
import scipy.io.wavfile as wavfile

try:
    import soundcard as sc
except ImportError:
    sc = None

try:
    import pywebrtc_audio as pwa
except ImportError:
    pwa = None

from logger import log_info, log_warning, log_error, log_debug

DIAGNOSTIC_DIR = Path(__file__).parent / "outputs" / "diagnostic"

class AECState(Enum):
    WARMING_UP = "WARMING_UP"
    ACTIVE = "ACTIVE"
    DEGRADED = "DEGRADED"
    BYPASS = "BYPASS"
    FAILED = "FAILED"

class AECFrontend:
    """
    Аудиофронтенд локального эхоподавления WebRTC APM.
    Принимает микрофон и опорный сигнал колонок (WASAPI Loopback),
    выполняет синхронизацию и возвращает очищенный сигнал.
    """
    def __init__(
        self,
        enabled: bool = True,
        sample_rate: int = 16000,
        frame_size: int = 160,          # 10 мс при 16 кГц
        stream_delay_ms: int = 35,       # базовая задержка распространения звука в комнате
        high_pass_filter: bool = True
    ):
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.stream_delay_ms = stream_delay_ms
        self.high_pass_filter = high_pass_filter

        self._enabled = enabled
        self._state = AECState.WARMING_UP if enabled else AECState.BYPASS
        self._lock = threading.Lock()
        self._running = False

        # WebRTC APM процессор
        self._processor: Optional[Any] = None
        self._init_processor()

        # Очередь опорного сигнала (WASAPI Loopback): 10 мс блоки
        # Максимальная длина 25 блоков (~250 мс) согласно спецификации для защиты от дрейфа
        self._far_queue = collections.deque(maxlen=25)
        self._loopback_thread: Optional[threading.Thread] = None
        self._loopback_active = False
        self._device_name = "Default Speaker"

        # Метрики и статистика
        self._blocks_processed = 0
        self._far_underruns = 0
        self._last_far_rms = 0.0
        self._last_near_rms = 0.0
        self._last_clean_rms = 0.0

        # Диагностическая запись (Раздел 10 спецификации)
        self._diag_active = False
        self._diag_end_time = 0.0
        self._diag_raw_mic: list = []
        self._diag_ref_far: list = []
        self._diag_clean_out: list = []
        self._diag_events: list = []

    def _init_processor(self):
        """Инициализация процессора WebRTC APM"""
        if pwa is None:
            log_warning("[AEC] Библиотека pywebrtc_audio не найдена, переход в режим BYPASS")
            self._state = AECState.BYPASS
            self._processor = None
            return

        try:
            self._processor = pwa.AudioProcessor(
                sample_rate=self.sample_rate,
                num_channels=1,
                echo_cancellation=True,
                noise_suppression=False,   # Чистый AEC без агрессивной резки частот
                high_pass_filter=self.high_pass_filter,
                stream_delay_ms=self.stream_delay_ms
            )
            log_info(f"[AEC] WebRTC APM инициализирован (SR={self.sample_rate}, delay={self.stream_delay_ms}ms, HPF={self.high_pass_filter})")
        except Exception as e:
            log_error(f"[AEC] Ошибка инициализации WebRTC AudioProcessor: {e}")
            self._processor = None
            self._state = AECState.FAILED

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, val: bool):
        self._enabled = bool(val)
        with self._lock:
            if not self._enabled:
                self._state = AECState.BYPASS
                log_info("[AEC] Эхоподавление отключено (режим BYPASS)")
            else:
                self._state = AECState.ACTIVE if self._loopback_active else AECState.WARMING_UP
                log_info("[AEC] Эхоподавление включено")

    def set_enabled(self, val: bool):
        """Программное включение / выключение эхоподавления"""
        self.enabled = val

    def get_last_diagnostic_summary(self) -> str:
        """Возвращает строку с последним замером ERLE или статус отсутствия замеров"""
        json_manifest = DIAGNOSTIC_DIR / "manifest.json"
        if json_manifest.exists():
            try:
                with open(json_manifest, "r", encoding="utf-8") as f:
                    data = json.load(f)
                erle = data.get("metrics", {}).get("suppression_erle_db", 0.0)
                dur = data.get("duration_seconds", 0.0)
                ts = data.get("timestamp", "")
                return f"{erle:.1f} дБ ({dur}с, {ts})"
            except Exception:
                pass
        return "тест еще не проводился"

    @property
    def state(self) -> AECState:
        return self._state

    @property
    def state_description(self) -> str:
        names = {
            AECState.ACTIVE: "Активно (WebRTC AEC3)",
            AECState.WARMING_UP: "Калибровка...",
            AECState.DEGRADED: "Ограничено (нет loopback)",
            AECState.BYPASS: "Выключено (сквозной пропуск)",
            AECState.FAILED: "Ошибка модуля"
        }
        return names.get(self._state, str(self._state.value))

    def start(self):
        """Запуск фонового потока захвата системного звука (WASAPI Loopback)"""
        if self._running:
            return
        self._running = True
        self._loopback_thread = threading.Thread(target=self._loopback_worker, daemon=True)
        self._loopback_thread.start()
        log_info("[AEC] Фоновый поток захвата WASAPI Loopback запущен")

    def stop(self):
        """Остановка фронтенда"""
        self._running = False
        if self._loopback_thread and self._loopback_thread.is_alive():
            self._loopback_thread.join(timeout=1.0)
        self._loopback_active = False
        log_info("[AEC] Фронтенд эхоподавления остановлен")

    def _loopback_worker(self):
        """
        Фоновый воркер: непрерывно захватывает PCM из звуковой карты (WASAPI Loopback)
        и складывает 10-мс блоки в кольцевую очередь.
        """
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.ole32.CoInitialize(None)
            except Exception:
                pass

        if sc is None:
            log_warning("[AEC] Модуль soundcard не установлен, loopback недоступен")
            self._state = AECState.DEGRADED
            return

        while self._running:
            try:
                spk = sc.default_speaker()
                self._device_name = spk.name
                loop_mic = sc.get_microphone(spk.name, include_loopback=True)
                log_info(f"[AEC] Подключен WASAPI Loopback для устройства: «{spk.name}»")
                
                # Захватываем блоками по 10 мс (160 сэмплов при 16 кГц)
                with loop_mic.recorder(samplerate=self.sample_rate, channels=2, blocksize=self.frame_size) as rec:
                    self._loopback_active = True
                    if self._enabled and self._processor is not None:
                        self._state = AECState.ACTIVE

                    while self._running:
                        block = rec.record(self.frame_size)
                        if block is None or len(block) == 0:
                            continue

                        # Сведение стерео-сигнала колонок в моно (L + R) / 2
                        mono = np.mean(block, axis=1, dtype=np.float32)
                        now = time.perf_counter()
                        self._far_queue.append((now, mono))

            except Exception as e:
                self._loopback_active = False
                if self._enabled:
                    self._state = AECState.DEGRADED
                log_warning(f"[AEC] Сбой захвата WASAPI Loopback ({e}). Повторное подключение через 2.5с...")
                time.sleep(2.5)

    def process_mic_block(self, raw_mic_block: np.ndarray) -> np.ndarray:
        """
        Обработка блока микрофона произвольной длины (например, 960 отсчетов = 60 мс).
        Нарезает вход на 10-мс фреймы (160 отсчетов), выравнивает с опорным сигналом
        и возвращает массив очищенного звука того же размера.
        """
        if not self._enabled or self._processor is None or self._state == AECState.BYPASS:
            return raw_mic_block

        total_samples = len(raw_mic_block)
        if total_samples < self.frame_size:
            return raw_mic_block

        clean_chunks = []
        num_frames = total_samples // self.frame_size

        for i in range(num_frames):
            near_chunk = raw_mic_block[i * self.frame_size : (i + 1) * self.frame_size]

            # Получаем опорный блок из очереди
            far_chunk = None
            if self._far_queue:
                _, far_chunk = self._far_queue.popleft()
            else:
                self._far_underruns += 1
                far_chunk = np.zeros(self.frame_size, dtype=np.float32)

            # Эхоподавление
            try:
                clean_chunk = self._processor.process(near_chunk, far_chunk)
            except Exception as e:
                log_debug(f"[AEC] Ошибка вызова processor.process: {e}")
                clean_chunk = near_chunk

            clean_chunks.append(clean_chunk)

            # Диагностическая запись
            if self._diag_active:
                self._diag_raw_mic.append(near_chunk.copy())
                self._diag_ref_far.append(far_chunk.copy())
                self._diag_clean_out.append(clean_chunk.copy())

        # Если блок не был строго кратен 160 сэмплам, сохраняем остаток
        remainder_len = total_samples % self.frame_size
        if remainder_len > 0:
            remainder = raw_mic_block[num_frames * self.frame_size:]
            clean_chunks.append(remainder)

        self._blocks_processed += 1
        return np.concatenate(clean_chunks)

    def start_diagnostic_session(self, duration_sec: float = 8.0, duration_s: Optional[float] = None) -> Path:
        """
        Запуск диагностической сессии (Раздел 10 спецификации).
        Записывает 3 потока: сырой микрофон, опорный сигнал колонок и очищенный выход.
        """
        if duration_s is not None:
            duration_sec = duration_s

        with self._lock:
            self._diag_active = True
            self._diag_end_time = time.time() + duration_sec
            self._diag_raw_mic.clear()
            self._diag_ref_far.clear()
            self._diag_clean_out.clear()
            self._diag_events.clear()
            DIAGNOSTIC_DIR.mkdir(parents=True, exist_ok=True)
            log_info(f"[AEC] Запущена диагностическая запись на {duration_sec}с в {DIAGNOSTIC_DIR}")
        return DIAGNOSTIC_DIR

    def update_diagnostic_state(self) -> Optional[Dict[str, Any]]:
        """Проверяет окончание сессии диагностики и сохраняет файлы на диск"""
        if not self._diag_active:
            return None

        if time.time() < self._diag_end_time:
            return None

        with self._lock:
            self._diag_active = False
            raw_arr = np.concatenate(self._diag_raw_mic) if self._diag_raw_mic else np.zeros(160, dtype=np.float32)
            ref_arr = np.concatenate(self._diag_ref_far) if self._diag_ref_far else np.zeros(160, dtype=np.float32)
            clean_arr = np.concatenate(self._diag_clean_out) if self._diag_clean_out else np.zeros(160, dtype=np.float32)

            # Нормализация в 16-bit PCM для стандартных WAV
            def _to_int16(arr):
                clipped = np.clip(arr, -1.0, 1.0)
                return (clipped * 32767).astype(np.int16)

            wav_raw = DIAGNOSTIC_DIR / "capture_raw.wav"
            wav_ref = DIAGNOSTIC_DIR / "render_reference.wav"
            wav_clean = DIAGNOSTIC_DIR / "capture_clean.wav"
            json_manifest = DIAGNOSTIC_DIR / "manifest.json"

            wavfile.write(str(wav_raw), self.sample_rate, _to_int16(raw_arr))
            wavfile.write(str(wav_ref), self.sample_rate, _to_int16(ref_arr))
            wavfile.write(str(wav_clean), self.sample_rate, _to_int16(clean_arr))

            # Расчёт подавления эха (ERLE)
            rms_raw = float(np.sqrt(np.mean(raw_arr**2))) if len(raw_arr) else 0.0
            rms_clean = float(np.sqrt(np.mean(clean_arr**2))) if len(clean_arr) else 0.0
            rms_ref = float(np.sqrt(np.mean(ref_arr**2))) if len(ref_arr) else 0.0

            erle_db = 0.0
            if rms_clean > 1e-6 and rms_raw > 1e-6:
                erle_db = 20.0 * np.log10(rms_raw / rms_clean)

            manifest_data = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sample_rate": self.sample_rate,
                "frame_size": self.frame_size,
                "stream_delay_ms": self.stream_delay_ms,
                "speaker_device": self._device_name,
                "duration_seconds": round(len(clean_arr) / self.sample_rate, 2),
                "metrics": {
                    "rms_raw_mic": round(rms_raw, 5),
                    "rms_loopback_ref": round(rms_ref, 5),
                    "rms_clean_output": round(rms_clean, 5),
                    "suppression_erle_db": round(erle_db, 2)
                },
                "files": {
                    "raw": str(wav_raw.name),
                    "reference": str(wav_ref.name),
                    "clean": str(wav_clean.name)
                }
            }

            with open(json_manifest, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, ensure_ascii=False, indent=2)

            log_info(f"[AEC] Диагностика сохранена: ERLE={erle_db:.1f} dB в {DIAGNOSTIC_DIR}")
            return manifest_data
