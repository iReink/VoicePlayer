"""
Модуль голосового ввода на базе Whisper-Small (GPU CUDA / CPU) с поддержкой логирования и нечеткого триггера
"""
import os
import sys
import re
import time
import threading
import collections
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
import sounddevice as sd
import ctranslate2
from faster_whisper import WhisperModel
from vad_engine import get_stateful_vad, StatefulSileroVAD

from logger import log_debug, log_info, log_warning, log_error
from kws_engine import KWSEngine, play_wake_chime_async
from aec_engine import AECFrontend

MODEL_DIR = Path(__file__).parent / "models" / "whisper-small"

def setup_cuda_dlls():
    """Добавляет каталоги с NVIDIA DLL (cublas, cudnn) в PATH и DLL search directory"""
    try:
        base_dir = Path(__file__).parent
        possible_dirs = [
            base_dir / ".venv" / "Lib" / "site-packages" / "nvidia",
            base_dir / "runtime" / "Lib" / "site-packages" / "nvidia",
            Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia",
        ]
        for nvidia_dir in possible_dirs:
            if nvidia_dir.exists():
                for bin_dir in nvidia_dir.glob("*/bin"):
                    if bin_dir.is_dir():
                        bin_path = str(bin_dir)
                        if bin_path not in os.environ.get("PATH", ""):
                            os.environ["PATH"] = bin_path + os.pathsep + os.environ.get("PATH", "")
                        try:
                            os.add_dll_directory(bin_path)
                        except Exception:
                            pass
    except Exception:
        pass

setup_cuda_dlls()

RESERVED_COMMAND_WORDS = {
    'стоп', 'пауза', 'включи', 'включить', 'поставь', 'сыграй', 'играй', 'запусти',
    'дальше', 'назад', 'следующий', 'предыдущий', 'громче', 'тише', 'лайк', 'дизлайк',
    'хватит', 'слушай', 'песня', 'трек', 'музыка', 'плейлист', 'волна', 'stop', 'pause', 'play'
}

WHISPER_HALLUCINATION_PATTERNS = [
    r'субтитр',
    r'редактор\s+субтитр',
    r'бойков[аеу]',
    r'переводчик',
    r'перевод\s+и\s+озвучк',
    r'озвучен[оа]',
    r'dimatorzok',
    r'спасибо\s+за\s+(?:просмотр|внимание)',
    r'ставит?е?\s+лайк',
    r'подписывайтесь',
    r'продолжение\s+следует',
    r'до\s+(?:скорых?|новых)\s+встреч',
    r'до\s+свидания',
    r'всем\s+пока',
    r'конец\s+фильма',
    r'популярные\s+артисты',
    r'^команды[\.\s]*$',
]

def is_whisper_hallucination(text: str) -> bool:
    """
    Проверяет, является ли распознанный текст типичной галлюцинацией Whisper
    (субтитры из обучающего датасета YouTube/фильмов или артефакты тишины/эха)
    """
    if not text:
        return True
    t = text.lower().strip()
    
    # 1. Проверка по шаблонам субтитров и титров
    for pattern in WHISPER_HALLUCINATION_PATTERNS:
        if re.search(pattern, t):
            return True
            
    # 2. Повторяющиеся бессмысленные слова-зацикливания Whisper ("дальше дальше дальше")
    words = t.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
        
    return False

WAKE_WORD_PREFIX_REGEX = re.compile(
    r'^(?:'
    r'(?:(?:эй|hey|хей|и|ei)\s*[,.\-—:]*\s*(?:висп[а-яa-z]*|wisp|vispa?|вис\s+по))'
    r'|(?:висп[а-яa-z]*|wisp|vispa)'
    r'|(?:к[ао]рлуш[а-яa-z]*)'
    r'|(?:[гм]ор[сф]ик[а-яa-z]*)'
    r'|(?:karlusha|morsik)'
    r'|(?:эй|hey|хей)'
    r')(?:\b|[^\w])',
    re.IGNORECASE
)

def strip_wake_word(text: str) -> str:
    """
    Удаляет слова активации («Морсик», «Карлуша», «Эй Висп», «Висп» и их фонетические варианты)
    из начала произнесённой фразы. Если во фразе было только имя ассистента — возвращает пустую строку.
    """
    cleaned = text.strip()
    while True:
        m = WAKE_WORD_PREFIX_REGEX.match(cleaned)
        if not m:
            break
        cleaned = cleaned[m.end():].strip()
        cleaned = re.sub(r'^[^\w\s]+', '', cleaned).strip()
    cleaned = re.sub(r'[^\w\s]+$', '', cleaned).strip()
    return cleaned

def longest_common_substring(s1: str, s2: str) -> int:
    """Возвращает длину наибольшей непрерывной общей подстроки между двумя строками"""
    return SequenceMatcher(None, s1, s2).find_longest_match(0, len(s1), 0, len(s2)).size

def is_trigger_match(word: str, trigger: str, threshold: float = 0.58) -> bool:
    """
    Универсальный нечеткий алгоритм сопоставления слова с триггером:
    - Точное совпадение
    - Исключение резервных командных слов (чтобы 'стоп' или 'включи' не принимались за триггер)
    - Ограничение по относительной длине слова (не более чем в 1.8 раза и не менее 50% длины)
    - Сходство SequenceMatcher.ratio >= threshold (0.58)
    - Или наибольшая общая подстрока (LCS) >= 50% длины триггера и >= 45% длины произнесенного слова
    """
    w = re.sub(r'^[^\w]+|[^\w]+$', '', word).lower().strip()
    t = re.sub(r'^[^\w]+|[^\w]+$', '', trigger).lower().strip()
    if not w or not t:
        return False
    if w == t:
        return True

    # Резервные слова не могут быть триггером случайно
    if w in RESERVED_COMMAND_WORDS and t != w:
        return False

    # Слишком большая разница по длине слова исключает совпадение
    if len(w) < max(2, len(t) * 0.5) or len(w) > len(t) * 1.8:
        return False

    # Прямой коэффициент схожести строк (SequenceMatcher)
    ratio = SequenceMatcher(None, w, t).ratio()
    if ratio >= threshold:
        return True

    # Проверка наибольшей общей непрерывной подстроки (LCS)
    lcs = longest_common_substring(w, t)
    min_lcs_len = min(3, len(t))
    if lcs >= min_lcs_len:
        if (lcs / len(t) >= 0.50) and (lcs / len(w) >= 0.45):
            return True

    return False

def find_trigger_in_phrase(phrase: str, trigger: str) -> Tuple[bool, str]:
    """
    Ищет триггер в начале фразы (среди первых 3-4 слов):
    1. Проверяет одиночные слова
    2. Проверяет склейку двух соседних слов (на случай разбивки Whisper: 'эр глуша', 'кар луша')
    Возвращает (найден_ли, оставшаяся_команда)
    """
    words = phrase.split()
    if not words:
        return False, ""

    clean_words = [re.sub(r'^[^\w]+|[^\w]+$', '', w).lower() for w in words]
    check_len = min(len(words), 4)

    # 1. Проверяем одиночные слова
    for i in range(check_len):
        if is_trigger_match(clean_words[i], trigger):
            remains = " ".join(words[i+1:]).strip()
            return True, remains

    # 2. Проверяем пары соседних слов (склейка разбитых слогов)
    for i in range(min(len(words) - 1, 3)):
        pair = clean_words[i] + clean_words[i+1]
        if is_trigger_match(pair, trigger):
            remains = " ".join(words[i+2:]).strip()
            return True, remains

    return False, ""

def get_wasapi_input_device() -> Optional[int]:
    """Поиск индекса микрофона по умолчанию в подсистеме Windows WASAPI для минимальной задержки (~3 мс)"""
    try:
        host_apis = sd.query_hostapis()
        wasapi_idx = next((i for i, api in enumerate(host_apis) if 'WASAPI' in api['name']), None)
        if wasapi_idx is None:
            return None
        default_in = sd.query_devices(kind='input')
        def_name = default_in['name'].strip()
        # 1. Точное совпадение
        for i, d in enumerate(sd.query_devices()):
            if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                if d['name'].strip() == def_name:
                    return i
        # 2. Нестрогое совпадение
        for i, d in enumerate(sd.query_devices()):
            if d['hostapi'] == wasapi_idx and d['max_input_channels'] > 0:
                if def_name in d['name'] or d['name'] in def_name:
                    return i
        return host_apis[wasapi_idx].get('default_input')
    except Exception:
        return None

class VoiceListener:
    def __init__(
        self,
        config_manager,
        on_command: Callable[[str, str], None],
        on_status: Optional[Callable[[str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[bool], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        is_music_playing: Optional[Callable[[], bool]] = None,
        on_wake_word: Optional[Callable[[str], None]] = None
    ):
        self.config = config_manager
        self.on_command = on_command
        self.on_status = on_status
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_log = on_log
        self.is_music_playing = is_music_playing
        self.on_wake_word = on_wake_word

        self.kws_engine = KWSEngine(threshold=0.85)
        self.aec = AECFrontend(enabled=getattr(self.config, 'aec_enabled', True))

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[WhisperModel] = None
        self._is_tts_active: bool = False
        self._force_command_mode: bool = False
        self._listen_command_window_until: float = 0.0
        self._is_paused: bool = False

    def open_command_window(self, duration: float = 6.0):
        """
        Открывает временное окно прямого прослушивания команды без повторного wake-word.
        Используется после ответа ассистента «Слушаю вас».
        """
        self._listen_command_window_until = time.time() + duration
        log_info(f"Открыто окно ожидания голосовой команды на {duration}с (KWS не требуется)")
        self._notify_status("LISTENING")

    def pause_listening(self):
        """Временно освободить микрофон для сторонних сессий (например, Live API)"""
        self._is_paused = True
        log_info("VoiceListener приостановлен (микрофон освобождён)")

    def resume_listening(self):
        """Возобновить прослушивание микрофона"""
        self._is_paused = False
        log_info("VoiceListener возобновлён")

    def set_tts_active(self, active: bool):
        """Уведомляет о начале/окончании речи TTS для отключения записи собственного голоса"""
        self._is_tts_active = active

    def set_force_command_mode(self, enabled: bool):
        """Включает режим прямого голосового ответа (без wake-word) для меню выбора голоса"""
        self._force_command_mode = enabled

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            model_path = str(MODEL_DIR) if MODEL_DIR.exists() else "small"
            
            # Пробуем инициализировать на GPU (CUDA)
            if ctranslate2.get_cuda_device_count() > 0:
                try:
                    log_info(f"Инициализация WhisperModel на CUDA (compute_type=auto): {model_path}")
                    cuda_model = WhisperModel(model_path, device="cuda", compute_type="auto")
                    # Проверяем реальное выполнение матричных вычислений cuBLAS
                    dummy_test = np.zeros(1600, dtype=np.float32)
                    _ = list(cuda_model.transcribe(dummy_test)[0])
                    self._model = cuda_model
                    log_info(f"Whisper успешно проверен и запущен на GPU (CUDA, {self._model.model.compute_type})")
                except Exception as e:
                    log_warning(f"Не удалось запустить Whisper на CUDA ({e}). Переключение на CPU.")
                    self._model = None

            # Fallback на CPU
            if self._model is None:
                log_info(f"Инициализация WhisperModel на CPU (compute_type=int8): {model_path}")
                self._model = WhisperModel(model_path, device="cpu", compute_type="int8")
                log_info("Whisper успешно загружен на CPU")
                
        return self._model

    def start(self):
        """Запуск прослушивания микрофона в фоне"""
        if self._running:
            return
        self._running = True
        self.aec.start()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        log_info("Поток VoiceListener запущен")

    def stop(self):
        """Остановка прослушивания"""
        self._running = False
        self.aec.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        log_info("Остановка VoiceListener")

    def _notify_status(self, status: str):
        if self.on_status:
            try:
                self.on_status(status)
            except Exception:
                pass

    def _listen_loop(self):
        sample_rate = 16000
        # 960 сэмплов = 60 мс (кратно 160 сэмплам / 10 мс кадрам WebRTC APM AEC3)
        block_size = 960
        
        max_silence_blocks = int(0.70 / (block_size / sample_rate))  # 0.70 сек тишины
        min_speech_blocks = int(0.35 / (block_size / sample_rate))   # 0.35 сек речи
        max_speech_blocks = int(8.0 / (block_size / sample_rate))    # максимум 8 сек
        
        # Инициализация нейросетевого VAD (Silero) для устойчивого отсечения шумов вентиляторов, эха и колонок
        # Инициализация нейросетевого VAD (Silero) с сохранением состояния
        vad = get_stateful_vad()
        if vad is not None:
            log_info("Потоковый Silero VAD (stateful) успешно инициализирован")
        else:
            log_warning("Не удалось загрузить Silero VAD, используется адаптивный RMS детектор")

        consecutive_unrecognized = 0
        backoff_until = 0.0

        try:
            model = self._get_model()
        except Exception as e:
            log_error(f"Критическая ошибка загрузки WhisperModel: {e}")
            if self.on_log:
                self.on_log(f"[bold red]Ошибка загрузки Whisper:[/bold red] {e}")
            return

        while self._running:
            if self._is_paused:
                time.sleep(0.1)
                continue
            try:
                log_debug("Открытие аудиопотока микрофона...")
                wasapi_device = get_wasapi_input_device()
                if wasapi_device is not None:
                    try:
                        dev_name = sd.query_devices(wasapi_device)['name']
                        log_info(f"Используется WASAPI микрофон: '{dev_name}' (индекс {wasapi_device}, ~3 мс задержка)")
                    except Exception:
                        log_info(f"Используется WASAPI микрофон (индекс {wasapi_device})")
                else:
                    log_info("Используется системный микрофон по умолчанию")

                with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=block_size, device=wasapi_device) as stream:
                    # Кольцевой буфер KWS: 1.5 сек (25 блоков по 960 сэмплов)
                    ring_buffer_blocks = int(1.5 * sample_rate / block_size)
                    kws_ring_buffer = collections.deque(maxlen=ring_buffer_blocks)
                    pre_roll_buffer = collections.deque(maxlen=5)  # 5 блоков по 60 мс = 300 мс pre-roll
                    blocks_since_kws = 0
                    is_kws_command = False
                    kws_trigger_name = None

                    audio_buffer = []
                    is_recording = False
                    silence_blocks = 0
                    noise_floor = 0.005
                    log_info(f"Аудиопоток микрофона открыт и слушает (KWS активен: {self.kws_engine.has_models}, AEC: {self.aec.state_description})")

                    while self._running and not self._is_paused:
                        block, overflowed = stream.read(block_size)
                        if self._is_tts_active:
                            is_recording = False
                            audio_buffer = []
                            silence_blocks = 0
                            kws_ring_buffer.clear()
                            if vad is not None:
                                vad.reset()
                            continue

                        now = time.time()
                        if now < backoff_until and not is_recording:
                            continue

                        if overflowed:
                            log_warning("Audio buffer overflowed")

                        raw_block_flat = block.flatten().astype(np.float32)

                        # Локальное эхоподавление (WebRTC APM AEC3)
                        block_flat = self.aec.process_mic_block(raw_block_flat)
                        pre_roll_buffer.append(block_flat.copy())

                        # Проверка завершения фоновой сессии диагностики AEC
                        diag_result = self.aec.update_diagnostic_state()
                        if diag_result is not None:
                            erle = diag_result.get("erle_db") or diag_result.get("metrics", {}).get("suppression_erle_db", 0.0)
                            manifest = diag_result.get("manifest_path", "")
                            log_info(f"Сессия диагностики AEC завершена: ERLE = {erle:.2f} dB, манифест: {manifest}")
                            if self.on_log:
                                self.on_log(f"[bold green]✓ Диагностика AEC завершена:[/bold green] Подавление эха: [bold cyan]{erle:.1f} дБ[/bold cyan]")

                        rms = float(np.sqrt(np.mean(block_flat**2)))
                        music_active = self.is_music_playing() if self.is_music_playing else False

                        # Оценка вероятности речи нейросетью Silero VAD (потоковая, на каждом блоке)
                        vad_prob = 0.0
                        if vad is not None:
                            try:
                                vad_prob, _ = vad.process_block(block_flat)
                            except Exception:
                                vad_prob = 0.0

                        # Адаптация шумового фона для fallback
                        if not is_recording:
                            noise_floor = noise_floor * 0.96 + min(rms, 0.050) * 0.04

                        if self._listen_command_window_until > 0.0 and now >= self._listen_command_window_until:
                            log_debug("Окно прямого ожидания команды закрыто по таймауту")
                            self._listen_command_window_until = 0.0
                            if not is_recording and self.on_speech_end:
                                try:
                                    self.on_speech_end(False)
                                except Exception:
                                    pass

                        in_command_window = (now < self._listen_command_window_until)
                        use_kws = self.kws_engine.has_models and not self._force_command_mode and not in_command_window

                        if not is_recording:
                            if use_kws:
                                # KWS режим: накапливаем скользящее окно 1.5 сек и проверяем каждые 2 блока (~128 мс)
                                kws_ring_buffer.append(block_flat.copy())
                                blocks_since_kws += 1
                                if blocks_since_kws >= 2 and len(kws_ring_buffer) >= (ring_buffer_blocks // 2):
                                    blocks_since_kws = 0
                                    window_16k = np.concatenate(list(kws_ring_buffer))
                                    # Валидация человеческой речи через потоковый Silero VAD перед KWS
                                    # Исключает ложные срабатывания от шума кулеров, фонового шипения и шорохов
                                    if vad is not None:
                                        has_speech = vad.has_sustained_speech(threshold=0.50, min_frames=2)
                                    else:
                                        has_speech = (rms >= 0.012)

                                    if has_speech:
                                        kws_res = self.kws_engine.process_audio(window_16k)
                                        if kws_res is not None:
                                            log_info(f"🔔 KWS обнаружен: «{kws_res.model_name}» ({kws_res.probability*100:.1f}%)")
                                            kws_trigger_name = kws_res.model_name
                                            is_kws_command = True
                                            is_recording = True
                                            silence_blocks = 0

                                            if self.on_wake_word:
                                                try:
                                                    self.on_wake_word(kws_res.model_name)
                                                except Exception:
                                                    pass

                                            play_wake_chime_async()

                                            if self.on_speech_start:
                                                try:
                                                    self.on_speech_start()
                                                except Exception:
                                                    pass

                                            # Сохраняем последние 5 блоков (~300 мс) как pre-roll
                                            audio_buffer = list(kws_ring_buffer)[-5:] if len(kws_ring_buffer) >= 5 else list(kws_ring_buffer)
                                            kws_ring_buffer.clear()
                                            self._notify_status("RECORDING")
                                            continue
                            else:
                                # Legacy режим без KWS (или принудительный ввод команды в меню)
                                if vad is not None:
                                    min_prob = 0.70 if (music_active and consecutive_unrecognized >= 2) else 0.45
                                    is_voice = (vad_prob >= min_prob) and (rms >= 0.008)
                                else:
                                    threshold = max(0.012, noise_floor * (2.0 if music_active else 2.2))
                                    is_voice = rms > threshold

                                if is_voice:
                                    is_recording = True
                                    silence_blocks = 0
                                    audio_buffer = list(pre_roll_buffer)
                                    log_debug(f"Начало речи (RMS={rms:.4f}, VAD={vad_prob:.3f})")
                                    self._notify_status("RECORDING")
                                    if self.on_speech_start:
                                        try:
                                            self.on_speech_start()
                                        except Exception:
                                            pass
                        else:
                            # Идёт запись команды
                            audio_buffer.append(block_flat.copy())

                            if vad is not None:
                                is_voice = (vad_prob >= 0.25) or (vad_prob >= 0.20 and rms >= 0.008)
                            else:
                                is_voice = rms > max(0.010, noise_floor * 1.5)

                            if is_voice:
                                silence_blocks = 0
                            else:
                                silence_blocks += 1

                                if silence_blocks >= max_silence_blocks or len(audio_buffer) >= max_speech_blocks:
                                    is_recording = False
                                    kws_ring_buffer.clear()
                                    if vad is not None:
                                        vad.reset()
                                    duration_s = len(audio_buffer) * block_size / sample_rate
                                    log_debug(f"Окончание речи. Записано {len(audio_buffer)} блоков (~{duration_s:.2f} сек)")
                                    self._notify_status("TRANSCRIBING")
                                    
                                    was_in_command_window = in_command_window
                                    self._listen_command_window_until = 0.0

                                    recognized = False
                                    if len(audio_buffer) >= min_speech_blocks:
                                        audio_data = np.concatenate(audio_buffer, axis=0)
                                        recognized = self._process_audio(
                                             model,
                                             audio_data,
                                             kws_trigger=kws_trigger_name if is_kws_command else None,
                                             in_command_window=was_in_command_window
                                         )
                                    else:
                                        log_debug("Слишком короткий сегмент аудио, пропуск")

                                    if recognized:
                                        consecutive_unrecognized = 0
                                    else:
                                        consecutive_unrecognized += 1
                                        if consecutive_unrecognized >= 3 and not use_kws:
                                            backoff_until = time.time() + 0.40

                                    if self.on_speech_end:
                                        try:
                                            self.on_speech_end(recognized)
                                        except Exception:
                                            pass

                                    self._notify_status("IDLE")
                                    audio_buffer = []
                                    silence_blocks = 0
                                    is_kws_command = False
                                    kws_trigger_name = None
                                    kws_ring_buffer.clear()
                                    if vad is not None:
                                        vad.reset()

            except Exception as e:
                log_error(f"Ошибка в цикле микрофона: {e}")
                self._notify_status("IDLE")
                time.sleep(2.0)

    def _get_whisper_prompt(self) -> str:
        """Промпт для Whisper: фиксирует правильное русское написание триггеров и команд"""
        triggers = ["Морсик", "Карлуша", "Эй Висп", "Висп"]
        if self.config.trigger and self.config.trigger not in triggers:
            triggers.append(self.config.trigger)
        trig_str = ", ".join(triggers)
        return (
            f"Голосовой ассистент Яндекс Музыки. Активационные слова: {trig_str}. "
            "Команды: включи, пауза, стоп, дальше, назад, громче, тише, "
            "что сейчас играет, очередь, моя волна, лайк, дизлайк, помощь, отмена, смени голос."
        )

    def _process_audio(
        self,
        model: WhisperModel,
        audio_data: np.ndarray,
        kws_trigger: Optional[str] = None,
        in_command_window: bool = False
    ) -> bool:
        """Распознавание сегмента речи через Whisper с подробным логированием"""
        initial_prompt = self._get_whisper_prompt()

        t_start = time.time()
        try:
            segments, _ = model.transcribe(
                audio_data,
                language="ru",
                beam_size=3,
                initial_prompt=initial_prompt,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(threshold=0.5, min_silence_duration_ms=300),
                no_speech_threshold=0.45,
                log_prob_threshold=-0.85,
                compression_ratio_threshold=2.2
            )
            
            clean_segments = []
            for s in segments:
                if s.no_speech_prob > 0.45 or s.avg_logprob < -0.85:
                    continue
                clean_segments.append(s.text)
            raw_text = "".join(clean_segments).strip()
        except Exception as e:
            log_error(f"Ошибка выполнения model.transcribe: {e}")
            return False

        t_elapsed = time.time() - t_start
        log_info(f"Whisper распознал за {t_elapsed:.3f} сек: '{raw_text}'")

        # Очищаем текст от знаков препинания по краям
        clean_raw = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', raw_text).strip()
        text_lower = clean_raw.lower()

        # Фильтрация типичных галлюцинаций Whisper (субтитры, артефакты тишины)
        if is_whisper_hallucination(clean_raw):
            log_info(f"Отфильтрована галлюцинация Whisper: '{clean_raw}'")
            if kws_trigger:
                log_info(f"KWS «{kws_trigger}» активирован, но распознана галлюцинация тишины (пропуск)")
            return False

        if not clean_raw or len(clean_raw) < 2:
            if kws_trigger:
                log_info(f"Назван только триггер «{kws_trigger}»")
                self.on_command("триггер_активирован", raw_text)
                return True
            log_debug("Текст пустой или отсечён фильтром тишины")
            return False

        # 0. Режим прямого ответа без триггера (меню или активное окно после «Слушаю вас»)
        if self._force_command_mode or in_command_window:
            # Защита от зацикливания: если в окне команды тишина — закрываем окно, не вызывая повторно триггер
            if not text_lower or len(text_lower) < 2:
                log_debug("В окне команды тишина — закрытие окна")
                return False

            cmd_clean = strip_wake_word(text_lower)
            if not cmd_clean or len(cmd_clean) < 2:
                log_info("В окне команды повторно назван триггер")
                self.on_command("триггер_активирован", raw_text)
                return True
            log_info(f"Прямой ввод команды (без wake-word): '{cmd_clean}' (raw: '{raw_text}')")
            self.on_command(cmd_clean, raw_text)
            return True

        # 1. Если команда была инициирована KWS детектором:
        if kws_trigger:
            cmd_clean = strip_wake_word(text_lower)
            if not cmd_clean or len(cmd_clean) < 2:
                log_info(f"Назван только триггер «{kws_trigger}»")
                self.on_command("триггер_активирован", raw_text)
                return True

            log_info(f"Передача KWS-команды в обработчик: '{cmd_clean}' (raw: '{raw_text}', trigger: '{kws_trigger}')")
            self.on_command(cmd_clean, raw_text)
            return True

        # 2. Запрос «Какой сейчас триггер?» — Wake-word НЕ требуется
        if re.search(r'\b(?:какой|скажи|напомни)\s+(?:сейчас\s+)?триг+ер\b', text_lower):
            log_info("Распознан прямой запрос триггера (без wake-word)")
            self.on_command("какой сейчас триггер", raw_text)
            return True

        # 3. Fallback / legacy поиск триггера в фразе (когда KWS отключен)
        cmd_clean = strip_wake_word(text_lower)
        if cmd_clean != text_lower:
            if not cmd_clean or len(cmd_clean) < 2:
                log_info("Назван только триггер (fallback режим)")
                self.on_command("триггер_активирован", raw_text)
                return True
            log_info(f"Передача команды в обработчик: '{cmd_clean}' (raw: '{raw_text}')")
            self.on_command(cmd_clean, raw_text)
            return True

        current_trigger = self.config.trigger
        has_trigger, command_part = find_trigger_in_phrase(text_lower, current_trigger)
        if not has_trigger:
            log_info(f"Триггер '{current_trigger}' не найден в фразе: '{raw_text}'")
            if self.on_log:
                self.on_log(f"[dim]Услышано: '{raw_text}' (ожидался триггер '{current_trigger}')[/dim]")
            return False

        command_part = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', command_part).strip()
        if not command_part:
            log_info(f"Назван только триггер '{current_trigger}'")
            self.on_command("триггер_активирован", raw_text)
            return True

        # 4. Команда смены триггера: «Морсик, теперь тригер — Карлуша»
        change_regex = r'(?:теперь\s+|смени(?:ть)?\s+|новый\s+)?триг+ер\s*(?:теперь\s+)?(?:на\s+|—\s*|-\s*|:\s*|это\s*)?([а-яёa-z0-9]+)'
        change_match = re.search(change_regex, command_part)
        if change_match and any(w in command_part for w in ["смени", "теперь", "новый", "поменяй"]):
            new_trig = change_match.group(1).strip().capitalize()
            log_info(f"Команда смены триггера на: {new_trig}")
            self.on_command(f"смени триггер на {new_trig}", raw_text)
            return True

        # Обычная команда
        log_info(f"Передача команды в обработчик: '{command_part}' (raw: '{raw_text}')")
        self.on_command(command_part, raw_text)
        return True
