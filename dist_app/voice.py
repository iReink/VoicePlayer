"""
Модуль голосового ввода на базе Whisper-Small (GPU CUDA / CPU) с поддержкой логирования и нечеткого триггера
"""
import re
import time
import threading
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Callable, Tuple
import numpy as np
import sounddevice as sd
import ctranslate2
from faster_whisper import WhisperModel

from logger import log_debug, log_info, log_warning, log_error

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

class VoiceListener:
    def __init__(
        self,
        config_manager,
        on_command: Callable[[str, str], None],
        on_status: Optional[Callable[[str], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[bool], None]] = None,
        on_log: Optional[Callable[[str], None]] = None
    ):
        self.config = config_manager
        self.on_command = on_command
        self.on_status = on_status
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_log = on_log

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._model: Optional[WhisperModel] = None
        self._is_tts_active: bool = False
        self._force_command_mode: bool = False

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
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        log_info("Поток VoiceListener запущен")

    def stop(self):
        """Остановка прослушивания"""
        self._running = False
        log_info("Остановка VoiceListener")

    def _notify_status(self, status: str):
        if self.on_status:
            try:
                self.on_status(status)
            except Exception:
                pass

    def _listen_loop(self):
        sample_rate = 16000
        block_size = 1024  # ~64 ms
        
        base_threshold = 0.012
        noise_floor = 0.005
        max_silence_blocks = int(0.70 / (block_size / sample_rate))  # 0.70 сек тишины
        min_speech_blocks = int(0.35 / (block_size / sample_rate))   # 0.35 сек речи
        max_speech_blocks = int(8.0 / (block_size / sample_rate))    # максимум 8 сек
        
        try:
            model = self._get_model()
        except Exception as e:
            log_error(f"Критическая ошибка загрузки WhisperModel: {e}")
            if self.on_log:
                self.on_log(f"[bold red]Ошибка загрузки Whisper:[/bold red] {e}")
            return

        while self._running:
            try:
                log_debug("Открытие аудиопотока микрофона...")
                with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32', blocksize=block_size) as stream:
                    audio_buffer = []
                    is_recording = False
                    silence_blocks = 0
                    log_info("Аудиопоток микрофона открыт и слушает")

                    while self._running:
                        block, overflowed = stream.read(block_size)
                        if self._is_tts_active:
                            is_recording = False
                            audio_buffer = []
                            silence_blocks = 0
                            continue

                        if overflowed:
                            log_warning("Audio buffer overflowed")

                        rms = float(np.sqrt(np.mean(block**2)))

                        # Адаптация уровня шума (с верхним лимитом)
                        if not is_recording:
                            noise_floor = noise_floor * 0.96 + min(rms, 0.035) * 0.04
                            current_threshold = min(0.040, max(base_threshold, noise_floor * 2.0))
                        else:
                            current_threshold = min(0.035, max(base_threshold, noise_floor * 1.5))

                        if rms > current_threshold:
                            if not is_recording:
                                is_recording = True
                                silence_blocks = 0
                                audio_buffer = [block.copy()]
                                log_debug(f"Начало речи (RMS={rms:.4f}, порог={current_threshold:.4f})")
                                self._notify_status("RECORDING")
                                if self.on_speech_start:
                                    try:
                                        self.on_speech_start()
                                    except Exception:
                                        pass
                            else:
                                audio_buffer.append(block.copy())
                                silence_blocks = 0
                        else:
                            if is_recording:
                                audio_buffer.append(block.copy())
                                silence_blocks += 1

                                if silence_blocks >= max_silence_blocks or len(audio_buffer) >= max_speech_blocks:
                                    is_recording = False
                                    duration_s = len(audio_buffer) * block_size / sample_rate
                                    log_debug(f"Окончание речи. Записано {len(audio_buffer)} блоков (~{duration_s:.2f} сек)")
                                    self._notify_status("TRANSCRIBING")
                                    
                                    recognized = False
                                    if len(audio_buffer) >= min_speech_blocks:
                                        audio_data = np.concatenate(audio_buffer, axis=0).flatten()
                                        recognized = self._process_audio(model, audio_data)
                                    else:
                                        log_debug("Слишком короткий сегмент аудио, пропуск")

                                    if self.on_speech_end:
                                        try:
                                            self.on_speech_end(recognized)
                                        except Exception:
                                            pass

                                    self._notify_status("IDLE")
                                    audio_buffer = []
                                    silence_blocks = 0

            except Exception as e:
                log_error(f"Ошибка в цикле микрофона: {e}")
                self._notify_status("IDLE")
                time.sleep(2.0)

    def _process_audio(self, model: WhisperModel, audio_data: np.ndarray) -> bool:
        """Распознавание сегмента речи через Whisper с подробным логированием"""
        current_trigger = self.config.trigger
        
        initial_prompt = (
            f"Голосовой ассистент Яндекс Музыки. Триггер {current_trigger}. "
            "Популярные артисты: Queen, twenty one pilots, Imagine Dragons, Linkin Park, "
            "The Beatles, Metallica, Nirvana, Eminem, Billie Eilish, AC/DC, Rammstein, "
            "Red Hot Chili Peppers, Green Day, Coldplay, Depeche Mode, Michael Jackson. "
            "Команды: включи мой плейлист, включи трек, включи исполнителя, включи альбом, "
            "моя волна, плейлист дня, дежавю, премьера, тайник, любимые треки, "
            "пауза, стоп, дальше, назад, громче, тише, какой сейчас триггер."
        )

        t_start = time.time()
        try:
            segments, _ = model.transcribe(
                audio_data,
                beam_size=3,
                initial_prompt=initial_prompt,
                condition_on_previous_text=False
            )
            raw_text = "".join([s.text for s in segments]).strip()
        except Exception as e:
            log_error(f"Ошибка выполнения model.transcribe: {e}")
            return False

        t_elapsed = time.time() - t_start
        log_info(f"Whisper распознал за {t_elapsed:.3f} сек: '{raw_text}'")

        if not raw_text or len(raw_text) < 2:
            log_debug("Текст пустой или слишком короткий")
            return False

        # Очищаем текст от знаков препинания по краям
        clean_raw = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', raw_text).strip()
        text_lower = clean_raw.lower()

        # 0. Режим прямого ответа без триггера (например, голосовое меню выбора)
        if self._force_command_mode:
            log_info(f"Прямой ввод (меню без wake-word): '{clean_raw}'")
            self.on_command(clean_raw, raw_text)
            return True

        # 1. Запрос «Какой сейчас триггер?» — Wake-word НЕ требуется
        if re.search(r'\b(?:какой|скажи|напомни)\s+(?:сейчас\s+)?триг+ер\b', text_lower):
            log_info("Распознан прямой запрос триггера (без wake-word)")
            self.on_command("какой сейчас триггер", raw_text)
            return True

        # 2. Нечеткий поиск триггера в фразе
        has_trigger, command_part = find_trigger_in_phrase(text_lower, current_trigger)
        if not has_trigger:
            log_info(f"Триггер '{current_trigger}' не найден в фразе: '{raw_text}'")
            # Показываем пользователю в консоли, что именно микрофон услышал
            if self.on_log:
                self.on_log(f"[dim]Услышано: '{raw_text}' (ожидался триггер '{current_trigger}')[/dim]")
            return False

        command_part = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', command_part).strip()

        if not command_part:
            log_info(f"Назван только триггер '{current_trigger}'")
            self.on_command("триггер_активирован", raw_text)
            return True

        # 3. Команда смены триггера: «Морсик, теперь тригер — Карлуша»
        change_regex = r'(?:теперь\s+|смени(?:ть)?\s+|новый\s+)?триг+ер\s*(?:теперь\s+)?(?:на\s+|—\s*|-\s*|:\s*|это\s*)?([а-яёa-z0-9]+)'
        change_match = re.search(change_regex, command_part)
        if change_match and any(w in command_part for w in ["смени", "теперь", "новый", "поменяй"]):
            new_trig = change_match.group(1).strip().capitalize()
            log_info(f"Команда смены триггера на: {new_trig}")
            self.on_command(f"смени триггер на {new_trig}", raw_text)
            return True

        # 4. Обычная команда
        log_info(f"Передача команды в обработчик: '{command_part}' (raw: '{raw_text}')")
        self.on_command(command_part, raw_text)
        return True
