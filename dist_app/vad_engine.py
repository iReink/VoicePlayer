"""
Потоковый нейросетевой детектор речевой активности (Silero VAD v6) с сохранением состояния.

Особенности:
1. Сохраняет LSTM скрытое состояние (h, c) между 512-сэмпловыми чанками,
   что исключает потерю контекста и обеспечивает надёжное подавление стационарного шума
   (вентиляторы, гул кулеров, фоновое шипение снижаются с prob=0.67 до prob < 0.01).
2. Защита от in-place повреждения: никогда не модифицирует оригинальные массивы (строгий copy).
3. Произвольный размер входящих блоков: накапливает входящие блоки любой длины
   (например, 960 сэмплов = 60 мс WebRTC APM AEC3) и последовательно прогоняет по 512 сэмплов.
4. Отслеживание недавней устойчивой речи (recent_probs): фильтрует одиночные щелчки
   клавиатуры/мыши и транзиенты (импульс в 1 фрейм не считается речью).
"""

import collections
from typing import Tuple, List, Optional
import numpy as np

from logger import log_debug, log_info, log_warning, log_error

try:
    from faster_whisper.vad import get_vad_model
except ImportError:
    get_vad_model = None


class StatefulSileroVAD:
    """
    Потоковая обёртка над Silero VAD ONNX-сессией.
    Хранит скрытое состояние h, c и 64-сэмпловый контекст предыдущего фрейма.
    """
    def __init__(self, session=None):
        if session is None:
            if get_vad_model is None:
                raise RuntimeError("faster_whisper.vad.get_vad_model недоступен")
            raw_vad = get_vad_model()
            self.session = getattr(raw_vad, 'session', raw_vad)
        else:
            self.session = getattr(session, 'session', session)

        self.sample_rate = 16000
        self.context_size = 64
        self.chunk_size = 512

        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.context_size, dtype=np.float32)
        self.remainder = np.array([], dtype=np.float32)

        # Кольцевой буфер вероятностей последних ~500 мс (16 фреймов по 32 мс)
        self.recent_probs = collections.deque(maxlen=16)

    def reset(self):
        """Сброс внутреннего состояния (между речевыми репликами или при переключении сессии)"""
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(self.context_size, dtype=np.float32)
        self.remainder = np.array([], dtype=np.float32)
        self.recent_probs.clear()

    def process_chunk_512(self, chunk_512: np.ndarray) -> float:
        """
        Обработка ровно одного фрейма из 512 сэмплов (32 мс при 16 кГц).
        Гарантирует отсутствие in-place изменений во входном массиве.
        """
        assert len(chunk_512) == self.chunk_size, f"Требуется ровно {self.chunk_size} сэмплов"
        # Строгая копия: защита от модификации данных внешней библиотеки
        clean_512 = np.ascontiguousarray(chunk_512, dtype=np.float32).copy()

        # Формируем вход: 64 сэмпла контекста + 512 сэмплов текущего фрейма = 576
        inp = np.concatenate([self.context, clean_512])[np.newaxis, :]
        self.context = clean_512[-self.context_size:].copy()

        out, self.h, self.c = self.session.run(
            None,
            {'input': inp, 'h': self.h, 'c': self.c}
        )
        prob = float(out[0, 0] if out.ndim > 1 else out[0])
        self.recent_probs.append(prob)
        return prob

    def process_block(self, audio_block: np.ndarray) -> Tuple[float, List[float]]:
        """
        Обработка блока произвольной длины (например, 960 сэмплов от WebRTC APM AEC3).
        Остаток сохраняется в self.remainder для следующего блока.
        Возвращает (максимальная вероятность в блоке, список всех вероятностей фреймов).
        """
        clean_block = audio_block.flatten().astype(np.float32, copy=True)
        if len(self.remainder) > 0:
            combined = np.concatenate([self.remainder, clean_block])
        else:
            combined = clean_block

        probs = []
        i = 0
        limit = len(combined)
        while i + self.chunk_size <= limit:
            p = self.process_chunk_512(combined[i : i + self.chunk_size])
            probs.append(p)
            i += self.chunk_size

        self.remainder = combined[i:].copy()
        max_p = max(probs) if probs else (self.recent_probs[-1] if self.recent_probs else 0.0)
        return max_p, probs

    def has_sustained_speech(self, threshold: float = 0.50, min_frames: int = 2) -> bool:
        """
        Проверяет наличие устойчивой человеческой речи в недавних фреймах (~500 мс).
        Одиночные импульсы (щелчки мыши/клавиатуры) длятся 1 фрейм и отсекаются.
        """
        if not self.recent_probs:
            return False
        speech_frames = sum(1 for p in self.recent_probs if p >= threshold)
        return speech_frames >= min_frames

    def get_recent_speech_ratio(self, threshold: float = 0.50) -> float:
        """Доля речевых фреймов за последние ~500 мс (от 0.0 до 1.0)"""
        if not self.recent_probs:
            return 0.0
        speech_frames = sum(1 for p in self.recent_probs if p >= threshold)
        return speech_frames / len(self.recent_probs)


def get_stateful_vad() -> Optional[StatefulSileroVAD]:
    """Фабричная функция инициализации потокового Silero VAD"""
    if get_vad_model is None:
        log_warning("faster_whisper.vad не установлен, Silero VAD недоступен")
        return None
    try:
        vad = StatefulSileroVAD()
        log_info("Потоковый Silero VAD (stateful) успешно инициализирован")
        return vad
    except Exception as e:
        log_error(f"Ошибка инициализации потокового Silero VAD: {e}")
        return None
