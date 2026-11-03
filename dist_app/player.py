"""
Аудио-плеер на базе pygame с поддержкой очередей, предзагрузки и 'Моей волны'
"""
import io
import time
import threading
from typing import List, Optional, Callable, Dict, Any
import pygame
from yandex_music import Track

class PlaybackSnapshot:
    """Снимок состояния воспроизведения для истории и безопасной отмены (Undo)"""
    def __init__(self, queue: List[Track], index: int, is_my_wave: bool, playlist_title: str):
        self.queue: List[Track] = list(queue)
        self.index: int = index
        self.is_my_wave: bool = is_my_wave
        self.playlist_title: str = playlist_title
        self.timestamp: float = time.time()
        if 0 <= index < len(queue):
            t = queue[index]
            self.track_title: str = getattr(t, 'title', '') or "Без названия"
            artists = getattr(t, 'artists', None)
            if artists:
                self.artist_name: str = ", ".join(a.name for a in artists if hasattr(a, 'name'))
            else:
                self.artist_name: str = "Неизвестный исполнитель"
        else:
            self.track_title = ""
            self.artist_name = ""

class MusicPlayer:
    def __init__(self, music_service, on_track_start: Optional[Callable[[Track, int, int, str], None]] = None):
        self.service = music_service
        self.on_track_start = on_track_start

        # Инициализация звуковой подсистемы SDL/Pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)

        self._queue: List[Track] = []
        self._index: int = -1
        self._current_track: Optional[Track] = None
        self._current_stream: Optional[io.BytesIO] = None
        
        self._is_paused: bool = False
        self._is_stopped: bool = True
        self._is_loading: bool = False
        self._play_id: int = 0  # Счётчик поколений для защиты от race conditions
        self._volume: float = 0.7  # 70% по умолчанию
        self._saved_volume: float = 0.7
        self._is_ducked: bool = False
        self._is_my_wave: bool = False
        self._playlist_title: str = ""

        # Стек истории воспроизведения (до 20 снимков) и стек повтора (redo)
        self._history: List[PlaybackSnapshot] = []
        self._redo_history: List[PlaybackSnapshot] = []
        
        # Кэш предзагрузки {track_id: bytes}
        self._byte_cache: Dict[str, bytes] = {}
        self._lock = threading.RLock()

        pygame.mixer.music.set_volume(self._volume)

        # Фоновый мониторинг окончания треков
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return not self._is_stopped and not self._is_paused

    @property
    def is_paused(self) -> bool:
        with self._lock:
            return self._is_paused

    @property
    def volume(self) -> int:
        return int(round(self._volume * 100))

    @property
    def current_track(self) -> Optional[Track]:
        with self._lock:
            return self._current_track

    def _push_current_state(self):
        """Сохранить текущее состояние в стек истории (до 20 записей)"""
        if not self._queue or self._index < 0:
            return
        snapshot = PlaybackSnapshot(
            queue=self._queue,
            index=self._index,
            is_my_wave=self._is_my_wave,
            playlist_title=self._playlist_title
        )
        # Не добавляем дубликат, если текущее состояние идентично вершине стека
        if self._history:
            top = self._history[-1]
            if (top.playlist_title == snapshot.playlist_title and 
                top.index == snapshot.index and 
                top.track_title == snapshot.track_title and 
                len(top.queue) == len(snapshot.queue)):
                return
        self._history.append(snapshot)
        if len(self._history) > 20:
            self._history.pop(0)
        self._redo_history.clear()

    def can_undo(self) -> bool:
        """Есть ли сохраненные состояния в истории для отмены"""
        with self._lock:
            return len(self._history) > 0

    def can_redo(self) -> bool:
        """Есть ли состояния для повтора отменённого действия"""
        with self._lock:
            return len(self._redo_history) > 0

    def can_prev_track_in_queue(self) -> bool:
        """Есть ли предыдущий трек в текущей очереди"""
        with self._lock:
            return self._index > 0

    def undo_state(self) -> Optional[Dict[str, Any]]:
        """
        Откатить состояние плеера к предыдущему сохранённому в стеке истории.
        Возвращает словарь с информацией о восстановленном треке и очереди.
        """
        with self._lock:
            if not self._history:
                return None

            # Сохраняем текущее состояние в redo перед откатом
            if self._queue and self._index >= 0:
                cur_snap = PlaybackSnapshot(
                    queue=self._queue,
                    index=self._index,
                    is_my_wave=self._is_my_wave,
                    playlist_title=self._playlist_title
                )
                self._redo_history.append(cur_snap)
                if len(self._redo_history) > 20:
                    self._redo_history.pop(0)

            snapshot = self._history.pop()
            self._queue = list(snapshot.queue)
            self._index = snapshot.index
            self._is_my_wave = snapshot.is_my_wave
            self._playlist_title = snapshot.playlist_title
            self._byte_cache.clear()

        self._play_current()

        return {
            "track_title": snapshot.track_title,
            "artist_name": snapshot.artist_name,
            "playlist_title": snapshot.playlist_title,
            "index": snapshot.index + 1,
            "total": len(snapshot.queue),
            "history_remaining": len(self._history)
        }

    def redo_state(self) -> Optional[Dict[str, Any]]:
        """Повторить отменённое действие из стека redo"""
        with self._lock:
            if not self._redo_history:
                return None

            if self._queue and self._index >= 0:
                cur_snap = PlaybackSnapshot(
                    queue=self._queue,
                    index=self._index,
                    is_my_wave=self._is_my_wave,
                    playlist_title=self._playlist_title
                )
                self._history.append(cur_snap)
                if len(self._history) > 20:
                    self._history.pop(0)

            snapshot = self._redo_history.pop()
            self._queue = list(snapshot.queue)
            self._index = snapshot.index
            self._is_my_wave = snapshot.is_my_wave
            self._playlist_title = snapshot.playlist_title
            self._byte_cache.clear()

        self._play_current()

        return {
            "track_title": snapshot.track_title,
            "artist_name": snapshot.artist_name,
            "playlist_title": snapshot.playlist_title,
            "index": snapshot.index + 1,
            "total": len(snapshot.queue),
            "history_remaining": len(self._history)
        }

    def get_history_info(self) -> List[Dict[str, Any]]:
        """Получить список сохранённых состояний в истории от самых новых к старым"""
        with self._lock:
            items = []
            for s in reversed(self._history):
                items.append({
                    "playlist_title": s.playlist_title or "Очередь",
                    "track_title": s.track_title,
                    "artist_name": s.artist_name,
                    "index": s.index + 1,
                    "total": len(s.queue),
                    "timestamp": s.timestamp
                })
            return items

    def play_tracks(self, tracks: List[Track], is_my_wave: bool = False, playlist_title: str = "", save_to_history: bool = True):
        """Загрузить новую очередь треков и начать воспроизведение с первого трека"""
        if not tracks:
            return

        with self._lock:
            if save_to_history:
                self._push_current_state()

            self._queue = list(tracks)
            self._index = 0
            self._is_my_wave = is_my_wave
            self._playlist_title = playlist_title
            self._byte_cache.clear()

        self._play_current()

    def _preload_next(self):
        """Предзагрузка следующего трека в фоне для бесшовного перехода"""
        with self._lock:
            next_idx = self._index + 1
            if next_idx < len(self._queue):
                next_track = self._queue[next_idx]
                if next_track.id not in self._byte_cache:
                    def _loader(t):
                        try:
                            data = t.download_bytes()
                            with self._lock:
                                self._byte_cache[t.id] = data
                        except Exception:
                            pass
                    threading.Thread(target=_loader, args=(next_track,), daemon=True).start()

    def _play_current(self):
        """Запуск текущего трека по текущему индексу"""
        with self._lock:
            if not self._queue or self._index < 0 or self._index >= len(self._queue):
                self._is_stopped = True
                self._is_loading = False
                return

            self._play_id += 1
            current_play_id = self._play_id
            self._is_loading = True
            self._is_stopped = False
            self._is_paused = False
            
            track = self._queue[self._index]
            self._current_track = track
            cached_data = self._byte_cache.pop(track.id, None)

        # Скачивание аудиоданных вне блокировки
        data = cached_data
        if not data:
            try:
                data = track.download_bytes()
            except Exception:
                # Если скачивание не удалось (например, ограничение прав), переходим к следующему
                with self._lock:
                    if self._play_id == current_play_id:
                        self._is_loading = False
                self.next_track()
                return

        with self._lock:
            # Если пока мы качали, пользователь запросил другой трек или сменил очередь
            if self._play_id != current_play_id:
                return

            try:
                self._current_stream = io.BytesIO(data)
                pygame.mixer.music.load(self._current_stream)
                pygame.mixer.music.play()
                pygame.mixer.music.set_volume(self._volume)
            except Exception:
                self._is_loading = False
                self.next_track()
                return

            # Небольшая задержка, чтобы SDL audio буфер успел инициализироваться
            time.sleep(0.15)
            self._is_loading = False
            current_idx = self._index + 1
            total_count = len(self._queue)
            title = self._playlist_title

        # Оповещаем UI о старте трека
        if self.on_track_start:
            try:
                self.on_track_start(track, current_idx, total_count, title)
            except Exception:
                pass

        # Предзагружаем следующий трек
        self._preload_next()

    def pause(self) -> bool:
        """Поставить на паузу"""
        with self._lock:
            if not self._is_stopped and not self._is_paused:
                pygame.mixer.music.pause()
                self._is_paused = True
                return True
            return False

    def resume(self) -> bool:
        """Возобновить воспроизведение"""
        with self._lock:
            if not self._is_stopped and self._is_paused:
                pygame.mixer.music.unpause()
                self._is_paused = False
                return True
            return False

    def toggle_pause(self) -> bool:
        """Переключить паузу / плей"""
        with self._lock:
            if self._is_paused:
                return self.resume()
            else:
                return self.pause()

    def stop(self):
        """Остановить воспроизведение"""
        with self._lock:
            self._play_id += 1
            self._is_stopped = True
            self._is_paused = False
            self._is_loading = False
            pygame.mixer.music.stop()
            self._current_stream = None

    def next_track(self) -> bool:
        """Перейти к следующему треку"""
        with self._lock:
            if not self._queue:
                return False

            # Если это 'Моя волна' и треки подходят к концу, подгружаем еще
            if self._is_my_wave and self._index >= len(self._queue) - 2:
                try:
                    more_tracks = self.service.get_my_wave_batch()
                    self._queue.extend(more_tracks)
                except Exception:
                    pass

            if self._index + 1 < len(self._queue):
                self._index += 1
            else:
                # Достигли конца очереди
                self._is_stopped = True
                self._is_loading = False
                pygame.mixer.music.stop()
                return False

        self._play_current()
        return True

    def prev_track(self) -> bool:
        """Вернуться к предыдущему треку"""
        with self._lock:
            if not self._queue:
                return False
            if self._index > 0:
                self._index -= 1
            else:
                self._index = 0

        self._play_current()
        return True

    def duck(self, enable: bool):
        """Временное приглушение музыки во время ответа ассистента"""
        with self._lock:
            if enable and not self._is_ducked:
                self._is_ducked = True
                self._saved_volume = self._volume
                pygame.mixer.music.set_volume(min(self._volume, 0.15))
            elif not enable and self._is_ducked:
                self._is_ducked = False
                pygame.mixer.music.set_volume(self._saved_volume)

    def fade_out_pause(self, duration: float = 0.5) -> bool:
        """
        Плавное затухание громкости за duration секунд и пауза.
        Возвращает True, если трек играл и был поставлен на паузу.
        """
        with self._lock:
            if self._is_stopped or self._is_paused:
                return False
            # Если был активен ducking, снимаем его перед затуханием
            self._is_ducked = False
            start_vol = pygame.mixer.music.get_volume()
            if start_vol <= 0:
                start_vol = self._volume

        steps = 10
        step_time = max(0.01, duration / steps)
        for i in range(1, steps + 1):
            factor = 1.0 - (i / steps)
            current_v = max(0.0, start_vol * factor)
            pygame.mixer.music.set_volume(current_v)
            time.sleep(step_time)

        with self._lock:
            pygame.mixer.music.pause()
            self._is_paused = True
            pygame.mixer.music.set_volume(self._volume)
            return True

    def fade_in_resume(self, duration: float = 0.5) -> bool:
        """
        Плавное нарастание громкости от 0 до целевой за duration секунд после снятия с паузы.
        Возвращает True, если воспроизведение было возобновлено.
        """
        with self._lock:
            if self._is_stopped or not self._is_paused:
                return False
            target_vol = self._volume

        pygame.mixer.music.set_volume(0.0)
        with self._lock:
            pygame.mixer.music.unpause()
            self._is_paused = False

        steps = 10
        step_time = max(0.01, duration / steps)
        for i in range(1, steps + 1):
            factor = i / steps
            current_v = min(target_vol, target_vol * factor)
            pygame.mixer.music.set_volume(current_v)
            time.sleep(step_time)

        pygame.mixer.music.set_volume(target_vol)
        return True

    def get_next_track_info(self) -> Optional[Dict[str, Any]]:
        """Получить информацию о следующем треке в очереди"""
        with self._lock:
            next_idx = self._index + 1
            if 0 <= next_idx < len(self._queue):
                return self.service.format_track_info(self._queue[next_idx])
            return None

    def get_queue_count(self) -> int:
        """Получить общее количество треков в очереди"""
        with self._lock:
            return len(self._queue)

    def set_volume(self, percent: int) -> int:
        """Установить громкость от 0 до 100%"""
        percent = max(0, min(100, percent))
        self._volume = percent / 100.0
        with self._lock:
            pygame.mixer.music.set_volume(self._volume)
        return percent

    def change_volume(self, delta_percent: int) -> int:
        """Изменить громкость на +/- %"""
        current = self.volume
        return self.set_volume(current + delta_percent)

    def get_queue(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Получить список треков в очереди вокруг текущего"""
        with self._lock:
            results = []
            start = self._index
            end = min(len(self._queue), start + limit)
            for i in range(start, end):
                t = self._queue[i]
                info = self.service.format_track_info(t)
                info["is_current"] = (i == self._index)
                results.append(info)
            return results

    def _monitor_loop(self):
        """Фоновый цикл проверки естественного окончания трека"""
        while self._running:
            time.sleep(0.3)
            with self._lock:
                # Если трек остановлен, на паузе или прямо сейчас загружается - ничего не делаем
                if self._is_stopped or self._is_paused or self._is_loading:
                    continue
                is_busy = pygame.mixer.music.get_busy()

            # Если плеер сообщает, что не занят (трек завершился)
            if not is_busy:
                # Ждем 0.3s и перепроверяем, чтобы исключить редкие скачки буфера
                time.sleep(0.3)
                with self._lock:
                    if self._is_stopped or self._is_paused or self._is_loading:
                        continue
                    if not pygame.mixer.music.get_busy():
                        # Трек действительно закончил играть сам
                        self.next_track()

    def close(self):
        self._running = False
        self.stop()
        pygame.mixer.quit()
