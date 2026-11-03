"""
Модуль взаимодействия с API Яндекс.Музыки
"""
import os
from typing import List, Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from yandex_music import Client, Track

load_dotenv()

class MusicService:
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.getenv("YANDEX_MUSIC_TOKEN")
        if not self.token:
            raise ValueError("Токен Яндекс.Музыки не найден. Проверьте файл .env")
        
        self.client = Client(self.token).init()
        self.user = self.client.me.account
        self.user_name = self.user.full_name or self.user.login or "Пользователь"
        self.has_plus = self.client.me.plus.has_plus if hasattr(self.client.me, 'plus') else True

    def format_track_info(self, track: Track) -> Dict[str, Any]:
        """Возвращает форматированную информацию о треке"""
        artists = ", ".join([a.name for a in track.artists]) if track.artists else "Неизвестный исполнитель"
        albums = ", ".join([a.title for a in track.albums]) if track.albums else ""
        duration_ms = track.duration_ms or 0
        minutes = duration_ms // 60000
        seconds = (duration_ms % 60000) // 1000
        return {
            "id": track.id,
            "title": track.title,
            "artists": artists,
            "album": albums,
            "duration": f"{minutes:02d}:{seconds:02d}",
            "track_obj": track
        }

    def search_track(self, query: str, limit: int = 30) -> Tuple[List[Track], str]:
        """Поиск конкретного трека по названию"""
        search_res = self.client.search(query, type_='track')
        if not search_res.tracks or not search_res.tracks.results:
            return [], f"Трек по запросу '{query}' не найден"
        tracks = search_res.tracks.results[:limit]
        best_track = tracks[0]
        artists = ", ".join([a.name for a in best_track.artists]) if best_track.artists else ""
        return tracks, f"Трек: {artists} — {best_track.title}"

    def search_artist(self, query: str, limit: int = 40) -> Tuple[List[Track], str]:
        """Поиск артиста и загрузка его лучших треков"""
        search_res = self.client.search(query, type_='artist')
        if not search_res.artists or not search_res.artists.results:
            return [], f"Исполнитель '{query}' не найден"
        
        artist = search_res.artists.results[0]
        tracks_pager = artist.get_tracks(page_size=limit)
        tracks = tracks_pager.tracks if tracks_pager else []
        return tracks, f"Исполнитель: {artist.name} (топ {len(tracks)} треков)"

    def search_album(self, query: str) -> Tuple[List[Track], str]:
        """Поиск альбома и загрузка всех его треков"""
        search_res = self.client.search(query, type_='album')
        if not search_res.albums or not search_res.albums.results:
            return [], f"Альбом '{query}' не найден"
        
        album = search_res.albums.results[0]
        full_album = album.with_tracks()
        tracks = []
        if full_album and full_album.volumes:
            for volume in full_album.volumes:
                tracks.extend(volume)
        artists = ", ".join([a.name for a in album.artists]) if album.artists else ""
        return tracks, f"Альбом: {artists} — {album.title} ({len(tracks)} треков)"

    def search_playlist(self, query: str, limit: int = 50) -> Tuple[List[Track], str]:
        """Поиск публичного плейлиста"""
        search_res = self.client.search(query, type_='playlist')
        if not search_res.playlists or not search_res.playlists.results:
            return [], f"Плейлист по запросу '{query}' не найден"
        
        pl = search_res.playlists.results[0]
        short_tracks = pl.fetch_tracks()
        if not short_tracks:
            return [], f"Плейлист '{pl.title}' пуст"
        
        track_ids = [t.id for t in short_tracks[:limit]]
        full_tracks = self.client.tracks(track_ids)
        return full_tracks, f"Плейлист: {pl.title} ({len(full_tracks)} треков)"

    def get_liked_tracks(self, limit: int = 50) -> Tuple[List[Track], str]:
        """Получить понравившиеся треки (Мне нравится)"""
        likes = self.client.users_likes_tracks()
        if not likes or not likes.tracks:
            return [], "Список любимых треков пуст"
        
        track_ids = [t.id for t in likes.tracks[:limit]]
        full_tracks = self.client.tracks(track_ids)
        return full_tracks, f"Любимые треки ({len(full_tracks)} треков)"

    def search_user_playlist(self, query: str = "") -> Tuple[List[Track], str]:
        """Поиск по личным плейлистам пользователя"""
        try:
            playlists = self.client.users_playlists_list()
            q_clean = query.lower().strip()
            # Убираем лишние слова
            for w in ["включи", "мой", "плейлист", "свой", "поставь"]:
                q_clean = q_clean.replace(w, "").strip()

            # 1. Если указано конкретное название плейлиста
            if q_clean and len(q_clean) >= 2:
                for pl in playlists:
                    if q_clean in pl.title.lower() or pl.title.lower() in q_clean:
                        short_tracks = pl.fetch_tracks()
                        if short_tracks:
                            track_ids = [t.id for t in short_tracks[:50]]
                            full_tracks = self.client.tracks(track_ids)
                            return full_tracks, f"Мой плейлист: {pl.title} ({len(full_tracks)} треков)"

            # 2. Если просто "мой плейлист" - берем первый непустой личный плейлист
            for pl in playlists:
                if pl.track_count and pl.track_count > 0:
                    short_tracks = pl.fetch_tracks()
                    if short_tracks:
                        track_ids = [t.id for t in short_tracks[:50]]
                        full_tracks = self.client.tracks(track_ids)
                        return full_tracks, f"Мой плейлист: {pl.title} ({len(full_tracks)} треков)"
        except Exception:
            pass

        # Если личных плейлистов нет - включаем "Мне нравится"
        return self.get_liked_tracks()

    def get_generated_playlist(self, kind: str) -> Tuple[List[Track], str]:
        """
        Получить персональный плейлист из ленты:
        playlistOfTheDay - Плейлист дня
        neverHeard - Дежавю
        recentTracks - Премьера
        missedLikes - Тайник
        """
        feed = self.client.feed()
        target = None
        for item in feed.generated_playlists:
            if item.type == kind or (kind == "playlistOfTheDay" and "дня" in (item.data.title or "").lower()):
                target = item.data
                break
            elif kind == "neverHeard" and "дежавю" in (item.data.title or "").lower():
                target = item.data
                break
            elif kind == "recentTracks" and "премьера" in (item.data.title or "").lower():
                target = item.data
                break
            elif kind == "missedLikes" and "тайник" in (item.data.title or "").lower():
                target = item.data
                break
        
        if not target:
            return [], f"Персональный плейлист '{kind}' пока недоступен"
        
        short_tracks = target.fetch_tracks()
        track_ids = [t.id for t in short_tracks[:50]]
        full_tracks = self.client.tracks(track_ids)
        return full_tracks, f"{target.title} ({len(full_tracks)} треков)"

    def get_my_wave_batch(self) -> List[Track]:
        """Получить очередную пачку треков для 'Моей волны'"""
        rotor_tracks = self.client.rotor_station_tracks('user:onyourwave')
        tracks = []
        if rotor_tracks and rotor_tracks.sequence:
            for item in rotor_tracks.sequence:
                if item.track:
                    tracks.append(item.track)
        return tracks

    def smart_search(self, query: str) -> Tuple[List[Track], str]:
        """
        Умный поиск: анализирует общий результат поиска и выбирает лучший вариант
        (трек, исполнитель, альбом или плейлист).
        """
        search_res = self.client.search(query, type_='all')
        
        # 1. Проверяем точный/наилучший результат от поискового движка Яндекса
        if search_res.best:
            best_type = search_res.best.type
            best_item = search_res.best.result
            
            if best_type == 'artist' and hasattr(best_item, 'name'):
                tracks, desc = self.search_artist(best_item.name)
                if tracks:
                    return tracks, desc
                    
            elif best_type == 'album' and hasattr(best_item, 'title'):
                tracks, desc = self.search_album(best_item.title)
                if tracks:
                    return tracks, desc
                    
            elif best_type == 'playlist' and hasattr(best_item, 'title'):
                tracks, desc = self.search_playlist(best_item.title)
                if tracks:
                    return tracks, desc
                    
            elif best_type == 'track' and hasattr(best_item, 'title'):
                tracks = [best_item]
                if search_res.tracks and search_res.tracks.results:
                    for t in search_res.tracks.results:
                        if t.id != best_item.id:
                            tracks.append(t)
                artists = ", ".join([a.name for a in best_item.artists]) if best_item.artists else ""
                return tracks[:30], f"Трек: {artists} — {best_item.title}"

        # 2. Если best нет, проверяем совпадение с артистом
        if search_res.artists and search_res.artists.results:
            first_artist = search_res.artists.results[0]
            if first_artist.name.lower() in query.lower() or query.lower() in first_artist.name.lower():
                tracks, desc = self.search_artist(first_artist.name)
                if tracks:
                    return tracks, desc

        # 3. Треки
        if search_res.tracks and search_res.tracks.results:
            tracks = search_res.tracks.results[:30]
            best_track = tracks[0]
            artists = ", ".join([a.name for a in best_track.artists]) if best_track.artists else ""
            return tracks, f"Трек: {artists} — {best_track.title}"

        # 4. Альбомы
        if search_res.albums and search_res.albums.results:
            return self.search_album(search_res.albums.results[0].title)

        # 5. Плейлисты
        if search_res.playlists and search_res.playlists.results:
            return self.search_playlist(search_res.playlists.results[0].title)

        return [], f"По запросу '{query}' ничего не найдено"

    def like_track(self, track_id: str) -> bool:
        """Добавить трек в понравившиеся"""
        try:
            return bool(self.client.users_likes_tracks_add(track_id))
        except Exception:
            return False

    def dislike_track(self, track_id: str) -> bool:
        """Удалить трек из понравившихся"""
        try:
            return bool(self.client.users_likes_tracks_remove(track_id))
        except Exception:
            return False
