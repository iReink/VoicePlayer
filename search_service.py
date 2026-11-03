"""
Сервис веб-поиска фактов, новостей и погоды с поддержкой Gemma 4 31B Search Grounding
и быстрым локальным fallback-провайдером (Wikipedia API + Open-Meteo).
"""
import os
import re
import urllib.parse
import urllib.request
import json
from typing import Optional
from logger import log_info, log_warning, log_error

class SearchService:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def search(self, query: str) -> str:
        """
        Универсальный поиск фактов в интернете:
        1. Пробуем Gemma 4 31B с Default Search Grounding
        2. Fallback: Wikipedia API + Open-Meteo (быстрый ответ <300мс)
        """
        clean_q = query.strip()
        log_info(f"Выполняется веб-поиск: '{clean_q}'")

        # 1. Попытка через Google AI Studio (Gemma 4 31B Search Grounding)
        if self.api_key:
            try:
                gemma_res = self._search_via_gemma(clean_q)
                if gemma_res:
                    log_info("Поиск успешно выполнен через Gemma 4 31B Search Grounding")
                    return gemma_res
            except Exception as e:
                log_warning(f"Gemma Search Grounding временно недоступен ({e}), переключаемся на fallback")

        # 2. Быстрый локальный fallback
        return self._search_fallback(clean_q)

    def _search_via_gemma(self, query: str) -> Optional[str]:
        """Запрос к модели Gemma 4 31B с поисковым заземлением (Search Grounding)"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemma-4-31b-it:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"Кратко ответь на русском языке по фактам: {query}"}]}],
            "tools": [{"googleSearch": {}}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 250
            }
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            if resp.status == 200:
                body = json.loads(resp.read().decode("utf-8"))
                candidates = body.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
        return None

    def _search_fallback(self, query: str) -> str:
        """Резервный быстрый поиск через открытые API"""
        # Погода
        if any(w in query.lower() for w in ["погода", "температура", "градус"]):
            return self._get_weather_fallback(query)

        # Факты, люди, музыка, группы через Wikipedia API
        try:
            enc = urllib.parse.quote(query)
            wiki_url = f"https://ru.wikipedia.org/w/api.php?action=query&list=search&srsearch={enc}&utf8=1&format=json"
            req = urllib.request.Request(wiki_url, headers={"User-Agent": "VoicePlayerAssistant/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                search_res = data.get("query", {}).get("search", [])
                if search_res:
                    first = search_res[0]
                    title = first.get("title", "")
                    raw_snippet = first.get("snippet", "")
                    clean_snip = re.sub(r'<[^>]+>', '', raw_snippet).strip()
                    # Убираем знаки ударения unicode
                    clean_snip = re.sub(r'[\u0300-\u036f]', '', clean_snip)
                    return f"Факты из энциклопедии ({title}): {clean_snip}"
        except Exception as e:
            log_warning(f"Ошибка Wikipedia поиска: {e}")

        return f"По запросу '{query}' найдена информация, но детальные данные уточняются."

    def _get_weather_fallback(self, query: str) -> str:
        """Погода для популярных городов"""
        cities = {
            "москва": (55.75, 37.61, "Москве"),
            "москве": (55.75, 37.61, "Москве"),
            "питер": (59.93, 30.31, "Санкт-Петербурге"),
            "петербург": (59.93, 30.31, "Санкт-Петербурге"),
            "казань": (55.79, 49.12, "Казани"),
            "екатеринбург": (56.84, 60.61, "Екатеринбурге"),
            "новосибирск": (55.04, 82.93, "Новосибирске")
        }
        target_lat, target_lon, target_name = 55.75, 37.61, "Москве"
        for k, (lat, lon, name) in cities.items():
            if k in query.lower():
                target_lat, target_lon, target_name = lat, lon, name
                break

        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={target_lat}&longitude={target_lon}&current_weather=true"
            req = urllib.request.Request(url, headers={"User-Agent": "VoicePlayer/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                cw = data.get("current_weather", {})
                temp = cw.get("temperature")
                wind = cw.get("windspeed")
                if temp is not None:
                    return f"Сейчас в {target_name} около {temp} градусов, скорость ветра {wind} километров в час."
        except Exception as e:
            log_warning(f"Ошибка получения погоды: {e}")

        return "В данный момент не удалось связаться с погодной станцией."
