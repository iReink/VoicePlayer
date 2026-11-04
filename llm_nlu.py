"""
Модуль гибридной NLU-классификации с каскадом нейросетей:
Gemini 3.5 Flash Lite -> Gemini 3.1 Flash Lite -> Локальный парсер CommandParser
"""
import os
import re
import time
import json
from typing import Dict, Any, Optional
import requests
from dotenv import load_dotenv

from parser import CommandParser
from logger import log_info, log_warning, log_error, log_debug

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

MODELS_CASCADE = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]

SYSTEM_PROMPT = """Ты — интеллектуальный NLU-классификатор голосового ассистента Яндекс.Музыки («Морсик» / «Карлуша» / «Эй Висп»).
Твоя задача — проанализировать запрос пользователя, исправить возможные опечатки/неточности распознавания речи и вернуть результат строго в формате JSON:
{
  "intent": "<ИМЯ_ИНТЕНТА>",
  "params": { ... },
  "tts_response": "<короткая естественная фраза на русском для озвучки голосом, 3-7 слов, или null>"
}

ВАЖНЕЙШЕЕ ПРАВИЛО:
Если запрос пользователя состоит ТОЛЬКО из обращения к ассистенту по имени («морсик», «карлуша», «эй висп», «висп», «эй», «слушай», «привет», «ei vispa», «hey wisp»), верни:
{
  "intent": "TRIGGER_ACTIVATED",
  "params": {},
  "tts_response": "Слушаю вас"
}
НИКОГДА не пытайся искать исполнителей или треки («Wisp», «Visa», «Карлуша») на простое обращение к ассистенту!

СПИСОК ДОСТУПНЫХ ИНТЕНТОВ (intent):
1. PLAY_TRACK: конкретный трек или трек по описанию.
   params: {"query": "Исполнитель — Название"}
   tts_response: "Включаю {Название}" или "Ставлю {Артист} — {Название}"

2. PLAY_ARTIST: включить дискографию/популярные песни артиста или группы.
   params: {"query": "Имя артиста"}
   tts_response: "Включаю популярные треки {Артист}"

3. PLAY_ALBUM: включить музыкальный альбом.
   params: {"query": "Название альбома"}
   tts_response: "Включаю альбом {Альбом}"

4. PLAY_PLAYLIST: тематический, жанровый или под настроение плейлист.
   params: {"query": "Тематика или название плейлиста"}
   tts_response: "Включаю плейлист {Плейлист}"

5. PLAY_MY_PLAYLIST: собственный плейлист пользователя из его профиля.
   params: {"query": "Название плейлиста"}
   tts_response: "Включаю ваш плейлист {Плейлист}"

6. PLAY_MY_WAVE: бесконечный персональный поток «Моя волна» (радио, поток, включи что-нибудь).
   params: {}
   tts_response: "Запускаю Мою волну"

7. PLAY_LIKED: любимые треки / плейлист мне нравится.
   params: {}
   tts_response: "Включаю любимые треки"

8. PLAY_EDITORIAL: специальные персонализированные подборки Яндекса.
   params: {"kind": "daily" | "neverHeard" | "recentTracks" | "missedLikes", "title": "Плейлист дня" | "Дежавю" | "Премьера" | "Тайник"}
   tts_response: "Включаю {title}"

9. PAUSE: пауза, стоп, замолчи, останови, тишина.
   params: {}
   tts_response: "Пауза"

10. RESUME: продолжить, возобновить, играй дальше, сними с паузы.
    params: {}
    tts_response: "Продолжаю"

11. NEXT: следующий трек, пропусти, скип, переключи.
    params: {}
    tts_response: "Следующий трек"

12. PREV: предыдущий трек, верни назад.
    params: {}
    tts_response: "Предыдущий трек"

13. SET_VOLUME: установить точную громкость от 0 до 100% (на полную / на максимум -> 100, половина -> 50, еле слышно -> 15).
    params: {"value": 0..100}
    tts_response: "Громкость {value} процентов"

14. CHANGE_VOLUME: изменить громкость относительно (громче -> +15, тише -> -15, сделай чуть потише -> -10).
    params: {"delta": +/- число}
    tts_response: "Делаю громче" или "Делаю тише"

15. LIKE: поставить лайк, добавить в избранное, мне нравится.
    params: {}
    tts_response: "Добавлено в любимые треки"

16. DISLIKE: поставить дизлайк, удалить из любимых, не нравится.
    params: {}
    tts_response: "Удалено из любимых, переключаю"

17. INFO: что сейчас играет, информация о треке, кто поёт.
    params: {}
    tts_response: null

18. QUEUE: очередь треков, что будет играть дальше.
    params: {}
    tts_response: null

19. OPEN_VOICE_MENU: сменить голос, выбор голоса, другой голос.
    params: {}
    tts_response: null

20. SET_VOICE: прямая смена голоса ассистента на конкретный.
    params: {"voice": "xenia" | "aidar" | "baya" | "eugene"}
    tts_response: "Голос изменён"

21. GET_VOICE: какой сейчас голос, кто ты.
    params: {}
    tts_response: null

22. SET_TRIGGER: сменить слово активации / триггер.
    params: {"new_trigger": "..."}
    tts_response: "Триггер изменён"

23. GET_TRIGGER: какой сейчас триггер / слово активации.
    params: {}
    tts_response: null

24. HELP: справка, список команд, что ты умеешь.
    params: {}
    tts_response: null

25. EXIT: завершение работы, выход, пока.
    params: {}
    tts_response: "До встречи!"

26. GENERAL_QUESTION: ЛЮБЫЕ вопросы, не связанные с музыкой (расчёты, рецепты, факты, справки, погода, общение).
    params: {"answer": "<краткий, ёмкий ответ 1-2 предложения, строго без markdown-разметки и спецсимволов для чистого чтения вслух>"}
    tts_response: "<тот же краткий ответ для озвучки голосом>"

27. UNDO: отмена последнего действия, возврат к предыдущему плейлисту, треку или очереди.
    Фразы: «отмена», «отмени», «нет нет нет», «не то!», «не то включил», «верни как было», «верни назад», «верни плейлист», «я не то просил».
    params: {}
    tts_response: "Отменяю"

28. REDO: повторить отмененное действие, верни вперед.
    Фразы: «повтори», «верни вперед», «redo».
    params: {}
    tts_response: "Повторяю действие"

29. HISTORY: история воспроизведения, что играло до этого.
    Фразы: «история», «история треков», «что играло».
    params: {}
    tts_response: null

30. CHANGE_YANDEX_TOKEN: сменить, обновить или ввести токен Яндекс.Музыки.
    Фразы: «сменить токен», «обновить токен яндекса», «ввести токен яндекс».
    params: {}
    tts_response: "Пожалуйста, введите новый токен Яндекс Музыки в окне терминала"

31. CHANGE_GEMINI_KEY: сменить, обновить или ввести API-ключ Google AI Studio.
    Фразы: «сменить api ключ», «обновить ключ gemini», «ввести ключ google ai».
    params: {}
    tts_response: "Пожалуйста, введите новый API ключ Google в окне терминала"

32. START_LIVE_CHAT: начать непрерывный свободный голосовой диалог (Live API режим).
    Фразы: «поговори со мной», «давай поболтаем», «режим разговора», «режим диалога».
    params: {}
    tts_response: "Включаю режим разговора"

33. RESUME_LIVE_CHAT: возобновить только что завершённый диалог по фразе 'Морсик, подожди!'.
    Фразы: «подожди», «морсик подожди», «не уходи», «стой».
    params: {}
    tts_response: "Я тут, продолжай"

34. TRIGGER_ACTIVATED: только обращение к ассистенту по имени без команды («Морсик», «Карлуша», «Эй Висп», «Висп»).
    params: {}
    tts_response: "Слушаю вас"
"""

class HybridNLU:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY
        self._cooldowns: Dict[str, float] = {}  # {model_name: timestamp_when_available}

    def _is_model_available(self, model: str) -> bool:
        until = self._cooldowns.get(model, 0)
        return time.time() >= until

    def _mark_model_cooldown(self, model: str, seconds: float = 60.0):
        self._cooldowns[model] = time.time() + seconds
        log_warning(f"Модель {model} помещена в cooldown на {seconds:.0f}с (лимит квоты исчерпан)")

    def parse(self, raw_text: str) -> Dict[str, Any]:
        """
        Главная точка входа:
        1. Fast-Path (0 мс) для мгновенных сервисных команд (пауза, дальше, меню)
        2. Каскадный опрос Gemini 3.5 Flash Lite -> Gemini 3.1 Flash Lite
        3. Fallback на локальный CommandParser при отсутствии сети / исчерпании квот
        """
        clean_text = raw_text.strip().lower()
        if not clean_text:
            return {"action": "NONE"}

        # --- 1. FAST-PATH (Мгновенное локальное исполнение без сетевого запроса) ---
        fast_result = self._check_fast_path(clean_text)
        if fast_result is not None:
            log_debug(f"Fast-path сработал для '{clean_text}': {fast_result['action']}")
            return fast_result

        # --- 2. ОБЛАЧНЫЙ КАСКАД LLM (Gemini 3.5 -> 3.1) ---
        if self.api_key:
            for model_name in MODELS_CASCADE:
                if not self._is_model_available(model_name):
                    continue

                try:
                    log_debug(f"Попытка классификации через {model_name}...")
                    t0 = time.time()
                    llm_result = self._call_gemini(model_name, raw_text)
                    t_el = time.time() - t0

                    if llm_result:
                        log_info(f"LLM ({model_name}) за {t_el:.2f}с определила интент: {llm_result['action']}")
                        return llm_result

                except Exception as e:
                    log_warning(f"Ошибка вызова {model_name}: {e}")

        # --- 3. FALLBACK: Локальный эвристический парсер ---
        log_info("Использование локального CommandParser (fallback)...")
        return CommandParser.parse(raw_text)

    def _check_fast_path(self, clean_text: str) -> Optional[Dict[str, Any]]:
        """Проверка тривиальных команд плеера, не требующих обращения к нейросети"""
        # Обращение по имени (триггер) без команды
        if clean_text in [
            "морсик", "морсика", "морсику", "морфик", "горсик",
            "карлуша", "корлуша", "карлушка", "карлуше",
            "эй висп", "висп", "виспа", "эй", "hey wisp", "wisp", "ei vispa", "morsik", "karlusha"
        ]:
            return {"action": "TRIGGER_ACTIVATED", "tts_response": "Слушаю вас"}

        # Отмена действия / Возврат к предыдущему состоянию (Undo)
        if clean_text in [
            "отмена", "отмени", "отменить", "откати", "откатить", "откат",
            "нет нет нет", "нет нет", "не то", "не то включил", "не туда",
            "верни как было", "верни назад", "верни прошлое", "верни плейлист",
            "верни старый трек", "предыдущее состояние", "верни предыдущее", "undo"
        ] or re.search(r'^(?:отмени|откат|верни)\s+(?:команду|действие|как\s+было|назад|плейлист|прошлое)$', clean_text):
            return {"action": "UNDO"}

        # Повтор отмененного действия (Redo)
        if clean_text in ["верни вперед", "повтори отмену", "redo"]:
            return {"action": "REDO"}

        # История
        if clean_text in ["история", "история треков", "история очередей", "что играло", "список истории"]:
            return {"action": "HISTORY"}

        # Смена токена Яндекс.Музыки и ключа Gemini
        if clean_text in [
            "сменить токен", "смени токен", "обновить токен", "обнови токен",
            "сменить токен яндекса", "смени токен яндекса", "токен яндекса", "токен яндекс"
        ] or re.search(r'\b(?:смени(?:ть)?|обнови(?:ть)?|введи(?:ть)?)\s+(?:токен|ключ)\s+яндекс\w*\b', clean_text):
            return {"action": "CHANGE_YANDEX_TOKEN"}

        if clean_text in [
            "сменить ключ", "смени ключ", "обновить ключ", "обнови ключ",
            "сменить api ключ", "смени api ключ", "сменить ключ gemini", "смени ключ gemini",
            "ключ gemini", "api ключ", "ключ гугл", "ключ google"
        ] or re.search(r'\b(?:смени(?:ть)?|обнови(?:ть)?|введи(?:ть)?)\s+(?:api\s+)?ключ\s*(?:гугл|google|gemini|джемини)?\b', clean_text):
            return {"action": "CHANGE_GEMINI_KEY"}

        # Live-режим общения и возобновление
        if clean_text in [
            "поговори со мной", "давай поболтаем", "поболтаем", "поговорим",
            "режим разговора", "режим диалога", "живой разговор", "включи разговор", "live"
        ] or re.search(r'\b(?:поговори\s+со\s+мной|давай\s+поболтаем|режим\s+(?:разговора|диалога))\b', clean_text):
            return {"action": "START_LIVE_CHAT"}

        if clean_text in ["подожди", "морсик подожди", "не уходи", "стой", "я тут", "вернись"]:
            return {"action": "RESUME_LIVE_CHAT"}

        # Пауза / Стоп
        if clean_text in ["пауза", "стоп", "останови", "замолчи", "тишина", "хватит", "pause", "stop", "top"]:
            return {"action": "PAUSE"}

        # Возобновление
        if clean_text in ["продолжи", "играй", "плей", "старт", "возобнови", "resume", "play"]:
            return {"action": "RESUME"}

        # Переключение
        if clean_text in ["дальше", "след", "следующий", "пропусти", "скип", "next", "skip", ">"]:
            return {"action": "NEXT"}

        if clean_text in ["назад", "предыдущий", "предыдущий трек", "prev", "back", "<"]:
            return {"action": "PREV"}

        # Выход
        if clean_text in ["выход", "выйти", "закрыть", "exit", "quit", "q"]:
            return {"action": "EXIT"}

        # Меню голоса
        if clean_text in ["сменить голос", "выбрать голос", "другой голос", "меню голоса"]:
            return {"action": "OPEN_VOICE_MENU"}

        # Запрос триггера
        if re.search(r'\b(?:какой|скажи|напомни)\s+(?:сейчас\s+)?триг+ер\b', clean_text):
            return {"action": "GET_TRIGGER"}

        return None

    def _call_gemini(self, model_name: str, user_text: str) -> Optional[Dict[str, Any]]:
        """Отправка запроса в Gemini REST API со строгим JSON-ответом"""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={self.api_key}"

        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_text}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "maxOutputTokens": 300
            }
        }

        try:
            resp = requests.post(url, json=payload, timeout=4.0)
            if resp.status_code == 200:
                body = resp.json()
                raw_json_str = body["candidates"][0]["content"]["parts"][0]["text"]
                data = json.loads(raw_json_str)
                return self._map_llm_json_to_command(data)

            elif resp.status_code in [429, 503]:
                # Исчерпание квоты или перегрузка
                log_warning(f"Gemini {model_name} вернул статус {resp.status_code}: {resp.text}")
                self._mark_model_cooldown(model_name, seconds=120.0)
                return None
            else:
                log_warning(f"Неожиданный ответ Gemini ({resp.status_code}): {resp.text}")
                return None

        except requests.Timeout:
            log_warning(f"Таймаут обращения к {model_name} (>4.0с)")
            return None
        except Exception as e:
            log_warning(f"Ошибка запроса к {model_name}: {e}")
            return None

    def _map_llm_json_to_command(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Преобразует JSON от Gemini во внутренний словарь действий ассистента"""
        intent = data.get("intent", "SMART_SEARCH").upper()
        params = data.get("params", {})
        tts_resp = data.get("tts_response")

        res: Dict[str, Any] = {
            "action": intent,
            "tts_response": tts_resp
        }

        if intent in ["PLAY_TRACK", "PLAY_ARTIST", "PLAY_ALBUM", "PLAY_PLAYLIST", "PLAY_MY_PLAYLIST"]:
            res["query"] = params.get("query", "")

        elif intent == "PLAY_EDITORIAL":
            res["kind"] = params.get("kind", "daily")
            res["title"] = params.get("title", "Плейлист дня")

        elif intent == "SET_VOLUME":
            res["value"] = int(params.get("value", 70))

        elif intent == "CHANGE_VOLUME":
            res["delta"] = int(params.get("delta", 10))

        elif intent == "SET_VOICE":
            res["voice"] = params.get("voice", "xenia")

        elif intent == "SET_TRIGGER":
            res["new_trigger"] = params.get("new_trigger", "Морсик").capitalize()

        elif intent == "GENERAL_QUESTION":
            answer = params.get("answer") or tts_resp or "Я не уверен в ответе."
            res["answer"] = answer
            if not tts_resp:
                res["tts_response"] = answer

        return res
