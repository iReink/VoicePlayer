"""
Парсер текстовых команд на естественном русском языке с поддержкой фонетических алиасов
"""
import re
from typing import Dict, Any

# Фонетическая нормализация популярных групп и исполнителей при распознавании
PHONETIC_ALIASES = [
    (r'\b(?:к\s+вин|квин|куин)\b', 'Queen'),
    (r'\b(?:21\s+пайлец|пайлец|двадцать\s+один\s+пайлотс|твенти\s+ван\s+пайлотс|21\s+пилотс)\b', 'twenty one pilots'),
    (r'\b(?:линкин\s+парк|линкинпарк)\b', 'Linkin Park'),
    (r'\b(?:имэджин\s+дрэгонс|имаджин\s+драгонс|имейджин\s+дрэгонс)\b', 'Imagine Dragons'),
    (r'\b(?:пинк\s+флойд)\b', 'Pink Floyd'),
    (r'\b(?:роллинг\s+стоунз|роллинг\s+стоунс)\b', 'The Rolling Stones'),
    (r'\b(?:битлз|зе\s+битлз|биттлз)\b', 'The Beatles'),
    (r'\b(?:рамштайн|раммштайн)\b', 'Rammstein'),
    (r'\b(?:металлика|металика)\b', 'Metallica'),
    (r'\b(?:нирвана)\b', 'Nirvana'),
    (r'\b(?:эминем|эминэм)\b', 'Eminem'),
    (r'\b(?:билли\s+айлиш)\b', 'Billie Eilish'),
    (r'\b(?:майкл\s+джексон)\b', 'Michael Jackson'),
    (r'\b(?:депеш\s+мод)\b', 'Depeche Mode'),
    (r'\b(?:эйси\s+диси|ас\s+дс)\b', 'AC/DC'),
    (r'\b(?:ред\s+хот\s+чилли\s+пепперс|перцы)\b', 'Red Hot Chili Peppers'),
    (r'\b(?:грин\s+дэй|гриндей)\b', 'Green Day'),
    (r'\b(?:колдплей|колдилей)\b', 'Coldplay'),
    (r'\b(?:би\s*2|би\s+два)\b', 'Би-2'),
]

class CommandParser:
    @staticmethod
    def parse(raw_text: str) -> Dict[str, Any]:
        raw = raw_text.strip().lstrip('\ufeff\u200b').strip().lower()
        if not raw:
            return {"action": "NONE"}

        # 1. Очистка от знаков препинания по краям (точки, многоточия, восклицания, кавычки)
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', raw).strip()
        # Заменяем знаки препинания внутри строки на пробелы
        clean_text = re.sub(r'[.,!?:;\"\'«»()—–-]+', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        if not clean_text:
            return {"action": "NONE"}

        # 1.1 Обращение к ассистенту по имени без команды (только триггер)
        if clean_text in [
            "морсик", "морсика", "морсику", "морфик", "горсик",
            "карлуша", "корлуша", "карлушка", "карлуше",
            "эй висп", "висп", "виспа", "эй", "hey wisp", "wisp", "ei vispa", "morsik", "karlusha"
        ]:
            return {"action": "TRIGGER_ACTIVATED"}

        # 2. Запросы триггера (wake-word)
        if re.search(r'\b(?:какой|скажи|напомни)\s+(?:сейчас\s+)?триг+ер\b', clean_text):
            return {"action": "GET_TRIGGER"}

        change_regex = r'(?:теперь\s+|смени(?:ть)?\s+|новый\s+)?триг+ер\s*(?:теперь\s+)?(?:на\s+|—\s*|-\s*|:\s*|это\s*)?([а-яёa-z0-9]+)'
        change_trig = re.search(change_regex, clean_text)
        if change_trig and any(w in clean_text for w in ["смени", "теперь", "новый", "поменяй"]):
            new_val = change_trig.group(1).strip().capitalize()
            return {"action": "SET_TRIGGER", "new_trigger": new_val}

        # 3. Голос ассистента и голосовое меню
        if re.search(r'\b(?:какой|скажи|напомни)\s+(?:сейчас\s+)?голос\b', clean_text):
            return {"action": "GET_VOICE"}

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

        # Прямая смена голоса: «смени голос на айдар», «голос ксения»
        direct_voice = re.search(r'(?:смени(?:ть)?|выбери|поставь|включи)?\s*голос\s*(?:на)?\s*([а-яa-z]+)', clean_text)
        if direct_voice:
            target_voice = direct_voice.group(1).lower()
            voice_map = {
                "ксения": "xenia", "ксюша": "xenia", "xenia": "xenia",
                "айдар": "aidar", "aidar": "aidar",
                "бая": "baya", "боя": "baya", "baya": "baya",
                "евгений": "eugene", "женя": "eugene", "eugene": "eugene",
                "классик": "kseniya", "kseniya": "kseniya"
            }
            if target_voice in voice_map:
                return {"action": "SET_VOICE", "voice": voice_map[target_voice]}

        # Live-режим свободного голосового общения
        if clean_text in [
            "поговори со мной", "давай поболтаем", "поболтаем", "поговорим",
            "режим разговора", "режим диалога", "живой разговор", "включи разговор", "live"
        ] or re.search(r'\b(?:поговори\s+со\s+мной|давай\s+поболтаем|режим\s+(?:разговора|диалога))\b', clean_text):
            return {"action": "START_LIVE_CHAT"}

        # Возобновление диалога при льготном периоде «Морсик, подожди!»
        if clean_text in ["подожди", "морсик подожди", "не уходи", "стой", "я тут", "вернись"]:
            return {"action": "RESUME_LIVE_CHAT"}

        # Открытие меню голосов
        if clean_text in [
            "сменить голос", "смени голос", "выбрать голос", "выбери голос",
            "другой голос", "поменяй голос", "настройки голоса", "выбор голоса",
            "голос", "голоса", "какие голоса", "список голосов", "voice", "voices"
        ]:
            return {"action": "OPEN_VOICE_MENU"}

        # 3. Служебные команды управления плеером
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

        # История воспроизведения
        if clean_text in ["история", "история треков", "история очередей", "что играло", "список истории"]:
            return {"action": "HISTORY"}

        # Пауза / Стоп
        if clean_text in ["пауза", "стоп", "останови", "замолчи", "тишина", "хватит", "pause", "stop", "top"] or \
           clean_text in ["стоп музыка", "музыка стоп", "пауза музыка", "останови музыку"]:
            return {"action": "PAUSE"}

        # Возобновление
        if clean_text in ["продолжи", "играй", "плей", "старт", "возобнови", "resume", "play", "продолжить"]:
            return {"action": "RESUME"}

        # Переключение треков
        if clean_text in ["дальше", "след", "следующий", "следующий трек", "пропусти", "скип", "next", "skip", ">"]:
            return {"action": "NEXT"}

        if clean_text in ["назад", "предыдущий", "предыдущий трек", "предыд", "prev", "back", "<"]:
            return {"action": "PREV"}

        # Инфо / Очередь / Справка / Выход
        if clean_text in ["выход", "выйти", "закрыть", "exit", "quit", "q"]:
            return {"action": "EXIT"}

        if clean_text in ["помощь", "справка", "хелп", "help", "?"]:
            return {"action": "HELP"}

        if clean_text in ["что играет", "текущий трек", "инфо", "информация", "сейчас играет", "np", "info"]:
            return {"action": "INFO"}

        if clean_text in ["очередь", "список", "треклист", "queue"]:
            return {"action": "QUEUE"}

        if clean_text in ["лайк", "нравится", "добавь в любимые", "в избранное", "кайф", "like", "+"]:
            return {"action": "LIKE"}

        if clean_text in ["дизлайк", "удали из любимых", "не нравится", "dislike", "-"]:
            return {"action": "DISLIKE"}

        # 4. Громкость
        vol_match = re.search(r'(?:громкость|звук)\s*(?:на\s*)?(\d{1,3})%?', clean_text)
        if vol_match:
            val = int(vol_match.group(1))
            return {"action": "SET_VOLUME", "value": val}

        if any(w in clean_text for w in ["тише", "потише", "сделай тише", "убавь", "уменьши звук", "тихий"]):
            return {"action": "CHANGE_VOLUME", "delta": -10}

        if any(w in clean_text for w in ["громче", "погромче", "сделай громче", "прибавь", "увеличь звук"]):
            return {"action": "CHANGE_VOLUME", "delta": 10}

        # 5. Персональные подборки и радио
        if any(w in clean_text for w in ["моя волна", "мою волну", "волну", "радио", "my wave"]):
            return {"action": "PLAY_MY_WAVE"}

        # Личные плейлисты пользователя
        if re.search(r'\b(?:мой\s+плейлист|мои\s+плейлисты|свой\s+плейлист)\b', clean_text):
            return {"action": "PLAY_MY_PLAYLIST", "query": clean_text}

        if any(w in clean_text for w in ["мои треки", "любимые треки", "любимая музыка", "мне нравится", "моя музыка", "любимое"]):
            return {"action": "PLAY_LIKED"}

        if "плейлист дня" in clean_text:
            return {"action": "PLAY_EDITORIAL", "kind": "playlistOfTheDay", "title": "Плейлист дня"}

        if "дежавю" in clean_text:
            return {"action": "PLAY_EDITORIAL", "kind": "neverHeard", "title": "Дежавю"}

        if "премьера" in clean_text or "премьеру" in clean_text:
            return {"action": "PLAY_EDITORIAL", "kind": "recentTracks", "title": "Премьера"}

        if "тайник" in clean_text:
            return {"action": "PLAY_EDITORIAL", "kind": "missedLikes", "title": "Тайник"}

        # 6. Удаление вводных слов типа "включи", "включит", "включите", "поставь", "сыграй"
        prefix_pattern = r'^(?:включ[иеит]+(?:-ка)?|постав[ьите]+(?:-ка)?|сыграй(?:те)?|запусти(?:те)?|найди(?:те)?|хочу\s+послушать|play|вруби(?:те)?)\s+'
        query_text = re.sub(prefix_pattern, '', clean_text).strip()

        # Фонетическая замена русских калек на оригинальные названия групп
        for pattern, replacement in PHONETIC_ALIASES:
            query_text = re.sub(pattern, replacement, query_text, flags=re.IGNORECASE)

        # Поиск по конкретной сущности (исполнитель / альбом / плейлист / трек)
        artist_match = re.search(r'^(?:исполнител[яей]|артист[а]?|групп[уы]|песни|треки)\s+(.+)$', query_text)
        if artist_match:
            return {"action": "PLAY_ARTIST", "query": artist_match.group(1).strip()}

        album_match = re.search(r'^(?:альбом)\s+(.+)$', query_text)
        if album_match:
            return {"action": "PLAY_ALBUM", "query": album_match.group(1).strip()}

        playlist_match = re.search(r'^(?:плейлист|сборник)\s+(.+)$', query_text)
        if playlist_match:
            pl_name = playlist_match.group(1).strip()
            if any(w in pl_name for w in ["мой", "свой"]):
                return {"action": "PLAY_MY_PLAYLIST", "query": pl_name}
            return {"action": "PLAY_PLAYLIST", "query": pl_name}

        track_match = re.search(r'^(?:трек|песн[юя])\s+(.+)$', query_text)
        if track_match:
            return {"action": "PLAY_TRACK", "query": track_match.group(1).strip()}

        # 7. Умный общий поиск (если не указана точная категория)
        return {"action": "SMART_SEARCH", "query": query_text}

    @staticmethod
    def parse_menu_response(raw_text: str) -> Dict[str, Any]:
        """Парсинг ответа пользователя в голосовом меню выбора голоса"""
        raw = raw_text.strip().lower()
        clean = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', raw).strip()

        # Отмена
        if clean in ["отмена", "отменить", "назад", "выход", "не надо", "стоп", "хватит", "cancel", "нет"]:
            return {"action": "CANCEL"}

        # Повторить список
        if clean in ["повтори", "повторить", "ещё раз", "еще раз", "что там", "какие варианты", "список", "repeat"]:
            return {"action": "REPEAT"}

        # Вариант 1 - Ксения
        if clean in ["1", "один", "первый", "первая", "ксения", "ксюша", "xenia"]:
            return {"action": "CHOOSE", "voice": "xenia", "name": "Ксения"}

        # Вариант 2 - Айдар
        if clean in ["2", "два", "второй", "айдар", "aidar"]:
            return {"action": "CHOOSE", "voice": "aidar", "name": "Айдар"}

        # Вариант 3 - Бая
        if clean in ["3", "три", "третий", "третья", "бая", "боя", "baya"]:
            return {"action": "CHOOSE", "voice": "baya", "name": "Бая"}

        # Вариант 4 - Евгений
        if clean in ["4", "четыре", "четвёртый", "четвертый", "евгений", "женя", "eugene"]:
            return {"action": "CHOOSE", "voice": "eugene", "name": "Евгений"}

        return {"action": "UNKNOWN", "raw": clean}
