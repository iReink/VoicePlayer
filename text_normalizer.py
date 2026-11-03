"""
Модуль фонетической нормализации текста, преобразования чисел в слова и транскрибации латиницы (G2P) для Silero TTS v5.
"""
import re

# Словарь популярных терминов, брендов, исполнителей и музыкальных терминов
COMMON_TERMS = {
    # Популярные группы и артисты
    "queen": "Квин",
    "queens": "Квинс",
    "twenty": "тв+енти",
    "one": "ван",
    "pilots": "п+айлотс",
    "pilot": "п+айлот",
    "imagine": "им+эджин",
    "dragons": "др+эгонс",
    "dragon": "др+эгон",
    "linkin": "л+инкин",
    "park": "парк",
    "the": "",
    "beatles": "б+итлз",
    "metallica": "мет+аллика",
    "nirvana": "нирв+ана",
    "eminem": "эмин+эм",
    "billie": "б+илли",
    "eilish": "+айлиш",
    "ac/dc": "эй-си ди-си",
    "acdc": "эй-си ди-си",
    "rammstein": "рамшт+айн",
    "chili": "ч+или",
    "peppers": "п+епперс",
    "coldplay": "к+олдплей",
    "depeche": "деп+еш",
    "mode": "мод",
    "michael": "м+айкл",
    "jackson": "дж+ексон",
    "green": "грин",
    "day": "дэй",
    "pink": "пинк",
    "floyd": "флойд",
    "rolling": "р+оллинг",
    "stones": "ст+оунз",
    "scorpions": "ск+орпионс",
    "radiohead": "радиох+ед",
    "muse": "мьюз",
    "arctic": "+арктик",
    "monkeys": "м+анкиз",
    "daft": "дафт",
    "punk": "панк",
    "lady": "л+еди",
    "gaga": "г+ага",
    "weeknd": "у+икенд",
    "weekend": "у+икенд",

    # Названия треков и музыкальные слова
    "bohemian": "бох+емиан",
    "rhapsody": "рапс+одия",
    "rapsody": "рапс+оди",
    "stressed": "стр+ессед",
    "out": "аут",
    "believer": "бел+ивер",
    "numb": "намб",
    "in": "ин",
    "end": "энд",
    "smells": "смеллс",
    "like": "лайк",
    "teen": "тин",
    "spirit": "сп+ирит",
    "rock": "рок",
    "pop": "поп",
    "rap": "рэп",
    "jazz": "джаз",
    "metal": "м+етал",
    "track": "трек",
    "song": "песня",
    "album": "альб+ом",
    "single": "сингл",
    "play": "плей",
    "pause": "п+ауза",
    "stop": "стоп",
    "next": "некст",
    "skip": "скип",
    "queue": "кью",
    "volume": "гр+омкость",
    "live": "лайв",
    "remix": "р+емикс",
    "feat": "фит",
    "featuring": "ф+ичеринг",
    "love": "лав",
    "night": "найт",
    "time": "тайм",
    "life": "лайф",
    "world": "ворлд",
    "radio": "р+адио",

    # Бренды и сервисы
    "yandex": "Яндекс",
    "youtube": "Ют+уб",
    "google": "Гугл",
    "apple": "Эппл",
    "spotify": "Спотиф+ай",
    "telegram": "Телегр+ам",
    "bluetooth": "Блют+ус",
    "wi-fi": "Вай-ф+ай",
    "wifi": "Вай-ф+ай",
}

PHONETIC_DIGRAPHS = [
    ("tion", "шн"), ("sion", "жн"), ("igh", "ай"), ("all", "ол"),
    ("ch", "ч"), ("sh", "ш"), ("th", "т"), ("ph", "ф"),
    ("ck", "к"), ("qu", "кв"), ("wh", "в"), ("ee", "и"),
    ("oo", "у"), ("ea", "и"), ("ai", "эй"), ("ay", "эй"),
    ("oy", "ой"), ("oi", "ой"), ("ow", "оу"), ("ou", "ау"),
]

LETTER_MAP = {
    "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
    "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
    "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
    "y": "и", "z": "з"
}

def number_to_russian(n: int, feminine: bool = False) -> str:
    """Преобразует целое число от 0 до 999 999 999 в русские слова."""
    if n == 0:
        return "ноль"
    if n < 0:
        return "минус " + number_to_russian(-n, feminine=feminine)

    units_m = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
               "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
               "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
    units_f = ["", "одна", "две", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
               "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
               "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
    tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
    hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

    def triad_to_ru(val: int, is_fem: bool = False) -> str:
        words = []
        u = units_f if is_fem else units_m
        h_val = val // 100
        t_val = val % 100
        if h_val:
            words.append(hundreds[h_val])
        if t_val < 20:
            if t_val > 0:
                words.append(u[t_val])
        else:
            words.append(tens[t_val // 10])
            rem = t_val % 10
            if rem:
                words.append(u[rem])
        return " ".join(words)

    parts = []
    # Миллиарды
    b = n // 1_000_000_000
    if b:
        triad_str = triad_to_ru(b, is_fem=False)
        last2 = b % 100
        last1 = b % 10
        ending = "миллиардов"
        if not (11 <= last2 <= 19):
            if last1 == 1:
                ending = "миллиард"
            elif 2 <= last1 <= 4:
                ending = "миллиарда"
        parts.append(f"{triad_str} {ending}")

    # Миллионы
    m = (n % 1_000_000_000) // 1_000_000
    if m:
        triad_str = triad_to_ru(m, is_fem=False)
        last2 = m % 100
        last1 = m % 10
        ending = "миллионов"
        if not (11 <= last2 <= 19):
            if last1 == 1:
                ending = "миллион"
            elif 2 <= last1 <= 4:
                ending = "миллиона"
        parts.append(f"{triad_str} {ending}")

    # Тысячи
    th = (n % 1_000_000) // 1_000
    if th:
        triad_str = triad_to_ru(th, is_fem=True)
        last2 = th % 100
        last1 = th % 10
        ending = "тысяч"
        if not (11 <= last2 <= 19):
            if last1 == 1:
                ending = "тысяча"
            elif 2 <= last1 <= 4:
                ending = "тысячи"
        parts.append(f"{triad_str} {ending}")

    # Единицы
    rem = n % 1000
    if rem or not parts:
        parts.append(triad_to_ru(rem, is_fem=feminine))

    return " ".join(p for p in parts if p).strip()

def transcribe_english_word(word: str) -> str:
    """Транскрибирует отдельное английское слово в кириллическую фонетику."""
    clean = word.strip().lower()
    if not clean:
        return ""
    if clean in COMMON_TERMS:
        val = COMMON_TERMS[clean]
        if not val:
            return ""
        return val.capitalize() if word[0].isupper() else val

    s = clean
    for pat, rep in PHONETIC_DIGRAPHS:
        s = s.replace(pat, rep)

    s = re.sub(r'c([еий])', r'с\1', s)
    s = re.sub(r'g([еий])', r'дж\1', s)

    chars = [LETTER_MAP.get(ch, ch) for ch in s]
    res = "".join(chars)
    return res.capitalize() if word[0].isupper() else res

def normalize_text_for_silero(text: str) -> str:
    """
    Полная подготовка текста для чистого синтеза в Silero v5:
    1. Замена спецсимволов и формул (&, +, %)
    2. Преобразование чисел и цифр (включая '21 pilots', '80%', '№ 5', тайминги 03:22) в русские слова
    3. Транскрибация латиницы (Queen -> Квин)
    4. Очистка пунктуации
    """
    if not text:
        return ""

    # Специфические популярные группы с цифрами
    text = re.sub(r'\b21\s+pilots\b', 'двадцать один п+айлотс', text, flags=re.IGNORECASE)
    text = re.sub(r'\bби[\s-]*2\b', 'Би два', text, flags=re.IGNORECASE)
    text = re.sub(r'\bu\s*2\b', 'Ю ту', text, flags=re.IGNORECASE)
    text = re.sub(r'\bblink[\s-]*182\b', 'Блинк сто восемьдесят два', text, flags=re.IGNORECASE)
    text = re.sub(r'\bsum[\s-]*41\b', 'Сам сорок один', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmaroon[\s-]*5\b', 'Мар+ун пять', text, flags=re.IGNORECASE)

    # Замена символов и знаков (математический плюс, проценты, номера)
    text = text.replace("&", " и ")
    text = re.sub(r'(?<=\d)\s*\+\s*(?=\d)|(?<=\s)\+(?=\s)', ' плюс ', text)
    text = re.sub(r'№\s*(\d+)', r'номер \1', text)
    text = re.sub(r'(\d+)\s*%', r'\1 процентов', text)
    text = text.replace("%", " процентов")

    # Преобразование таймингов типа 03:22 или 3:22
    def _time_repl(m):
        mins = int(m.group(1))
        secs = int(m.group(2))
        return f"{number_to_russian(mins, feminine=True)} мин {number_to_russian(secs, feminine=True)} сек"
    text = re.sub(r'\b(\d{1,2}):(\d{2})\b', _time_repl, text)

    # Преобразование всех остальных чисел в русские слова
    def _num_repl(m):
        try:
            val = int(m.group(0))
            return number_to_russian(val)
        except Exception:
            return m.group(0)
    text = re.sub(r'\b\d+\b', _num_repl, text)

    # Преобразуем латинские слова
    if re.search(r'[A-Za-z]', text):
        text = re.sub(r'[A-Za-z]+(?:\'[A-Za-z]+)?', lambda m: transcribe_english_word(m.group(0)), text)

    # Чистим пунктуацию
    text = re.sub(r'\s*—\s*', ' — ', text)
    text = re.sub(r'\s*-\s*', ' — ', text)
    text = re.sub(r'[^\w\s,.!?«»"—+]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
