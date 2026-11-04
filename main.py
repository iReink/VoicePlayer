"""
Главный модуль консольного ассистента Яндекс.Музыки с поддержкой Whisper STT, TTS, клавиатурного ввода и логирования
"""
import sys
import os
import queue
import threading
import time
from typing import Optional

# Скрываем стандартное приветствие pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Гарантируем корректный UTF-8 ввод/вывод в консоли Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from logger import log_info, log_error, log_debug, LOG_FILE
from config import ConfigManager
from env_manager import ensure_credentials_interactive, save_env_var, YANDEX_TOKEN_URL, GEMINI_KEY_URL
from service import MusicService
from player import MusicPlayer
from parser import CommandParser
from llm_nlu import HybridNLU
from tts import TTSEngine, AVAILABLE_VOICES
from voice import VoiceListener
from live_session import LiveSessionManager

console = Console()

def print_help(current_trigger: str):
    table = Table(title="[Команды и примеры запросов]", border_style="cyan")
    table.add_column("Категория", style="bold yellow", width=18)
    table.add_column("Примеры промптов (голосом или текстом)", style="green")
    
    table.add_row("Триггер-слово", f"«{current_trigger}, теперь тригер — Карлуша» / «какой сейчас триггер?»")
    table.add_row("Голос ассистента", "«сменить голос» / «выбрать голос» / «голос Айдар»")
    table.add_row("Мой плейлист", f"«{current_trigger}, включи мой плейлист» / «плейлист Travel»")
    table.add_row("Любой трек", f"«{current_trigger}, включи Queen Bohemian Rhapsody» / «Цой Группа крови»")
    table.add_row("Исполнитель", f"«{current_trigger}, включи исполнителя Король и Шут» / «артист Linkin Park»")
    table.add_row("Альбом", f"«{current_trigger}, включи альбом Meteora» / «альбом Hybrid Theory»")
    table.add_row("Плейлист", f"«{current_trigger}, включи плейлист рок» / «плейлист для тренировок»")
    table.add_row("Моя волна", f"«{current_trigger}, моя волна» / «волну» / «радио»")
    table.add_row("Подборки Яндекса", "плейлист дня / дежавю / премьера / тайник")
    table.add_row("Любимые треки", "мои треки / любимая музыка / мне нравится")
    table.add_row("Управление", "пауза / стоп / продолжи / дальше / назад")
    table.add_row("Отмена (Undo)", "отмена! / нет нет нет / верни как было / не то")
    table.add_row("История состояний", "история / что играло")
    table.add_row("Громкость", "громкость 80 / тише / громче")
    table.add_row("Лайки", "лайк (+) / дизлайк (-)")
    table.add_row("Информация", "что играет / инфо / очередь")
    table.add_row("Live-диалог", "«поговори со мной» / «давай поболтаем» / «подожди»")
    table.add_row("Ключи и токены", "сменить токен / сменить api ключ")
    table.add_row("Выход", "выход / exit / q")
    
    console.print(table)

def main():
    console.clear()
    log_info("=== Запуск Яндекс.Музыка Ассистент ===")
    console.print("[bold yellow]Инициализация Яндекс.Музыки и модулей...[/bold yellow]")

    # 1. Загрузка конфигурации
    config = ConfigManager()
    log_info(f"Конфигурация загружена: trigger='{config.trigger}', volume={config.volume}")

    # 1.1. Проверка наличия и ввод ключей при первом запуске
    ensure_credentials_interactive(console)

    # 2. Сервис Яндекс.Музыки
    try:
        service = MusicService()
        log_info(f"Яндекс.Музыка подключена. Пользователь: {service.user_name}")
    except Exception as e:
        log_error(f"Ошибка инициализации сервиса Яндекс.Музыки: {e}")
        console.print(f"[bold red]Ошибка инициализации сервиса Яндекс.Музыки:[/bold red] {e}")
        console.print(f"[dim]Подробности записаны в: {LOG_FILE}[/dim]")
        return

    # 3. Аудио-плеер
    def on_track_start(track, current_idx, total_count, playlist_title):
        info = service.format_track_info(track)
        status_line = f"[bold green]▶ Сейчас играет:[/bold green] [bold white]{info['artists']} — {info['title']}[/bold white]"
        sub_line = f"[dim]Альбом: {info['album']} | Длительность: {info['duration']} | Очередь: {current_idx}/{total_count}[/dim]"
        if playlist_title:
            sub_line += f" [dim]({playlist_title})[/dim]"
        
        console.print()
        console.print(Panel(f"{status_line}\n{sub_line}", border_style="green", expand=False))
        log_info(f"Старт трека: {info['artists']} — {info['title']} ({current_idx}/{total_count})")

    player = MusicPlayer(music_service=service, on_track_start=on_track_start)
    player.set_volume(config.volume)

    # 4. Синтез речи Silero TTS v5
    tts = TTSEngine(config_manager=config, player=player, device_mode="auto")

    # 5. Гибридный NLU классификатор (Gemini Flash Lite cascade -> локальный парсер)
    nlu = HybridNLU()

    # Очередь общих команд
    cmd_queue = queue.Queue()

    # 5. Обработчики голосового ввода
    def on_voice_status(status: str):
        if status == "RECORDING":
            console.print("[bold red][REC] [Запись речи...][/bold red]")
        elif status == "TRANSCRIBING":
            console.print("[bold yellow][~] [Распознавание речи...][/bold yellow]")

    def on_wake_word(model_name: str):
        names = {
            "morsik": "🐾 «Морсик»",
            "karlusha": "🦅 «Карлуша»",
            "hey_wisp": "⚡ «Эй, Висп!»"
        }
        pretty_name = names.get(model_name, f"«{model_name}»")
        console.print(f"[bold cyan]🔔 [KWS Активация: {pretty_name}][/bold cyan]")
        player.duck(True)

    def on_speech_start():
        player.duck(True)

    def on_speech_end(recognized: bool):
        if not recognized:
            player.duck(False)

    def on_voice_command(cmd_text: str, raw_text: str):
        cmd_queue.put(("VOICE", cmd_text, raw_text))

    def on_voice_log(msg: str):
        console.print(msg)

    voice_listener = VoiceListener(
        config_manager=config,
        on_command=on_voice_command,
        on_status=on_voice_status,
        on_speech_start=on_speech_start,
        on_speech_end=on_speech_end,
        on_log=on_voice_log,
        is_music_playing=lambda: player.is_playing,
        on_wake_word=on_wake_word
    )
    voice_listener.start()

    # Связываем TTS с микрофоном для предотвращения самозаписи
    tts.on_speech_state_change = voice_listener.set_tts_active

    # Фоновый прогрев модели Silero и озвучка готовности
    tts.warmup(announce_text="Голосовой ассистент готов к работе.")

    # Поток клавиатурного ввода
    def keyboard_input_loop():
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                text = line.strip()
                if text:
                    cmd_queue.put(("TEXT", text, None))
            except Exception:
                break

    kbd_thread = threading.Thread(target=keyboard_input_loop, daemon=True)
    kbd_thread.start()

    # Приветственный баннер
    active_v_name = AVAILABLE_VOICES.get(config.voice, {}).get("name", config.voice)
    banner_text = (
        f"[bold cyan]=== Голосовой и текстовый ассистент Яндекс.Музыки ===[/bold cyan]\n"
        f"Пользователь: [bold white]{service.user_name}[/bold white] | "
        f"Подписка Плюс: [bold green]{'Активна' if service.has_plus else 'Нет'}[/bold green] | "
        f"Громкость: [bold yellow]{player.volume}%[/bold yellow]\n"
        f"Триггер: [bold magenta]{config.trigger}[/bold magenta] | "
        f"Голос: [bold cyan]{active_v_name}[/bold cyan] (скажите: «сменить голос»)\n"
        f"[dim]Логирование активно в: {LOG_FILE.name} | 'помощь' — список команд[/dim]"
    )
    console.print(Panel(banner_text, border_style="cyan"))
    console.print("[dim]Готов к работе. Слушаю микрофон...[/dim]\n")

    def run_voice_menu():
        """Интерактивное меню выбора голоса ассистента с плавным затуханием музыки"""
        was_playing = player.is_playing
        if was_playing:
            player.fade_out_pause(duration=0.5)

        menu_table = Table(title="[Меню выбора голоса ассистента]", border_style="magenta")
        menu_table.add_column("№", style="bold cyan", width=4)
        menu_table.add_column("Голос", style="bold white", width=16)
        menu_table.add_column("Описание", style="yellow")
        menu_table.add_column("Статус", style="green")

        current_v = config.voice
        menu_items = [
            ("1", "xenia", "Ксения", "Живой разговорный женский тембр (эталон)"),
            ("2", "aidar", "Айдар", "Спокойный глубокий мужской баритон"),
            ("3", "baya", "Бая", "Мягкий и тёплый женский тембр"),
            ("4", "eugene", "Евгений", "Энергичный молодой мужской голос")
        ]
        for num, vid, name, desc in menu_items:
            status = "▶ Текущий" if vid == current_v else ""
            menu_table.add_row(num, name, desc, status)

        console.print()
        console.print(menu_table)
        console.print("[dim]Назовите номер или имя голоса, либо 'повтори' / 'отмена' (без триггера)[/dim]\n")

        voice_listener.set_force_command_mode(True)
        prompt_text = "Доступны четыре голоса: один — Ксения, два — Айдар, три — Бая, четыре — Евгений. Назовите номер или имя голоса, либо скажите отмена."
        tts.say(prompt_text, pause_music=False)

        menu_active = True
        menu_start_time = time.time()
        timeout_seconds = 25.0

        try:
            while menu_active:
                if time.time() - menu_start_time > timeout_seconds:
                    console.print("[yellow]Время ожидания выбора голоса истекло.[/yellow]")
                    tts.say("Время ожидания истекло.", pause_music=False)
                    break

                try:
                    source, raw_input, original_text = cmd_queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                if source == "VOICE":
                    console.print(f"[bold magenta][Голос]:[/bold magenta] [bold white]{original_text}[/bold white]")
                else:
                    console.print(f"[bold cyan][Текст]:[/bold cyan] [bold white]{raw_input}[/bold white]")

                parsed = CommandParser.parse_menu_response(raw_input)
                m_action = parsed["action"]

                if m_action == "CHOOSE":
                    chosen_id = parsed["voice"]
                    chosen_name = parsed["name"]
                    config.voice = chosen_id
                    console.print(f"[bold green][OK] Голос успешно переключен на:[/bold green] [bold yellow]{chosen_name}[/bold yellow] ({chosen_id})")
                    tts.say(f"Голос изменён на {chosen_name}. Приятно познакомиться!", voice=chosen_id, pause_music=False)
                    menu_active = False

                elif m_action == "REPEAT":
                    tts.say(prompt_text, pause_music=False)
                    menu_start_time = time.time()

                elif m_action == "CANCEL":
                    console.print("[yellow]Выбор голоса отменён.[/yellow]")
                    tts.say("Выбор голоса отменён.", pause_music=False)
                    menu_active = False

                else:
                    tts.say("Не распознано. Назовите номер от одного до четырёх или скажите отмена.", pause_music=False)
                    menu_start_time = time.time()

        finally:
            voice_listener.set_force_command_mode(False)
            if was_playing:
                player.fade_in_resume(duration=0.5)

    def run_live_dialog(resume_context: bool = False):
        """Запуск сессии живого общения Gemini Multimodal Live API"""
        log_info(f"Запуск Live-диалога (resume={resume_context})")
        voice_listener.pause_listening()
        live_mgr = LiveSessionManager(
            config_manager=config,
            player=player,
            music_service=service,
            whisper_model=voice_listener._get_model(),
            console=console
        )
        try:
            live_mgr.run_sync(resume_context=resume_context)
        finally:
            voice_listener.resume_listening()

    try:
        while True:
            try:
                source, cmd_text, raw_text = cmd_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                # Отображаем команду в терминале
                if source == "VOICE":
                    console.print(f"[bold magenta][Голос]:[/bold magenta] [bold white]{raw_text}[/bold white]")
                    log_info(f"Получена голосовая команда: raw='{raw_text}', parsed='{cmd_text}'")
                else:
                    console.print(f"[bold cyan][Текст]:[/bold cyan] [bold white]{cmd_text}[/bold white]")
                    log_info(f"Получена текстовая команда: '{cmd_text}'")

                if cmd_text == "триггер_активирован":
                    tts.say("Слушаю вас")
                    continue

                cmd = nlu.parse(cmd_text)
                action = cmd["action"]
                log_debug(f"Action: {action}, data: {cmd}")

                # Выход
                if action == "EXIT":
                    console.print("[yellow]Завершение работы... До встречи![/yellow]")
                    tts.say("До встречи!")
                    log_info("Выход по команде пользователя")
                    time.sleep(0.8)
                    break

                # Общий вопрос или факт (не связанный с музыкой)
                elif action == "GENERAL_QUESTION":
                    answer = cmd.get("answer") or cmd.get("tts_response", "")
                    console.print(Panel(
                        f"[bold white]{answer}[/bold white]",
                        title="[Ответ ассистента]",
                        border_style="yellow",
                        expand=False
                    ))
                    tts.say(answer, pause_music=True)

                # Запуск Live-диалога
                elif action == "START_LIVE_CHAT":
                    run_live_dialog(resume_context=False)

                # Возобновление диалога при фразе «Морсик, подожди!»
                elif action == "RESUME_LIVE_CHAT":
                    if LiveSessionManager.can_resume():
                        console.print("[bold green][OK] Возобновляем Live-диалог с сохранённым контекстом![/bold green]")
                        tts.say("Я тут, продолжаем!", pause_music=False)
                        run_live_dialog(resume_context=True)
                    else:
                        console.print("[dim]С момента окончания диалога прошло слишком много времени[/dim]")
                        tts.say("Диалог уже завершён. Скажите «поговори со мной» для нового разговора.", pause_music=False)

                # Запрос текущего триггера
                elif action == "GET_TRIGGER":
                    cur_trig = config.trigger
                    console.print(f"[bold cyan]Текущий триггер:[/bold cyan] [bold yellow]{cur_trig}[/bold yellow]")
                    tts.say(f"Текущий триггер — {cur_trig}", pause_music=False)

                # Смена триггера
                elif action == "SET_TRIGGER":
                    new_trig = cmd["new_trigger"]
                    config.trigger = new_trig
                    console.print(f"[bold green][OK] Триггер успешно изменён на:[/bold green] [bold yellow]{new_trig}[/bold yellow]")
                    tts.say(f"Триггер изменён на {new_trig}", pause_music=False)
                    log_info(f"Триггер изменен на: {new_trig}")

                # Запрос текущего голоса
                elif action == "GET_VOICE":
                    cur_v_name = AVAILABLE_VOICES.get(config.voice, {}).get("name", config.voice)
                    console.print(f"[bold cyan]Текущий голос:[/bold cyan] [bold yellow]{cur_v_name}[/bold yellow]")
                    tts.say(f"Текущий голос ассистента — {cur_v_name}", pause_music=False)

                # Прямая смена голоса
                elif action == "SET_VOICE":
                    new_v = cmd["voice"]
                    config.voice = new_v
                    v_name = AVAILABLE_VOICES.get(new_v, {}).get("name", new_v)
                    console.print(f"[bold green][OK] Голос успешно изменён на:[/bold green] [bold yellow]{v_name}[/bold yellow] ({new_v})")
                    tts.say(f"Голос изменён на {v_name}", voice=new_v, pause_music=False)
                    log_info(f"Голос изменен на: {new_v}")

                # Открытие меню голосов
                elif action == "OPEN_VOICE_MENU":
                    run_voice_menu()

                # Смена токена Яндекс.Музыки
                elif action == "CHANGE_YANDEX_TOKEN":
                    was_playing = player.is_playing
                    if was_playing:
                        player.fade_out_pause(duration=0.4)
                    tts.say("Пожалуйста, введите новый токен Яндекс Музыки в окне терминала.", pause_music=False)
                    console.print()
                    console.print(Panel(
                        f"[bold cyan]Смена токена Яндекс.Музыки[/bold cyan]\n\n"
                        f"1. Получите токен по ссылке: [bold white]{YANDEX_TOKEN_URL}[/bold white]\n"
                        f"2. Вставьте его ниже (или напишите 'отмена'):",
                        title="[Яндекс.Музыка]",
                        border_style="cyan"
                    ))
                    voice_listener.set_force_command_mode(True)
                    try:
                        new_tok = input(" Новый токен: ").strip().strip("'\"")
                        if new_tok and new_tok.lower() not in ["отмена", "cancel"]:
                            from yandex_music import Client
                            new_client = Client(new_tok).init()
                            save_env_var("YANDEX_MUSIC_TOKEN", new_tok)
                            service.token = new_tok
                            service.client = new_client
                            service.user = new_client.me.account
                            service.user_name = service.user.full_name or service.user.login or "Пользователь"
                            service.has_plus = new_client.me.plus.has_plus if hasattr(new_client.me, 'plus') else True
                            console.print(f"[bold green][OK] Токен Яндекс.Музыки обновлён! Пользователь: {service.user_name}[/bold green]")
                            tts.say("Токен Яндекс Музыки успешно обновлён.", pause_music=False)
                        else:
                            console.print("[yellow]Смена токена отменена.[/yellow]")
                            tts.say("Смена токена отменена.", pause_music=False)
                    except Exception as e:
                        console.print(f"[bold red]Ошибка применения токена:[/bold red] {e}")
                        tts.say("Не удалось применить токен, произошла ошибка.", pause_music=False)
                    finally:
                        voice_listener.set_force_command_mode(False)
                        if was_playing:
                            player.fade_in_resume(duration=0.4)

                # Смена API-ключа Google AI Studio
                elif action == "CHANGE_GEMINI_KEY":
                    was_playing = player.is_playing
                    if was_playing:
                        player.fade_out_pause(duration=0.4)
                    tts.say("Пожалуйста, введите новый API ключ Google в окне терминала.", pause_music=False)
                    console.print()
                    console.print(Panel(
                        f"[bold yellow]Смена API-ключа Google AI Studio[/bold yellow]\n\n"
                        f"1. Получите ключ по ссылке: [bold white]{GEMINI_KEY_URL}[/bold white]\n"
                        f"2. Вставьте его ниже (или напишите 'отмена'):",
                        title="[Google AI Studio]",
                        border_style="yellow"
                    ))
                    voice_listener.set_force_command_mode(True)
                    try:
                        new_key = input(" Новый API-ключ: ").strip().strip("'\"")
                        if new_key and new_key.lower() not in ["отмена", "cancel"]:
                            save_env_var("GEMINI_API_KEY", new_key)
                            nlu.api_key = new_key
                            console.print("[bold green][OK] API-ключ Google AI Studio успешно обновлён![/bold green]")
                            tts.say("Ключ Google успешно обновлён.", pause_music=False)
                        else:
                            console.print("[yellow]Смена ключа отменена.[/yellow]")
                            tts.say("Смена ключа отменена.", pause_music=False)
                    except Exception as e:
                        console.print(f"[bold red]Ошибка сохранения ключа:[/bold red] {e}")
                    finally:
                        voice_listener.set_force_command_mode(False)
                        if was_playing:
                            player.fade_in_resume(duration=0.4)

                # Справка
                elif action == "HELP":
                    print_help(config.trigger)
                    tts.say(
                        "Вы можете попросить включить трек, исполнителя или плейлист, запустить Мою волну, управлять громкостью, ставить лайки и менять голос ассистента.",
                        pause_music=True
                    )

                # Пауза / Стоп
                elif action == "PAUSE":
                    if player.pause():
                        console.print("[yellow]⏸ Пауза[/yellow]")
                        tts.say("Пауза", pause_music=False)
                    else:
                        console.print("[dim]Воспроизведение уже остановлено или на паузе[/dim]")

                # Возобновить
                elif action == "RESUME":
                    if player.resume():
                        console.print("[green]▶ Возобновлено[/green]")
                        tts.say("Продолжаю", pause_music=False)
                    else:
                        console.print("[dim]Нет трека для возобновления[/dim]")

                # Отмена действия / Возврат к предыдущему состоянию (Undo)
                elif action == "UNDO":
                    if player.can_undo():
                        console.print("[bold yellow]↩ Отмена команды! Возвращаемся к предыдущему состоянию...[/bold yellow]")
                        restored = player.undo_state()
                        if restored:
                            p_title = restored.get("playlist_title") or "предыдущую очередь"
                            artist = restored.get("artist_name") or ""
                            title = restored.get("track_title") or ""
                            t_desc = f"{artist} — {title}".strip(" —")
                            remaining = restored.get("history_remaining", 0)

                            console.print(Panel(
                                f"[bold green][OK] Восстановлено предыдущее состояние:[/bold green]\n"
                                f"Источник: [bold cyan]{p_title}[/bold cyan]\n"
                                f"Трек: [bold white]{t_desc}[/bold white] (№ {restored.get('index')} из {restored.get('total')})\n"
                                f"[dim]В истории сохранено ещё {remaining} состояний[/dim]",
                                title="[Отмена / Undo]",
                                border_style="yellow",
                                expand=False
                            ))
                            tts_msg = f"Отменяю. Возвращаю {p_title}: {t_desc}" if t_desc else f"Отменяю. Возвращаю {p_title}"
                            tts.say(tts_msg, pause_music=False)
                    else:
                        console.print("[yellow]История переключений пуста. Нечего отменять.[/yellow]")
                        tts.say("История переключений пуста, нечего отменять", pause_music=False)

                # Повтор отмененного действия (Redo)
                elif action == "REDO":
                    if player.can_redo():
                        console.print("[bold green][Redo] Повтор действия...[/bold green]")
                        restored = player.redo_state()
                        if restored:
                            p_title = restored.get("playlist_title") or "следующую очередь"
                            artist = restored.get("artist_name") or ""
                            title = restored.get("track_title") or ""
                            t_desc = f"{artist} — {title}".strip(" —")
                            console.print(f"[bold green][OK] Повторено:[/bold green] [bold white]{p_title}[/bold white] ({t_desc})")
                            tts.say(f"Повторяю {p_title}: {t_desc}", pause_music=False)
                    else:
                        console.print("[yellow]Нет отменённых действий для повтора.[/yellow]")
                        tts.say("Нет действий для повтора", pause_music=False)

                # История состояний
                elif action == "HISTORY":
                    hist = player.get_history_info()
                    if not hist:
                        console.print("[dim]История состояний пуста[/dim]")
                        tts.say("История состояний пуста", pause_music=False)
                    else:
                        h_table = Table(title="[История воспроизведения] (доступно для отмены)", border_style="yellow")
                        h_table.add_column("#", style="dim", width=4)
                        h_table.add_column("Источник / Плейлист", style="cyan")
                        h_table.add_column("Трек", style="bold white")
                        h_table.add_column("Позиция", style="green")
                        for idx, h in enumerate(hist, 1):
                            h_table.add_row(str(idx), h["playlist_title"], f"{h['artist_name']} — {h['track_title']}", f"№ {h['index']} из {h['total']}")
                        console.print(h_table)
                        tts.say(f"В истории сохранено {len(hist)} предыдущих состояний. Скажите отмена или назад для возврата.", pause_music=True)

                # Следующий / Предыдущий
                elif action == "NEXT":
                    console.print("[cyan]⏭ Следующий трек...[/cyan]")
                    if player.next_track():
                        cur = player.current_track
                        if cur:
                            info = service.format_track_info(cur)
                            tts.say(f"Следующий трек: {info['artists']} — {info['title']}", pause_music=False)
                    else:
                        console.print("[yellow]Очередь закончилась[/yellow]")
                        tts.say("Очередь воспроизведения закончилась", pause_music=False)

                elif action == "PREV":
                    # Если в текущей очереди есть предыдущий трек - переключаем
                    if player.can_prev_track_in_queue():
                        console.print("[cyan]⏮ Предыдущий трек...[/cyan]")
                        if player.prev_track():
                            cur = player.current_track
                            if cur:
                                info = service.format_track_info(cur)
                                tts.say(f"Предыдущий трек: {info['artists']} — {info['title']}", pause_music=False)
                    # Если это первый трек в очереди, но есть история предыдущих состояний (плейлистов)
                    elif player.can_undo():
                        console.print("[cyan]⏮ Первый трек в очереди. Возвращаемся к предыдущему плейлисту/состоянию...[/cyan]")
                        restored = player.undo_state()
                        if restored:
                            p_title = restored.get("playlist_title") or "предыдущую подборку"
                            artist = restored.get("artist_name") or ""
                            title = restored.get("track_title") or ""
                            t_desc = f"{artist} — {title}".strip(" —")
                            console.print(f"[bold green][OK] Возврат к предыдущему состоянию:[/bold green] [bold white]{p_title}[/bold white] ({t_desc})")
                            tts.say(f"Возвращаю {p_title}: {t_desc}", pause_music=False)
                    else:
                        console.print("[dim]Это первый трек в очереди, предыдущих состояний нет[/dim]")
                        tts.say("Это первый трек в очереди, предыдущих треков нет", pause_music=False)

                # Громкость
                elif action == "SET_VOLUME":
                    new_vol = player.set_volume(cmd["value"])
                    config.volume = new_vol
                    console.print(f"[yellow][Громкость]: {new_vol}%[/yellow]")
                    tts_text = cmd.get("tts_response") or f"Громкость {new_vol} процентов"
                    tts.say(tts_text, pause_music=False)

                elif action == "CHANGE_VOLUME":
                    new_vol = player.change_volume(cmd["delta"])
                    config.volume = new_vol
                    console.print(f"[yellow][Громкость]: {new_vol}%[/yellow]")
                    tts_text = cmd.get("tts_response") or f"Громкость {new_vol} процентов"
                    tts.say(tts_text, pause_music=False)

                # Инфо о текущем треке
                elif action == "INFO":
                    cur = player.current_track
                    if cur:
                        info = service.format_track_info(cur)
                        state = "На паузе" if player.is_paused else ("Играет" if player.is_playing else "Остановлен")
                        console.print(Panel(
                            f"[bold white]{info['artists']} — {info['title']}[/bold white]\n"
                            f"Альбом: {info['album']}\n"
                            f"Длительность: {info['duration']}\n"
                            f"Статус: [bold green]{state}[/bold green] | Громкость: [bold yellow]{player.volume}%[/bold yellow]",
                            title="[Информация о треке]",
                            border_style="blue",
                            expand=False
                        ))
                        tts.say(f"Сейчас играет: {info['artists']}, трек {info['title']}. Альбом: {info['album']}", pause_music=True)
                    else:
                        console.print("[dim]Сейчас ничего не играет[/dim]")
                        tts.say("Сейчас ничего не играет", pause_music=False)

                # Очередь
                elif action == "QUEUE":
                    q = player.get_queue(limit=10)
                    total_count = player.get_queue_count()
                    next_info = player.get_next_track_info()
                    if not q:
                        console.print("[dim]Очередь воспроизведения пуста[/dim]")
                        tts.say("Очередь воспроизведения пуста", pause_music=False)
                    else:
                        table = Table(title="[Очередь воспроизведения]", border_style="blue")
                        table.add_column("#", style="dim", width=4)
                        table.add_column("Трек", style="bold white")
                        table.add_column("Исполнитель", style="cyan")
                        table.add_column("Длит.", style="dim")
                        for idx, t in enumerate(q, 1):
                            prefix = "▶ " if t["is_current"] else "  "
                            title = f"{prefix}{t['title']}"
                            table.add_row(str(idx), title, t["artists"], t["duration"])
                        console.print(table)
                        # Озвучиваем только следующий трек с паузой
                        if next_info:
                            tts.say(f"В очереди {total_count} треков. Следующим играет: {next_info['artists']} — {next_info['title']}", pause_music=True)
                        else:
                            tts.say("В очереди остался только текущий трек.", pause_music=True)

                # Лайки
                elif action == "LIKE":
                    cur = player.current_track
                    if cur:
                        if service.like_track(cur.id):
                            console.print(f"[bold red][+][/bold red] Трек [bold]{cur.title}[/bold] добавлен в любимые!")
                            tts.say("Добавлено в любимые", pause_music=False)
                        else:
                            console.print("[red]Не удалось поставить лайк[/red]")
                    else:
                        console.print("[dim]Сейчас ничего не играет[/dim]")

                elif action == "DISLIKE":
                    cur = player.current_track
                    if cur:
                        if service.dislike_track(cur.id):
                            console.print(f"[-] Трек [bold]{cur.title}[/bold] удален из любимых.")
                            tts.say("Удалено из любимых, переключаю", pause_music=False)
                            player.next_track()
                        else:
                            console.print("[red]Не удалось удалить лайк[/red]")
                    else:
                        console.print("[dim]Сейчас ничего не играет[/dim]")

                # Моя волна
                elif action == "PLAY_MY_WAVE":
                    console.print("[bold cyan][Моя волна] Запускаем Мою волну...[/bold cyan]")
                    tts_text = cmd.get("tts_response") or "Запускаю Мою волну"
                    tts.say(tts_text, pause_music=False)
                    tracks = service.get_my_wave_batch()
                    if tracks:
                        player.play_tracks(tracks, is_my_wave=True, playlist_title="Моя волна")
                    else:
                        console.print("[red]Не удалось загрузить Мою волну[/red]")
                        tts.say("Не удалось загрузить Мою волну", pause_music=False)

                # Личные плейлисты пользователя
                elif action == "PLAY_MY_PLAYLIST":
                    q = cmd.get("query", "")
                    console.print("[bold green][Плейлист] Загружаем ваш плейлист...[/bold green]")
                    tracks, desc = service.search_user_playlist(q)
                    if tracks:
                        tts_text = cmd.get("tts_response") or f"Включаю ваш плейлист {desc}"
                        tts.say(tts_text, pause_music=False)
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say("Не удалось найти ваш плейлист", pause_music=False)

                # Любимые треки
                elif action == "PLAY_LIKED":
                    console.print("[bold red][Любимые] Загружаем любимые треки...[/bold red]")
                    tts_text = cmd.get("tts_response") or "Включаю любимые треки"
                    tts.say(tts_text, pause_music=False)
                    tracks, desc = service.get_liked_tracks()
                    if tracks:
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say("Любимые треки пусты", pause_music=False)

                # Персональные подборки Яндекса
                elif action == "PLAY_EDITORIAL":
                    title = cmd["title"]
                    console.print(f"[bold green][Подборка] Загружаем '{title}'...[/bold green]")
                    tts_text = cmd.get("tts_response") or f"Включаю {title}"
                    tts.say(tts_text, pause_music=False)
                    tracks, desc = service.get_generated_playlist(cmd["kind"])
                    if tracks:
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say(f"Не удалось загрузить подборку {title}", pause_music=False)

                # Исполнитель
                elif action == "PLAY_ARTIST":
                    q = cmd["query"]
                    console.print(f"[cyan]Поиск исполнителя: [bold]{q}[/bold]...[/cyan]")
                    tracks, desc = service.search_artist(q)
                    if tracks:
                        tts_text = cmd.get("tts_response") or f"Включаю исполнителя {q}"
                        tts.say(tts_text, pause_music=False)
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say(f"Исполнитель {q} не найден", pause_music=False)

                # Альбом
                elif action == "PLAY_ALBUM":
                    q = cmd["query"]
                    console.print(f"[cyan]Поиск альбома: [bold]{q}[/bold]...[/cyan]")
                    tracks, desc = service.search_album(q)
                    if tracks:
                        tts_text = cmd.get("tts_response") or f"Включаю альбом {q}"
                        tts.say(tts_text, pause_music=False)
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say(f"Альбом {q} не найден", pause_music=False)

                # Плейлист
                elif action == "PLAY_PLAYLIST":
                    q = cmd["query"]
                    console.print(f"[cyan]Поиск плейлиста: [bold]{q}[/bold]...[/cyan]")
                    tracks, desc = service.search_playlist(q)
                    if tracks:
                        tts_text = cmd.get("tts_response") or f"Включаю плейлист {q}"
                        tts.say(tts_text, pause_music=False)
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say(f"Плейлист {q} не найден", pause_music=False)

                # Трек
                elif action == "PLAY_TRACK":
                    q = cmd["query"]
                    console.print(f"[cyan]Поиск трека: [bold]{q}[/bold]...[/cyan]")
                    tracks, desc = service.search_track(q)
                    if tracks:
                        tts_text = cmd.get("tts_response") or f"Включаю трек {q}"
                        tts.say(tts_text, pause_music=False)
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say(f"Трек {q} не найден", pause_music=False)

                # Умный поиск
                elif action == "SMART_SEARCH":
                    q = cmd["query"]
                    console.print(f"[cyan]Ищем: [bold]{q}[/bold]...[/cyan]")
                    tracks, desc = service.smart_search(q)
                    if tracks:
                        tts_text = cmd.get("tts_response") or f"Включаю {q}"
                        tts.say(tts_text, pause_music=False)
                        player.play_tracks(tracks, is_my_wave=False, playlist_title=desc)
                    else:
                        console.print(f"[yellow]{desc}[/yellow]")
                        tts.say(f"По запросу {q} ничего не найдено", pause_music=False)

            except Exception as e:
                log_error(f"Исключение при выполнении команды '{cmd_text}': {e}")
                console.print(f"[bold red]Ошибка обработки команды:[/bold red] {e} (подробности в {LOG_FILE.name})")

    except KeyboardInterrupt:
        console.print("\n[yellow]Завершение работы (Ctrl+C)... До встречи![/yellow]")
        log_info("Завершение работы по KeyboardInterrupt (Ctrl+C)")

    finally:
        voice_listener.stop()
        tts.close()
        player.close()
        log_info("=== Завершение сессии ===")

if __name__ == "__main__":
    main()
