"""
Модуль безопасного управления персональными токенами и API-ключами (.env)
"""
import os
import re
from pathlib import Path
from typing import Optional, Tuple
from rich.console import Console
from rich.panel import Panel

ENV_PATH = Path(__file__).parent / ".env"

YANDEX_TOKEN_URL = "https://ym.marshal.dev/token/"
GEMINI_KEY_URL = "https://aistudio.google.com/api-keys"

def load_env_file(path: Optional[Path] = None) -> dict:
    """Загрузить переменные из .env файла"""
    p = path or ENV_PATH
    data = {}
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    data[k.strip()] = v.strip().strip("'\"")
    return data

def save_env_var(key: str, value: str, path: Optional[Path] = None):
    """Безопасно записать или обновить переменную в .env файле"""
    p = path or ENV_PATH
    lines = []
    found = False

    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()

    new_lines = []
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for line in lines:
        if pattern.match(line):
            new_lines.append(f"{key}={value}\n")
            found = True
        else:
            new_lines.append(line)

    if not found:
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines.append("\n")
        new_lines.append(f"{key}={value}\n")

    with open(p, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Обновляем в текущем окружении процесса
    os.environ[key] = value

def get_yandex_token() -> Optional[str]:
    return os.getenv("YANDEX_MUSIC_TOKEN") or load_env_file().get("YANDEX_MUSIC_TOKEN")

def get_gemini_key() -> Optional[str]:
    return os.getenv("GEMINI_API_KEY") or load_env_file().get("GEMINI_API_KEY")

def ensure_credentials_interactive(console: Console, tts=None) -> Tuple[str, str]:
    """
    Проверка наличия токена Яндекс.Музыки и ключа Gemini при старте.
    Если отсутствуют — интерактивный запрос в терминале с подсказками и ссылками.
    """
    ym_token = get_yandex_token()
    gemini_key = get_gemini_key()

    # 1. Проверка токена Яндекс.Музыки
    while not ym_token:
        console.print()
        console.print(Panel(
            f"[bold red]Токен Яндекс.Музыки не найден![/bold red]\n\n"
            f"Для воспроизведения музыки необходим персональный токен аккаунта Яндекс.\n"
            f"1. Откройте в браузере: [bold cyan]{YANDEX_TOKEN_URL}[/bold cyan]\n"
            f"2. Авторизуйтесь и скопируйте полученный токен\n"
            f"3. Вставьте его ниже:",
            title="🔑 Авторизация Яндекс.Музыки",
            border_style="red"
        ))
        if tts:
            tts.say("Для работы ассистента нужен токен Яндекс Музыки. Пожалуйста, перейдите по ссылке в окне терминала и введите токен.")
        
        try:
            val = input(" Введите токен Яндекс.Музыки: ").strip().strip("'\"")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]Прервано пользователем.[/yellow]")
            break

        if val:
            # Проверяем валидность токена через yandex_music Client
            try:
                from yandex_music import Client
                test_client = Client(val).init()
                u_name = test_client.me.account.full_name or test_client.me.account.login or "Пользователь"
                save_env_var("YANDEX_MUSIC_TOKEN", val)
                ym_token = val
                console.print(f"[bold green]✓ Токен подтверждён! Пользователь: {u_name}[/bold green]\n")
                if tts:
                    tts.say("Токен Яндекс Музыки успешно сохранён.")
                break
            except Exception as e:
                console.print(f"[bold red]Ошибка проверки токена:[/bold red] {e}. Попробуйте ещё раз.")
        else:
            console.print("[yellow]Токен не может быть пустым.[/yellow]")

    # 2. Проверка API-ключа Gemini
    while not gemini_key:
        console.print()
        console.print(Panel(
            f"[bold yellow]API-ключ Google AI Studio не найден![/bold yellow]\n\n"
            f"Для естественного распознавания команд и умного Live-диалога нужен бесплатный ключ.\n"
            f"1. Откройте в браузере: [bold cyan]{GEMINI_KEY_URL}[/bold cyan]\n"
            f"2. Нажмите [bold white]'Create API Key'[/bold white] и скопируйте ключ\n"
            f"3. Вставьте его ниже (или нажмите Enter для работы только с оффлайн-парсером):",
            title="🔑 Google AI Studio API Key",
            border_style="yellow"
        ))
        if tts:
            tts.say("Для умного понимания команд нужен ключ Google AI Studio. Пожалуйста, введите его в терминале.")

        try:
            val = input(" Введите ключ Google AI Studio (или Enter для пропуска): ").strip().strip("'\"")
        except (KeyboardInterrupt, EOFError):
            break

        if val:
            save_env_var("GEMINI_API_KEY", val)
            gemini_key = val
            console.print("[bold green]✓ Ключ Google AI Studio успешно сохранён![/bold green]\n")
            if tts:
                tts.say("Ключ Google успешно сохранён.")
            break
        else:
            console.print("[dim]Пропущено. Будет использоваться только локальный оффлайн-парсер команд.[/dim]\n")
            break

    return ym_token or "", gemini_key or ""
