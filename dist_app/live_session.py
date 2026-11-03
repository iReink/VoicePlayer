"""
Модуль полнодуплексного диалогового режима Gemini Multimodal Live API (WebSockets)
с нативным потоковым звуком, Function Calling и текстовой транскрибацией в консоли.
"""
import asyncio
import os
import sys
import json
import base64
import time
import math
import struct
import threading
import queue
from typing import Optional, Dict, Any, List, Callable

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except Exception:
        pass

import sounddevice as sd
import websockets
from rich.console import Console
from rich.panel import Panel

from logger import log_info, log_warning, log_error
from search_service import SearchService

# Доступные проверенные голоса в Gemini Live API
LIVE_VOICES = {
    "aoede": "Aoede",      # Тёплый выразительный женский голос
    "kore": "Kore",        # Мягкий женский голос
    "fenrir": "Fenrir",    # Уверенный мужской баритон
    "puck": "Puck",        # Энергичный молодой мужской голос
    "charon": "Charon"     # Глубокий спокойный мужской бас
}

class LiveSessionManager:
    # Сохранённая история для возобновления при «Морсик, подожди!»
    _last_history: List[Dict[str, Any]] = []
    _last_session_time: float = 0.0
    _grace_period_active: bool = False

    def __init__(self, config_manager, player, music_service, console: Optional[Console] = None):
        self.config = config_manager
        self.player = player
        self.service = music_service
        self.console = console or Console()
        self.search_service = SearchService()

        self.api_key = os.getenv("GEMINI_API_KEY")
        self.model_name = getattr(self.config, "live_model", "models/gemini-3.8-live")
        self.selected_voice = self._resolve_voice()

        self.is_running = False
        self.should_exit = False
        self.inactivity_timeout = 25.0
        self.last_speech_time = time.time()
        
        self.tokens_used = 0
        self.max_tokens = 20000

        # Аудио-буферы и очереди
        self.audio_out_queue = queue.Queue()
        self.is_assistant_speaking = False
        self.output_stream = None

    def _resolve_voice(self) -> str:
        """Автоматический выбор голоса Live API под пол голоса Silero"""
        cfg_voice = getattr(self.config, "live_voice", "auto")
        if cfg_voice and cfg_voice.lower() in LIVE_VOICES:
            return LIVE_VOICES[cfg_voice.lower()]

        # Синхронизация с текущим голосом Silero
        silero_v = self.config.voice.lower()
        if silero_v in ["xenia", "baya"]:
            return "Aoede"   # Женский
        else:
            return "Fenrir"  # Мужской

    @classmethod
    def can_resume(cls) -> bool:
        """Проверка, активен ли льготный период (15 сек) для фразы 'Морсик, подожди!'"""
        if not cls._last_history:
            return False
        return (time.time() - cls._last_session_time) <= 15.0

    def run_sync(self, resume_context: bool = False):
        """Синхронная точка входа для запуска асинхронного цикла Live-сессии"""
        try:
            asyncio.run(self.run_async(resume_context=resume_context))
        except Exception as e:
            log_error(f"Ошибка в Live-сессии: {e}")

    async def run_async(self, resume_context: bool = False):
        if not self.api_key:
            self.console.print("[bold red]Для запуска Live-диалога требуется GEMINI_API_KEY![/bold red]")
            return

        self.is_running = True
        self.should_exit = False
        self.last_speech_time = time.time()

        # Приглушаем музыку до 10%
        was_playing = self.player.is_playing
        if was_playing:
            self.player.duck(True)

        voice_display = f"{self.selected_voice} (авто-пол под {self.config.voice})"
        banner = (
            f"[bold cyan]🎙️ Live-диалог с Gemini 3.8 активирован[/bold cyan]\n"
            f"Модель: [bold yellow]{self.model_name}[/bold yellow] | Голос: [bold magenta]{voice_display}[/bold magenta]\n"
            f"[dim]Говорите свободно без триггера. Для выхода скажите «положи трубку» или нажмите Esc[/dim]"
        )
        self.console.print()
        self.console.print(Panel(banner, border_style="cyan"))

        uri = f"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={self.api_key}"

        try:
            async with websockets.connect(uri, max_size=10*1024*1024) as ws:
                # 1. Отправка Setup сообщения
                setup_msg = self._build_setup_message(resume_context=resume_context)
                await ws.send(json.dumps(setup_msg))

                resp = await asyncio.wait_for(ws.recv(), timeout=5.0)
                setup_resp = json.loads(resp)
                if "setupComplete" not in setup_resp:
                    log_warning(f"Неожиданный ответ setup: {resp}")

                log_info("Gemini Live API WebSocket соединение успешно установлено")

                # 2. Инициализация аудиовывода (24kHz Mono 16-bit PCM)
                self.output_stream = sd.RawOutputStream(
                    samplerate=24000,
                    channels=1,
                    dtype='int16',
                    blocksize=1024
                )
                self.output_stream.start()

                # Фоновый поток вывода звука
                out_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
                out_thread.start()

                # 3. Запуск одновременных задач чтения и записи сокета
                send_task = asyncio.create_task(self._audio_sender_loop(ws))
                recv_task = asyncio.create_task(self._ws_receiver_loop(ws))

                # Ожидание завершения любой из задач
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()

        except Exception as e:
            log_error(f"Исключение WebSocket Live API: {e}")
            self.console.print(f"[yellow]Связь с Live API завершена: {e}[/yellow]")

        finally:
            self.is_running = False
            if self.output_stream:
                try:
                    self.output_stream.stop()
                    self.output_stream.close()
                except Exception:
                    pass

            # Сохраняем время окончания для периода «Морсик, подожди!»
            LiveSessionManager._last_session_time = time.time()
            log_info("Live-сессия завершена")

            # Плавно восстанавливаем музыку
            if was_playing:
                self.player.duck(False)
            
            self.console.print("[dim]Выход из Live-режима. Ассистент снова слушает обычные команды.[/dim]\n")

    def _audio_playback_worker(self):
        """Фоновый воркер воспроизведения входящих аудио-пакетов из Gemini"""
        while self.is_running:
            try:
                chunk = self.audio_out_queue.get(timeout=0.1)
                if chunk and self.output_stream and not self.output_stream.closed:
                    self.is_assistant_speaking = True
                    self.output_stream.write(chunk)
                else:
                    self.is_assistant_speaking = False
            except queue.Empty:
                self.is_assistant_speaking = False
            except Exception:
                self.is_assistant_speaking = False

    async def _audio_sender_loop(self, ws):
        """Чтение микрофона (16kHz PCM), локальный VAD и отправка в сокет"""
        loop = asyncio.get_running_loop()
        input_queue = asyncio.Queue()

        def mic_callback(indata, frames, time_info, status):
            if self.is_running:
                # Копируем байты для передачи в async цикл
                loop.call_soon_threadsafe(input_queue.put_nowait, bytes(indata))

        # Открываем микрофон 16kHz
        with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', blocksize=1024, callback=mic_callback):
            while self.is_running and not self.should_exit:
                # Проверка таймаута неактивности
                if time.time() - self.last_speech_time > self.inactivity_timeout:
                    self.console.print("\n[yellow]⏳ Таймаут молчания 25 секунд. Завершаем диалог...[/yellow]")
                    self.should_exit = True
                    break

                try:
                    pcm_data = await asyncio.wait_for(input_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

                # Вычисляем RMS для проверки энергии голоса
                count = len(pcm_data) // 2
                if count == 0:
                    continue
                shorts = struct.unpack(f"<{count}h", pcm_data)
                sum_sq = sum(s * s for s in shorts)
                rms = math.sqrt(sum_sq / count) / 32768.0

                # Адаптивный порог: если говорит ассистент, порог выше (для надежного Barge-in)
                threshold = 0.035 if self.is_assistant_speaking else 0.012

                if rms > threshold:
                    self.last_speech_time = time.time()

                    # Перебивание (Barge-in): если пользователь заговорил во время речи ассистента
                    if self.is_assistant_speaking and rms > 0.035:
                        # Очищаем очередь воспроизведения
                        while not self.audio_out_queue.empty():
                            try:
                                self.audio_out_queue.get_nowait()
                            except Exception:
                                break
                        self.is_assistant_speaking = False

                    # Отправляем аудио-чанк в Gemini
                    b64_audio = base64.b64encode(pcm_data).decode("utf-8")
                    msg = {
                        "realtime_input": {
                            "media_chunks": [
                                {
                                    "mime_type": "audio/pcm;rate=16000",
                                    "data": b64_audio
                                }
                            ]
                        }
                    }
                    await ws.send(json.dumps(msg))

    async def _ws_receiver_loop(self, ws):
        """Приём ответов от Gemini: текст, аудио, вызовы инструментов и токены"""
        role_label_printed = False
        current_ai_text = []

        while self.is_running and not self.should_exit:
            try:
                msg_raw = await ws.recv()
                data = json.loads(msg_raw)
            except Exception:
                break

            # 1. Отслеживание расхода токенов (скользящее окно 20 000)
            meta = data.get("usageMetadata", {})
            if meta:
                total = meta.get("totalTokenCount", 0)
                if total:
                    self.tokens_used = total
                    if total >= self.max_tokens:
                        log_warning(f"Достигнут предел скользящего окна токенов: {total}/{self.max_tokens}")

            # 2. Обработка контента от сервера
            server_content = data.get("serverContent", {})
            if server_content:
                model_turn = server_content.get("modelTurn", {})
                parts = model_turn.get("parts", [])
                
                for part in parts:
                    # Входящий звук
                    if "inlineData" in part:
                        b64_bytes = part["inlineData"].get("data", "")
                        if b64_bytes:
                            raw_audio = base64.b64decode(b64_bytes)
                            self.audio_out_queue.put(raw_audio)

                    # Входящий текст транскрибации ответа
                    if "text" in part:
                        t = part["text"]
                        if not role_label_printed:
                            token_str = f"[dim cyan]({self.tokens_used}/{self.max_tokens} токенов)[/dim cyan]"
                            self.console.print(f"\n[bold cyan]🤖 [Морсик]:[/bold cyan] {token_str} ", end="")
                            role_label_printed = True
                        self.console.print(f"[white]{t}[/white]", end="")
                        current_ai_text.append(t)

                if server_content.get("turnComplete"):
                    if role_label_printed:
                        self.console.print()
                        role_label_printed = False
                        current_ai_text.clear()

            # 3. Обработка Function Calling (вызовы команд плеера и поиска)
            tool_call = data.get("toolCall", {})
            if tool_call:
                calls = tool_call.get("functionCalls", [])
                responses = []

                for fc in calls:
                    call_id = fc.get("id")
                    name = fc.get("name")
                    args = fc.get("args", {})
                    log_info(f"Live ToolCall: {name}({args})")

                    res_output = await self._execute_tool(name, args)
                    responses.append({
                        "response": {"output": res_output},
                        "id": call_id
                    })

                tool_resp = {
                    "tool_response": {
                        "function_responses": responses
                    }
                }
                await ws.send(json.dumps(tool_resp))

    async def _execute_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Исполнение вызова инструмента на локальном плеере или сервисе поиска"""
        if name == "set_volume":
            lvl = int(args.get("level", 50))
            new_v = self.player.set_volume(lvl)
            self.console.print(f"\n[yellow]🔊 [Команда]: Громкость {new_v}%[/yellow]")
            return {"status": "success", "new_volume": new_v}

        elif name == "change_volume":
            delta = int(args.get("delta", 10))
            new_v = self.player.change_volume(delta)
            self.console.print(f"\n[yellow]🔊 [Команда]: Громкость {new_v}%[/yellow]")
            return {"status": "success", "new_volume": new_v}

        elif name == "pause_music":
            self.player.pause()
            self.console.print("\n[yellow]⏸ [Команда]: Музыка на паузе[/yellow]")
            return {"status": "paused"}

        elif name == "resume_music":
            self.player.resume()
            self.console.print("\n[green]▶ [Команда]: Музыка возобновлена[/green]")
            return {"status": "playing"}

        elif name == "next_track":
            ok = self.player.next_track()
            cur = self.player.current_track
            info = self.service.format_track_info(cur) if cur else {}
            self.console.print(f"\n[cyan]⏭ [Команда]: Следующий трек: {info.get('artists')} — {info.get('title')}[/cyan]")
            return {"status": "success", "now_playing": f"{info.get('artists')} — {info.get('title')}"}

        elif name == "prev_track":
            self.player.prev_track()
            cur = self.player.current_track
            info = self.service.format_track_info(cur) if cur else {}
            self.console.print(f"\n[cyan]⏮ [Команда]: Предыдущий трек: {info.get('artists')} — {info.get('title')}[/cyan]")
            return {"status": "success", "now_playing": f"{info.get('artists')} — {info.get('title')}"}

        elif name == "like_current_track":
            cur = self.player.current_track
            if cur:
                self.service.like_track(cur.id)
                self.console.print(f"\n[red]❤️ [Команда]: Трек {cur.title} добавлен в любимые[/red]")
                return {"status": "liked", "track": cur.title}
            return {"status": "error", "message": "Nothing playing"}

        elif name == "play_music":
            q = args.get("query", "")
            tracks, desc = self.service.smart_search(q)
            if tracks:
                self.player.play_tracks(tracks, playlist_title=desc)
                self.console.print(f"\n[green]🎵 [Команда]: Включаем {desc}[/green]")
                return {"status": "success", "description": desc}
            return {"status": "not_found", "message": f"Не найдено по запросу {q}"}

        elif name == "get_player_state":
            cur = self.player.current_track
            info = self.service.format_track_info(cur) if cur else {}
            next_t = self.player.get_next_track_info()
            return {
                "is_playing": self.player.is_playing,
                "volume": self.player.volume,
                "current_track": f"{info.get('artists')} — {info.get('title')}" if cur else "Ничего не играет",
                "next_track": f"{next_t.get('artists')} — {next_t.get('title')}" if next_t else "Конец очереди"
            }

        elif name == "web_search":
            q = args.get("query", "")
            self.console.print(f"\n[bold yellow]🔍 [Поиск в интернете]: {q}...[/bold yellow]")
            search_res = self.search_service.search(q)
            return {"search_results": search_res}

        elif name == "end_conversation":
            self.console.print("\n[yellow]Завершение диалога по просьбе пользователя...[/yellow]")
            self.should_exit = True
            return {"status": "conversation_ended"}

        return {"status": "unknown_function"}

    def _build_setup_message(self, resume_context: bool = False) -> Dict[str, Any]:
        """Формирование стартового пакета Setup: голос, системный промпт с контекстом и инструменты"""
        cur = self.player.current_track
        info = self.service.format_track_info(cur) if cur else None
        next_t = self.player.get_next_track_info()

        player_context = (
            f"- Состояние плеера: {'Играет' if self.player.is_playing else 'Пауза/Остановлен'}\n"
            f"- Громкость: {self.player.volume}%\n"
            f"- Сейчас играет: {info['artists']} — {info['title']} (Альбом: {info['album']})\n" if info else "- Музыка не играет\n"
        )
        if next_t:
            player_context += f"- Следующий трек в очереди: {next_t['artists']} — {next_t['title']}\n"

        system_instruction = (
            f"Ты — Морсик, живой, остроумный и дружелюбный голосовой ассистент Яндекс.Музыки.\n"
            f"Ты разговариваешь с пользователем напрямую голосом на естественном русском языке.\n"
            f"Отвечай кратко, живо, без длинных лекций и книжных штампов (1-3 ёмких предложения на реплику).\n\n"
            f"ТЕКУЩИЙ КОНТЕКСТ МУЗЫКАЛЬНОГО ПЛЕЕРА:\n{player_context}\n"
            f"ИНСТРУКЦИИ:\n"
            f"1. Если пользователь просит управлять музыкой (громче, тише, пауза, следующий трек, включи песню) — обязательно вызывай соответствующий инструмент (toolCall).\n"
            f"2. Если пользователь спрашивает свежие факты, погоду, новости, дискографию — вызывай инструмент `web_search`.\n"
            f"3. Если пользователь прощается или просит положить трубку ('пока', 'хватит', 'отбой', 'положи трубку') — вызывай `end_conversation`.\n"
        )

        tools_declaration = [
            {
                "name": "set_volume",
                "description": "Установить точный уровень громкости (0-100)",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"level": {"type": "INTEGER", "description": "0-100"}},
                    "required": ["level"]
                }
            },
            {
                "name": "change_volume",
                "description": "Изменить громкость относительно текущей на +/- процентов",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"delta": {"type": "INTEGER", "description": "Например 10 или -15"}},
                    "required": ["delta"]
                }
            },
            {
                "name": "pause_music",
                "description": "Поставить музыку на паузу",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "resume_music",
                "description": "Снять музыку с паузы и продолжить воспроизведение",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "next_track",
                "description": "Переключить на следующий трек в очереди",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "prev_track",
                "description": "Вернуться к предыдущему треку",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "like_current_track",
                "description": "Поставить лайк текущему треку",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "play_music",
                "description": "Включить трек, артиста, альбом или плейлист по названию",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING", "description": "Поисковый запрос трека или исполнителя"}},
                    "required": ["query"]
                }
            },
            {
                "name": "get_player_state",
                "description": "Получить актуальный статус плеера, текущий и следующий треки",
                "parameters": {"type": "OBJECT", "properties": {}}
            },
            {
                "name": "web_search",
                "description": "Поиск в интернете свежих новостей, фактов, погоды и информации",
                "parameters": {
                    "type": "OBJECT",
                    "properties": {"query": {"type": "STRING", "description": "Поисковый запрос"}},
                    "required": ["query"]
                }
            },
            {
                "name": "end_conversation",
                "description": "Завершить разговор и закрыть сессию диалога",
                "parameters": {"type": "OBJECT", "properties": {}}
            }
        ]

        return {
            "setup": {
                "model": self.model_name,
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {
                                "voiceName": self.selected_voice
                            }
                        }
                    }
                },
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
                "tools": [
                    {"functionDeclarations": tools_declaration}
                ]
            }
        }
