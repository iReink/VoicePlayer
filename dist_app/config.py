"""
Управление настройками ассистента (триггер-слово, TTS, громкость)
"""
import os
import json
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.json"

DEFAULT_SETTINGS = {
    "trigger": "Морсик",
    "tts_enabled": True,
    "volume": 70,
    "voice": "xenia",
    "live_model": "models/gemini-3.8-live",
    "live_voice": "auto"
}

class ConfigManager:
    def __init__(self):
        self.settings = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        if SETTINGS_FILE.exists():
            try:
                with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.settings.update(data)
            except Exception:
                pass
        else:
            self.save()

    def save(self):
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    @property
    def trigger(self) -> str:
        return self.settings.get("trigger", "Морсик")

    @trigger.setter
    def trigger(self, value: str):
        self.settings["trigger"] = value.strip().capitalize()
        self.save()

    @property
    def tts_enabled(self) -> bool:
        return self.settings.get("tts_enabled", True)

    @tts_enabled.setter
    def tts_enabled(self, value: bool):
        self.settings["tts_enabled"] = bool(value)
        self.save()

    @property
    def voice(self) -> str:
        return self.settings.get("voice", "xenia")

    @voice.setter
    def voice(self, value: str):
        self.settings["voice"] = value.strip().lower()
        self.save()

    @property
    def volume(self) -> int:
        return self.settings.get("volume", 70)

    @volume.setter
    def volume(self, value: int):
        self.settings["volume"] = max(0, min(100, int(value)))
        self.save()

    @property
    def live_model(self) -> str:
        return self.settings.get("live_model", "models/gemini-3.8-live")

    @live_model.setter
    def live_model(self, value: str):
        self.settings["live_model"] = value.strip()
        self.save()

    @property
    def live_voice(self) -> str:
        return self.settings.get("live_voice", "auto")

    @live_voice.setter
    def live_voice(self, value: str):
        self.settings["live_voice"] = value.strip()
        self.save()
