"""
Скрипт обучения и экспорта KWS моделей в ONNX.
Поддерживает первичное обучение на синтетическом датасете и дообучение на реальных сэмплах.
"""
import os
import sys
import time
import json
import pathlib
import datetime
import numpy as np
import torch

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from model_def import KWSNet, export_to_onnx, train_kws_head, ONNXWakeWordDetector
from audio_features import extract_mel_spectrogram

BASE_DIR = pathlib.Path(__file__).parent
DATASET_CACHE_PATH = BASE_DIR / "dataset_cache.npz"
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
TRAIN_HISTORY_FILE = BASE_DIR / "training_history.json"

MODELS_CONFIG = {
    "morsik": {"name": "Морсик", "file": "morsik.onnx"},
    "karlusha": {"name": "Карлуша", "file": "karlusha.onnx"},
    "hey_wisp": {"name": "Эй, Висп!", "file": "hey_wisp.onnx"}
}

def load_user_samples(samples_dir: pathlib.Path = SAMPLES_DIR):
    """Загружает реальные пользовательские сэмплы из папок samples/<label>/"""
    user_samples = {k: [] for k in MODELS_CONFIG}
    user_samples["negative"] = []

    if not samples_dir.exists():
        return user_samples

    for cat_dir in samples_dir.iterdir():
        if not cat_dir.is_dir():
            continue
        cat_name = cat_dir.name
        for wav_path in cat_dir.glob("*.npy"):
            try:
                # Если сохранен готовый mel
                mel = np.load(str(wav_path))
                if cat_name in user_samples:
                    user_samples[cat_name].append(mel)
                else:
                    user_samples["negative"].append(mel)
            except Exception:
                pass
    return user_samples

def train_all_models(epochs: int = 10, lr: float = 0.003, is_fine_tune: bool = False):
    """
    Обучает 3 независимые модели на синтетике + реальных пользовательских данных.
    Возвращает словарь с результатами обучения для журнала.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_CACHE_PATH.exists():
        print("[!] Кэш датасета не найден, генерация...")
        from dataset_generator import build_dataset_cache
        build_dataset_cache()

    data = np.load(str(DATASET_CACHE_PATH))
    user_samples = load_user_samples()
    
    total_user_samples = sum(len(v) for v in user_samples.values())
    results_summary = {}

    print(f"\n=======================================================", flush=True)
    print(f"  {'Дообучение' if is_fine_tune else 'Обучение'} KWS моделей (Пользовательских сэмплов: {total_user_samples})", flush=True)
    print(f"=======================================================", flush=True)

    for target_key in MODELS_CONFIG:
        print(f"\n[*] Обучение модели: «{MODELS_CONFIG[target_key]['name']}» ({target_key})...", flush=True)
        
        # 1. Положительные примеры
        pos_list = [data[f"pos_{target_key}"]]
        if user_samples.get(target_key):
            # Дублируем реальные сэмплы пользователя с высоким весом (х8..х10),
            # чтобы голос реального человека доминировал над синтетикой Silero
            user_pos = np.array(user_samples[target_key], dtype=np.float32)
            repeat_pos = 10 if is_fine_tune else 6
            for _ in range(repeat_pos):
                pos_list.append(user_pos)
            print(f"  -> Добавлено {len(user_samples[target_key])} реальных сэмплов (вес x{repeat_pos})", flush=True)

        pos_all = np.concatenate(pos_list, axis=0)

        # 2. Отрицательные примеры (общие негативы + кросс-триггеры других моделей)
        neg_list = [data["negatives"]]
        for other_key in MODELS_CONFIG:
            if other_key != target_key:
                neg_list.append(data[f"pos_{other_key}"])
                if user_samples.get(other_key):
                    user_other = np.array(user_samples[other_key], dtype=np.float32)
                    for _ in range(6):
                        neg_list.append(user_other)

        if user_samples.get("negative"):
            user_neg = np.array(user_samples["negative"], dtype=np.float32)
            for _ in range(6):
                neg_list.append(user_neg)

        neg_all = np.concatenate(neg_list, axis=0)

        print(f"  -> Итого обучающая выборка: {len(pos_all)} POS / {len(neg_all)} NEG", flush=True)

        # 3. Обучение PyTorch модели
        model = KWSNet()
        actual_epochs = 12 if is_fine_tune and total_user_samples > 0 else epochs
        actual_lr = lr

        t0 = time.time()
        losses = train_kws_head(model, pos_all, neg_all, epochs=actual_epochs, lr=actual_lr, batch_size=64)
        train_time = time.time() - t0

        # 4. Экспорт в ONNX
        onnx_file = MODELS_DIR / MODELS_CONFIG[target_key]["file"]
        export_to_onnx(model, str(onnx_file))
        file_size_kb = onnx_file.stat().st_size / 1024

        # 5. Быстрая валидация
        det = ONNXWakeWordDetector(str(onnx_file))
        sample_pos = [det.predict(p) for p in pos_all[:50]]
        sample_neg = [det.predict(n) for n in neg_all[:50]]
        
        avg_pos_prob = float(np.mean(sample_pos))
        avg_neg_prob = float(np.mean(sample_neg))
        acc = float(np.mean([p >= 0.5 for p in sample_pos] + [n < 0.5 for n in sample_neg])) * 100.0

        print(f"  [OK] Экспортирован: {onnx_file.name} ({file_size_kb:.1f} КБ)", flush=True)
        print(f"  [OK] Время обучения: {train_time:.1f}с | Final Loss: {losses[-1]:.4f} | Точность теста: {acc:.1f}%", flush=True)
        print(f"  [OK] Средняя вероятность (POS): {avg_pos_prob:.3f} | Средняя вероятность (NEG): {avg_neg_prob:.3f}", flush=True)

        results_summary[target_key] = {
            "name": MODELS_CONFIG[target_key]["name"],
            "accuracy": round(acc, 1),
            "final_loss": round(losses[-1], 4),
            "avg_pos": round(avg_pos_prob, 3),
            "avg_neg": round(avg_neg_prob, 3),
            "file": onnx_file.name
        }

    # Логирование в журнал обучений
    log_entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "Дообучение (Fine-tune)" if is_fine_tune else "Первичное обучение (Cold Train)",
        "user_samples_count": total_user_samples,
        "results": results_summary
    }

    history = []
    if TRAIN_HISTORY_FILE.exists():
        try:
            with open(TRAIN_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.insert(0, log_entry)
    with open(TRAIN_HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Запись сохранена в журнал обучений: {TRAIN_HISTORY_FILE}")
    return log_entry

if __name__ == "__main__":
    train_all_models()
