"""
Веб-сервер испытательного полигона KWS (KWS Polygon) на базе aiohttp.
Поддерживает HTTPS (для доступа к микрофону по локальной сети с других ПК),
инференс ONNX моделей, сохранение сэмплов разметки, расчёт метрик и дообучение на лету.
"""
import os
import sys
import json
import ssl
import time
import socket
import pathlib
import datetime
import io
import wave
import traceback
import numpy as np
import scipy.io.wavfile
import scipy.signal
from aiohttp import web

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Добавляем текущую директорию в sys.path
BASE_DIR = pathlib.Path(__file__).parent.resolve()
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from model_def import ONNXWakeWordDetector
from audio_features import extract_mel_spectrogram
from train import train_all_models, MODELS_CONFIG, TRAIN_HISTORY_FILE

MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
TEMP_DIR = BASE_DIR / "temp_audio"
SAMPLES_DB_FILE = BASE_DIR / "samples_db.json"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
for k in list(MODELS_CONFIG.keys()) + ["negative"]:
    (SAMPLES_DIR / k).mkdir(parents=True, exist_ok=True)

# Инициализация детектора и моделей
detectors = {}
temp_cache = {}  # audio_id -> { "mel": np.ndarray, "wav_bytes": bytes, "results": dict }

def reload_detectors():
    global detectors
    detectors = {}
    for key, cfg in MODELS_CONFIG.items():
        model_path = MODELS_DIR / cfg["file"]
        if model_path.exists():
            detectors[key] = ONNXWakeWordDetector(str(model_path))
            print(f"[OK] Загружена модель: {cfg['name']} -> {model_path.name}")
        else:
            print(f"[!] Файл модели не найден: {model_path}")

reload_detectors()

def get_local_lan_ip() -> str:
    """Определяет IP адрес текущего ПК в домашней локальной сети (предпочитая 192.168.*)"""
    candidate = '127.0.0.1'
    try:
        # Проверяем все доступные адреса сокета
        hostname = socket.gethostname()
        for ip in socket.gethostbyname_ex(hostname)[2]:
            if ip.startswith('192.168.'):
                return ip
            elif ip.startswith('10.') and not candidate.startswith('192.168.'):
                candidate = ip
            elif ip != '127.0.0.1' and candidate == '127.0.0.1':
                candidate = ip
    except Exception:
        pass

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        if candidate == '127.0.0.1' or candidate.startswith('172.'):
            return ip
    except Exception:
        pass
        
    return candidate

LOCAL_IP = get_local_lan_ip()
PORT = 8443

def ensure_ssl_certs():
    """
    Генерирует самоподписанный SSL-сертификат для HTTPS,
    чтобы браузеры на других ПК в локальной сети разрешали доступ к микрофону
    (getUserMedia требует HTTPS при доступе не через localhost).
    """
    cert_path = BASE_DIR / "cert.pem"
    key_path = BASE_DIR / "key.pem"

    if not cert_path.exists() or not key_path.exists():
        print("[*] Генерация локального SSL-сертификата для HTTPS (доступа к микрофону по LAN)...")
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        import ipaddress

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, u"KWS-Polygon"),
        ])
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1))
            .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=3650))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName(u"localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                    x509.IPAddress(ipaddress.IPv4Address(LOCAL_IP)),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256())
        )

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        print(f"[OK] SSL-сертификат создан: {cert_path}")

    return str(cert_path), str(key_path)

# --- МАРШРУТЫ API ---

async def handle_index(request):
    """Отдаёт главную страницу"""
    index_file = BASE_DIR / "static" / "index.html"
    return web.FileResponse(index_file)

async def handle_info(request):
    """Информация о сервере и адресах подключения"""
    return web.json_response({
        "status": "ok",
        "local_ip": LOCAL_IP,
        "port": PORT,
        "lan_url": f"https://{LOCAL_IP}:{PORT}",
        "local_url": f"https://localhost:{PORT}"
    })

async def handle_predict(request):
    """
    Принимает WAV аудиозапись из браузера,
    извлекает Mel-спектрограмму и возвращает вероятности всех трёх моделей.
    """
    reader = await request.multipart()
    field = await reader.next()
    if field.name != 'audio':
        return web.json_response({"status": "error", "error": "Поле audio не найдено"}, status=400)

    audio_bytes = await field.read()
    if len(audio_bytes) < 100:
        return web.json_response({"status": "error", "error": "Слишком короткое аудио"}, status=400)

    try:
        # Читаем WAV файл из памяти
        sr, audio_int16 = scipy.io.wavfile.read(io.BytesIO(audio_bytes))
        audio_float = audio_int16.astype(np.float32) / 32768.0

        if len(audio_float.shape) > 1:
            audio_float = np.mean(audio_float, axis=1)

        # Если частота не 16кГц, ресемплим
        if sr != 16000:
            num_16k = int(len(audio_float) * 16000 / sr)
            audio_float = scipy.signal.resample(audio_float, num_16k).astype(np.float32)

        mel = extract_mel_spectrogram(audio_float)
    except Exception as e:
        return web.json_response({"status": "error", "error": f"Ошибка декодирования аудио: {e}"}, status=400)

    # Запускаем все 3 модели
    results = {}
    try:
        threshold = float(request.query.get("threshold", 0.80))
    except Exception:
        threshold = 0.80

    for key, det in detectors.items():
        try:
            prob = det.predict(mel)
        except Exception:
            prob = 0.0
        results[key] = {
            "name": MODELS_CONFIG[key]["name"],
            "prob": round(prob, 4),
            "detected": prob >= threshold
        }

    # Генерируем уникальный ID для возможности валидации и сохранения
    audio_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_") + str(int(time.time() * 1000) % 10000)
    
    # Сохраняем во временный кэш
    temp_cache[audio_id] = {
        "mel": mel,
        "audio_bytes": audio_bytes,
        "results": results,
        "timestamp": datetime.datetime.now().isoformat()
    }

    # Очищаем старый кэш если накопилось > 50
    if len(temp_cache) > 60:
        keys_to_del = list(temp_cache.keys())[:-40]
        for k in keys_to_del:
            del temp_cache[k]

    return web.json_response({
        "status": "ok",
        "audio_id": audio_id,
        "results": results
    })

async def handle_validate(request):
    """
    Принимает подтверждение от пользователя (ground truth)
    и сохраняет размеченный сэмпл для будущего дообучения.
    """
    data = await request.json()
    audio_id = data.get("audio_id")
    label = data.get("label")  # 'morsik' | 'karlusha' | 'hey_wisp' | 'negative'

    if not audio_id or audio_id not in temp_cache:
        return web.json_response({"status": "error", "error": "Сессия аудио не найдена или устарела"}, status=404)

    item = temp_cache[audio_id]
    mel = item["mel"]
    audio_bytes = item["audio_bytes"]
    results = item["results"]

    # 1. Сохраняем файл на диск в папку samples/<label>/
    target_dir = SAMPLES_DIR / label
    target_dir.mkdir(parents=True, exist_ok=True)
    
    wav_path = target_dir / f"{audio_id}.wav"
    npy_path = target_dir / f"{audio_id}.npy"

    with open(wav_path, "wb") as f:
        f.write(audio_bytes)
    np.save(str(npy_path), mel)

    # 2. Оцениваем корректность моделей по эталону
    evaluation = {}
    try:
        threshold = float(data.get("threshold", 0.80))
    except Exception:
        threshold = 0.80

    for key in MODELS_CONFIG:
        prob = results[key]["prob"]
        detected = prob >= threshold
        is_target = (key == label)

        if is_target:
            if detected:
                evaluation[key] = {"correct": True, "type": "TP"}   # True Positive
            else:
                evaluation[key] = {"correct": False, "type": "FN"}  # Ошибка II рода (пропуск)
        else:
            if not detected:
                evaluation[key] = {"correct": True, "type": "TN"}   # True Negative
            else:
                evaluation[key] = {"correct": False, "type": "FP"}  # Ошибка I рода (ложная тревога)

    # 3. Записываем в базу samples_db.json
    db_entries = []
    if SAMPLES_DB_FILE.exists():
        try:
            with open(SAMPLES_DB_FILE, "r", encoding="utf-8") as f:
                db_entries = json.load(f)
        except Exception:
            db_entries = []

    db_entries.append({
        "audio_id": audio_id,
        "label": label,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "results": {k: results[k]["prob"] for k in results},
        "evaluation": evaluation
    })

    with open(SAMPLES_DB_FILE, "w", encoding="utf-8") as f:
        json.dump(db_entries, f, ensure_ascii=False, indent=2)

    return web.json_response({
        "status": "ok",
        "audio_id": audio_id,
        "evaluation": evaluation
    })

async def handle_stats(request):
    """Возвращает общую статистику точности и ошибок по моделям"""
    db_entries = []
    if SAMPLES_DB_FILE.exists():
        try:
            with open(SAMPLES_DB_FILE, "r", encoding="utf-8") as f:
                db_entries = json.load(f)
        except Exception:
            db_entries = []

    total_tests = len(db_entries)
    models_stats = {}
    sample_counts = {k: 0 for k in list(MODELS_CONFIG.keys()) + ["negative"]}

    try:
        threshold = float(request.query.get("threshold", 0.80))
    except Exception:
        threshold = 0.80

    for key, cfg in MODELS_CONFIG.items():
        tp = sum(1 for e in db_entries if e.get("label") == key and e.get("results", {}).get(key, 0.0) >= threshold)
        fn = sum(1 for e in db_entries if e.get("label") == key and e.get("results", {}).get(key, 0.0) < threshold)
        fp = sum(1 for e in db_entries if e.get("label") != key and e.get("results", {}).get(key, 0.0) >= threshold)
        tn = sum(1 for e in db_entries if e.get("label") != key and e.get("results", {}).get(key, 0.0) < threshold)

        total = tp + tn + fp + fn
        acc = round((tp + tn) / total * 100.0, 1) if total > 0 else 100.0
        fp_rate = round(fp / total * 100.0, 1) if total > 0 else 0.0
        fn_rate = round(fn / (fn + tp) * 100.0, 1) if (fn + tp) > 0 else 0.0
        recall = round(tp / (tp + fn) * 100.0, 1) if (tp + fn) > 0 else 100.0

        models_stats[key] = {
            "name": cfg["name"],
            "total": total,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "accuracy": acc,
            "fp_rate": fp_rate,
            "fn_rate": fn_rate,
            "recall": recall
        }

    # Подсчёт физических сохранённых файлов
    for label in sample_counts:
        cat_dir = SAMPLES_DIR / label
        if cat_dir.exists():
            sample_counts[label] = len(list(cat_dir.glob("*.npy")))

    total_fp = sum(m["fp"] for m in models_stats.values())
    total_fn = sum(m["fn"] for m in models_stats.values())
    avg_acc = round(sum(m["accuracy"] for m in models_stats.values()) / max(1, len(models_stats)), 1)

    return web.json_response({
        "status": "ok",
        "total_tests": total_tests,
        "avg_accuracy": avg_acc,
        "total_fp": total_fp,
        "total_fn": total_fn,
        "models": models_stats,
        "sample_counts": sample_counts,
        "total_user_samples": sum(sample_counts.values())
    })

async def handle_reset_stats(request):
    """
    Сбрасывает историю испытаний (samples_db.json), архивируя её,
    но оставляет физические аудиосэмплы в папке samples/ для дообучения.
    """
    try:
        if SAMPLES_DB_FILE.exists():
            archive_file = BASE_DIR / "samples_db_archive.json"
            try:
                with open(SAMPLES_DB_FILE, "r", encoding="utf-8") as f:
                    current_entries = json.load(f)
            except Exception:
                current_entries = []

            if current_entries:
                existing_archive = []
                if archive_file.exists():
                    try:
                        with open(archive_file, "r", encoding="utf-8") as f:
                            existing_archive = json.load(f)
                    except Exception:
                        existing_archive = []
                
                existing_archive.append({
                    "archived_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "total_entries": len(current_entries),
                    "entries": current_entries
                })
                with open(archive_file, "w", encoding="utf-8") as f:
                    json.dump(existing_archive, f, ensure_ascii=False, indent=2)

            with open(SAMPLES_DB_FILE, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=2)

        return web.json_response({
            "status": "ok",
            "message": "Статистика успешно сброшена. Накопленные сэмплы для дообучения сохранены."
        })
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"status": "error", "error": str(e)}, status=500)

async def handle_retrain(request):
    """Запускает дообучение на накопленных сэмплах и перезагружает модели в память"""
    try:
        print("\n[*] Запрос на дообучение KWS моделей от пользователя...")
        log_entry = train_all_models(is_fine_tune=True)
        # Горячая перезагрузка моделей в память
        reload_detectors()
        return web.json_response({"status": "ok", "log": log_entry})
    except Exception as e:
        traceback.print_exc()
        return web.json_response({"status": "error", "error": str(e)}, status=500)

async def handle_train_history(request):
    """Отдаёт историю всех обучений"""
    history = []
    if TRAIN_HISTORY_FILE.exists():
        try:
            with open(TRAIN_HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []
    return web.json_response(history)

def create_app():
    app = web.Application(client_max_size=15 * 1024 * 1024)
    app.router.add_get('/', handle_index)
    app.router.add_get('/api/info', handle_info)
    app.router.add_post('/api/predict', handle_predict)
    app.router.add_post('/api/validate', handle_validate)
    app.router.add_get('/api/stats', handle_stats)
    app.router.add_post('/api/reset_stats', handle_reset_stats)
    app.router.add_post('/api/retrain', handle_retrain)
    app.router.add_get('/api/train_history', handle_train_history)
    app.router.add_static('/static', path=str(BASE_DIR / 'static'), name='static')
    return app

if __name__ == '__main__':
    cert_file, key_file = ensure_ssl_certs()
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)

    app = create_app()

    banner = f"""
======================================================================
  🎙️  KWS POLYGON: ИСПЫТАТЕЛЬНЫЙ ПОЛИГОН WAKE-WORD МОДЕЛЕЙ
======================================================================
  1. Локально на этом ПК:
     👉 https://localhost:{PORT}

  2. С других компьютеров / ноутбуков / телефонов в домашней сети:
     👉 https://{LOCAL_IP}:{PORT}

  🔒 Браузер покажет разовое предупреждение о сертификате:
     Нажмите «Дополнительно» -> «Перейти на сайт (небезопасно)».
     Это разблокирует доступ к микрофону по Wi-Fi / LAN!
======================================================================
"""
    print(banner)
    if sys.platform == "win32":
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        def _silence_winerror_10054(loop, context):
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)) or getattr(exc, 'winerror', None) == 10054:
                return
            loop.default_exception_handler(context)
        loop.set_exception_handler(_silence_winerror_10054)
        web.run_app(app, host='0.0.0.0', port=PORT, ssl_context=ssl_context, loop=loop)
    else:
        web.run_app(app, host='0.0.0.0', port=PORT, ssl_context=ssl_context)
