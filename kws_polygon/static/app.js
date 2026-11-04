/**
 * KWS Polygon Frontend Controller
 * Web Audio API Oscilloscope + Push-To-Talk + Multi-Model Scoring & Validation
 */

let audioContext = null;
let mediaStream = null;
let analyserNode = null;
let scriptProcessor = null;
let isRecording = false;
let recordedAudioChunks = [];
let totalRecordedSamples = 0;
let currentAudioId = null;
let currentPredictions = null;
let spaceKeyLocked = false;
let autoStopTimer = null;

const RECORD_SAMPLE_RATE = 16000;
const MAX_RECORD_SECONDS = 3.0;

// DOM Elements
const canvas = document.getElementById('waveformCanvas');
const canvasCtx = canvas.getContext('2d');
const pttBtn = document.getElementById('pttBtn');
const pttBtnText = document.getElementById('pttBtnText');
const recIndicator = document.getElementById('recIndicator');
const visStatusText = document.getElementById('visStatusText');
const playbackWrapper = document.getElementById('playbackWrapper');
const audioPreview = document.getElementById('audioPreview');
const valStatusText = document.getElementById('valStatusText');

const cards = {
    morsik: document.getElementById('card-morsik'),
    karlusha: document.getElementById('card-karlusha'),
    hey_wisp: document.getElementById('card-hey_wisp')
};

const vals = {
    morsik: document.getElementById('val-morsik'),
    karlusha: document.getElementById('val-karlusha'),
    hey_wisp: document.getElementById('val-hey_wisp')
};

const fills = {
    morsik: document.getElementById('fill-morsik'),
    karlusha: document.getElementById('fill-karlusha'),
    hey_wisp: document.getElementById('fill-hey_wisp')
};

const statuses = {
    morsik: document.getElementById('status-morsik'),
    karlusha: document.getElementById('status-karlusha'),
    hey_wisp: document.getElementById('status-hey_wisp')
};

const verdicts = {
    morsik: document.getElementById('verdict-morsik'),
    karlusha: document.getElementById('verdict-karlusha'),
    hey_wisp: document.getElementById('verdict-hey_wisp')
};

const valButtons = document.querySelectorAll('.btn-val');

// 1. INITIALIZE WEB AUDIO API & CANVAS
async function initAudio() {
    if (audioContext) return true;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: RECORD_SAMPLE_RATE,
                echoCancellation: false,
                noiseSuppression: false,
                autoGainControl: false
            }
        });
        mediaStream = stream;
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: RECORD_SAMPLE_RATE
        });
        const source = audioContext.createMediaStreamSource(stream);
        analyserNode = audioContext.createAnalyser();
        analyserNode.fftSize = 512;
        source.connect(analyserNode);

        // ScriptProcessor для перехвата сырых семплов (16 кГц)
        scriptProcessor = audioContext.createScriptProcessor(2048, 1, 1);
        source.connect(scriptProcessor);
        scriptProcessor.connect(audioContext.destination);

        scriptProcessor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const inputData = e.inputBuffer.getChannelData(0);
            recordedAudioChunks.push(new Float32Array(inputData));
            totalRecordedSamples += inputData.length;

            const recSec = (totalRecordedSamples / RECORD_SAMPLE_RATE).toFixed(1);
            visStatusText.textContent = `🎙️ Запись (${recSec}с / макс ${MAX_RECORD_SECONDS}с)... Отпустите Пробел для анализа`;

            // Автоостановка при превышении лимита
            if (totalRecordedSamples >= RECORD_SAMPLE_RATE * MAX_RECORD_SECONDS) {
                stopRecording();
            }
        };

        drawWaveform();
        visStatusText.textContent = "Микрофон подключен (Удерживайте Пробел)";
        return true;
    } catch (err) {
        console.error("Microphone error:", err);
        visStatusText.textContent = "Ошибка доступа к микрофону: " + err.message;
        alert("Не удалось получить доступ к микрофону. Проверьте разрешения браузера или используйте HTTPS.");
        return false;
    }
}

// 2. OSCILLOSCOPE DRAWING
function drawWaveform() {
    requestAnimationFrame(drawWaveform);
    if (!analyserNode) return;

    const bufferLength = analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyserNode.getByteTimeDomainData(dataArray);

    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = canvas.parentElement.clientHeight;

    canvasCtx.fillStyle = '#090d14';
    canvasCtx.fillRect(0, 0, canvas.width, canvas.height);

    canvasCtx.lineWidth = isRecording ? 3 : 2;
    canvasCtx.strokeStyle = isRecording ? '#f85149' : '#388bfd';
    canvasCtx.beginPath();

    const sliceWidth = canvas.width * 1.0 / bufferLength;
    let x = 0;

    for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = v * canvas.height / 2;
        if (i === 0) {
            canvasCtx.moveTo(x, y);
        } else {
            canvasCtx.lineTo(x, y);
        }
        x += sliceWidth;
    }

    canvasCtx.lineTo(canvas.width, canvas.height / 2);
    canvasCtx.stroke();
}

// 3. RECORDING LIFECYCLE (PUSH-TO-TALK)
async function startRecording() {
    if (isRecording) return;
    const ok = await initAudio();
    if (!ok) return;

    if (audioContext && audioContext.state === 'suspended') {
        await audioContext.resume();
    }

    resetCardVisuals();
    recordedAudioChunks = [];
    totalRecordedSamples = 0;
    isRecording = true;

    pttBtn.classList.add('recording');
    pttBtnText.textContent = "🔴 Идёт запись (Отпустите для анализа)...";
    recIndicator.classList.add('recording');
    visStatusText.textContent = "🎙️ Запись... Произнесите слово и отпустите Пробел";

    // Блокируем кнопки валидации на время новой записи
    valButtons.forEach(btn => {
        btn.disabled = true;
        btn.classList.remove('selected');
    });
    valStatusText.textContent = "Говорите в микрофон...";
}

function stopRecording() {
    if (!isRecording) return;
    isRecording = false;

    pttBtn.classList.remove('recording');
    pttBtnText.textContent = "Зажмите для записи (Пробел)";
    recIndicator.classList.remove('recording');
    visStatusText.textContent = "Анализ нейросетями...";

    processRecordedAudio();
}

// 4. CONVERT RECORDED FLOAT32 TO WAV
function encodeWAV(samples, sampleRate = RECORD_SAMPLE_RATE) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);

    function writeString(view, offset, string) {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    }

    /* RIFF identifier */
    writeString(view, 0, 'RIFF');
    /* file length */
    view.setUint32(4, 36 + samples.length * 2, true);
    /* RIFF type */
    writeString(view, 8, 'WAVE');
    /* format chunk identifier */
    writeString(view, 12, 'fmt ');
    /* format chunk length */
    view.setUint32(16, 16, true);
    /* sample format (raw) */
    view.setUint16(20, 1, true);
    /* channel count */
    view.setUint16(22, 1, true);
    /* sample rate */
    view.setUint32(24, sampleRate, true);
    /* byte rate (sample rate * block align) */
    view.setUint32(28, sampleRate * 2, true);
    /* block align (channel count * bytes per sample) */
    view.setUint16(32, 2, true);
    /* bits per sample */
    view.setUint16(34, 16, true);
    /* data chunk identifier */
    writeString(view, 36, 'data');
    /* data chunk length */
    view.setUint32(40, samples.length * 2, true);

    // Float32 to Int16
    let offset = 44;
    for (let i = 0; i < samples.length; i++, offset += 2) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }

    return new Blob([view], { type: 'audio/wav' });
}

// 5. SEND AUDIO TO BACKEND FOR PREDICTION
async function processRecordedAudio() {
    if (recordedAudioChunks.length === 0) return;

    // Склеиваем все чанки
    const totalLength = recordedAudioChunks.reduce((acc, c) => acc + c.length, 0);
    if (totalLength < RECORD_SAMPLE_RATE * 0.25) {
        visStatusText.textContent = "⚠️ Слишком короткая запись (< 0.25 с). Удерживайте Пробел дольше.";
        return;
    }
    const fullAudio = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of recordedAudioChunks) {
        fullAudio.set(chunk, offset);
        offset += chunk.length;
    }

    // Создаём WAV файл
    const wavBlob = encodeWAV(fullAudio, RECORD_SAMPLE_RATE);

    // Локальный предпросмотр аудио
    const audioUrl = URL.createObjectURL(wavBlob);
    audioPreview.src = audioUrl;
    playbackWrapper.style.display = 'flex';

    // Отправляем на сервер
    const formData = new FormData();
    formData.append('audio', wavBlob, 'recording.wav');

    try {
        const resp = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();

        if (data.status === 'ok') {
            currentAudioId = data.audio_id;
            currentPredictions = data.results;
            displayPredictions(data.results);
            visStatusText.textContent = "Анализ завершён. Подтвердите сказанное справа ➡️";
            valStatusText.textContent = "Выберите слово или «Ошибка»";
            // Включаем кнопки валидации
            valButtons.forEach(btn => btn.disabled = false);
        } else {
            visStatusText.textContent = "Ошибка анализа: " + (data.error || "Неизвестная ошибка");
        }
    } catch (err) {
        console.error("Predict error:", err);
        visStatusText.textContent = "Ошибка отправки на сервер";
    }
}

// 6. DISPLAY MODEL PREDICTIONS
function displayPredictions(results) {
    const threshold = 0.50;

    for (const [key, res] of Object.entries(results)) {
        const pct = Math.round(res.prob * 100);
        vals[key].textContent = pct + "%";
        fills[key].style.width = pct + "%";

        if (res.prob >= threshold) {
            statuses[key].textContent = "🔔 СРАБАТЫВАНИЕ (" + pct + "%)";
            statuses[key].style.color = "var(--accent-green)";
            fills[key].style.backgroundColor = "var(--accent-green)";
        } else {
            statuses[key].textContent = "ТИШИНА (" + pct + "%)";
            statuses[key].style.color = "var(--text-secondary)";
            fills[key].style.backgroundColor = "var(--accent-blue)";
        }
    }
}

// 7. USER VALIDATION (GROUND TRUTH CONFIRMATION)
async function submitValidation(label, targetBtn = null) {
    if (!currentAudioId) {
        valStatusText.textContent = "⚠️ Сначала запишите звук (удерживайте Пробел)";
        return;
    }
    if (isRecording) {
        valStatusText.textContent = "⏳ Дождитесь окончания записи";
        return;
    }

    const btn = targetBtn || document.querySelector(`.btn-val[data-label="${label}"]`);
    valButtons.forEach(b => b.classList.remove('selected'));
    if (btn) btn.classList.add('selected');

    valStatusText.textContent = "Сохранение эталона...";

    try {
        const resp = await fetch('/api/validate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                audio_id: currentAudioId,
                label: label
            })
        });
        const data = await resp.json();

        if (data.status === 'ok') {
            applyValidationFeedback(data.evaluation, label);
            const labelNames = {
                morsik: '«Морсик»',
                karlusha: '«Карлуша»',
                hey_wisp: '«Эй, Висп!»',
                negative: 'Ошибка / Шум'
            };
            valStatusText.textContent = `✅ Учтено: ${labelNames[label] || label}! Сэмпл сохранён.`;
            loadStats(); // Обновляем статистику
        } else {
            valStatusText.textContent = "Ошибка: " + (data.error || "Не удалось сохранить");
        }
    } catch (err) {
        console.error("Validate error:", err);
        valStatusText.textContent = "Ошибка сохранения валидации";
    }
}

valButtons.forEach(btn => {
    btn.addEventListener('click', () => {
        const label = btn.getAttribute('data-label');
        submitValidation(label, btn);
    });
});

// Подсветка карточек: ЗЕЛЁНЫЙ если модель была права, КРАСНЫЙ если ошиблась
function applyValidationFeedback(evaluation, groundTruthLabel) {
    for (const [key, evalData] of Object.entries(evaluation)) {
        const card = cards[key];
        card.classList.remove('state-correct', 'state-wrong');

        if (evalData.correct) {
            card.classList.add('state-correct');
            if (evalData.type === 'TP') {
                verdicts[key].textContent = "✅ ВЕРНО ОБНАРУЖЕНО (True Positive)";
                verdicts[key].style.color = "var(--accent-green)";
            } else {
                verdicts[key].textContent = "✅ ВЕРНО МОЛЧИТ (True Negative)";
                verdicts[key].style.color = "var(--accent-green)";
            }
        } else {
            card.classList.add('state-wrong');
            if (evalData.type === 'FP') {
                verdicts[key].textContent = "❌ ЛОЖНАЯ ТРЕВОГА (Ошибка I рода / FP)";
                verdicts[key].style.color = "var(--accent-red)";
            } else {
                verdicts[key].textContent = "❌ ПРОПУСК ЦЕЛИ (Ошибка II рода / FN)";
                verdicts[key].style.color = "var(--accent-red)";
            }
        }
    }
}

function resetCardVisuals() {
    for (const key of ['morsik', 'karlusha', 'hey_wisp']) {
        cards[key].classList.remove('state-correct', 'state-wrong');
        verdicts[key].textContent = "";
        statuses[key].textContent = "Ожидание...";
        statuses[key].style.color = "var(--text-secondary)";
    }
}

// Карта хоткеев валидации: 1, 2, 3, 4 (как стандартные цифры, так и Numpad)
const VALIDATION_HOTKEYS = {
    'Digit1': 'morsik',
    'Numpad1': 'morsik',
    'Digit2': 'karlusha',
    'Numpad2': 'karlusha',
    'Digit3': 'hey_wisp',
    'Numpad3': 'hey_wisp',
    'Digit4': 'negative',
    'Numpad4': 'negative'
};

// 8. KEYBOARD LISTENERS (SPACEBAR & HOTKEYS 1-4)
window.addEventListener('keydown', (e) => {
    // Игнорируем ввод, если фокус в поле формы
    if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;

    // 1. Пробел -> Push-to-Talk
    if (e.code === 'Space') {
        if (e.repeat) {
            e.preventDefault();
            return;
        }
        if (!spaceKeyLocked) {
            e.preventDefault();
            spaceKeyLocked = true;
            startRecording();
        }
        return;
    }

    // 2. Хоткеи 1, 2, 3, 4 для валидации (Digit и Numpad)
    let label = VALIDATION_HOTKEYS[e.code];
    if (!label) {
        // Fallback по символу клавиши
        if (e.key === '1') label = 'morsik';
        else if (e.key === '2') label = 'karlusha';
        else if (e.key === '3') label = 'hey_wisp';
        else if (e.key === '4') label = 'negative';
    }

    if (label) {
        e.preventDefault();
        const btn = document.querySelector(`.btn-val[data-label="${label}"]`);
        if (btn && !btn.disabled) {
            // Анимация нажатия клавиши на кнопке
            btn.classList.add('active');
            setTimeout(() => btn.classList.remove('active'), 150);
            submitValidation(label, btn);
        } else if (!currentAudioId) {
            valStatusText.textContent = "⚠️ Сначала запишите звук (удерживайте Пробел)";
        }
    }
});

window.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
        if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
        e.preventDefault();
        spaceKeyLocked = false;
        if (isRecording) {
            stopRecording();
        }
    }
});

// Mouse & Touch events on PTT Button
pttBtn.addEventListener('mousedown', (e) => {
    e.preventDefault();
    startRecording();
});

window.addEventListener('mouseup', () => {
    if (isRecording && !spaceKeyLocked) {
        stopRecording();
    }
});

pttBtn.addEventListener('touchstart', (e) => {
    e.preventDefault();
    startRecording();
});

pttBtn.addEventListener('touchend', (e) => {
    e.preventDefault();
    stopRecording();
});

// 9. TAB SWITCHING
const tabBtns = document.querySelectorAll('.tab-btn');
const tabPanes = document.querySelectorAll('.tab-pane');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        tabBtns.forEach(b => b.classList.remove('active'));
        tabPanes.forEach(p => p.classList.remove('active'));

        btn.classList.add('active');
        const targetTab = btn.getAttribute('data-tab');
        document.getElementById('pane-' + targetTab).classList.add('active');

        if (targetTab === 'stats') loadStats();
        if (targetTab === 'train') loadTrainingHistory();
    });
});

// 10. STATS LOADING
async function loadStats() {
    try {
        const resp = await fetch('/api/stats');
        const data = await resp.json();

        document.getElementById('statTotalTests').textContent = data.total_tests;
        document.getElementById('statAvgAccuracy').textContent = data.avg_accuracy + "%";
        document.getElementById('statTotalFP').textContent = data.total_fp;
        document.getElementById('statTotalFN').textContent = data.total_fn;

        const tbody = document.getElementById('statsTableBody');
        tbody.innerHTML = '';

        for (const [key, m] of Object.entries(data.models)) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong>${m.name}</strong> (${key})</td>
                <td>${m.total}</td>
                <td><strong style="color: var(--accent-green);">${m.accuracy}%</strong></td>
                <td><span style="color: var(--accent-red);">${m.fp} (${m.fp_rate}%)</span></td>
                <td><span style="color: var(--accent-yellow);">${m.fn} (${m.fn_rate}%)</span></td>
                <td>${m.recall}%</td>
            `;
            tbody.appendChild(tr);
        }

        // Badges
        const badgesContainer = document.getElementById('samplesCountBadges');
        badgesContainer.innerHTML = '';
        for (const [label, count] of Object.entries(data.sample_counts)) {
            const div = document.createElement('div');
            div.className = 'sample-badge-item';
            div.innerHTML = `${label}: <strong>${count} шт</strong>`;
            badgesContainer.appendChild(div);
        }

        document.getElementById('readySamplesCount').textContent = `Готово новых сэмплов пользователя: ${data.total_user_samples}`;
    } catch (err) {
        console.error("Stats load error:", err);
    }
}

// 11. RE-TRAINING & HISTORY
async function loadTrainingHistory() {
    try {
        const resp = await fetch('/api/train_history');
        const history = await resp.json();

        const tbody = document.getElementById('historyTableBody');
        tbody.innerHTML = '';

        if (!history || history.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" class="text-center">История обучений пуста</td></tr>';
            return;
        }

        for (const h of history) {
            const r = h.results || {};
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${h.timestamp}</td>
                <td><span class="badge ${h.type.includes('Fine-tune') ? 'badge-accent' : 'badge-warning'}">${h.type}</span></td>
                <td><strong>${h.user_samples_count} шт</strong></td>
                <td>Acc: <strong>${r.morsik?.accuracy || 0}%</strong> (Loss: ${r.morsik?.final_loss || '-'})</td>
                <td>Acc: <strong>${r.karlusha?.accuracy || 0}%</strong> (Loss: ${r.karlusha?.final_loss || '-'})</td>
                <td>Acc: <strong>${r.hey_wisp?.accuracy || 0}%</strong> (Loss: ${r.hey_wisp?.final_loss || '-'})</td>
            `;
            tbody.appendChild(tr);
        }
    } catch (err) {
        console.error("Train history error:", err);
    }
}

const btnStartRetrain = document.getElementById('btnStartRetrain');
const retrainProgressCard = document.getElementById('retrainProgressCard');
const progressStatusText = document.getElementById('progressStatusText');

btnStartRetrain.addEventListener('click', async () => {
    if (!confirm("Запустить цикл дообучения KWS-моделей на накопленных сэмплах?")) return;

    btnStartRetrain.disabled = true;
    retrainProgressCard.style.display = 'block';
    progressStatusText.textContent = "Обучение моделей в процессе (PyTorch -> ONNX)...";

    try {
        const resp = await fetch('/api/retrain', { method: 'POST' });
        const res = await resp.json();

        if (res.status === 'ok') {
            progressStatusText.textContent = "✅ Модели успешно дообучены и перезагружены в память!";
            setTimeout(() => {
                retrainProgressCard.style.display = 'none';
            }, 3000);
            loadTrainingHistory();
            loadStats();
        } else {
            progressStatusText.textContent = "❌ Ошибка обучения: " + (res.error || "Неизвестная ошибка");
        }
    } catch (err) {
        progressStatusText.textContent = "❌ Сбой сети при вызове дообучения";
    } finally {
        btnStartRetrain.disabled = false;
    }
});

// Load LAN IP
async function loadNetworkInfo() {
    try {
        const resp = await fetch('/api/info');
        const data = await resp.json();
        document.getElementById('lanUrl').textContent = `Сеть: ${data.lan_url}`;
    } catch (e) {
        document.getElementById('lanUrl').textContent = `Локально: ${window.location.host}`;
    }
}

// Initial calls
loadNetworkInfo();
loadStats();
