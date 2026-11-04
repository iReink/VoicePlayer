"""
Архитектура нейросети Keyword Spotting (KWSNet) на базе Depthwise-Separable CNN.
Легковесная (около 35 000 параметров, ~140 КБ), инференс на CPU < 0.5 мс.
Экспорт в ONNX и дообучение на лету.
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import onnxruntime as ort

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class KWSNet(nn.Module):
    """
    KWS Классификатор: принимает Mel-спектрограмму (Batch, 1, 40, 151)
    Выдаёт вероятность [0.0 - 1.0] конкретного триггера.
    """
    def __init__(self):
        super().__init__()
        # Вход: (1, 40, 151)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Сжатие по частоте и времени
        self.block1 = DepthwiseSeparableConv(16, 32, stride=2)   # (32, 20, 76)
        self.block2 = DepthwiseSeparableConv(32, 48, stride=2)   # (48, 10, 38)
        self.block3 = DepthwiseSeparableConv(48, 64, stride=2)   # (64, 5, 19)
        self.block4 = DepthwiseSeparableConv(64, 64, stride=1)   # (64, 5, 19)
        
        # Глобальный пулинг
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Классификаторная голова
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.head(x)

def export_to_onnx(model: KWSNet, output_path: str):
    """Экспорт обученной PyTorch модели в ONNX формат"""
    import os, sys
    model.eval()
    dummy_input = torch.randn(1, 1, 40, 151, dtype=torch.float32)
    # Гарантируем корректную кодировку для логов PyTorch
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['mel_input'],
        output_names=['prob'],
        dynamic_axes={'mel_input': {0: 'batch'}, 'prob': {0: 'batch'}},
        opset_version=18,
        do_constant_folding=True
    )

def train_kws_head(
    model: KWSNet,
    pos_features: np.ndarray,
    neg_features: np.ndarray,
    epochs: int = 8,
    lr: float = 0.003,
    batch_size: int = 64,
    neg_weight_val: float = 2.5
) -> List[float]:
    """
    Обучает/дообучает бинарную KWS голову на признаках (Mel-спектрограммах).
    pos_features: (N, 40, 151)
    neg_features: (M, 40, 151)
    neg_weight_val: штраф за ложные тревоги (False Positives)
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    num_pos = len(pos_features)
    num_neg = len(neg_features)
    
    # Объединяем датасет
    X = np.concatenate([pos_features, neg_features], axis=0)  # (N+M, 40, 151)
    y = np.concatenate([np.ones((num_pos, 1), dtype=np.float32), np.zeros((num_neg, 1), dtype=np.float32)], axis=0)
    
    # Добавляем измерение канала (Batch, 1, 40, 151)
    X = np.expand_dims(X, axis=1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset_size = len(X)
    loss_history = []

    for epoch in range(epochs):
        indices = np.random.permutation(dataset_size)
        epoch_loss = 0.0
        batches = 0

        for i in range(0, dataset_size, batch_size):
            batch_idx = indices[i:i+batch_size]
            bx = X_tensor[batch_idx]
            by = y_tensor[batch_idx]

            optimizer.zero_grad()
            preds = model(bx)
            
            # Асимметричный штраф: за ошибку на негативе (False Positive) штрафуем жестче в 2.5 раза
            weights = torch.where(by == 0.0, torch.tensor(neg_weight_val), torch.tensor(1.0))
            loss = nn.functional.binary_cross_entropy(preds, by, weight=weights)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / max(1, batches)
        loss_history.append(avg_loss)
        print(f"    [Epoch {epoch+1}/{epochs}] Loss: {avg_loss:.4f}", flush=True)

    model.eval()
    return loss_history

class ONNXWakeWordDetector:
    """Обертка для быстрого инференса через ONNX Runtime"""
    def __init__(self, onnx_path: str):
        self.path = onnx_path
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.enable_cpu_mem_arena = False
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, mel_spec: np.ndarray) -> float:
        """
        mel_spec: shape (40, 151)
        Возвращает вероятность (float от 0.0 до 1.0)
        """
        inp = np.expand_dims(mel_spec, axis=(0, 1)).astype(np.float32)  # (1, 1, 40, 151)
        res = self.session.run(None, {self.input_name: inp})
        return float(res[0][0, 0])
