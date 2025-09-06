#!/usr/bin/env python3
"""
EEG判別機関数
前処理されたLSLデータ1.2秒分を入力として、3つの判別クラスを出力する関数
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from typing import Union, Tuple, Optional

class EEGDeepConvNet(nn.Module):
    """
    論文準拠のDeepConvNet（分類器用）
    プロジェクト内のlsl_classification.pyと同じ構造
    """
    
    def __init__(self, n_channels=32, n_classes=3, input_window_samples=300):
        super(EEGDeepConvNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        
        # プロジェクト内のDeepConvNetClassifierと同じ構造
        # Block 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, 25, (1, 10), padding=(0, 4))
        
        # Block 2: Spatial Convolution  
        self.conv2 = nn.Conv2d(25, 25, (n_channels, 1), padding=0)
        self.batch_norm1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d((1, 3))
        
        # Block 3: Separable Convolution
        self.conv3 = nn.Conv2d(25, 50, (1, 10), padding=(0, 4))
        self.batch_norm2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 3))
        
        # Block 4: Separable Convolution
        self.conv4 = nn.Conv2d(50, 100, (1, 10), padding=(0, 4))
        self.batch_norm3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 3))
        
        # Block 5: Separable Convolution
        self.conv5 = nn.Conv2d(100, 200, (1, 10), padding=(0, 4))
        self.batch_norm4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d((1, 3))
        
        # 適応的プーリング（可変長入力対応）
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(200 * 8, 256),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(0.15),
            nn.Linear(128, n_classes)
        )
    
    def forward(self, x):
        """順伝播"""
        # x: (batch_size, 1, channels, samples)
        
        # Block 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm1(x)
        x = F.elu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv3(x)
        x = self.batch_norm2(x)
        x = F.elu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv4(x)
        x = self.batch_norm3(x)
        x = F.elu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv5(x)
        x = self.batch_norm4(x)
        x = F.elu(x)
        x = self.pool4(x)
        
        # Global Average Pooling
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.classifier(x)
        return x


def classify_eeg_epoch(
    eeg_data: np.ndarray,
    model_path: str,
    class_names: Optional[list] = None,
    device: Optional[str] = None
) -> Tuple[str, int, float]:
    """
    EEGエポックデータを3クラス分類する関数
    
    Args:
        eeg_data (np.ndarray): 前処理済みEEGデータ (300, 32) または (32, 300)
        model_path (str): 学習済みモデルのパス
        class_names (list, optional): クラス名のリスト。デフォルトは ['Correct', 'UnderGrip', 'OverGrip']
        device (str, optional): 計算デバイス ('cpu', 'cuda')。Noneの場合は自動選択
    
    Returns:
        Tuple[str, int, float]: (予測クラス名, 予測クラスID, 信頼度)
    
    Raises:
        FileNotFoundError: モデルファイルが見つからない場合
        ValueError: 入力データの形状が不正な場合
        RuntimeError: モデル読み込み・推論エラー
    
    Example:
        >>> eeg_data = np.random.randn(300, 32)  # 1.2秒分のEEGデータ
        >>> model_path = './models/best_eeg_classifier.pth'
        >>> class_name, class_id, confidence = classify_eeg_epoch(eeg_data, model_path)
        >>> print(f"予測: {class_name} (ID: {class_id}, 信頼度: {confidence:.3f})")
    """
    
    # デフォルト値設定
    if class_names is None:
        class_names = ['Correct', 'UnderGrip', 'OverGrip']
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    try:
        # 1. モデルファイル存在確認
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        # 2. 入力データ検証・変換
        eeg_data = np.array(eeg_data, dtype=np.float32)
        
        # データ形状確認
        if eeg_data.ndim != 2:
            raise ValueError(f"入力データは2次元である必要があります。実際: {eeg_data.shape}")
        
        # (32, 300) → (300, 32) 変換
        if eeg_data.shape == (32, 300):
            eeg_data = eeg_data.T
        elif eeg_data.shape != (300, 32):
            # サイズ調整
            target_shape = (300, 32)
            
            # 時間軸調整
            if eeg_data.shape[0] != 300:
                if eeg_data.shape[0] > 300:
                    eeg_data = eeg_data[:300, :]
                else:
                    # ゼロパディング
                    padding = np.zeros((300 - eeg_data.shape[0], eeg_data.shape[1]))
                    eeg_data = np.vstack([eeg_data, padding])
            
            # チャンネル軸調整
            if eeg_data.shape[1] != 32:
                if eeg_data.shape[1] > 32:
                    eeg_data = eeg_data[:, :32]
                else:
                    # ゼロパディング
                    padding = np.zeros((eeg_data.shape[0], 32 - eeg_data.shape[1]))
                    eeg_data = np.hstack([eeg_data, padding])
        
        # 3. モデル読み込み
        model = EEGDeepConvNet(
            n_channels=32,
            n_classes=len(class_names),
            input_window_samples=300
        )
        
        # 学習済み重みの読み込み
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        
        # 評価モードに設定
        model.to(device)
        model.eval()
        
        # 4. テンソル変換 (1, 1, 32, 300)
        # CNN期待形式: (batch, channels, height, width) = (1, 1, 32, 300)
        epoch_tensor = torch.from_numpy(eeg_data.T).float()  # (32, 300)
        epoch_tensor = epoch_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 300)
        epoch_tensor = epoch_tensor.to(device)
        
        # 5. 推論実行
        with torch.no_grad():
            outputs = model(epoch_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # 予測結果
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_id].item()
            predicted_class_name = class_names[predicted_class_id]
        
        return predicted_class_name, predicted_class_id, confidence
        
    except Exception as e:
        raise RuntimeError(f"EEG分類エラー: {e}")


def batch_classify_eeg_epochs(
    eeg_data_list: list,
    model_path: str,
    class_names: Optional[list] = None,
    device: Optional[str] = None,
    batch_size: int = 32
) -> list:
    """
    複数のEEGエポックを一括分類する関数
    
    Args:
        eeg_data_list (list): EEGデータのリスト。各要素は (300, 32) または (32, 300)
        model_path (str): 学習済みモデルのパス
        class_names (list, optional): クラス名のリスト
        device (str, optional): 計算デバイス
        batch_size (int): バッチサイズ
    
    Returns:
        list: [(予測クラス名, 予測クラスID, 信頼度), ...] のリスト
    
    Example:
        >>> eeg_data_list = [np.random.randn(300, 32) for _ in range(10)]
        >>> results = batch_classify_eeg_epochs(eeg_data_list, model_path)
        >>> for i, (class_name, class_id, confidence) in enumerate(results):
        ...     print(f"データ{i}: {class_name} ({confidence:.3f})")
    """
    
    if not eeg_data_list:
        return []
    
    # デフォルト値設定
    if class_names is None:
        class_names = ['Correct', 'UnderGrip', 'OverGrip']
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    device = torch.device(device)
    
    try:
        # モデル読み込み（一度だけ）
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
        
        model = EEGDeepConvNet(
            n_channels=32,
            n_classes=len(class_names),
            input_window_samples=300
        )
        
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        results = []
        
        # バッチ処理
        for i in range(0, len(eeg_data_list), batch_size):
            batch_data = eeg_data_list[i:i+batch_size]
            batch_tensors = []
            
            # バッチデータ準備
            for eeg_data in batch_data:
                eeg_data = np.array(eeg_data, dtype=np.float32)
                
                # 形状調整（classify_eeg_epochと同じロジック）
                if eeg_data.shape == (32, 300):
                    eeg_data = eeg_data.T
                elif eeg_data.shape != (300, 32):
                    # サイズ調整処理（省略、必要に応じて実装）
                    pass
                
                # テンソル変換
                epoch_tensor = torch.from_numpy(eeg_data.T).float()  # (32, 300)
                epoch_tensor = epoch_tensor.unsqueeze(0)  # (1, 32, 300)
                batch_tensors.append(epoch_tensor)
            
            # バッチテンソル作成
            batch_tensor = torch.stack(batch_tensors).to(device)  # (batch_size, 1, 32, 300)
            
            # 推論実行
            with torch.no_grad():
                outputs = model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                predicted_classes = torch.argmax(probabilities, dim=1)
                confidences = torch.max(probabilities, dim=1)[0]
                
                # 結果をリストに変換
                for j in range(len(batch_data)):
                    class_id = predicted_classes[j].item()
                    confidence = confidences[j].item()
                    class_name = class_names[class_id]
                    
                    results.append((class_name, class_id, confidence))
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"バッチEEG分類エラー: {e}")


# 使用例とテスト
if __name__ == "__main__":
    # テスト用ダミーデータ
    print("🧠 EEG判別機関数テスト")
    
    # 1. 単一エポック分類テスト
    print("\n1. 単一エポック分類テスト")
    dummy_eeg = np.random.randn(300, 32)  # 1.2秒分のEEGデータ
    model_path = './models/best_eeg_classifier.pth'
    
    try:
        class_name, class_id, confidence = classify_eeg_epoch(dummy_eeg, model_path)
        print(f"✅ 成功: {class_name} (ID: {class_id}, 信頼度: {confidence:.3f})")
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    # 2. バッチ分類テスト
    print("\n2. バッチ分類テスト")
    dummy_eeg_list = [np.random.randn(300, 32) for _ in range(5)]
    
    try:
        results = batch_classify_eeg_epochs(dummy_eeg_list, model_path)
        print(f"✅ バッチ分類成功:")
        for i, (class_name, class_id, confidence) in enumerate(results):
            print(f"   データ{i}: {class_name} (ID: {class_id}, 信頼度: {confidence:.3f})")
    except Exception as e:
        print(f"❌ バッチ分類エラー: {e}")
    
    print("\n🎯 テスト完了")