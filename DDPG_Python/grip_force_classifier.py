#!/usr/bin/env python3
"""
EEG把持力分類システム
tcp_lsl_sync_systemから保存されたCSVデータを使って把持力の分類器を学習し、
リアルタイムでEEGデータから把持力レベルを判定するシステム

把持力分類:
- UnderGrip: 把持力 < 8N
- Success: 8N <= 把持力 <= 15N  
- OverGrip: 把持力 > 15N
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import sys
import time
import threading
import queue
import json
import pickle
from datetime import datetime
from collections import deque, Counter
from typing import List, Tuple, Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# tcp_lsl_sync_systemから必要なモジュールをインポート
try:
    from eeg_receiver import LSLEEGReceiver
    from unity_tcp_interface import EEGTCPInterface
    from eeg_neuroadaptation_preprocessor import NeuroadaptationEEGPreprocessor
except ImportError:
    print("⚠️ tcp_lsl_sync_systemのモジュールが見つかりません")
    print("   このスクリプトをDDPG_Pythonディレクトリで実行してください")


class EEGNetClassifier(nn.Module):
    """
    EEGNet: 把持力分類用のCNNモデル
    3つのクラス（UnderGrip, Success, OverGrip）を分類
    """
    
    def __init__(self, n_channels=32, n_classes=3, input_window_samples=300, 
                 dropout=0.25, kernLength=64, F1=8, D=2, F2=16):
        super(EEGNetClassifier, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        
        # Block 1: Temporal Convolution
        self.firstconv = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength//2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise Convolution
        self.depthwiseConv = nn.Conv2d(F1, F1*D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1*D)
        self.activation1 = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 3: Separable Convolution
        self.separableConv = nn.Conv2d(F1*D, F2, (1, 16), padding=(0, 8), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.activation2 = nn.ELU()
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # 動的に分類器サイズを計算
        self.flatten = nn.Flatten()
        self._setup_classifier(F2)
        
        print(f"🧠 EEGNet初期化完了: {n_channels}ch, {input_window_samples}samples → {n_classes}classes")
    
    def _setup_classifier(self, F2):
        """分類器の動的セットアップ"""
        # ダミー入力で特徴量サイズを計算
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.n_channels, self.input_window_samples)
            features = self._forward_features(dummy_input)
            feature_size = features.shape[1]
        
        self.classifier = nn.Linear(feature_size, self.n_classes)
    
    def _forward_features(self, x):
        """特徴抽出部分の順伝播"""
        # Block 1
        x = self.firstconv(x)
        x = self.batchnorm1(x)
        
        # Block 2
        x = self.depthwiseConv(x)
        x = self.batchnorm2(x)
        x = self.activation1(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.separableConv(x)
        x = self.batchnorm3(x)
        x = self.activation2(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Flatten
        x = self.flatten(x)
        return x
    
    def forward(self, x):
        """順伝播処理"""
        # x: (batch_size, 1, channels, samples)
        features = self._forward_features(x)
        x = self.classifier(features)
        return x


class GripForceDataset(Dataset):
    """把持力分類用データセット"""
    
    def __init__(self, eeg_data, grip_force_labels, transform=None):
        """
        Args:
            eeg_data: List of (300, 32) EEGエポック
            grip_force_labels: List of grip force labels (0: UnderGrip, 1: Success, 2: OverGrip)
        """
        self.eeg_data = eeg_data
        self.grip_force_labels = grip_force_labels
        self.transform = transform
        
        # ラベル統計表示
        label_counts = Counter(grip_force_labels)
        label_names = ['UnderGrip', 'Success', 'OverGrip']
        print(f"📊 データセット統計:")
        for i, name in enumerate(label_names):
            print(f"   {name}: {label_counts.get(i, 0)}件")
    
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        eeg_epoch = self.eeg_data[idx]  # (300, 32)
        label = self.grip_force_labels[idx]
        
        # データ拡張処理
        if self.transform:
            eeg_epoch = self.transform(eeg_epoch)
        
        # EEGNetの入力形式に変換: (1, 32, 300)
        eeg_tensor = torch.from_numpy(eeg_epoch.T).float()  # (32, 300)
        eeg_tensor = eeg_tensor.unsqueeze(0)  # (1, 32, 300)
        
        return eeg_tensor, torch.tensor(label, dtype=torch.long)


class GripForceClassifierTrainer:
    """把持力分類器の学習・評価・保存を管理"""
    
    def __init__(self, model, learning_rate=0.001, weight_decay=1e-4):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 最適化設定
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # 学習履歴
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        self.best_val_accuracy = 0.0
        
        print(f"🎯 学習環境: {self.device}")
    
    def train_epoch(self, train_loader):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """検証"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = test_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, all_preds, all_targets

    def train_full(self, train_loader, val_loader, epochs=100, early_stopping=None):
        """完全学習ループ

        Args:
            train_loader: 学習用データローダ
            val_loader: 検証用データローダ
            epochs: 学習エポック数
            early_stopping: 連続エポック改善なしで停止する回数。
                None の場合は早期終了を行わない
        """
        es_text = "なし" if early_stopping is None else early_stopping
        print(f"🎓 学習開始: {epochs}エポック, Early Stopping={es_text}")

        best_epoch = 0
        epochs_without_improvement = 0

        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # 学習
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 検証
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            
            # 学習率調整
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 学習率変更の表示
            if current_lr != old_lr:
                print(f"📉 学習率調整: {old_lr:.6f} → {current_lr:.6f}")
            
            # 履歴保存
            self.train_history['loss'].append(train_loss)
            self.train_history['accuracy'].append(train_acc)
            self.val_history['loss'].append(val_loss)
            self.val_history['accuracy'].append(val_acc)
            
            print(f"学習   : Loss={train_loss:.4f}, Acc={train_acc:.1f}%")
            print(f"検証   : Loss={val_loss:.4f}, Acc={val_acc:.1f}%, LR={current_lr:.6f}")
            
            # ベストモデル更新
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                # モデル保存
                os.makedirs('models', exist_ok=True)
                torch.save(self.model.state_dict(), 'models/best_grip_force_classifier.pth')
                print(f"🎯 ベストモデル更新! 精度: {val_acc:.1f}%")
            else:
                epochs_without_improvement += 1

            # Early Stopping
            if early_stopping is not None and epochs_without_improvement >= early_stopping:
                print(f"⏰ Early Stopping: {early_stopping}エポック改善なし")
                break
        
        print(f"\n✅ 学習完了!")
        print(f"   ベスト検証精度: {self.best_val_accuracy:.1f}% (Epoch {best_epoch})")
        return self.best_val_accuracy
    
    def evaluate_final(self, test_loader):
        """最終評価"""
        # ベストモデルをロード
        if os.path.exists('models/best_grip_force_classifier.pth'):
            self.model.load_state_dict(torch.load('models/best_grip_force_classifier.pth'))
        
        test_loss, test_acc, test_preds, test_targets = self.validate(test_loader)
        
        print(f"\n📊 最終テスト結果:")
        print(f"   テスト精度: {test_acc:.1f}%")
        print(f"   テスト損失: {test_loss:.4f}")
        
        # 分類レポート
        class_names = ['UnderGrip', 'Success', 'OverGrip']
        print(f"\n📋 分類レポート:")
        print(classification_report(test_targets, test_preds, target_names=class_names))
        
        # 混同行列を保存（表示はオプション）
        try:
            cm = confusion_matrix(test_targets, test_preds)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('把持力分類 - 混同行列')
            plt.xlabel('予測')
            plt.ylabel('実際')
            plt.tight_layout()
            
            cm_path = f'models/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(cm_path)
            print(f"混同行列保存: {cm_path}")
            plt.close()  # GUI環境でない場合のエラーを避けるため
        except Exception as e:
            print(f"混同行列保存エラー（無視可能）: {e}")
        
        return test_acc


def load_csv_data(csv_dir: str) -> Tuple[List[np.ndarray], List[int]]:
    """
    tcp_lsl_sync_systemで保存されたCSVデータを読み込み
    
    Args:
        csv_dir: CSVファイルが保存されているディレクトリ
        
    Returns:
        eeg_data_list: EEGデータのリスト [(300, 32), ...]
        grip_force_labels: 把持力ラベルのリスト [0, 1, 2, ...]
    """
    print(f"📂 CSVデータ読み込み開始: {csv_dir}")
    
    eeg_data_list = []
    grip_force_labels = []
    
    # エピソード情報ファイルを検索
    info_files = glob.glob(os.path.join(csv_dir, "*_info.csv"))
    
    if not info_files:
        print(f"❌ エピソード情報ファイルが見つかりません: {csv_dir}")
        return eeg_data_list, grip_force_labels
    
    print(f"📋 発見したエピソード: {len(info_files)}件")
    
    for info_file in sorted(info_files):
        try:
            # エピソード情報を読み込み
            info_df = pd.read_csv(info_file)
            episode_id = info_df['episode_id'].iloc[0]
            grip_force = info_df['grip_force'].iloc[0]
            
            # 把持力からラベルを決定
            if grip_force < 8.0:
                label = 0  # UnderGrip
                label_name = "UnderGrip"
            elif grip_force > 15.0:
                label = 2  # OverGrip  
                label_name = "OverGrip"
            else:
                label = 1  # Success
                label_name = "Success"
            
            # 対応するEEGファイルを読み込み
            eeg_file = info_file.replace('_info.csv', '_eeg.csv')
            if os.path.exists(eeg_file):
                eeg_df = pd.read_csv(eeg_file)
                
                # EEGデータ部分を抽出（チャンネル列のみ）
                channel_cols = [col for col in eeg_df.columns if col.startswith('ch_')]
                eeg_data = eeg_df[channel_cols].values  # (samples, channels)
                
                # 300サンプル（1.2秒）に調整
                if eeg_data.shape[0] >= 300:
                    eeg_data = eeg_data[:300, :]  # 最初の300サンプル
                    
                    # 32チャンネルに調整
                    if eeg_data.shape[1] >= 32:
                        eeg_data = eeg_data[:, :32]
                    else:
                        # チャンネル数が足りない場合はゼロパディング
                        padding = np.zeros((300, 32 - eeg_data.shape[1]))
                        eeg_data = np.hstack([eeg_data, padding])
                    
                    eeg_data_list.append(eeg_data)
                    grip_force_labels.append(label)
                    
                    print(f"   Episode {episode_id:04d}: 把持力={grip_force:.1f}N → {label_name}")
                
            else:
                print(f"⚠️ EEGファイルが見つかりません: {eeg_file}")
                
        except Exception as e:
            print(f"⚠️ ファイル読み込みエラー: {info_file}, {e}")
            continue
    
    print(f"✅ データ読み込み完了: {len(eeg_data_list)}件")
    
    # ラベル分布確認
    label_counts = Counter(grip_force_labels)
    label_names = ['UnderGrip', 'Success', 'OverGrip']
    print(f"📊 ラベル分布:")
    for i, name in enumerate(label_names):
        print(f"   {name}: {label_counts.get(i, 0)}件")
    
    return eeg_data_list, grip_force_labels


def train_grip_force_classifier(csv_dir: str, model_save_path: str = 'models/grip_force_classifier.pth'):
    """
    保存されたCSVデータから把持力分類器を学習
    
    Args:
        csv_dir: CSVデータのディレクトリ
        model_save_path: 学習済みモデルの保存パス
        
    Returns:
        dict: 学習結果の詳細情報
    """
    print(f"🎓 把持力分類器学習開始")
    print(f"=" * 60)
    
    # データ読み込み
    eeg_data_list, grip_force_labels = load_csv_data(csv_dir)
    
    if len(eeg_data_list) == 0:
        print(f"❌ 学習データがありません")
        return None
    
    # 最小サンプル数チェック
    label_counts = Counter(grip_force_labels)
    min_samples = min(label_counts.values()) if label_counts else 0
    
    if min_samples < 1:  # 最低3件（train, val, testに1件ずつ）
        print(f"⚠️ 各クラスのサンプル数が少なすぎます (最小: {min_samples}件)")
        print(f"   各クラス最低5件以上推奨")
        return None
    
    # データセット作成
    dataset = GripForceDataset(eeg_data_list, grip_force_labels)
    
    # データ分割 (70% train, 15% val, 15% test)
    total_size = len(dataset)
    train_size = max(1, int(0.7 * total_size))
    val_size = max(1, int(0.15 * total_size))
    test_size = total_size - train_size - val_size
    
    # test_sizeが0になる場合の調整
    if test_size <= 0:
        test_size = 1
        val_size = max(1, total_size - train_size - test_size)
        train_size = total_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"📊 データ分割: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # DataLoader作成（batch_sizeを動的調整）
    batch_size = min(8, train_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # モデル作成
    model = EEGNetClassifier(n_channels=32, n_classes=3, input_window_samples=300)
    trainer = GripForceClassifierTrainer(model)
    
    # 学習実行（エポック数を動的調整）
    epochs = min(100, max(20, total_size * 2))

    print(f"🚀 学習開始... (epochs={epochs}, early_stopping=None)")
    best_val_accuracy = trainer.train_full(
        train_loader, val_loader,
        epochs=epochs,
    )
    
    # 最終評価
    print(f"\n🔍 最終テスト評価...")
    test_accuracy = trainer.evaluate_final(test_loader)
    
    # モデル保存
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    
    # 結果サマリー
    result = {
        'model_path': model_save_path,
        'test_accuracy': test_accuracy,
        'val_accuracy': best_val_accuracy,
        'total_samples': total_size,
        'train_samples': train_size,
        'test_samples': test_size,
        'class_distribution': dict(label_counts),
        'timestamp': datetime.now().isoformat()
    }
    
    # 結果保存
    result_file = model_save_path.replace('.pth', '_result.json')
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 学習完了!")
    print(f"   モデル保存: {model_save_path}")
    print(f"   結果保存: {result_file}")
    print(f"   テスト精度: {test_accuracy:.1f}%")
    
    return result


class RealtimeGripForceClassifier:
    """リアルタイム把持力分類システム"""
    
    def __init__(self, model_path: str, lsl_stream_name: str = 'X.on-102807-0109', 
                 tcp_host: str = '127.0.0.1', tcp_port: int = 12345):
        """
        Args:
            model_path: 学習済みモデルのパス
            lsl_stream_name: LSLストリーム名
            tcp_host: TCPホスト
            tcp_port: TCPポート
        """
        self.model_path = model_path
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        
        # モデル読み込み
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EEGNetClassifier(n_channels=32, n_classes=3, input_window_samples=300)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ['UnderGrip', 'Success', 'OverGrip']
        
        # LSL受信システム（ImportErrorに対応）
        try:
            self.eeg_receiver = LSLEEGReceiver(stream_name=lsl_stream_name)
            self.eeg_preprocessor = NeuroadaptationEEGPreprocessor(
                sampling_rate=250,
                enable_asr=True,
                enable_ica=False
            )
            self.tcp_interface = EEGTCPInterface(host=tcp_host, port=tcp_port)
            self.system_available = True
        except Exception as e:
            print(f"⚠️ リアルタイムシステム初期化エラー: {e}")
            self.system_available = False
        
        # データバッファ
        self.lsl_data_buffer = deque(maxlen=2500)  # 10秒分
        self.lsl_timestamp_buffer = deque(maxlen=2500)
        
        # 実行制御
        self.is_running = False
        self.buffer_lock = threading.Lock()
        self.classification_count = 0
        
        print(f"🧠 リアルタイム把持力分類器初期化完了")
        print(f"   モデル: {model_path}")
        print(f"   デバイス: {self.device}")
        print(f"   クラス: {self.class_names}")
        print(f"   システム利用可能: {self.system_available}")
    
    def classify_eeg_epoch(self, eeg_data: np.ndarray) -> Tuple[str, int, float]:
        """
        EEGエポックを分類
        
        Args:
            eeg_data: (300, 32) のEEGデータ
            
        Returns:
            (予測クラス名, 予測クラスID, 信頼度)
        """
        try:
            # データ前処理
            if eeg_data.shape[0] != 300 or eeg_data.shape[1] != 32:
                # サイズ調整
                if eeg_data.shape[0] >= 300:
                    eeg_data = eeg_data[:300, :]
                else:
                    # パディング
                    padding = np.zeros((300 - eeg_data.shape[0], eeg_data.shape[1]))
                    eeg_data = np.vstack([eeg_data, padding])
                
                if eeg_data.shape[1] >= 32:
                    eeg_data = eeg_data[:, :32]
                else:
                    padding = np.zeros((300, 32 - eeg_data.shape[1]))
                    eeg_data = np.hstack([eeg_data, padding])
            
            # テンソル変換: (1, 1, 32, 300)
            eeg_tensor = torch.from_numpy(eeg_data.T).float()  # (32, 300)
            eeg_tensor = eeg_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 300)
            eeg_tensor = eeg_tensor.to(self.device)
            
            # 推論実行
            with torch.no_grad():
                outputs = self.model(eeg_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                predicted_class_id = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class_id].item()
                predicted_class_name = self.class_names[predicted_class_id]
            
            return predicted_class_name, predicted_class_id, confidence
            
        except Exception as e:
            print(f"⚠️ 分類エラー: {e}")
            return "ERROR", -1, 0.0
    
    def test_with_dummy_data(self):
        """ダミーデータでの分類テスト"""
        print(f"🧪 ダミーデータテスト開始")
        
        # ダミーEEGデータ作成
        dummy_eeg = np.random.randn(300, 32)
        
        # 分類実行
        prediction, class_id, confidence = self.classify_eeg_epoch(dummy_eeg)
        
        print(f"✅ テスト結果:")
        print(f"   予測: {prediction}")
        print(f"   クラスID: {class_id}")
        print(f"   信頼度: {confidence:.3f}")
        
        return prediction != "ERROR"


def main():
    """メイン実行関数"""
    print(f"🧠 EEG把持力分類システム")
    print(f"=" * 60)
    print(f"選択してください:")
    print(f"1. CSVデータから分類器を学習")
    print(f"2. 学習済み分類器でリアルタイム判定")
    print(f"3. 両方実行（学習→判定）")
    print(f"4. 分類器テスト（ダミーデータ）")
    
    choice = input(f"選択 (1-4): ").strip()
    
    if choice == "1":
        # 学習モード
        print(f"\n📚 学習モード選択")
        csv_dir = input(f"CSVデータディレクトリパス: ").strip()
        
        if not csv_dir:
            # デフォルトパス検索
            log_dirs = glob.glob("DDPG_Python/logs/episodes_*")
            if log_dirs:
                csv_dir = max(log_dirs)  # 最新のログディレクトリ
                print(f"デフォルト使用: {csv_dir}")
            else:
                print(f"❌ CSVディレクトリが見つかりません")
                return
        
        if not os.path.exists(csv_dir):
            print(f"❌ ディレクトリが存在しません: {csv_dir}")
            return
        
        # 学習実行
        result = train_grip_force_classifier(csv_dir)
        
        if result:
            print(f"\n✅ 学習完了!")
            print(f"   モデル: {result['model_path']}")
            print(f"   テスト精度: {result['test_accuracy']:.1f}%")
        else:
            print(f"❌ 学習失敗")
    
    elif choice == "2":
        # リアルタイム判定モード
        print(f"\n🎯 リアルタイム判定モード選択")
        model_path = input(f"学習済みモデルパス: ").strip()
        
        if not model_path:
            model_path = "models/best_grip_force_classifier.pth"
            print(f"デフォルト使用: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ モデルファイルが存在しません: {model_path}")
            return
        
        # リアルタイム分類器作成
        classifier = RealtimeGripForceClassifier(
            model_path=model_path,
            lsl_stream_name='MockEEG',
            tcp_host='127.0.0.1',
            tcp_port=12345
        )
        
        if not classifier.system_available:
            print(f"❌ リアルタイムシステムが利用できません")
            print(f"   tcp_lsl_sync_systemのモジュールを確認してください")
            return
        
        # 分類開始
        if classifier.start_classification():
            try:
                print(f"\n💡 システム稼働中...")
                print(f"   TCPからメッセージを送信すると分類実行")
                print(f"   Ctrl+C で終了")
                
                while True:
                    time.sleep(1.0)
                    
            except KeyboardInterrupt:
                print(f"\n⏹️ ユーザー停止")
            finally:
                classifier.stop_classification()
        
    elif choice == "3":
        # 両方実行モード
        print(f"\n🔄 学習→判定モード選択")
        csv_dir = input(f"CSVデータディレクトリパス: ").strip()
        
        if not csv_dir:
            log_dirs = glob.glob("DDPG_Python/logs/episodes_*")
            if log_dirs:
                csv_dir = max(log_dirs)
                print(f"デフォルト使用: {csv_dir}")
            else:
                print(f"❌ CSVディレクトリが見つかりません")
                return
        
        if not os.path.exists(csv_dir):
            print(f"❌ ディレクトリが存在しません: {csv_dir}")
            return
        
        # Step 1: 学習
        print(f"\n🎓 Step 1: 分類器学習")
        result = train_grip_force_classifier(csv_dir)
        
        if not result:
            print(f"❌ 学習失敗")
            return
        
        print(f"✅ 学習完了! テスト精度: {result['test_accuracy']:.1f}%")
        
        # Step 2: リアルタイム判定
        if result['test_accuracy'] > 30:  # 30%以上で実用可能（少データ対応）
            print(f"\n🎯 Step 2: リアルタイム判定開始")
            
            classifier = RealtimeGripForceClassifier(
                model_path=result['model_path'],
                lsl_stream_name='MockEEG',
                tcp_host='127.0.0.1',
                tcp_port=12345
            )
            
            if classifier.system_available and classifier.start_classification():
                try:
                    print(f"\n💡 両方のシステム稼働中...")
                    print(f"   学習完了 → リアルタイム分類中")
                    print(f"   TCPからメッセージを送信すると分類実行")
                    print(f"   Ctrl+C で終了")
                    
                    while True:
                        time.sleep(1.0)
                        
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    classifier.stop_classification()
            else:
                print(f"⚠️ リアルタイムシステムが利用できません")
                print(f"   モデルは学習済みです: {result['model_path']}")
        else:
            print(f"⚠️ 分類精度が低すぎます ({result['test_accuracy']:.1f}%)")
            print(f"   より多くのデータ収集が必要です")
    
    elif choice == "4":
        # テストモード
        print(f"\n🧪 分類器テストモード")
        model_path = input(f"学習済みモデルパス: ").strip()
        
        if not model_path:
            model_path = "models/best_grip_force_classifier.pth"
            print(f"デフォルト使用: {model_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ モデルファイルが存在しません: {model_path}")
            return
        
        # テスト実行
        classifier = RealtimeGripForceClassifier(
            model_path=model_path,
            lsl_stream_name='MockEEG',
            tcp_host='127.0.0.1',
            tcp_port=12345
        )
        
        success = classifier.test_with_dummy_data()
        
        if success:
            print(f"✅ 分類器テスト成功!")
        else:
            print(f"❌ 分類器テスト失敗")
    
    else:
        print(f"❌ 無効な選択です")


# 単体テスト用関数
def test_csv_loading():
    """CSV読み込みテスト"""
    print(f"🧪 CSV読み込みテスト")
    
    # 最新のログディレクトリを検索
    log_dirs = glob.glob("DDPG_Python/logs/episodes_*")
    if not log_dirs:
        print(f"❌ テスト用CSVディレクトリが見つかりません")
        return False
    
    csv_dir = max(log_dirs)
    print(f"テスト対象: {csv_dir}")
    
    eeg_data_list, grip_force_labels = load_csv_data(csv_dir)
    
    if eeg_data_list:
        print(f"✅ テスト成功!")
        print(f"   データ件数: {len(eeg_data_list)}")
        print(f"   EEG形状: {eeg_data_list[0].shape}")
        print(f"   ラベル例: {grip_force_labels[:5]}")
        return True
    else:
        print(f"❌ テスト失敗")
        return False


def test_model_inference():
    """モデル推論テスト"""
    print(f"🧪 モデル推論テスト")
    
    try:
        # ダミーデータでモデル作成
        model = EEGNetClassifier(n_channels=32, n_classes=3, input_window_samples=300)
        model.eval()
        
        # ダミーEEGデータ
        dummy_eeg = np.random.randn(300, 32)
        
        # テンソル変換
        eeg_tensor = torch.from_numpy(dummy_eeg.T).float()  # (32, 300)
        eeg_tensor = eeg_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, 32, 300)
        
        # 推論実行
        with torch.no_grad():
            outputs = model(eeg_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            predicted_class_id = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class_id].item()
        
        class_names = ['UnderGrip', 'Success', 'OverGrip']
        predicted_class_name = class_names[predicted_class_id]
        
        print(f"✅ 推論テスト成功!")
        print(f"   入力形状: {eeg_tensor.shape}")
        print(f"   出力形状: {outputs.shape}")
        print(f"   予測クラス: {predicted_class_name}")
        print(f"   信頼度: {confidence:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ 推論テスト失敗: {e}")
        return False


if __name__ == "__main__":
    import os
    import sys
    import glob
    import time
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datetime import datetime
    
    # 必要な関数やクラスをインポート（実際のファイルから）
    # from grip_force_classifier import train_grip_force_classifier, RealtimeGripForceClassifier, EEGNetClassifier, load_csv_data
    
    print(f"🧠 EEG把持力分類システム")
    print(f"tcp_lsl_sync_systemのCSVデータを使用してEEGNetで把持力を分類")
    print(f"UnderGrip(<8N), Success(8-15N), OverGrip(>15N)")
    print(f"")
    
    # コマンドライン引数での実行
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_csv":
            test_csv_loading()
        elif sys.argv[1] == "test_model":
            test_model_inference()
        elif sys.argv[1] == "test_all":
            print("🧪 全テスト実行")
            csv_ok = test_csv_loading()
            print("")
            model_ok = test_model_inference()
            print(f"\n結果: CSV={csv_ok}, Model={model_ok}")
        else:
            main()
    else:
        main()