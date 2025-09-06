#!/usr/bin/env python3
"""
EEG認知競合分類器学習システム
TCPからの明示的フィードバックを教師データとして、
LSLのEEGデータから認知競合を分類する判別器を作成・学習

学習フロー:
1. Unity TCP → 明示的フィードバック (success/under_grip/over_grip)
2. LSL EEG → 接触時前後1.2秒のエポックデータ 
3. (フィードバック, EEGエポック) ペアを蓄積
4. DeepConvNet CNNで分類器を学習
5. 学習済み分類器で新しいEEGデータを自動分類
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from scipy import signal
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import csv
import os
import time
from collections import deque, Counter
import pickle
from datetime import datetime

# 既存システムから拡張
from systems.episode_contact_sync_system import EpisodeContactSynchronizer
from lsl_data_send.eeg_neuroadaptation_preprocessor import NeuroadaptationEEGPreprocessor

class EEGClassifierDataset(Dataset):
    """
    EEG分類器用データセット
    (EEGエポック, 明示的フィードバック) のペアを管理
    """
    
    def __init__(self, eeg_epochs, feedback_labels, transform=None):
        """
        Args:
            eeg_epochs: List of (300, 32) EEGエポック
            feedback_labels: List of feedback labels (0: success, 1: under_grip, 2: over_grip)
            transform: データ拡張・前処理関数
        """
        self.eeg_epochs = eeg_epochs
        self.feedback_labels = feedback_labels
        self.transform = transform
        
        # ラベル統計
        self.label_counts = Counter(feedback_labels)
        print(f"📊 データセット統計: {dict(self.label_counts)}")
    
    def __len__(self):
        return len(self.eeg_epochs)
    
    def __getitem__(self, idx):
        epoch = self.eeg_epochs[idx]  # (300, 32)
        label = self.feedback_labels[idx]
        
        if self.transform:
            epoch = self.transform(epoch)
            
        # CNNの入力形状に変換: (1, channels, samples)
        epoch_tensor = torch.FloatTensor(epoch.T).unsqueeze(0)  # (1, 32, 300)
        label_tensor = torch.LongTensor([label])
        
        return epoch_tensor, label_tensor.squeeze()

class DeepConvNetClassifier(nn.Module):
    """
    DeepConvNet分類器（論文準拠）
    EEGエポック → 認知競合3クラス分類
    """
    
    def __init__(self, n_channels=32, n_samples=300, n_classes=3, dropout=0.5):
        super(DeepConvNetClassifier, self).__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        
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
            nn.Dropout(dropout),
            nn.Linear(200 * 8, 256),
            nn.ELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, n_classes)
        )
        
        print(f"🧠 DeepConvNet初期化: {n_channels}ch, {n_samples}samples → {n_classes}classes")
        
    def forward(self, x):
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

class EEGClassifierTrainer:
    """
    EEG分類器の学習・評価・保存を管理
    """
    
    def __init__(self, model, device='auto'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device
        self.model = model.to(self.device)
        
        # 学習設定
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.5)
        
        # 学習履歴
        self.train_history = {'loss': [], 'accuracy': []}
        self.val_history = {'loss': [], 'accuracy': []}
        self.best_val_accuracy = 0.0
        
        print(f"🎓 学習環境: {self.device}")
    
    def train_epoch(self, train_loader):
        """1エポックの学習"""
        self.model.train()
        total_loss = 0
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
            
            if batch_idx % 10 == 0:
                print(f'   Batch {batch_idx:3d}: Loss={loss.item():.4f}, Acc={100.*correct/total:.1f}%')
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """検証"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                all_preds.extend(pred.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = val_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, all_preds, all_targets
    
    def train_full(self, train_loader, val_loader, epochs=100, early_stopping=15):
        """完全学習ループ"""
        print(f"🎓 学習開始: {epochs}エポック, Early Stopping={early_stopping}")
        
        best_epoch = 0
        epochs_without_improvement = 0
        
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch+1}/{epochs} ---")
            
            # 学習
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 検証
            val_loss, val_acc, val_preds, val_targets = self.validate(val_loader)
            
            # 学習率調整
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
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
                torch.save(self.model.state_dict(), 'models/best_eeg_classifier.pth')
                print(f"🎯 ベストモデル更新! Val Acc: {val_acc:.1f}%")
            else:
                epochs_without_improvement += 1
            
            # Early Stopping
            if epochs_without_improvement >= early_stopping:
                print(f"⏰ Early Stopping at epoch {epoch+1}")
                print(f"   ベスト: Epoch {best_epoch}, Val Acc: {self.best_val_accuracy:.1f}%")
                break
        
        return self.best_val_accuracy
    
    def evaluate_final(self, test_loader, class_names=['Success', 'Under-grip', 'Over-grip']):
        """最終評価（混同行列、分類レポート）"""
        print("\n🔍 最終評価実行...")
        
        # ベストモデル読み込み
        self.model.load_state_dict(torch.load('models/best_eeg_classifier.pth'))
        
        # テスト
        test_loss, test_acc, test_preds, test_targets = self.validate(test_loader)
        
        print(f"📊 テスト結果: Loss={test_loss:.4f}, Accuracy={test_acc:.1f}%")
        
        # 🔍 クラス分布チェック
        unique_targets = np.unique(test_targets)
        unique_preds = np.unique(test_preds)
        
        print(f"🔍 データ分布確認:")
        print(f"   真のクラス: {unique_targets}")
        print(f"   予測クラス: {unique_preds}")
        print(f"   クラス数: 真={len(unique_targets)}, 予測={len(unique_preds)}")
        
        # 単一クラス問題の検出
        if len(unique_targets) <= 1:
            print(f"⚠️ 警告: 単一クラスのみのデータです")
            print(f"   学習には最低2クラス以上のデータが必要です")
            print(f"   現在のクラス: {unique_targets}")
            
            # 単一クラス用の簡単な評価
            if len(unique_targets) == 1:
                single_class = unique_targets[0]
                class_name = class_names[single_class] if single_class < len(class_names) else f'Class_{single_class}'
                print(f"\n📈 単一クラス分析:")
                print(f"   クラス: {class_name}")
                print(f"   サンプル数: {len(test_targets)}")
                print(f"   正解率: {test_acc:.1f}%")
                
                return test_acc
        
        # 通常の詳細分析（複数クラス時）
        try:
            # 利用可能なクラス名のみ使用
            available_class_names = [class_names[i] for i in unique_targets if i < len(class_names)]
            
            print("\n📈 分類レポート:")
            report = classification_report(test_targets, test_preds, 
                                         target_names=available_class_names,
                                         labels=unique_targets)
            print(report)
            
            # 混同行列
            cm = confusion_matrix(test_targets, test_preds, labels=unique_targets)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=available_class_names, 
                       yticklabels=available_class_names)
            plt.title(f'混同行列 (Test Accuracy: {test_acc:.1f}%)')
            plt.ylabel('真の分類')
            plt.xlabel('予測分類')
            plt.tight_layout()
            
            os.makedirs('plots', exist_ok=True)
            plt.savefig(f'plots/confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            print(f"💾 混同行列保存: plots/confusion_matrix_*.png")
            # plt.show()  # コメントアウト（自動実行時のエラー回避）
            
        except Exception as e:
            print(f"⚠️ 詳細分析エラー: {e}")
            print(f"   基本評価のみ実行")
        
        # 学習曲線
        try:
            self._plot_training_curves()
        except:
            print(f"⚠️ 学習曲線描画スキップ")
        
        return test_acc
    
    def _plot_training_curves(self):
        """学習曲線の描画"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        ax1.plot(self.train_history['loss'], label='Train')
        ax1.plot(self.val_history['loss'], label='Validation')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.train_history['accuracy'], label='Train')
        ax2.plot(self.val_history['accuracy'], label='Validation')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'plots/training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.show()

class EEGClassifierDataCollector(EpisodeContactSynchronizer):
    """
    EEG分類器用データ収集システム
    TCPからの明示的フィードバック + LSLのEEGエポック ペアを収集
    """
    
    def __init__(self, *args, **kwargs):
        # データ収集設定
        self.collect_training_data = kwargs.pop('collect_training_data', True)
        self.min_samples_per_class = kwargs.pop('min_samples_per_class', 100)

        super().__init__(*args, **kwargs)

        # EEG前処理器を初期化
        self.eeg_preprocessor = NeuroadaptationEEGPreprocessor(
            sampling_rate=self.sampling_rate,
            n_channels=self.n_channels,
            epoch_duration=self.epoch_duration
        )

        # 学習データ蓄積
        self.training_data = {
            'eeg_epochs': [],           # List of (300, 32) arrays
            'feedback_labels': [],      # List of integers (0, 1, 2)
            'timestamps': [],           # List of timestamps
            'episode_info': []          # List of episode metadata
        }
        
        # フィードバック分類マッピング
        self.feedback_mapping = {
            'success': 0,
            'normal': 0,
            'good': 0,
            'correct': 0,
            'under_grip': 1,
            'weak': 1,
            'insufficient': 1,
            'light': 1,
            'over_grip': 2,
            'strong': 2,
            'excessive': 2,
            'crush': 2,
            'deform': 2
        }
        
        # 収集統計
        self.collection_stats = Counter()
        
        print(f"📚 EEG分類器データ収集システム初期化:")
        print(f"   目標: 各クラス{self.min_samples_per_class}サンプル以上")
        print(f"   フィードバックマッピング: {self.feedback_mapping}")

    def _parse_explicit_feedback(self, tcp_data):
        """
        TCPデータから明示的フィードバック解析（grip_force判定版）

        フィードバックルール:
        - 8 <= grip_force <= 15 → success (成功)
        - grip_force > 15      → over_grip (強すぎ)
        - grip_force < 8       → under_grip (弱すぎ)

        Args:
            tcp_data: Unity TCPデータ

        Returns:
            feedback_info: 解析されたフィードバック情報
        """
        feedback_info = None

        try:
            grip_force = tcp_data.get('grip_force', None)

            print(f"🔍 フィードバック状態解析:")
            print(f"   grip_force: {grip_force}")

            if grip_force is not None:
                if isinstance(grip_force, str):
                    try:
                        grip_force = float(grip_force)
                    except ValueError:
                        grip_force = None
                elif isinstance(grip_force, (int, float)):
                    grip_force = float(grip_force)

            if grip_force is not None:
                # 🎯 フィードバック判定ルール
                if 8 <= grip_force <= 15:
                    feedback_class = 'success'
                    feedback_label = 0
                    confidence = 0.9
                    reasoning = "適切な把持力（8-15N）"
                elif grip_force > 15:
                    feedback_class = 'over_grip'
                    feedback_label = 2
                    confidence = 0.95
                    reasoning = "把持力過剰（>15N）"
                else:
                    feedback_class = 'under_grip'
                    feedback_label = 1
                    confidence = 0.9
                    reasoning = "把持力不足（<8N）"

                feedback_info = {
                    'class': feedback_class,
                    'label': feedback_label,
                    'confidence': confidence,
                    'explicit': True,
                    'raw_feedback': f"grip_force={grip_force}",
                    'source_field': 'grip_force',
                    'generation_rule': reasoning
                }

                print(f"✅ フィードバック判定成功:")
                print(f"   分類: {feedback_class} (label={feedback_label})")
                print(f"   信頼度: {confidence:.2f}")
                print(f"   根拠: {reasoning}")
            else:
                print(f"❌ 必要なフィールドが見つかりません:")
                print(f"   grip_force フィールド: {grip_force}")
                print(f"   利用可能フィールド: {list(tcp_data.keys())}")

        except Exception as e:
            print(f"❌ フィードバック解析エラー: {e}")
            import traceback

            print(f"   詳細: {traceback.format_exc()}")

        return feedback_info

    def _create_synchronized_event(self, tcp_event: dict, lsl_event: dict = None):
        """
        データ収集版の同期イベント作成
        明示的フィードバック + EEGエポック ペアを収集
        """
        try:
            tcp_timestamp = tcp_event['system_time']
            episode_info = tcp_event['episode_info']
            episode_num = episode_info['episode']
            
            # 1. EEGエポック抽出
            epoch_data, epoch_timestamps, sync_latency = self._extract_epoch_around_time(
                tcp_timestamp, episode_num
            )
            
            if epoch_data is None:
                print(f"⚠️ Episode {episode_num}: EEGエポックデータ不足")
                return None
            
            # 2. エポック前処理
            preprocess_result = self.eeg_preprocessor.preprocess_epoch(epoch_data)
            epoch_data = preprocess_result['processed_epoch']


            # 基本品質評価と高度な品質指標を統合
            basic_quality = self._assess_epoch_quality(epoch_data)
            advanced_quality = preprocess_result.get('quality_metrics', {})
            epoch_quality = {**basic_quality, **advanced_quality}


            # 3. 明示的フィードバック解析
            feedback_info = self._parse_explicit_feedback(tcp_event['data'])

            if feedback_info is None:
                print(f"⚠️ Episode {episode_num}: 明示的フィードバック未検出")
                # ランダムフィードバックは生成するが学習データには使わない
                feedback_value = self._generate_random_feedback(episode_info)
            else:
                # 明示的フィードバックを学習データとして蓄積
                if self.collect_training_data:
                    self.training_data['eeg_epochs'].append(epoch_data.copy())
                    self.training_data['feedback_labels'].append(feedback_info['label'])
                    self.training_data['timestamps'].append(tcp_timestamp)
                    self.training_data['episode_info'].append(episode_info.copy())

                    # 統計更新
                    self.collection_stats[feedback_info['class']] += 1

                    print(f"💾 学習データ蓄積: Episode {episode_num}")
                    print(f"   フィードバック: {feedback_info['class']} (信頼度: {feedback_info['confidence']:.2f})")
                    print(f"   収集統計: {dict(self.collection_stats)}")

                # フィードバック値は実際の分類結果ベースで生成
                reward_mapping = {0: 25.0, 1: 15.0, 2: 5.0}  # success, under_grip, over_grip
                feedback_value = self._generate_random_feedback(episode_info)

            # 4. 同期イベント作成
            collection_sync_event = {
                'episode_number': episode_info['episode'],
                'contact_timestamp': tcp_timestamp,
                'epoch_data': epoch_data,
                'epoch_timestamps': epoch_timestamps,
                'episode_info': episode_info,
                'tcp_data': tcp_event['data'],
                
                # フィードバック情報
                'feedback_info': feedback_info,
                'feedback_value': feedback_value,
                'has_explicit_feedback': feedback_info is not None,

                # 品質評価
                'sync_latency': sync_latency,
                'epoch_quality': epoch_quality
            }
            
            return collection_sync_event
            
        except Exception as e:
            print(f"❌ データ収集同期イベント作成エラー: {e}")
            return None

    def save_training_data(self, filename=None):
        """学習データをファイルに保存"""
        if filename is None:
            filename = f'training_data/eeg_training_data_{self.session_id}.pkl'
        
        try:
            os.makedirs('training_data', exist_ok=True)
            
            # データ整合性チェック
            n_epochs = len(self.training_data['eeg_epochs'])
            n_labels = len(self.training_data['feedback_labels'])
            
            if n_epochs != n_labels:
                print(f"⚠️ データ不整合: エポック数={n_epochs}, ラベル数={n_labels}")
                return False
            
            # 保存
            with open(filename, 'wb') as f:
                pickle.dump(self.training_data, f)
            
            print(f"💾 学習データ保存完了: {filename}")
            print(f"   総サンプル数: {n_epochs}")
            print(f"   クラス別分布: {dict(self.collection_stats)}")
            
            # CSVサマリーも保存
            summary_file = filename.replace('.pkl', '_summary.csv')
            self._save_data_summary(summary_file)
            
            return True
            
        except Exception as e:
            print(f"❌ 学習データ保存エラー: {e}")
            return False

    def _save_data_summary(self, filename):
        """データ収集サマリーをCSV保存"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['episode_number', 'timestamp', 'feedback_class', 'feedback_label', 'confidence', 'sync_latency_ms', 'epoch_quality'])
                
                for i, (timestamp, episode_info, feedback_label) in enumerate(zip(
                    self.training_data['timestamps'],
                    self.training_data['episode_info'], 
                    self.training_data['feedback_labels']
                )):
                    # ラベルからクラス名を逆引き
                    class_name = {v: k for k, v in self.feedback_mapping.items() if v == feedback_label}
                    class_name = list(class_name.keys())[0] if class_name else 'unknown'
                    
                    writer.writerow([
                        episode_info['episode'],
                        timestamp,
                        class_name,
                        feedback_label,
                        1.0,  # confidence (デフォルト)
                        0.0,  # sync_latency_ms (計算省略)
                        'good'  # epoch_quality (簡略化)
                    ])
                    
        except Exception as e:
            print(f"❌ サマリーCSV保存エラー: {e}")

    def check_data_sufficiency(self):
        """データ収集十分性チェック（改良版）"""
        sufficient = True
        print(f"\n📊 データ収集状況チェック:")
        
        # 実際に収集されたラベルの分布
        actual_labels = Counter(self.training_data['feedback_labels'])
        total_samples = sum(actual_labels.values())
        
        print(f"   総サンプル数: {total_samples}")
        
        # 各クラスの状況
        class_names = {0: 'success', 1: 'under_grip', 2: 'over_grip'}
        for label, class_name in class_names.items():
            count = actual_labels.get(label, 0)
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            needed = max(0, self.min_samples_per_class - count)
            status = "OK" if count >= self.min_samples_per_class else "NG"
            
            print(f"   {class_name:12s}: {count:3d}/{self.min_samples_per_class} ({percentage:4.1f}%) {status}")
            if needed > 0:
                print(f"                    → あと{needed}サンプル必要")
                sufficient = False
        
        # 単一クラス問題の警告
        n_classes_present = len([c for c in actual_labels.values() if c > 0])
        if n_classes_present <= 1:
            print(f"重要: 現在{n_classes_present}クラスのみ検出")
            print(f"   分類器学習には最低2クラス以上必要です")
            print(f"   現在の自動ラベル生成ルールを確認してください")
            sufficient = False
        elif n_classes_present == 2:
            print(f"注意: 現在{n_classes_present}クラスのみ検出")
            print(f"   より良い性能には3クラス全てが推奨されます")
        
        if sufficient:
            print(f"データ収集完了！分類器学習を開始できます。")
        else:
            if n_classes_present <= 1:
                print(f"データ不足: より多様なシナリオでデータ収集が必要")
            else:
                print(f"データ収集継続中...")
        
        return sufficient and n_classes_present >= 2
        

    def run_data_collection_session(self, duration_seconds=18000, auto_train=True):
        """
        データ収集セッション実行
        
        Args:
            duration_seconds: 最大収集時間（デフォルト30分）
            auto_train: データ十分時に自動学習開始
        """
        if not self.start_synchronization_system():
            return None
            
        print(f"📚 EEG分類器データ収集セッション開始")
        print(f"⏱️ 最大収集時間: {duration_seconds}秒 ({duration_seconds//60}分)")
        print(f"🎯 各クラス目標: {self.min_samples_per_class}サンプル")
        
        start_time = time.time()
        last_check_time = start_time
        
        try:
            while self.is_running:
                elapsed = time.time() - start_time
                
                # 30秒ごとに進捗確認
                if elapsed - (last_check_time - start_time) >= 30:
                    sufficient = self.check_data_sufficiency()
                    
                    if sufficient and auto_train:
                        print(f"🎓 データ十分！自動学習開始...")
                        break
                    
                    last_check_time = time.time()
                
                # 終了条件チェック
                if elapsed >= duration_seconds:
                    print(f"⏰ 制限時間到達（{duration_seconds}秒）")
                    break
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print(f"\n⚡ ユーザー中断")
        finally:
            print(f"🔚 データ収集終了...")
            self.stop_synchronization_system()
            
            # データ保存
            saved = self.save_training_data()
            
            if saved and auto_train:
                # 自動学習実行
                return self.train_classifier_from_collected_data()
            
            return saved

    def train_classifier_from_collected_data(self):
        """収集したデータから分類器を学習"""
        print(f"\n🎓 EEG分類器学習開始...")
        
        if len(self.training_data['eeg_epochs']) == 0:
            print(f"❌ 学習データがありません")
            return None
        
        # データセット作成
        dataset = EEGClassifierDataset(
            self.training_data['eeg_epochs'],
            self.training_data['feedback_labels']
        )
        
        # データ分割
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size) 
        test_size = total_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size]
        )
        
        print(f"📊 データ分割: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # DataLoader作成
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # モデル作成
        model = DeepConvNetClassifier(n_channels=32, n_samples=300, n_classes=3)
        trainer = EEGClassifierTrainer(model)
        
        # 学習実行
        print(f"🚀 学習開始...")
        best_accuracy = trainer.train_full(
            train_loader, val_loader, 
            epochs=100, early_stopping=15
        )
        
        # 最終評価
        print(f"\n🔍 最終テスト評価...")
        test_accuracy = trainer.evaluate_final(test_loader)
        
        print(f"\n🎉 学習完了!")
        print(f"   ベスト検証精度: {best_accuracy:.1f}%")
        print(f"   最終テスト精度: {test_accuracy:.1f}%")
        
        # 学習済みモデル情報を返す
        return {
            'model_path': 'models/best_eeg_classifier.pth',
            'val_accuracy': best_accuracy,
            'test_accuracy': test_accuracy,
            'total_samples': total_size,
            'class_distribution': dict(self.collection_stats)
        }

class EEGDataAugmentation:
    """
    EEGデータ拡張クラス
    少ないデータから学習効果を最大化
    """
    
    @staticmethod
    def add_noise(epoch, noise_level=0.05):
        """ガウシアンノイズ追加"""
        noise = np.random.normal(0, noise_level, epoch.shape)
        return epoch + noise
    
    @staticmethod
    def time_shift(epoch, max_shift_samples=10):
        """時間軸シフト"""
        shift = np.random.randint(-max_shift_samples, max_shift_samples + 1)
        if shift > 0:
            shifted = np.zeros_like(epoch)
            shifted[shift:, :] = epoch[:-shift, :]
        elif shift < 0:
            shifted = np.zeros_like(epoch)
            shifted[:shift, :] = epoch[-shift:, :]
        else:
            shifted = epoch
        return shifted
    
    @staticmethod
    def channel_dropout(epoch, dropout_prob=0.1):
        """チャンネルドロップアウト"""
        mask = np.random.random(epoch.shape[1]) > dropout_prob
        augmented = epoch.copy()
        augmented[:, ~mask] = 0
        return augmented
    
    @staticmethod
    def amplitude_scaling(epoch, scale_range=(0.8, 1.2)):
        """振幅スケーリング"""
        scale = np.random.uniform(*scale_range)
        return epoch * scale

def preprocess_eeg_epoch_for_training(epoch_data, sampling_rate=250):
    """
    学習用EEGエポック前処理（論文準拠）
    
    Args:
        epoch_data: (300, 32) EEGエポック
        sampling_rate: サンプリング周波数
        
    Returns:
        processed_epoch: 前処理済みエポック
    """
    # バンドパスフィルタ (2-50Hz)
    sos = signal.butter(5, [2, 50], btype='band', fs=sampling_rate, output='sos')
    filtered_epoch = np.zeros_like(epoch_data)
    
    for ch in range(epoch_data.shape[1]):
        try:
            filtered_epoch[:, ch] = signal.sosfilt(sos, epoch_data[:, ch])
        except:
            filtered_epoch[:, ch] = epoch_data[:, ch]  # フィルタリング失敗時は元データ
    
    # Artifact Subspace Reconstruction (ASR) の簡易版
    # 極端な外れ値チャンネルの除去
    for ch in range(filtered_epoch.shape[1]):
        ch_data = filtered_epoch[:, ch]
        if np.std(ch_data) > 0:
            z_scores = np.abs((ch_data - np.mean(ch_data)) / np.std(ch_data))
            if np.max(z_scores) > 5:  # 5σを超える外れ値
                print(f"⚠️ Channel {ch}: 外れ値検出、ゼロ化")
                filtered_epoch[:, ch] = 0
    
    # 正規化 (チャンネルごとZ-score)
    for ch in range(filtered_epoch.shape[1]):
        ch_data = filtered_epoch[:, ch]
        if np.std(ch_data) > 1e-10:  # ゼロ除算回避
            filtered_epoch[:, ch] = (ch_data - np.mean(ch_data)) / np.std(ch_data)
    
    return filtered_epoch

# 🎓 学習実行スクリプト
def run_complete_training_pipeline():
    """
    完全な学習パイプラインの実行
    1. データ収集 → 2. 学習 → 3. 評価 → 4. 強化学習統合
    """
    print("🎓 EEG認知競合分類器 完全学習パイプライン")
    print("=" * 70)
    
    # Step 1: データ収集
    print("\n📚 Step 1: 学習データ収集")
    data_collector = EEGClassifierDataCollector(
        tcp_host='127.0.0.1',
        tcp_port=12345,
        lsl_stream_name='MockEEG',
        contact_buffer_duration=1.5,
        min_samples_per_class=50,  # 各クラス50サンプル
        collect_training_data=True
    )
    
    # データ収集セッション
    training_result = data_collector.run_data_collection_session(
        duration_seconds=1800,  # 30分間
        auto_train=True
    )
    
    if training_result is None:
        print("❌ データ収集失敗")
        return
    
    # Step 2: 学習結果確認
    print(f"\n✅ 学習完了!")
    print(f"   モデルパス: {training_result['model_path']}")
    print(f"   テスト精度: {training_result['test_accuracy']:.1f}%")
    print(f"   学習サンプル数: {training_result['total_samples']}")
    
    # Step 3: 強化学習統合システムで実用化
    if training_result['test_accuracy'] > 70:  # 70%以上で実用可能
        print(f"\n🤖 Step 3: 強化学習システム統合")
        
        eeg_rl_system = EEGReinforcementLearningSystem(
            tcp_host='127.0.0.1',
            tcp_port=12345,
            lsl_stream_name='MockEEG',
            contact_buffer_duration=1.5,
            eeg_model_path=training_result['model_path'],
            enable_eeg_classification=True
        )
        
        print(f"🚀 EEG強化学習セッション開始...")
        eeg_rl_system.run_eeg_reinforcement_learning_session(
            duration_seconds=1200,  # 20分間
            target_episodes=100
        )
    else:
        print(f"⚠️ 分類精度が低すぎます ({training_result['test_accuracy']:.1f}%)")
        print(f"   より多くのデータ収集が必要です")

if __name__ == "__main__":
    print("🧠 EEG認知競合分類器学習システム")
    print("TCPからの明示的フィードバックでLSL EEGデータを学習")
    
    # 選択メニュー
    print("\n選択してください:")
    print("1. データ収集のみ実行")
    print("2. 既存データから学習のみ実行") 
    
    choice = input("選択 (1/2): ").strip()
    
    if choice == "1":
        # データ収集のみ
        collector = EEGClassifierDataCollector(
            tcp_host='127.0.0.1',
            tcp_port=12345,
            lsl_stream_name='MockEEG',
            contact_buffer_duration=1.5,
            min_samples_per_class=100
        )
        collector.run_data_collection_session(duration_seconds=1800, auto_train=False)
        
    elif choice == "2":
        # 既存データから学習
        data_file = input("データファイルパス (.pkl): ").strip()
        
        try:
            with open(data_file, 'rb') as f:
                training_data = pickle.load(f)
            
            dataset = EEGClassifierDataset(
                training_data['eeg_epochs'],
                training_data['feedback_labels']
            )
            
            # データ分割・学習実行
            total_size = len(dataset)
            train_size = int(0.7 * total_size)
            val_size = int(0.15 * total_size)
            test_size = total_size - train_size - val_size
            
            train_dataset, val_dataset, test_dataset = random_split(
                dataset, [train_size, val_size, test_size]
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            model = DeepConvNetClassifier()
            trainer = EEGClassifierTrainer(model)
            
            trainer.train_full(train_loader, val_loader, epochs=100)
            trainer.evaluate_final(test_loader)
            
        except Exception as e:
            print(f"❌ 学習エラー: {e}")
    
    elif choice == "3":
        # 完全パイプライン
        run_complete_training_pipeline()
    
    else:
        print("❌ 無効な選択です")

"""
🧠 使用方法:

1. **データ収集フェーズ**:
   - Unity側で明示的フィードバック送信
   - TCP: {"feedback": "success"} / {"feedback": "over_grip"} 等
   - 自動でEEGエポックとペアにして蓄積

2. **学習フェーズ**: 
   - 収集データでDeepConvNet CNN学習
   - 70%以上の精度達成を目標

3. **実用フェーズ**:
   - 学習済み分類器で新しいEEGを自動分類
   - CC-DDPG強化学習でロボット制御最適化

📊 **期待される精度**: 論文では80-90%の分類精度を達成
🎯 **最小データ要件**: 各クラス50-100サンプル（合計150-300エピソード）
⚡ **リアルタイム性能**: 接触検出から50ms以内で分類・制御
"""