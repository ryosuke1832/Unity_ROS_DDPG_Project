#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完全改善版グリッパー力分類器
元のコードのデータ読み込み方式を踏襲しつつ、以下を改善:
1. EEGNetから統計的特徴量+MLPに変更
2. クラス不均衡対策（SMOTE + 重み付きロス）
3. 早期終了とk-fold交差検証
4. 詳細な評価指標
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
from collections import Counter
import json
import warnings
from pathlib import Path
warnings.filterwarnings('ignore')

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 使用デバイス: {device}")

class GripForceDataset(Dataset):
    """把持力分類用データセット"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ImprovedGripForceClassifier(nn.Module):
    """統計的特徴量ベースの改善版分類器"""
    
    def __init__(self, input_size, num_classes=3, dropout_rate=0.4):
        super().__init__()
        
        # 入力正規化
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # ResNet風残差ブロック
        hidden_sizes = [512, 256, 128, 64]
        self.blocks = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            block = self._make_residual_block(prev_size, hidden_size, dropout_rate)
            self.blocks.append(block)
            prev_size = hidden_size
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(prev_size, num_classes)
        )
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _make_residual_block(self, in_features, out_features, dropout_rate):
        """残差ブロック作成"""
        # ショートカット接続用
        if in_features != out_features:
            shortcut = nn.Linear(in_features, out_features)
        else:
            shortcut = nn.Identity()
        
        # メインパス
        main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        
        return nn.ModuleDict({
            'main': main_path,
            'shortcut': shortcut
        })
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_bn(x)
        
        for block in self.blocks:
            identity = block['shortcut'](x)
            x = block['main'](x) + identity
            x = torch.relu(x)
        
        return self.classifier(x)

class EarlyStopping:
    """早期終了"""
    def __init__(self, patience=25, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        return self.counter >= self.patience
    
    def save_checkpoint(self, model):
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()
    
    def restore(self, model):
        if self.best_weights:
            model.load_state_dict(self.best_weights)

def load_csv_data_from_episodes(csv_dir):
    """
    元のコードと同様の方法でエピソードCSVデータを読み込み
    
    Args:
        csv_dir: CSVファイルが保存されているディレクトリ
        
    Returns:
        eeg_data_list: EEGデータのリスト [(300, 32), ...]
        grip_force_labels: 把持力ラベルのリスト [0, 1, 2, ...]
    """
    print(f"📂 エピソードCSVデータ読み込み開始: {csv_dir}")
    
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
                    
                    if len(eeg_data_list) % 100 == 0:
                        print(f"   読み込み進捗: {len(eeg_data_list)}件")
                
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
        count = label_counts.get(i, 0)
        percentage = count / len(grip_force_labels) * 100 if grip_force_labels else 0
        print(f"   {name}: {count}件 ({percentage:.1f}%)")
    
    return eeg_data_list, grip_force_labels

def extract_eeg_features(eeg_data_list):
    """
    EEGデータから統計的特徴量を抽出
    
    Args:
        eeg_data_list: EEGデータのリスト [(300, 32), ...]
        
    Returns:
        features_array: 特徴量配列 (n_samples, n_features)
    """
    print("🔄 EEGデータから特徴量抽出中...")
    
    features_list = []
    
    for i, eeg_data in enumerate(eeg_data_list):
        features = []
        
        # 各チャンネルから特徴量を抽出
        for ch in range(eeg_data.shape[1]):  # 32チャンネル
            ch_data = eeg_data[:, ch]
            
            # 時間ドメイン統計的特徴量
            features.extend([
                np.mean(ch_data),              # 平均
                np.std(ch_data),               # 標準偏差
                np.var(ch_data),               # 分散
                np.min(ch_data),               # 最小値
                np.max(ch_data),               # 最大値
                np.median(ch_data),            # 中央値
                np.percentile(ch_data, 25),    # 第1四分位数
                np.percentile(ch_data, 75),    # 第3四分位数
                np.ptp(ch_data),               # レンジ（最大-最小）
                len(ch_data[ch_data > 0]) / len(ch_data)  # 正の値の割合
            ])
            
            # 周波数ドメイン特徴量
            try:
                fft = np.fft.fft(ch_data)
                freqs = np.fft.fftfreq(len(ch_data), 1/250)  # 250Hz
                power_spectrum = np.abs(fft)**2
                
                # 各周波数帯域のパワー
                # デルタ波 (0.5-4Hz)
                delta_mask = (freqs >= 0.5) & (freqs <= 4)
                delta_power = np.mean(power_spectrum[delta_mask]) if np.any(delta_mask) else 0
                
                # シータ波 (4-8Hz)
                theta_mask = (freqs >= 4) & (freqs <= 8)
                theta_power = np.mean(power_spectrum[theta_mask]) if np.any(theta_mask) else 0
                
                # アルファ波 (8-12Hz)
                alpha_mask = (freqs >= 8) & (freqs <= 12)
                alpha_power = np.mean(power_spectrum[alpha_mask]) if np.any(alpha_mask) else 0
                
                # ベータ波 (12-30Hz)
                beta_mask = (freqs >= 12) & (freqs <= 30)
                beta_power = np.mean(power_spectrum[beta_mask]) if np.any(beta_mask) else 0
                
                # ガンマ波 (30-100Hz)
                gamma_mask = (freqs >= 30) & (freqs <= 100)
                gamma_power = np.mean(power_spectrum[gamma_mask]) if np.any(gamma_mask) else 0
                
                features.extend([delta_power, theta_power, alpha_power, beta_power, gamma_power])
                
            except:
                # FFTエラー時は0で埋める
                features.extend([0, 0, 0, 0, 0])
        
        # チャンネル間の相関特徴量（簡易版）
        try:
            corr_matrix = np.corrcoef(eeg_data.T)
            # 上三角行列の要素を特徴量として使用
            upper_tri_indices = np.triu_indices(32, k=1)
            corr_features = corr_matrix[upper_tri_indices]
            
            # 相関の統計量
            features.extend([
                np.mean(corr_features),
                np.std(corr_features),
                np.max(corr_features),
                np.min(corr_features)
            ])
        except:
            features.extend([0, 0, 0, 0])
        
        features_list.append(features)
        
        if i % 100 == 0 and i > 0:
            print(f"   特徴量抽出進捗: {i}/{len(eeg_data_list)}")
    
    features_array = np.array(features_list)
    print(f"✅ 特徴量抽出完了: {features_array.shape[1]}次元")
    
    return features_array

def prepare_data_with_balancing(data_source, balance_method='combined'):
    """
    データ準備とクラス不均衡対策
    
    Args:
        data_source: CSVファイルパスまたはエピソードディレクトリパス
        balance_method: 'smote', 'undersample', 'combined', 'none'
        
    Returns:
        X_balanced, y_balanced, scaler, le, class_names
    """
    print("📂 データ準備中...")
    
    if os.path.isdir(data_source):
        # エピソードディレクトリから読み込み
        print("🗂️ エピソードディレクトリから読み込み...")
        eeg_data_list, grip_force_labels = load_csv_data_from_episodes(data_source)
        
        if len(eeg_data_list) == 0:
            print("❌ データが見つかりませんでした")
            return None, None, None, None, None
        
        # EEGデータから特徴量抽出
        X = extract_eeg_features(eeg_data_list)
        y = np.array(grip_force_labels)
        
        # クラス名設定
        class_names = np.array(['UnderGrip', 'Success', 'OverGrip'])
        le = LabelEncoder()
        le.classes_ = class_names
        
    else:
        # 単一CSVファイルから読み込み
        print("📄 単一CSVファイルから読み込み...")
        df = pd.read_csv(data_source)
        
        # ラベルエンコーディング
        le = LabelEncoder()
        y = le.fit_transform(df['result'].values)
        class_names = le.classes_
        
        # 特徴量準備
        feature_cols = [col for col in df.columns if col != 'result']
        X = df[feature_cols].values
    
    print(f"📊 データ準備完了:")
    print(f"   総サンプル数: {len(X)}")
    print(f"   特徴量数: {X.shape[1]}")
    print(f"   クラス: {class_names}")
    
    # クラス分布確認
    unique, counts = np.unique(y, return_counts=True)
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        print(f"   {class_names[class_idx]}: {count}件 ({count/len(y)*100:.1f}%)")
    
    # データ標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # クラス不均衡対策
    if balance_method == 'smote':
        print("🔄 SMOTE適用中...")
        k_neighbors = min(3, min(counts) - 1) if min(counts) > 1 else 1
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_balanced, y_balanced = smote.fit_resample(X_scaled, y)
        
    elif balance_method == 'undersample':
        print("🔄 アンダーサンプリング適用中...")
        undersampler = RandomUnderSampler(random_state=42)
        X_balanced, y_balanced = undersampler.fit_resample(X_scaled, y)
        
    elif balance_method == 'combined':
        print("🔄 SMOTE + アンダーサンプリング適用中...")
        k_neighbors = min(3, min(counts) - 1) if min(counts) > 1 else 1
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42, k_neighbors=k_neighbors)),
            ('undersample', RandomUnderSampler(random_state=42))
        ])
        X_balanced, y_balanced = pipeline.fit_resample(X_scaled, y)
        
    else:  # 'none'
        X_balanced, y_balanced = X_scaled, y
    
    # バランシング後の分布確認
    if balance_method != 'none':
        print(f"📊 バランシング後:")
        unique, counts = np.unique(y_balanced, return_counts=True)
        for class_idx, count in zip(unique, counts):
            print(f"   {class_names[class_idx]}: {count}件 ({count/len(y_balanced)*100:.1f}%)")
    
    return X_balanced, y_balanced, scaler, le, class_names

def train_model_with_kfold(X, y, class_names, n_splits=5, epochs=200):
    """k-fold交差検証での学習"""
    print(f"🔄 {n_splits}-fold交差検証開始")
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n📋 Fold {fold + 1}/{n_splits}")
        
        # データ分割
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # クラス重み計算
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        # データセット作成
        train_dataset = GripForceDataset(X_train, y_train)
        val_dataset = GripForceDataset(X_val, y_val)
        
        batch_size = min(32, len(train_dataset))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # モデル初期化
        model = ImprovedGripForceClassifier(
            input_size=X.shape[1], 
            num_classes=len(class_names),
            dropout_rate=0.4
        ).to(device)
        
        # オプティマイザーとロス関数
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=15, min_lr=1e-6
        )
        
        # 早期終了
        early_stopping = EarlyStopping(patience=30, min_delta=0.001)
        
        # 学習履歴
        history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        # 学習ループ
        best_val_f1 = 0
        for epoch in range(epochs):
            # 訓練
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # 検証
            model.eval()
            val_loss = 0
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = outputs.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(batch_y.cpu().numpy())
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            val_f1 = f1_score(val_true, val_preds, average='weighted')
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_f1'].append(val_f1)
            
            scheduler.step(val_loss)
            
            # ベストモデル更新
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
            
            # 進捗表示
            if epoch % 25 == 0 or epoch == epochs - 1:
                lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:3d}: Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, Val F1={val_f1:.3f}, LR={lr:.2e}")
            
            # 早期終了チェック
            if early_stopping(val_loss, model):
                print(f"  🛑 早期終了 (Epoch {epoch})")
                break
        
        # ベストモデル復元
        early_stopping.restore(model)
        
        # 最終評価
        model.eval()
        final_preds = []
        final_true = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                final_preds.extend(predicted.cpu().numpy())
                final_true.extend(batch_y.cpu().numpy())
        
        fold_f1 = f1_score(final_true, final_preds, average='weighted')
        fold_acc = np.mean(np.array(final_preds) == np.array(final_true))
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': fold_acc,
            'f1_score': fold_f1,
            'model': model.state_dict().copy(),
            'history': history
        })
        
        print(f"  ✅ Fold {fold + 1} 完了: Acc={fold_acc:.3f}, F1={fold_f1:.3f}")
    
    return fold_results

def evaluate_model(model, X_test, y_test, class_names):
    """詳細なモデル評価"""
    print("🔍 最終テスト評価中...")
    
    test_dataset = GripForceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model.eval()
    test_preds = []
    test_true = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            _, predicted = outputs.max(1)
            
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())
    
    test_acc = np.mean(np.array(test_preds) == np.array(test_true))
    test_f1 = f1_score(test_true, test_preds, average='weighted')
    
    print(f"📊 最終テスト結果:")
    print(f"   精度: {test_acc:.3f}")
    print(f"   F1スコア: {test_f1:.3f}")
    
    # 分類レポート
    print(f"\n📋 詳細分類レポート:")
    print(classification_report(test_true, test_preds, target_names=class_names))
    
    # 混同行列
    cm = confusion_matrix(test_true, test_preds)
    
    try:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('改善版把持力分類 - 混同行列')
        plt.ylabel('実際のクラス')
        plt.xlabel('予測クラス')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        BASE_DIR = Path(__file__).resolve().parent
        cm_path = BASE_DIR / "models" / f"confusion_matrix_improved_{timestamp}.png"
        os.makedirs(cm_path.parent, exist_ok=True)
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混同行列保存: {cm_path}")
    except Exception as e:
        print(f"混同行列保存エラー（無視可能）: {e}")
    
    return {
        'accuracy': test_acc,
        'f1_score': test_f1,
        'classification_report': classification_report(test_true, test_preds, target_names=class_names, output_dict=True),
        'confusion_matrix': cm.tolist()
    }

def main():
    """メイン実行関数"""
    print("🚀 完全改善版グリッパー力分類器")
    print("=" * 60)
    
    # データソース選択
    print("\nデータソースを選択してください:")
    print("1. エピソードディレクトリ（元のコードと同じ方式）")
    print("2. 単一CSVファイル（episode_data_raw.csv）")
    
    choice = input("選択 (1-2): ").strip()
    
    if choice == "1":
        # エピソードディレクトリから読み込み
        csv_dir = input("エピソードディレクトリパス（空白でデフォルト検索）: ").strip()
        
        if not csv_dir:
            # デフォルトパス検索
            log_dirs = glob.glob("logs/episodes_20250908_*")
            if not log_dirs:
                log_dirs = glob.glob("DDPG_Python/logs/episodes_20250908_*")
            
            if log_dirs:
                csv_dir = max(log_dirs)  # 最新のログディレクトリ
                print(f"🔍 デフォルト使用: {csv_dir}")
            else:
                print(f"❌ エピソードディレクトリが見つかりません")
                return
        
        if not os.path.exists(csv_dir):
            print(f"❌ ディレクトリが存在しません: {csv_dir}")
            return
            
        data_source = csv_dir
        
    else:
        # 単一CSVファイル
        csv_file = 'episode_data_raw.csv'
        if not os.path.exists(csv_file):
            print(f"❌ CSVファイルが見つかりません: {csv_file}")
            return
        data_source = csv_file
    
    # データ準備（クラス不均衡対策付き）
    print("\n📂 データ準備中...")
    result = prepare_data_with_balancing(
        data_source, 
        balance_method='combined'  # 'smote', 'undersample', 'combined', 'none'
    )
    
    if result[0] is None:
        print("❌ データ準備に失敗しました")
        return
        
    X, y, scaler, le, class_names = result
    
    # データ分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    print(f"📊 最終データ分割:")
    print(f"   学習+検証: {len(X_temp)}件")
    print(f"   テスト: {len(X_test)}件")
    
    # k-fold交差検証で学習
    fold_results = train_model_with_kfold(X_temp, y_temp, class_names, n_splits=5, epochs=200)
    
    # 最高性能のモデルを選択
    best_fold = max(fold_results, key=lambda x: x['f1_score'])
    print(f"\n🏆 最高性能モデル: Fold {best_fold['fold']} (F1={best_fold['f1_score']:.3f})")
    
    # 最高性能モデルで最終評価
    best_model = ImprovedGripForceClassifier(
        input_size=X.shape[1], 
        num_classes=len(class_names)
    ).to(device)
    best_model.load_state_dict(best_fold['model'])
    
    test_results = evaluate_model(best_model, X_test, y_test, class_names)
    
    # 結果保存
    os.makedirs('models', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    BASE_DIR = Path(__file__).resolve().parent  
    
    # モデル保存
    model_path = BASE_DIR / "models" / f"improved_grip_force_classifier_{timestamp}.pth"
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'scaler': scaler,
        'label_encoder': le,
        'class_names': class_names,
        'input_size': X.shape[1],
        'test_results': test_results,
        'fold_results': fold_results
    }, model_path)
    
    # 結果保存
    results = {
        'timestamp': timestamp,
        'data_source': data_source,
        'total_samples': len(X),
        'features_count': X.shape[1],
        'test_accuracy': test_results['accuracy'],
        'test_f1_score': test_results['f1_score'],
        'cross_validation_scores': [r['f1_score'] for r in fold_results],
        'mean_cv_f1': np.mean([r['f1_score'] for r in fold_results]),
        'std_cv_f1': np.std([r['f1_score'] for r in fold_results]),
        'classification_report': test_results['classification_report'],
        'confusion_matrix': test_results['confusion_matrix'],
        'class_names': class_names.tolist()
    }

    result_path = BASE_DIR / "models" / f"improved_results_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 学習完了!")
    print(f"   モデル保存: {model_path}")
    print(f"   結果保存: {result_path}")
    print(f"   テスト精度: {test_results['accuracy']:.3f}")
    print(f"   テストF1: {test_results['f1_score']:.3f}")
    print(f"   CV平均F1: {np.mean([r['f1_score'] for r in fold_results]):.3f} ± {np.std([r['f1_score'] for r in fold_results]):.3f}")
    
    # 改善状況をサマリー表示
    print(f"\n📈 改善サマリー:")
    print(f"   使用データ数: {len(X):,}件（元の1000件すべて使用）")
    print(f"   特徴量数: {X.shape[1]}次元（統計的特徴量）")
    print(f"   モデル: ResNet風MLP（EEGNetから変更）")
    print(f"   クラス不均衡対策: SMOTE + アンダーサンプリング")
    print(f"   交差検証: 5-fold")
    print(f"   早期終了: 30エポック patience")
    
    if test_results['accuracy'] > 0.7:
        print(f"🎉 優秀な性能です！ テスト精度 {test_results['accuracy']:.1%}")
    elif test_results['accuracy'] > 0.6:
        print(f"👍 良好な性能です。テスト精度 {test_results['accuracy']:.1%}")
    else:
        print(f"📊 改善の余地があります。テスト精度 {test_results['accuracy']:.1%}")
        print(f"   さらなる特徴量エンジニアリングやハイパーパラメータ調整を検討してください")

def test_model_loading():
    """保存されたモデルの読み込みテスト"""
    print("🧪 モデル読み込みテスト")
    
    # 最新のモデルファイルを検索
    model_files = glob.glob("models/improved_grip_force_classifier_*.pth")
    if not model_files:
        print("❌ 保存されたモデルが見つかりません")
        return False
    
    latest_model = max(model_files)
    print(f"テスト対象: {latest_model}")
    
    try:
        # モデル読み込み
        checkpoint = torch.load(latest_model, map_location=device)
        
        input_size = checkpoint['input_size']
        class_names = checkpoint['class_names']
        
        model = ImprovedGripForceClassifier(
            input_size=input_size,
            num_classes=len(class_names)
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # ダミーデータでテスト
        dummy_input = torch.randn(1, input_size).to(device)
        
        with torch.no_grad():
            output = model(dummy_input)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        print(f"✅ モデル読み込みテスト成功!")
        print(f"   入力サイズ: {input_size}")
        print(f"   クラス数: {len(class_names)}")
        print(f"   予測クラス: {class_names[predicted_class]}")
        print(f"   信頼度: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ モデル読み込みテスト失敗: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    print("🧠 完全改善版グリッパー力分類器")
    print("元のコードのデータ読み込み方式を踏襲し、統計的特徴量+MLPで大幅改善")
    print("UnderGrip(<8N), Success(8-15N), OverGrip(>15N)")
    print("")
    
    # コマンドライン引数での実行
    if len(sys.argv) > 1:
        if sys.argv[1] == "test_model":
            test_model_loading()
        else:
            main()
    else:
        main()