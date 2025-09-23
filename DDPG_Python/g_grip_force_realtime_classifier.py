#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リアルタイム把持力分類システム

f_grip_force_classifier_improved.py で学習した分類機を使用して、
e_tcp_lsl_sync_system.py で取得したリアルタイムエピソードデータを分類

機能:
1. 改善版分類機モデルの読み込み
2. リアルタイムLSL/TCPデータの取得
3. エピソードごとの把持力分類（UnderGrip/Success/OverGrip）
4. 分類結果のリアルタイム表示・保存
5. 統計情報とパフォーマンス分析
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import json
import time
import threading
import queue
from datetime import datetime
from collections import deque, Counter
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 既存モジュールをインポート
from e_tcp_lsl_sync_system import LSLTCPEpisodeCollector, Episode
from f_grip_force_classifier_improved import ImprovedGripForceClassifier

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 デバイス: {device}")

class RealtimeGripForceClassifier:
    """リアルタイム把持力分類システム"""
    
    def __init__(self, model_path=None, lsl_stream_name='MockEEG', tcp_host='127.0.0.1', tcp_port=12345):
        """
        初期化
        
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
        
        # モデル関連
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = None
        self.input_size = None
        
        # データ収集システム
        self.episode_collector = None
        
        # 分類結果保存
        self.classification_results = []
        self.classification_queue = queue.Queue()
        
        # 実行制御
        self.is_running = False
        self.classification_thread = None
        
        # 統計情報
        self.stats = {
            'total_episodes': 0,
            'total_classifications': 0,
            'class_counts': {'UnderGrip': 0, 'Success': 0, 'OverGrip': 0},
            'avg_confidence': 0.0,
            'avg_processing_time_ms': 0.0,
            'start_time': None
        }
        
        # 結果保存ディレクトリ
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"DDPG_Python/logs/realtime_classification_{self.session_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🧠 リアルタイム把持力分類システム初期化")
        print(f"   セッションID: {self.session_id}")
        print(f"   出力ディレクトリ: {self.output_dir}")

    def classify_features(self, features: np.ndarray):
        """
        特徴量から分類確率を取得（ドキュメント5で要求されたメソッド）
        
        Args:
            features: 抽出された特徴量
            
        Returns:
            dict: 分類結果（probabilities含む）
        """
        if self.model is None or self.scaler is None or self.class_names is None:
            print("⚠️ 分類器が初期化されていません - フォールバック確率を返します")
            return {
                'probabilities': {
                    'UnderGrip': 1/3, 
                    'Success': 1/3, 
                    'OverGrip': 1/3
                }
            }
        
        try:
            # 入力次元合わせ
            if features.shape[0] != self.input_size:
                if features.shape[0] < self.input_size:
                    features = np.pad(features, (0, self.input_size - features.shape[0]), 'constant')
                else:
                    features = features[:self.input_size]
            
            # 正規化
            X = self.scaler.transform(features.reshape(1, -1))
            
            # 推論実行
            with torch.no_grad():
                logits = self.model(torch.FloatTensor(X).to(device))
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
            # 結果整形
            probabilities = dict(zip(self.class_names, probs))
            
            return {
                'probabilities': probabilities
            }
            
        except Exception as e:
            print(f"⚠️ 分類実行エラー: {e}")
            return {
                'probabilities': {
                    'UnderGrip': 1/3, 
                    'Success': 1/3, 
                    'OverGrip': 1/3
                }
            }
    
    def load_model(self, model_path=None):
        """学習済みモデルの読み込み"""
        if model_path is None:
            # 最新のモデルファイルを自動検索
            model_files = glob.glob("DDPG_Python/models/improved_grip_force_classifier_*.pth")
            if not model_files:
                model_files = glob.glob("models/improved_grip_force_classifier_*.pth")
            
            if not model_files:
                print("❌ 学習済みモデルが見つかりません")
                print("   f_grip_force_classifier_improved.py で分類機を学習してください")
                return False
            
            model_path = max(model_files)  # 最新のモデル
            print(f"🔍 最新モデル自動選択: {model_path}")
        
        try:
            print(f"📂 モデル読み込み中: {model_path}")
            
            # モデルファイル読み込み（PyTorch 2.6対応）
            try:
                # 方法1: 安全なグローバルを指定して読み込み
                with torch.serialization.safe_globals([
                    StandardScaler, 
                    LabelEncoder,
                    np.ndarray,
                    np.float64,
                    np.int64
                ]):
                    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
                print(f"✅ 安全モードで読み込み成功")
            except Exception as e1:
                print(f"⚠️ 安全モード読み込み失敗: {e1}")
                try:
                    # 方法2: weights_only=Falseで読み込み（信頼できるソース）
                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                    print(f"✅ 互換モードで読み込み成功")
                except Exception as e2:
                    print(f"❌ 互換モード読み込み失敗: {e2}")
                    raise e2
            
            # モデル情報取得
            self.input_size = checkpoint['input_size']
            self.class_names = checkpoint['class_names']
            self.scaler = checkpoint['scaler']
            self.label_encoder = checkpoint['label_encoder']
            
            # モデル構築
            self.model = ImprovedGripForceClassifier(
                input_size=self.input_size,
                num_classes=len(self.class_names)
            ).to(device)
            
            # 学習済み重み読み込み
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # テスト結果取得（参考用）
            test_results = checkpoint.get('test_results', {})
            test_accuracy = test_results.get('accuracy', 'unknown')
            test_f1 = test_results.get('f1_score', 'unknown')
            
            print(f"✅ モデル読み込み完了:")
            print(f"   入力サイズ: {self.input_size}次元")
            print(f"   クラス数: {len(self.class_names)}クラス")
            print(f"   クラス名: {list(self.class_names)}")
            print(f"   学習時テスト精度: {test_accuracy}")
            print(f"   学習時F1スコア: {test_f1}")
            
            self.model_path = model_path
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込み失敗: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extract_eeg_features(self, eeg_data):
        """
        EEGデータから特徴量を抽出（改善版分類機と同じ処理）
        
        Args:
            eeg_data: EEGデータ (samples, channels)
            
        Returns:
            features: 特徴量配列 (n_features,)
        """
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
        
        return np.array(features)
    
    def classify_episode(self, episode):
        """
        エピソードデータを分類
        
        Args:
            episode: Episodeオブジェクト
            
        Returns:
            classification_result: 分類結果辞書
        """
        start_time = time.time()
        
        try:
            # EEGデータから特徴量抽出
            eeg_features = self.extract_eeg_features(episode.lsl_data)
            
            # 特徴量次元チェック
            if len(eeg_features) != self.input_size:
                print(f"⚠️ 特徴量次元不一致: {len(eeg_features)} != {self.input_size}")
                # 次元調整
                if len(eeg_features) < self.input_size:
                    eeg_features = np.pad(eeg_features, (0, self.input_size - len(eeg_features)), 'constant')
                else:
                    eeg_features = eeg_features[:self.input_size]
            
            # 特徴量正規化
            eeg_features_scaled = self.scaler.transform(eeg_features.reshape(1, -1))
            
            # Pytorchテンソルに変換
            features_tensor = torch.FloatTensor(eeg_features_scaled).to(device)
            
            # 分類実行
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class_idx].item()
            
            predicted_class = self.class_names[predicted_class_idx]
            
            # 処理時間計算
            processing_time_ms = (time.time() - start_time) * 1000
            
            # TCPデータから実際の把持力を取得
            actual_grip_force = episode.tcp_data.get('grip_force', 0.0)
            
            # 実際のラベルを計算（参考用）
            if actual_grip_force < 8.0:
                actual_label = "UnderGrip"
            elif actual_grip_force > 15.0:
                actual_label = "OverGrip"
            else:
                actual_label = "Success"
            
            # 分類結果作成
            classification_result = {
                'episode_id': episode.episode_id,
                'timestamp': time.time(),
                'predicted_class': predicted_class,
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'probabilities': {
                    class_name: prob.item() 
                    for class_name, prob in zip(self.class_names, probabilities[0])
                },
                'actual_grip_force': actual_grip_force,
                'actual_label': actual_label,
                'correct_prediction': predicted_class == actual_label,
                'processing_time_ms': processing_time_ms,
                'eeg_data_shape': episode.lsl_data.shape,
                'tcp_data': episode.tcp_data,
                'sync_latency_ms': episode.sync_latency
            }
            
            return classification_result
            
        except Exception as e:
            print(f"❌ 分類エラー (Episode {episode.episode_id}): {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'episode_id': episode.episode_id,
                'timestamp': time.time(),
                'error': str(e),
                'processing_time_ms': (time.time() - start_time) * 1000
            }
    
    def start_classification(self):
        """リアルタイム分類開始"""
        print(f"🚀 リアルタイム分類開始")
        
        if self.model is None:
            print(f"❌ モデルが読み込まれていません")
            return False
        
        # エピソード収集システム初期化
        self.episode_collector = LSLTCPEpisodeCollector(
            lsl_stream_name=self.lsl_stream_name,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port,
            save_to_csv=False  # リアルタイム分類では無効
        )
        
        # エピソード収集開始
        if not self.episode_collector.start_collection():
            print(f"❌ エピソード収集開始失敗")
            return False
        
        # 分類スレッド開始
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.classification_thread = threading.Thread(target=self._classification_loop, daemon=True)
        self.classification_thread.start()
        
        print(f"✅ リアルタイム分類開始完了")
        print(f"💡 Unity側でエピソードを実行してください")
        return True
    
    def _classification_loop(self):
        """分類処理ループ"""
        print(f"🔄 分類処理ループ開始")
        
        last_episode_count = 0
        
        while self.is_running:
            try:
                # 新しいエピソードをチェック
                current_episode_count = len(self.episode_collector.episodes)
                
                if current_episode_count > last_episode_count:
                    # 新しいエピソードを処理
                    for i in range(last_episode_count, current_episode_count):
                        episode = self.episode_collector.episodes[i]
                        
                        print(f"🆕 新エピソード検出: Episode {episode.episode_id}")
                        
                        # 分類実行
                        classification_result = self.classify_episode(episode)
                        
                        # 結果処理
                        self._process_classification_result(classification_result)
                        
                        # 統計更新
                        self._update_statistics(classification_result)
                    
                    last_episode_count = current_episode_count
                
                time.sleep(0.1)  # 100ms待機
                
            except Exception as e:
                print(f"❌ 分類ループエラー: {e}")
                time.sleep(1.0)
        
        print(f"🔄 分類処理ループ終了")
    
    def _process_classification_result(self, result):
        """分類結果の処理"""
        if 'error' in result:
            print(f"❌ 分類失敗 (Episode {result['episode_id']}): {result['error']}")
            return
        
        # 結果保存
        self.classification_results.append(result)
        
        # リアルタイム表示
        print(f"🎯 分類結果 (Episode {result['episode_id']}):")
        print(f"   予測クラス: {result['predicted_class']} (信頼度: {result['confidence']:.3f})")
        print(f"   実際の把持力: {result['actual_grip_force']:.2f}N")
        print(f"   実際のラベル: {result['actual_label']}")
        print(f"   正解: {'✅' if result['correct_prediction'] else '❌'}")
        print(f"   処理時間: {result['processing_time_ms']:.1f}ms")
        
        # 詳細な確率表示
        print(f"   クラス確率:")
        for class_name, prob in result['probabilities'].items():
            print(f"     {class_name}: {prob:.3f}")
        
        # CSV保存（逐次追記）
        self._save_result_to_csv(result)
        
        print()  # 空行
    
    def _save_result_to_csv(self, result):
        """分類結果をCSVに保存"""
        try:
            csv_file = os.path.join(self.output_dir, "realtime_classifications.csv")
            
            # CSVデータ準備
            csv_data = {
                'episode_id': result['episode_id'],
                'timestamp': result['timestamp'],
                'predicted_class': result['predicted_class'],
                'predicted_class_idx': result['predicted_class_idx'],
                'confidence': result['confidence'],
                'prob_undergrip': result['probabilities'].get('UnderGrip', 0),
                'prob_success': result['probabilities'].get('Success', 0),
                'prob_overgrip': result['probabilities'].get('OverGrip', 0),
                'actual_grip_force': result['actual_grip_force'],
                'actual_label': result['actual_label'],
                'correct_prediction': result['correct_prediction'],
                'processing_time_ms': result['processing_time_ms'],
                'sync_latency_ms': result['sync_latency_ms']
            }
            
            # ファイルが存在しない場合はヘッダー付きで作成
            if not os.path.exists(csv_file):
                pd.DataFrame([csv_data]).to_csv(csv_file, index=False)
            else:
                pd.DataFrame([csv_data]).to_csv(csv_file, mode='a', header=False, index=False)
                
        except Exception as e:
            print(f"⚠️ CSV保存エラー: {e}")
    
    def _update_statistics(self, result):
        """統計情報を更新"""
        if 'error' in result:
            return
        
        self.stats['total_episodes'] += 1
        self.stats['total_classifications'] += 1
        
        # クラス別カウント
        predicted_class = result['predicted_class']
        if predicted_class in self.stats['class_counts']:
            self.stats['class_counts'][predicted_class] += 1
        
        # 平均信頼度
        prev_avg_conf = self.stats['avg_confidence']
        n = self.stats['total_classifications']
        self.stats['avg_confidence'] = (prev_avg_conf * (n-1) + result['confidence']) / n
        
        # 平均処理時間
        prev_avg_time = self.stats['avg_processing_time_ms']
        self.stats['avg_processing_time_ms'] = (prev_avg_time * (n-1) + result['processing_time_ms']) / n
    
    def stop_classification(self):
        """分類処理停止"""
        print(f"🛑 リアルタイム分類停止中...")
        
        self.is_running = False
        
        # エピソード収集停止
        if self.episode_collector:
            self.episode_collector.stop_collection()
        
        # 最終統計表示
        self._print_final_statistics()
        
        # 結果分析
        if len(self.classification_results) > 0:
            self._analyze_results()
        
        print(f"🛑 リアルタイム分類停止完了")
    
    def _print_final_statistics(self):
        """最終統計情報表示"""
        print(f"\n📊 リアルタイム分類統計:")
        
        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
            print(f"   稼働時間: {total_time:.1f}秒")
        
        print(f"   総エピソード数: {self.stats['total_episodes']}")
        print(f"   総分類数: {self.stats['total_classifications']}")
        print(f"   平均信頼度: {self.stats['avg_confidence']:.3f}")
        print(f"   平均処理時間: {self.stats['avg_processing_time_ms']:.1f}ms")
        
        print(f"   クラス別予測数:")
        for class_name, count in self.stats['class_counts'].items():
            percentage = count / self.stats['total_classifications'] * 100 if self.stats['total_classifications'] > 0 else 0
            print(f"     {class_name}: {count}件 ({percentage:.1f}%)")
        
        print(f"   出力ディレクトリ: {self.output_dir}")
    
    def _analyze_results(self):
        """分類結果の詳細分析"""
        print(f"\n📈 分類結果分析:")
        
        # 正解率計算
        correct_predictions = [r for r in self.classification_results if r.get('correct_prediction', False)]
        accuracy = len(correct_predictions) / len(self.classification_results) * 100
        print(f"   リアルタイム精度: {accuracy:.1f}% ({len(correct_predictions)}/{len(self.classification_results)})")
        
        # クラス別正解率
        class_accuracies = {}
        for class_name in self.class_names:
            class_results = [r for r in self.classification_results if r.get('actual_label') == class_name]
            if class_results:
                class_correct = [r for r in class_results if r.get('correct_prediction', False)]
                class_acc = len(class_correct) / len(class_results) * 100
                class_accuracies[class_name] = class_acc
                print(f"   {class_name}精度: {class_acc:.1f}% ({len(class_correct)}/{len(class_results)})")
        
        # 信頼度分布
        confidences = [r['confidence'] for r in self.classification_results if 'confidence' in r]
        if confidences:
            print(f"   信頼度分布:")
            print(f"     最大: {max(confidences):.3f}")
            print(f"     最小: {min(confidences):.3f}")
            print(f"     平均: {np.mean(confidences):.3f}")
            print(f"     標準偏差: {np.std(confidences):.3f}")
        
        # 処理時間統計
        processing_times = [r['processing_time_ms'] for r in self.classification_results if 'processing_time_ms' in r]
        if processing_times:
            print(f"   処理時間統計:")
            print(f"     平均: {np.mean(processing_times):.1f}ms")
            print(f"     最大: {max(processing_times):.1f}ms")
            print(f"     最小: {min(processing_times):.1f}ms")
        
        # 混同行列の簡易表示
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            
            actual_labels = [r['actual_label'] for r in self.classification_results if 'actual_label' in r]
            predicted_labels = [r['predicted_class'] for r in self.classification_results if 'predicted_class' in r]
            
            if len(actual_labels) == len(predicted_labels) and len(set(actual_labels)) > 1:
                print(f"\n🎯 混同行列:")
                cm = confusion_matrix(actual_labels, predicted_labels, labels=list(self.class_names))
                
                for i, true_class in enumerate(self.class_names):
                    row_str = f"   {true_class:10}: "
                    for j, pred_class in enumerate(self.class_names):
                        row_str += f"{cm[i,j]:3d} "
                    print(row_str)
                
                print(f"   予測→        " + "".join([f"{cls[:3]:>4}" for cls in self.class_names]))
        except ImportError:
            print(f"   混同行列表示にはscikit-learnが必要です")
        except Exception as e:
            print(f"   混同行列計算エラー: {e}")
    
    def run_demo(self):
        """デモ実行"""
        print(f"🚀 リアルタイム把持力分類デモ開始")
        
        # モデル読み込み
        if not self.load_model():
            print(f"❌ モデル読み込み失敗")
            return
        
        # 分類開始
        if not self.start_classification():
            print(f"❌ 分類開始失敗")
            return
        
        try:
            print(f"\n💡 リアルタイム分類実行中:")
            print(f"   1. LSLデータ受信中 ({self.lsl_stream_name})")
            print(f"   2. TCP接続待機中 ({self.tcp_host}:{self.tcp_port})")
            print(f"   3. エピソードごとに自動分類")
            print(f"   4. Ctrl+C で終了")
            print(f"\n🎮 Unity側でエピソードを実行してください:")
            print(f"   1. ロボット状態データ送信")
            print(f"   2. 'EPISODE_END' トリガー送信")
            print(f"   → 自動的に把持力分類が実行されます")
            
            # メインループ
            while self.is_running:
                time.sleep(5.0)
                
                # 定期的な統計表示
                if self.stats['total_classifications'] > 0:
                    print(f"💻 進捗: 分類済み {self.stats['total_classifications']}件, "
                          f"平均信頼度 {self.stats['avg_confidence']:.3f}, "
                          f"平均処理時間 {self.stats['avg_processing_time_ms']:.1f}ms")
                else:
                    print(f"⏳ エピソード待機中...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ デモ停止（Ctrl+C）")
        finally:
            self.stop_classification()


class BatchClassificationTester:
    """バッチ分類テスター（過去のCSVデータでテスト）"""
    
    def __init__(self, classifier):
        self.classifier = classifier
    
    def test_with_saved_episodes(self, csv_dir):
        """保存されたエピソードCSVでテスト"""
        print(f"🧪 保存エピソードでバッチテスト開始: {csv_dir}")
        
        if not os.path.exists(csv_dir):
            print(f"❌ ディレクトリが見つかりません: {csv_dir}")
            return False
        
        # エピソードCSVファイル検索
        info_files = glob.glob(os.path.join(csv_dir, "*_info.csv"))
        eeg_files = glob.glob(os.path.join(csv_dir, "*_eeg.csv"))
        
        print(f"📋 発見エピソード: {len(info_files)}件")
        
        if len(info_files) == 0:
            print(f"❌ エピソードファイルが見つかりません")
            return False
        
        test_results = []
        
        for info_file in sorted(info_files):
            try:
                # エピソード情報読み込み
                info_df = pd.read_csv(info_file)
                episode_id = info_df['episode_id'].iloc[0]
                
                # EEGデータ読み込み
                eeg_file = info_file.replace('_info.csv', '_eeg.csv')
                if not os.path.exists(eeg_file):
                    print(f"⚠️ EEGファイル未発見: {eeg_file}")
                    continue
                
                eeg_df = pd.read_csv(eeg_file)
                channel_cols = [col for col in eeg_df.columns if col.startswith('ch_')]
                eeg_data = eeg_df[channel_cols].values[:300, :32]  # 300サンプル、32チャンネル
                
                # モック Episodeオブジェクト作成
                mock_episode = type('Episode', (), {
                    'episode_id': episode_id,
                    'lsl_data': eeg_data,
                    'tcp_data': {
                        'grip_force': info_df['grip_force'].iloc[0],
                        'contact': info_df['contact'].iloc[0],
                        'broken': info_df['broken'].iloc[0]
                    },
                    'sync_latency': info_df.get('sync_latency_ms', [0]).iloc[0]
                })()
                
                # 分類実行
                result = self.classifier.classify_episode(mock_episode)
                test_results.append(result)
                
                # 進捗表示
                if len(test_results) % 10 == 0:
                    print(f"   テスト進捗: {len(test_results)}/{len(info_files)}")
                
            except Exception as e:
                print(f"⚠️ エピソード{episode_id}テスト失敗: {e}")
                continue
        
        print(f"✅ バッチテスト完了: {len(test_results)}件")
        
        # 結果分析
        self._analyze_batch_results(test_results)
        
        return test_results
    
    def _analyze_batch_results(self, results):
        """バッチテスト結果分析"""
        print(f"\n📊 バッチテスト分析:")
        
        valid_results = [r for r in results if 'error' not in r]
        print(f"   有効テスト数: {len(valid_results)}/{len(results)}")
        
        if len(valid_results) == 0:
            return
        
        # 精度計算
        correct = [r for r in valid_results if r.get('correct_prediction', False)]
        accuracy = len(correct) / len(valid_results) * 100
        print(f"   総合精度: {accuracy:.1f}% ({len(correct)}/{len(valid_results)})")
        
        # 平均信頼度
        confidences = [r['confidence'] for r in valid_results]
        avg_confidence = np.mean(confidences)
        print(f"   平均信頼度: {avg_confidence:.3f}")
        
        # 平均処理時間
        processing_times = [r['processing_time_ms'] for r in valid_results]
        avg_processing_time = np.mean(processing_times)
        print(f"   平均処理時間: {avg_processing_time:.1f}ms")
        
        # クラス別精度
        for class_name in ['UnderGrip', 'Success', 'OverGrip']:
            class_results = [r for r in valid_results if r.get('actual_label') == class_name]
            if class_results:
                class_correct = [r for r in class_results if r.get('correct_prediction', False)]
                class_acc = len(class_correct) / len(class_results) * 100
                print(f"   {class_name}精度: {class_acc:.1f}% ({len(class_correct)}/{len(class_results)})")


def main():
    """メイン実行関数"""
    print(f"🧠 リアルタイム把持力分類システム")
    print(f"=" * 60)
    print(f"改善版分類機 + リアルタイムエピソード分類")
    print(f"UnderGrip(<8N), Success(8-15N), OverGrip(>15N)")
    print(f"=" * 60)
    
    # 実行モード選択
    print(f"\n実行モードを選択してください:")
    print(f"1. リアルタイム分類デモ（デフォルト）")
    print(f"2. バッチテスト（過去のCSVデータで性能評価）")
    print(f"3. モデル情報表示のみ")
    
    choice = input("選択 (1-3): ").strip()
    
    # 分類器初期化
    classifier = RealtimeGripForceClassifier(
        lsl_stream_name='MockEEG',  # 必要に応じて変更
        tcp_host='127.0.0.1',
        tcp_port=12345
    )
    
    if choice == "2":
        # バッチテストモード
        print(f"\n🧪 バッチテストモード")
        
        # モデル読み込み
        if not classifier.load_model():
            return
        
        # テストディレクトリ選択
        test_dir = input("テスト用エピソードディレクトリパス（空白で自動検索）: ").strip()
        
        if not test_dir:
            # 自動検索
            log_dirs = glob.glob("DDPG_Python/logs/episodes_*")
            if not log_dirs:
                log_dirs = glob.glob("logs/episodes_*")
            
            if log_dirs:
                test_dir = max(log_dirs)  # 最新のログディレクトリ
                print(f"🔍 自動選択: {test_dir}")
            else:
                print(f"❌ テストディレクトリが見つかりません")
                return
        
        # バッチテスト実行
        tester = BatchClassificationTester(classifier)
        tester.test_with_saved_episodes(test_dir)
        
    elif choice == "3":
        # モデル情報表示のみ
        print(f"\n📋 モデル情報表示")
        classifier.load_model()
        
    else:
        # デフォルト：リアルタイムデモ
        print(f"\n🚀 リアルタイム分類デモ")
        classifier.run_demo()


if __name__ == "__main__":
    main()