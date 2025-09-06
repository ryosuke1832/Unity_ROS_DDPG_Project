#!/usr/bin/env python3
"""
EEGデータ前処理パイプライン

論文の前処理手順:
1. 250Hz リサンプリング
2. 2-50Hz バンドパスフィルタリング  
3. Artifact Subspace Reconstruction (ASR)
4. Independent Component Analysis (ICA)
5. ADJUST による自動アーティファクト除去
6. エポック抽出 (イベントから400ms後まで)
7. Power Spectral Density (PSD) 計算
8. ベースライン減算

本実装では、リアルタイム処理に適した簡易版を提供
"""

import numpy as np
import warnings
from scipy import signal
from sklearn.decomposition import FastICA
from collections import deque
import time

class NeuroadaptationEEGPreprocessor:
    """
    論文準拠EEG前処理クラス（リアルタイム対応版）
    """
    
    def __init__(self, 
                 sampling_rate=250,
                 n_channels=32,
                 epoch_duration=1.2,
                 filter_lowcut=2.0,
                 filter_highcut=50.0,
                 enable_asr=True,
                 enable_ica=False,  # 重い処理のため通常はFalse
                 asr_threshold=5.0):
        """
        初期化
        
        Args:
            sampling_rate: サンプリング周波数 (250Hz, 論文準拠)
            n_channels: チャンネル数 (32, 論文準拠)  
            epoch_duration: エポック長 (1.2秒)
            filter_lowcut: バンドパス低域カットオフ (2Hz, 論文準拠)
            filter_highcut: バンドパス高域カットオフ (50Hz, 論文準拠)
            enable_asr: ASR有効フラグ
            enable_ica: ICA有効フラグ (重い処理)
            asr_threshold: ASR閾値 (sigma数)
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.epoch_duration = epoch_duration
        self.epoch_samples = int(epoch_duration * sampling_rate)
        
        # フィルタ設定
        self.filter_lowcut = filter_lowcut
        self.filter_highcut = filter_highcut
        
        # 前処理オプション
        self.enable_asr = enable_asr
        self.enable_ica = enable_ica
        self.asr_threshold = asr_threshold
        
        # バンドパスフィルタ係数を事前計算
        self._setup_bandpass_filter()
        
        # ICA設定（有効時）
        if self.enable_ica:
            self.ica = FastICA(n_components=n_channels, random_state=42, max_iter=1000)
            self.ica_fitted = False
        
        # 統計情報
        self.processing_stats = {
            'total_epochs': 0,
            'asr_rejected_channels': 0,
            'ica_applications': 0,
            'avg_processing_time_ms': 0.0
        }
        
        print(f"🧠 Neuroadaptation EEG前処理システム初期化:")
        print(f"   サンプリング: {sampling_rate}Hz")
        print(f"   チャンネル数: {n_channels}ch")
        print(f"   エポック長: {epoch_duration}s ({self.epoch_samples}samples)")
        print(f"   バンドパス: {filter_lowcut}-{filter_highcut}Hz")
        print(f"   ASR有効: {enable_asr}")
        print(f"   ICA有効: {enable_ica}")
        
    def _setup_bandpass_filter(self):
        """バンドパスフィルタの設計（論文準拠: 2-50Hz）"""
        nyquist = self.sampling_rate / 2
        low = self.filter_lowcut / nyquist
        high = self.filter_highcut / nyquist
        
        # 論文では明記されていないが、5次Butterworthを使用
        self.filter_order = 5
        self.sos = signal.butter(self.filter_order, [low, high], 
                                btype='band', output='sos')
        
        print(f"   フィルタ設計完了: {self.filter_order}次Butterworth")
    
    def preprocess_epoch(self, epoch_data: np.ndarray) -> dict:
        """
        単一エポックの完全前処理（論文パイプライン）
        
        Args:
            epoch_data: (samples, channels) or (channels, samples) のEEGエポック
            
        Returns:
            dict: {
                'processed_epoch': 前処理済みデータ (samples, channels),
                'processing_info': 処理情報,
                'quality_metrics': 品質指標,
                'rejected_channels': 除去されたチャンネルID,
                'processing_time_ms': 処理時間
            }
        """
        start_time = time.time()
        
        # データ形状の標準化: (samples, channels)
        if epoch_data.ndim != 2:
            raise ValueError(f"エポックデータは2Dである必要があります: {epoch_data.shape}")
        
        # (channels, samples) → (samples, channels) 変換（必要に応じて）
        if epoch_data.shape[1] == self.n_channels and epoch_data.shape[0] != self.n_channels:
            # 既に (samples, channels) 形式
            processed_data = epoch_data.copy()
        elif epoch_data.shape[0] == self.n_channels:
            # (channels, samples) → (samples, channels) 変換
            processed_data = epoch_data.T.copy()
        else:
            raise ValueError(f"チャンネル数が不正です: {epoch_data.shape}, 期待値: {self.n_channels}")
        
        processing_info = {
            'original_shape': epoch_data.shape,
            'standardized_shape': processed_data.shape,
            'steps_applied': []
        }
        
        rejected_channels = []
        
        # Step 1: バンドパスフィルタリング (2-50Hz)
        processed_data = self._apply_bandpass_filter(processed_data)
        processing_info['steps_applied'].append('bandpass_filter')
        
        # Step 2: Artifact Subspace Reconstruction (ASR)
        if self.enable_asr:
            processed_data, asr_rejected = self._apply_asr(processed_data)
            rejected_channels.extend(asr_rejected)
            processing_info['steps_applied'].append('asr')
        
        # Step 3: Independent Component Analysis (ICA) - オプション
        if self.enable_ica:
            processed_data = self._apply_ica(processed_data)
            processing_info['steps_applied'].append('ica')
        
        # Step 4: 正規化 (Z-score)
        processed_data = self._apply_zscore_normalization(processed_data)
        processing_info['steps_applied'].append('zscore_normalization')
        
        # Step 5: 品質評価
        quality_metrics = self._assess_epoch_quality(processed_data)
        
        # 統計更新
        processing_time_ms = (time.time() - start_time) * 1000
        self.processing_stats['total_epochs'] += 1
        self.processing_stats['asr_rejected_channels'] += len(rejected_channels)
        
        # 処理時間の移動平均
        prev_avg = self.processing_stats['avg_processing_time_ms']
        n = self.processing_stats['total_epochs']
        self.processing_stats['avg_processing_time_ms'] = (prev_avg * (n-1) + processing_time_ms) / n
        
        return {
            'processed_epoch': processed_data,  # (samples, channels)
            'processing_info': processing_info,
            'quality_metrics': quality_metrics,
            'rejected_channels': rejected_channels,
            'processing_time_ms': processing_time_ms
        }
    
    def _apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """バンドパスフィルタ適用 (2-50Hz)"""
        filtered_data = np.zeros_like(data)
        
        for ch in range(data.shape[1]):
            try:
                # sosfilt を使用（より数値的に安定）
                filtered_data[:, ch] = signal.sosfilt(self.sos, data[:, ch])
            except Exception as e:
                warnings.warn(f"Channel {ch}: フィルタリング失敗 - {e}")
                filtered_data[:, ch] = data[:, ch]  # 元データを保持
                
        return filtered_data
    
    def _apply_asr(self, data: np.ndarray) -> tuple[np.ndarray, list]:
        """
        Artifact Subspace Reconstruction (ASR) の簡易実装
        極端な外れ値チャンネルの検出と修正
        """
        processed_data = data.copy()
        rejected_channels = []
        
        for ch in range(data.shape[1]):
            ch_data = data[:, ch]
            
            if np.std(ch_data) == 0:
                # 無信号チャンネル
                processed_data[:, ch] = 0
                rejected_channels.append(ch)
                continue
                
            # Z-score による外れ値検出
            z_scores = np.abs((ch_data - np.mean(ch_data)) / np.std(ch_data))
            max_z_score = np.max(z_scores)
            
            if max_z_score > self.asr_threshold:
                # 閾値を超える外れ値が存在
                if max_z_score > self.asr_threshold * 2:
                    # 非常に大きな外れ値 → チャンネル全体を除去
                    processed_data[:, ch] = 0
                    rejected_channels.append(ch)
                else:
                    # 中程度の外れ値 → 外れ値サンプルのみ除去
                    outlier_mask = z_scores > self.asr_threshold
                    processed_data[outlier_mask, ch] = np.median(ch_data)
        
        return processed_data, rejected_channels
    
    def _apply_ica(self, data: np.ndarray) -> np.ndarray:
        """Independent Component Analysis (ICA) 適用"""
        try:
            if not self.ica_fitted:
                # 初回学習
                self.ica.fit(data.T)  # ICAは (features, samples) を期待
                self.ica_fitted = True
                self.processing_stats['ica_applications'] += 1
            
            # ICA変換と逆変換（アーティファクト除去の簡易版）
            sources = self.ica.transform(data.T)
            
            # 簡易的なアーティファクト除去：極端な成分を減衰
            for i in range(sources.shape[0]):
                if np.std(sources[i, :]) > 3 * np.mean([np.std(sources[j, :]) for j in range(sources.shape[0])]):
                    sources[i, :] *= 0.1  # 90%減衰
            
            # 逆変換
            cleaned_data = self.ica.inverse_transform(sources).T
            return cleaned_data
            
        except Exception as e:
            warnings.warn(f"ICA処理失敗: {e}")
            return data  # ICA失敗時は元データを返す
    
    def _apply_zscore_normalization(self, data: np.ndarray) -> np.ndarray:
        """チャンネルごとZ-score正規化"""
        normalized_data = np.zeros_like(data)
        
        for ch in range(data.shape[1]):
            ch_data = data[:, ch]
            ch_mean = np.mean(ch_data)
            ch_std = np.std(ch_data)
            
            if ch_std > 1e-10:  # ゼロ除算回避
                normalized_data[:, ch] = (ch_data - ch_mean) / ch_std
            else:
                normalized_data[:, ch] = ch_data
                
        return normalized_data
    
    def _assess_epoch_quality(self, data: np.ndarray) -> dict:
        """エポック品質評価"""
        return {
            'snr_db': self._estimate_snr(data),
            'artifact_ratio': self._estimate_artifact_ratio(data),
            'channel_correlation': self._estimate_channel_correlation(data),
            'spectral_quality': self._estimate_spectral_quality(data)
        }
    
    def _estimate_snr(self, data: np.ndarray) -> float:
        """Signal-to-Noise Ratio 推定"""
        signal_power = np.mean(np.var(data, axis=0))
        noise_floor = np.mean([np.var(np.diff(data[:, ch])) for ch in range(data.shape[1])])
        
        if noise_floor > 0:
            snr_db = 10 * np.log10(signal_power / noise_floor)
        else:
            snr_db = float('inf')
            
        return min(snr_db, 40.0)  # 上限設定
    
    def _estimate_artifact_ratio(self, data: np.ndarray) -> float:
        """アーティファクト比率推定"""
        total_samples = data.shape[0] * data.shape[1]
        artifact_samples = 0
        
        for ch in range(data.shape[1]):
            z_scores = np.abs((data[:, ch] - np.mean(data[:, ch])) / np.std(data[:, ch]))
            artifact_samples += np.sum(z_scores > 3)
            
        return artifact_samples / total_samples
    
    def _estimate_channel_correlation(self, data: np.ndarray) -> float:
        """チャンネル間相関の平均"""
        try:
            corr_matrix = np.corrcoef(data.T)
            # 対角成分を除く
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)
            avg_correlation = np.mean(np.abs(corr_matrix[mask]))
            return avg_correlation
        except:
            return 0.0
    
    def _estimate_spectral_quality(self, data: np.ndarray) -> float:
        """スペクトル品質評価"""
        try:
            # 各チャンネルの2-50Hz帯域パワーの一様性
            freqs, psd = signal.welch(data, fs=self.sampling_rate, axis=0)
            target_band_mask = (freqs >= 2) & (freqs <= 50)
            
            band_powers = np.mean(psd[target_band_mask, :], axis=0)
            spectral_uniformity = 1.0 / (1.0 + np.std(band_powers) / np.mean(band_powers))
            
            return spectral_uniformity
        except:
            return 0.5  # デフォルト値
    
    def get_processing_statistics(self) -> dict:
        """前処理統計情報の取得"""
        return self.processing_stats.copy()
    
    def reset_statistics(self):
        """統計情報のリセット"""
        self.processing_stats = {
            'total_epochs': 0,
            'asr_rejected_channels': 0,
            'ica_applications': 0,
            'avg_processing_time_ms': 0.0
        }


class StreamingEEGPreprocessor(NeuroadaptationEEGPreprocessor):
    """
    リアルタイムストリーミング用前処理クラス
    連続データストリームに対応
    """
    
    def __init__(self, *args, buffer_duration=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.buffer_duration = buffer_duration
        self.buffer_samples = int(buffer_duration * self.sampling_rate)
        
        # 連続データバッファ
        self.continuous_buffer = deque(maxlen=self.buffer_samples)
        self.timestamps_buffer = deque(maxlen=self.buffer_samples)
        
        print(f"   ストリーミングバッファ: {buffer_duration}s ({self.buffer_samples}samples)")
    
    def add_sample(self, sample: np.ndarray, timestamp: float):
        """
        新しいサンプルをバッファに追加
        
        Args:
            sample: (n_channels,) の単一サンプル
            timestamp: サンプルのタイムスタンプ
        """
        if sample.shape[0] != self.n_channels:
            raise ValueError(f"サンプルのチャンネル数が不正: {sample.shape[0]}, 期待値: {self.n_channels}")
            
        self.continuous_buffer.append(sample.copy())
        self.timestamps_buffer.append(timestamp)
    
    def extract_and_preprocess_epoch(self, center_timestamp: float) -> dict:
        """
        指定タイムスタンプ周辺のエポックを抽出して前処理
        
        Args:
            center_timestamp: エポック中心のタイムスタンプ
            
        Returns:
            前処理結果 (preprocess_epochと同じ形式) または None
        """
        if len(self.continuous_buffer) < self.epoch_samples:
            return None
        
        # タイムスタンプから最適なエポック範囲を決定
        timestamps = list(self.timestamps_buffer)
        time_diffs = [abs(ts - center_timestamp) for ts in timestamps]
        
        if not time_diffs:
            return None
        
        center_idx = time_diffs.index(min(time_diffs))
        half_epoch = self.epoch_samples // 2
        
        start_idx = max(0, center_idx - half_epoch)
        end_idx = min(len(self.continuous_buffer), start_idx + self.epoch_samples)
        
        # エポックデータ抽出
        if end_idx - start_idx < self.epoch_samples:
            return None  # データ不足
        
        epoch_data = np.array([self.continuous_buffer[i] for i in range(start_idx, end_idx)])
        
        # 前処理実行
        result = self.preprocess_epoch(epoch_data)
        result['extraction_info'] = {
            'center_timestamp': center_timestamp,
            'center_idx': center_idx,
            'epoch_range': (start_idx, end_idx),
            'sync_latency': min(time_diffs)
        }
        
        return result


# 使用例とテスト関数
def demo_preprocessing():
    """前処理デモ"""
    print("🧠 EEG Neuroadaptation 前処理デモ")
    
    # 模擬EEGデータ生成 (1.2秒、32チャンネル、250Hz)
    np.random.seed(42)
    samples = 300  # 1.2秒 × 250Hz
    channels = 32
    
    # 模擬EEGデータ（ノイズ + アーティファクト含む）
    mock_eeg = np.random.randn(samples, channels) * 10
    
    # いくつかのチャンネルにアーティファクト追加
    mock_eeg[:, 5] += np.random.randn(samples) * 50  # 大きなノイズ
    mock_eeg[100:150, 10] = 100  # スパイクアーティファクト
    
    # 前処理実行
    preprocessor = NeuroadaptationEEGPreprocessor(
        enable_asr=True,
        enable_ica=False  # デモでは無効
    )
    
    result = preprocessor.preprocess_epoch(mock_eeg)
    
    print(f"\n📊 処理結果:")
    print(f"   処理時間: {result['processing_time_ms']:.2f}ms")
    print(f"   除去チャンネル: {result['rejected_channels']}")
    print(f"   品質指標:")
    for metric, value in result['quality_metrics'].items():
        print(f"     {metric}: {value:.3f}")
    
    print(f"\n📈 統計情報:")
    stats = preprocessor.get_processing_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo_preprocessing()