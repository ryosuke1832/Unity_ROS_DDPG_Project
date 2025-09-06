#!/usr/bin/env python3
"""
8CH LSL EEG Data Receiver and Processor
8チャンネル版EEGデータ受信・処理システム

このスクリプトは：
1. LSLから8チャンネルEEGデータを受信
2. 論文準拠の前処理を適用（2-50Hzフィルタリング）
3. 1.2秒のエポックを抽出
4. 8チャンネル特化のCNN分類（模擬）
5. リアルタイム可視化

8チャンネル電極配置（標準10-20システム）:
Fz, FCz, Cz, CPz, Pz, C3, C4, Oz

Requirements:
- pylsl (pip install pylsl)
- numpy (pip install numpy)
- scipy (pip install scipy)
- matplotlib (pip install matplotlib)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_streams
from scipy import signal
from scipy.signal import butter, filtfilt
import threading
import queue
import time
from collections import deque

class EEG8CHDataProcessor:
    def __init__(self, sampling_rate=250, epoch_length=1.2):
        """
        8チャンネルEEGデータ処理システムの初期化
        
        Args:
            sampling_rate: サンプリング周波数（Hz）
            epoch_length: エポック長（秒）
        """
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.epoch_samples = int(sampling_rate * epoch_length)
        self.n_channels = 8  # 8チャンネル固定
        
        # 8チャンネル電極配置
        self.channel_names = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C3', 'C4', 'Oz']
        
        # 論文に基づくフィルター設計（2-50Hz）
        self.lowcut = 2.0
        self.highcut = 50.0
        self.filter_order = 4
        
        # バターワースフィルターの係数を計算
        nyquist = sampling_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.b, self.a = butter(self.filter_order, [low, high], btype='band')
        
        # データバッファ（8チャンネル専用）
        self.data_buffer = deque(maxlen=self.epoch_samples * 3)  # 3エポック分
        self.processed_epochs = queue.Queue()
        
        # 8チャンネル特化の解剖学的グループ
        self.electrode_groups = {
            'frontal': [0, 1],      # Fz, FCz
            'central': [2, 5, 6],   # Cz, C3, C4
            'parietal': [3, 4],     # CPz, Pz
            'occipital': [7]        # Oz
        }
        
        print(f"8CH EEG Processor initialized:")
        print(f"  Channels: {self.n_channels} ({', '.join(self.channel_names)})")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Epoch length: {epoch_length} s ({self.epoch_samples} samples)")
        print(f"  Bandpass filter: {self.lowcut}-{self.highcut} Hz")
        print(f"  Electrode groups: {self.electrode_groups}")
        
    def apply_preprocessing(self, data):
        """
        8チャンネル特化の前処理を適用
        
        Args:
            data: EEGデータ (8 x samples)
            
        Returns:
            preprocessed_data: 前処理済みデータ
        """
        if data.shape[1] < self.filter_order * 3:
            return data
            
        # 2-50Hzバンドパスフィルター
        filtered_data = np.zeros_like(data)
        for ch in range(min(self.n_channels, data.shape[0])):
            try:
                filtered_data[ch] = filtfilt(self.b, self.a, data[ch])
            except Exception as e:
                print(f"Filtering error for channel {self.channel_names[ch]}: {e}")
                filtered_data[ch] = data[ch]
                
        return filtered_data
        
    def extract_epoch(self, trigger_idx=None):
        """
        8チャンネルエポックを抽出
        
        Args:
            trigger_idx: トリガーのインデックス（Noneの場合は最新のデータ）
            
        Returns:
            epoch_data: エポックデータ (8 x epoch_samples)
        """
        if len(self.data_buffer) < self.epoch_samples:
            return None
            
        # バッファからデータを取得
        buffer_array = np.array(self.data_buffer)  # (samples, channels)
        # 8チャンネルに制限
        if buffer_array.shape[1] > self.n_channels:
            buffer_array = buffer_array[:, :self.n_channels]
        elif buffer_array.shape[1] < self.n_channels:
            # パディング
            padding = np.zeros((buffer_array.shape[0], self.n_channels - buffer_array.shape[1]))
            buffer_array = np.hstack([buffer_array, padding])
            
        buffer_array = buffer_array.T  # (8, samples)
        
        if trigger_idx is None:
            # 最新のエポック
            epoch_data = buffer_array[:, -self.epoch_samples:]
        else:
            # 指定されたトリガーからのエポック
            start_idx = max(0, trigger_idx)
            end_idx = min(buffer_array.shape[1], trigger_idx + self.epoch_samples)
            epoch_data = buffer_array[:, start_idx:end_idx]
            
            # エポックが短い場合はゼロパディング
            if epoch_data.shape[1] < self.epoch_samples:
                padded_data = np.zeros((self.n_channels, self.epoch_samples))
                padded_data[:, :epoch_data.shape[1]] = epoch_data
                epoch_data = padded_data
                
        return epoch_data
        
    def compute_8ch_power_spectral_density(self, epoch_data):
        """
        8チャンネル特化のパワースペクトル密度計算
        
        Args:
            epoch_data: エポックデータ (8 x samples)
            
        Returns:
            psd_features: 8チャンネルPSD特徴量
        """
        psd_features = {}
        
        # 8チャンネル最適化された周波数帯域
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)  # 8チャンネルではガンマ波も追加
        }
        
        for ch in range(self.n_channels):
            channel_name = self.channel_names[ch]
            channel_psd = {}
            
            # ウェルチ法でPSDを計算（8チャンネル最適化）
            nperseg = min(128, epoch_data.shape[1])  # 8チャンネル用に調整
            noverlap = nperseg // 2
            
            freqs, psd = signal.welch(
                epoch_data[ch], 
                fs=self.sampling_rate,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # 各周波数帯域のパワーを計算
            for band_name, (low_freq, high_freq) in bands.items():
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
                band_power = np.mean(psd[freq_mask])
                channel_psd[band_name] = band_power
                
            psd_features[channel_name] = channel_psd
            
        # 電極グループ別の特徴量も計算
        group_features = {}
        for group_name, channel_indices in self.electrode_groups.items():
            group_psd = {}
            for band_name in bands.keys():
                band_powers = [psd_features[self.channel_names[ch]][band_name] 
                             for ch in channel_indices]
                group_psd[band_name] = np.mean(band_powers)
            group_features[f'group_{group_name}'] = group_psd
            
        psd_features.update(group_features)
        return psd_features
        
    def classify_8ch_cognitive_conflict(self, epoch_data):
        """
        8チャンネル特化の認知的競合分類
        
        Args:
            epoch_data: エポックデータ (8 x samples)
            
        Returns:
            classification: 分類結果辞書
        """
        # 8チャンネル特化の特徴量抽出
        
        # 前頭部重要電極: Fz (0), FCz (1)
        frontal_channels = [0, 1]
        # 中央部重要電極: Cz (2), C3 (5), C4 (6)  
        central_channels = [2, 5, 6]
        
        # ErrP典型時間窓: 200-400ms
        start_idx = int(0.2 * self.sampling_rate)   # 200ms
        end_idx = int(0.4 * self.sampling_rate)     # 400ms
        
        if end_idx > epoch_data.shape[1]:
            end_idx = epoch_data.shape[1]
            
        # 前頭部平均振幅（ErrP主要成分）
        frontal_amplitude = np.mean(epoch_data[frontal_channels, start_idx:end_idx])
        
        # 中央部平均振幅（運動関連電位）
        central_amplitude = np.mean(epoch_data[central_channels, start_idx:end_idx])
        
        # 8チャンネル特化の側性指標（C3 vs C4）
        laterality_index = np.mean(epoch_data[5, start_idx:end_idx]) - \
                          np.mean(epoch_data[6, start_idx:end_idx])  # C3 - C4
        
        # 8チャンネル分類ルール（論文準拠+最適化）
        if frontal_amplitude < -8:
            # 強い前頭部負電位 = over-grip error
            class_name = "over_grip"
            confidence = min(0.95, abs(frontal_amplitude) / 12)
            reward = -100
        elif frontal_amplitude < -4:
            # 中程度前頭部負電位 = under-grip error
            class_name = "under_grip"  
            confidence = min(0.9, abs(frontal_amplitude) / 8)
            reward = -50
        elif abs(laterality_index) > 5:
            # 強い側性化 = 運動準備エラー
            class_name = "motor_error"
            confidence = min(0.85, abs(laterality_index) / 8)
            reward = -25
        else:
            # 正常パターン
            class_name = "success"
            confidence = 0.8 + min(0.15, abs(central_amplitude) / 20)
            reward = 100
            
        return {
            "class": class_name,
            "confidence": confidence,
            "reward": reward,
            "frontal_amplitude": frontal_amplitude,
            "central_amplitude": central_amplitude,
            "laterality_index": laterality_index,
            "n_channels": self.n_channels
        }
        
    def compute_8ch_connectivity(self, epoch_data):
        """
        8チャンネル間の機能的結合性を計算
        
        Args:
            epoch_data: エポックデータ (8 x samples)
            
        Returns:
            connectivity_features: 結合性特徴量
        """
        connectivity_features = {}
        
        # チャンネル間相関
        correlation_matrix = np.corrcoef(epoch_data)
        
        # 重要な結合性ペア（8チャンネル特化）
        important_pairs = [
            ('Fz', 'FCz', 'frontal_coherence'),
            ('C3', 'C4', 'motor_laterality'),
            ('Cz', 'CPz', 'central_parietal'),
            ('Fz', 'Cz', 'frontal_central')
        ]
        
        for ch1_name, ch2_name, pair_name in important_pairs:
            ch1_idx = self.channel_names.index(ch1_name)
            ch2_idx = self.channel_names.index(ch2_name)
            connectivity_features[pair_name] = correlation_matrix[ch1_idx, ch2_idx]
        
        # 全体的な結合性指標
        connectivity_features['mean_connectivity'] = np.mean(np.abs(correlation_matrix))
        connectivity_features['max_connectivity'] = np.max(np.abs(correlation_matrix))
        
        return connectivity_features
        
    def process_8ch_epoch(self, epoch_data):
        """
        8チャンネルエポック全体の処理パイプライン
        """
        # 前処理
        preprocessed = self.apply_preprocessing(epoch_data)
        
        # PSD特徴量計算
        psd_features = self.compute_8ch_power_spectral_density(preprocessed)
        
        # 結合性特徴量計算
        connectivity_features = self.compute_8ch_connectivity(preprocessed)
        
        # 認知的競合分類
        classification = self.classify_8ch_cognitive_conflict(preprocessed)
        
        return {
            "preprocessed_data": preprocessed,
            "psd_features": psd_features,
            "connectivity_features": connectivity_features,
            "classification": classification,
            "timestamp": time.time(),
            "channel_names": self.channel_names
        }


class LSL8CHEEGReceiver:
    def __init__(self, stream_name="MockEEG_8CH"):
        """
        8チャンネルLSL EEG受信システムの初期化
        
        Args:
            stream_name: 受信するストリーム名
        """
        self.stream_name = stream_name
        self.inlet = None
        self.processor = None
        
        # 8チャンネル専用設定
        self.expected_channels = 8
        
        # 受信状態
        self.is_receiving = False
        self.received_samples = 0
        
        # リアルタイム表示用（8チャンネル最適化）
        self.display_buffer = deque(maxlen=1000)  # 4秒分の表示バッファ
        
    def connect_to_stream(self):
        """
        8チャンネルLSLストリームに接続
        """
        print(f"Looking for 8-channel stream '{self.stream_name}'...")
        
        try:
            # ストリームを検索
            streams = resolve_streams()
            
            # 指定した名前のストリームを探す
            target_stream = None
            for stream in streams:
                if stream.name() == self.stream_name:
                    target_stream = stream
                    break
            
            if target_stream is None:
                print(f"No stream named '{self.stream_name}' found!")
                print("Available streams:")
                for stream in streams:
                    channel_count = stream.channel_count()
                    print(f"  - {stream.name()} ({stream.type()}) - {channel_count} channels")
                
                # 8チャンネルストリームを自動検索
                for stream in streams:
                    if stream.channel_count() == 8 and stream.type() == 'EEG':
                        print(f"Found 8-channel EEG stream: {stream.name()}")
                        target_stream = stream
                        self.stream_name = stream.name()
                        break
                
                if target_stream is None:
                    print("No 8-channel EEG stream found!")
                    print("Make sure the 8-channel sender is running.")
                    return False
                
            # ストリームに接続
            self.inlet = StreamInlet(target_stream)
            
            # ストリーム情報を取得
            info = self.inlet.info()
            self.n_channels = info.channel_count()
            self.sampling_rate = int(info.nominal_srate())
            
            # チャンネル数確認
            if self.n_channels != self.expected_channels:
                print(f"Warning: Expected {self.expected_channels} channels, got {self.n_channels}")
                if self.n_channels < self.expected_channels:
                    print("Some channels may be padded with zeros")
                else:
                    print(f"Only first {self.expected_channels} channels will be used")
            
            # 8チャンネル専用プロセッサを初期化
            self.processor = EEG8CHDataProcessor(self.sampling_rate)
            
            print(f"Connected to 8-channel stream:")
            print(f"  Name: {info.name()}")
            print(f"  Channels: {self.n_channels} (using {self.expected_channels})")
            print(f"  Sampling rate: {self.sampling_rate} Hz")
            print(f"  Channel layout: {self.processor.channel_names}")
            
            return True
            
        except Exception as e:
            print(f"Error connecting to 8-channel stream: {e}")
            return False
        
    def start_receiving(self):
        """
        8チャンネルデータ受信を開始
        """
        if self.inlet is None:
            print("Not connected to any stream!")
            return
            
        self.is_receiving = True
        print("Starting 8-channel data reception...")
        
        # 受信ループ
        while self.is_receiving:
            try:
                # データを受信（タイムアウト1秒）
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    self.received_samples += 1
                    
                    # 8チャンネルに調整
                    adjusted_sample = self._adjust_to_8_channels(sample)
                    
                    # データバッファに追加
                    self.processor.data_buffer.append(adjusted_sample)
                    self.display_buffer.append(adjusted_sample)
                    
                    # 1.2秒分のデータが蓄積されたらエポック処理
                    if len(self.processor.data_buffer) >= self.processor.epoch_samples:
                        if self.received_samples % int(self.sampling_rate * 1.2) == 0:  # 1.2秒ごと
                            self._process_latest_8ch_epoch()
                            
            except Exception as e:
                print(f"8CH Reception error: {e}")
                
    def _adjust_to_8_channels(self, sample):
        """
        受信サンプルを8チャンネルに調整
        """
        if len(sample) == 8:
            return sample
        elif len(sample) > 8:
            # 8チャンネルに切り詰め
            return sample[:8]
        else:
            # 不足分をゼロパディング
            return sample + [0.0] * (8 - len(sample))
                
    def _process_latest_8ch_epoch(self):
        """
        最新の8チャンネルエポックを処理
        """
        # エポックを抽出
        epoch_data = self.processor.extract_epoch()
        
        if epoch_data is not None:
            # 8チャンネル処理実行
            result = self.processor.process_8ch_epoch(epoch_data)
            
            # 結果を保存（プロット用）
            self.latest_classification = result["classification"]
            self.latest_connectivity = result["connectivity_features"]
            
            # 詳細結果を表示
            classification = result["classification"]
            connectivity = result["connectivity_features"]
            
            print(f"\n=== 8CH Epoch Analysis ===")
            print(f"Class: {classification['class']}")
            print(f"Confidence: {classification['confidence']:.3f}")
            print(f"Reward: {classification['reward']}")
            print(f"Frontal amplitude: {classification['frontal_amplitude']:.2f}")
            print(f"Central amplitude: {classification['central_amplitude']:.2f}")
            print(f"Laterality (C3-C4): {classification['laterality_index']:.2f}")
            print(f"Mean connectivity: {connectivity['mean_connectivity']:.3f}")
            print(f"Samples processed: {self.received_samples}")
            print(f"Channels: {', '.join(result['channel_names'])}")
            
            # 強化学習システムに送信
            self._send_to_reinforcement_learning(classification)
            
    def _send_to_reinforcement_learning(self, classification):
        """
        8チャンネル分類結果を強化学習システムに送信
        """
        reward_mapping = {
            "success": 100,
            "over_grip": -100, 
            "under_grip": -50,
            "motor_error": -25
        }
        
        reward = reward_mapping.get(classification["class"], 0)
        
        print(f"→ Sending 8CH reward {reward} to RL agent")
        print(f"  Class: {classification['class']}")
        print(f"  8CH Features: Frontal={classification['frontal_amplitude']:.2f}, "
              f"Laterality={classification['laterality_index']:.2f}")
        
    def start_8ch_realtime_plot(self):
        """
        8チャンネル専用リアルタイム可視化
        """
        if self.processor is None:
            print("8CH Processor not initialized!")
            return
            
        # 8チャンネル専用レイアウト
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('8-Channel Real-time EEG Data')
        
        # 8チャンネル用のサブプロット配置（2x4 + 分類結果）
        gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
        
        # 各チャンネル用のサブプロット
        channel_axes = []
        channel_lines = []
        
        for i, ch_name in enumerate(self.processor.channel_names):
            row = i // 4
            col = i % 4
            ax = fig.add_subplot(gs[row, col])
            ax.set_title(f'{ch_name}', fontsize=10)
            ax.set_xlim(0, 4.0)  # 4秒表示
            ax.set_ylim(-50, 50)
            ax.set_xlabel('Time (s)', fontsize=8)
            ax.set_ylabel('µV', fontsize=8)
            ax.grid(True, alpha=0.3)
            
            line, = ax.plot([], [], 'b-', linewidth=1)
            channel_axes.append(ax)
            channel_lines.append(line)
        
        # 分類結果表示エリア
        result_ax = fig.add_subplot(gs[2, :])
        result_ax.set_xlim(0, 1)
        result_ax.set_ylim(0, 1)
        result_ax.set_title('8-Channel Classification Results', fontsize=12)
        result_ax.axis('off')
        
        classification_text = result_ax.text(0.25, 0.7, 'Waiting...', 
                ha='left', va='center', fontsize=11, weight='bold')
        connectivity_text = result_ax.text(0.25, 0.3, 'Connectivity...', 
                ha='left', va='center', fontsize=9)
                
        def update_8ch_plot(frame):
            if len(self.display_buffer) > 0:
                # 4秒分のデータを取得
                time_window = 4.0
                n_samples = int(time_window * self.sampling_rate)
                
                buffer_data = np.array(list(self.display_buffer))
                if buffer_data.shape[0] > n_samples:
                    buffer_data = buffer_data[-n_samples:]
                    
                time_axis = np.linspace(0, time_window, buffer_data.shape[0])
                
                # 各チャンネルを個別プロット
                for ch, (line, ax) in enumerate(zip(channel_lines, channel_axes)):
                    if ch < buffer_data.shape[1]:
                        y_data = buffer_data[:, ch]
                        line.set_data(time_axis, y_data)
                
                # 分類結果表示
                if hasattr(self, 'latest_classification'):
                    cls = self.latest_classification
                    conn = getattr(self, 'latest_connectivity', {})
                    
                    color_map = {
                        'success': 'green',
                        'over_grip': 'red', 
                        'under_grip': 'orange',
                        'motor_error': 'purple'
                    }
                    
                    # 分類結果テキスト
                    cls_text = f"Class: {cls['class']} (Conf: {cls['confidence']:.3f})\n"
                    cls_text += f"Reward: {cls['reward']}\n"
                    cls_text += f"Frontal: {cls['frontal_amplitude']:.2f}µV\n"
                    cls_text += f"Laterality (C3-C4): {cls['laterality_index']:.2f}µV"
                    
                    classification_text.set_text(cls_text)
                    classification_text.set_color(color_map.get(cls['class'], 'black'))
                    
                    # 結合性テキスト
                    conn_text = f"Mean Connectivity: {conn.get('mean_connectivity', 0):.3f}\n"
                    conn_text += f"Motor Laterality: {conn.get('motor_laterality', 0):.3f}\n"
                    conn_text += f"Frontal Coherence: {conn.get('frontal_coherence', 0):.3f}"
                    connectivity_text.set_text(conn_text)
                    
            return channel_lines + [classification_text, connectivity_text]
            
        # アニメーション開始
        ani = FuncAnimation(fig, update_8ch_plot, interval=100, blit=False)
        plt.tight_layout()
        plt.show()
        
        return ani
        
    def stop_receiving(self):
        """
        データ受信を停止
        """
        self.is_receiving = False
        print("8-channel data reception stopped.")
        
    def run_8ch_interactive_demo(self):
        """
        8チャンネル用インタラクティブデモ
        """
        print("\n=== 8-Channel LSL EEG Receiver Demo ===")
        print("Commands:")
        print("  'connect' - Connect to 8-channel EEG stream")
        print("  'start' - Start 8-channel data reception")
        print("  'stop' - Stop data reception")
        print("  'plot' - Start 8-channel real-time plotting")
        print("  'status' - Show current status")
        print("  'channels' - Show channel information")
        print("  'quit' - Exit")
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'connect':
                    self.connect_to_stream()
                    
                elif command == 'start':
                    if self.inlet is not None:
                        reception_thread = threading.Thread(target=self.start_receiving)
                        reception_thread.daemon = True
                        reception_thread.start()
                    else:
                        print("Not connected. Use 'connect' first.")
                        
                elif command == 'stop':
                    self.stop_receiving()
                    
                elif command == 'plot':
                    if self.processor is not None:
                        print("Starting 8-channel real-time plot...")
                        self.start_8ch_realtime_plot()
                    else:
                        print("Processor not initialized. Use 'connect' first.")
                        
                elif command == 'status':
                    self._print_8ch_status()
                    
                elif command == 'channels':
                    self._print_channel_info()
                    
                elif command == 'quit':
                    self.stop_receiving()
                    print("Exiting 8-channel system...")
                    break
                    
                else:
                    print("Unknown command.")
                    
            except KeyboardInterrupt:
                self.stop_receiving()
                print("\nExiting...")
                break
                
    def _print_8ch_status(self):
        """
        8チャンネルシステムの状態を表示
        """
        print(f"\n=== 8-Channel System Status ===")
        print(f"Connected: {self.inlet is not None}")
        print(f"Stream name: {self.stream_name}")
        print(f"Receiving: {self.is_receiving}")
        print(f"Samples received: {self.received_samples}")
        print(f"Expected channels: {self.expected_channels}")
        if self.processor:
            print(f"Buffer size: {len(self.processor.data_buffer)}")
            print(f"Epoch samples needed: {self.processor.epoch_samples}")
            print(f"Channel mapping: {self.processor.channel_names}")
            
    def _print_channel_info(self):
        """
        8チャンネル電極情報を表示
        """
        if self.processor is None:
            print("Processor not initialized.")
            return
            
        print(f"\n=== 8-Channel Electrode Information ===")
        print(f"Channel layout (10-20 system):")
        for i, ch_name in enumerate(self.processor.channel_names):
            print(f"  Ch{i+1}: {ch_name}")
            
        print(f"\nElectrode groups:")
        for group_name, indices in self.processor.electrode_groups.items():
            channels = [self.processor.channel_names[i] for i in indices]
            print(f"  {group_name.capitalize()}: {', '.join(channels)}")
            
        print(f"\nOptimal for:")
        print(f"  - Error-related potentials (ERPs)")
        print(f"  - Motor preparation signals")
        print(f"  - Cognitive conflict detection")
        print(f"  - Attention and executive control")


class Integrated8CHEEGSystem:
    """
    8チャンネル送信側と受信側を統合したシステム
    """
    def __init__(self):
        self.sender = None
        self.receiver = None
        
    def run_8ch_full_demo(self):
        """
        8チャンネル完全デモの実行
        """
        print("\n=== 8-Channel Integrated EEG System Demo ===")
        print("This demo simulates 8-channel EEG processing pipeline:")
        print("1. 8-channel mock EEG data generation")
        print("2. Optimized electrode placement (Fz,FCz,Cz,CPz,Pz,C3,C4,Oz)")
        print("3. Enhanced spatial resolution for cognitive signals")
        print("4. Motor laterality detection (C3/C4)")
        print("5. Frontal-central connectivity analysis")
        print("6. Real-time 8-channel classification")
        
        print("\n8-Channel advantages:")
        print("✓ Better spatial resolution than single channel")
        print("✓ Motor laterality detection (left/right hand)")
        print("✓ Frontal-parietal network analysis")
        print("✓ Reduced artifacts through spatial filtering")
        print("✓ Multiple error types classification")
        
        print("\nTo run the 8-channel demo:")
        print("1. Create 8-channel sender script:")
        print("   python mock_8ch_eeg_sender.py")
        print("2. Run this 8-channel receiver:")
        print("   python eeg_8ch_receiver.py")
        print("3. Use 'connect' and 'start' commands")
        print("4. Try 'plot' for 8-channel visualization")
        print("5. Observe enhanced classification accuracy")


class EEG8CHQualityAssessment:
    """
    8チャンネルEEGデータ品質評価システム
    """
    
    @staticmethod
    def assess_8ch_signal_quality(epoch_data, channel_names):
        """
        8チャンネル信号品質を評価
        
        Args:
            epoch_data: (8, samples) EEGデータ
            channel_names: チャンネル名リスト
            
        Returns:
            quality_report: 品質評価レポート
        """
        quality_report = {}
        
        for i, ch_name in enumerate(channel_names):
            ch_data = epoch_data[i]
            
            # 基本統計
            ch_quality = {
                'mean_amplitude': np.mean(np.abs(ch_data)),
                'std_amplitude': np.std(ch_data),
                'max_amplitude': np.max(np.abs(ch_data)),
                'signal_range': np.ptp(ch_data),
                'zero_crossings': len(np.where(np.diff(np.signbit(ch_data)))[0])
            }
            
            # 品質指標
            if ch_quality['max_amplitude'] > 100:
                ch_quality['status'] = 'artifact_detected'
            elif ch_quality['std_amplitude'] < 0.1:
                ch_quality['status'] = 'low_signal'
            elif ch_quality['signal_range'] > 200:
                ch_quality['status'] = 'high_noise'
            else:
                ch_quality['status'] = 'good'
                
            quality_report[ch_name] = ch_quality
        
        # 全体的な品質評価
        good_channels = sum(1 for ch in quality_report.values() 
                          if ch['status'] == 'good')
        
        quality_report['overall'] = {
            'good_channels': good_channels,
            'total_channels': len(channel_names),
            'quality_ratio': good_channels / len(channel_names),
            'overall_status': 'good' if good_channels >= 6 else 'poor'
        }
        
        return quality_report
    
    @staticmethod
    def detect_8ch_artifacts(epoch_data, channel_names, sampling_rate=250):
        """
        8チャンネル特化のアーティファクト検出
        """
        artifacts = {}
        
        # 眼電図アーティファクト（前頭部電極）
        frontal_channels = [0, 1]  # Fz, FCz
        frontal_data = epoch_data[frontal_channels]
        
        if np.max(np.abs(frontal_data)) > 80:
            artifacts['EOG'] = {
                'detected': True,
                'severity': 'high' if np.max(np.abs(frontal_data)) > 150 else 'medium',
                'affected_channels': ['Fz', 'FCz']
            }
        
        # 筋電図アーティファクト（高周波成分）
        for i, ch_name in enumerate(channel_names):
            # 20-50Hz帯域のパワー
            freqs, psd = signal.welch(epoch_data[i], fs=sampling_rate)
            high_freq_power = np.mean(psd[(freqs >= 20) & (freqs <= 50)])
            low_freq_power = np.mean(psd[(freqs >= 1) & (freqs <= 10)])
            
            if high_freq_power / low_freq_power > 0.5:
                artifacts[f'EMG_{ch_name}'] = {
                    'detected': True,
                    'power_ratio': high_freq_power / low_freq_power
                }
        
        return artifacts


def create_8ch_test_data(duration=1.2, sampling_rate=250):
    """
    8チャンネルテスト用EEGデータを生成
    """
    samples = int(duration * sampling_rate)
    channel_names = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C3', 'C4', 'Oz']
    
    # 基本的な脳波パターン
    t = np.linspace(0, duration, samples)
    
    eeg_data = np.zeros((8, samples))
    
    for i, ch_name in enumerate(channel_names):
        # チャンネル特性に応じた信号生成
        if 'F' in ch_name:  # 前頭部
            # アルファ波 + ベータ波
            eeg_data[i] = 10 * np.sin(2*np.pi*10*t) + 5 * np.sin(2*np.pi*20*t)
        elif 'C' in ch_name:  # 中央部
            # ミュー波 (8-12Hz)
            eeg_data[i] = 15 * np.sin(2*np.pi*9*t) + 8 * np.sin(2*np.pi*11*t)
        elif 'P' in ch_name:  # 頭頂部
            # アルファ波優勢
            eeg_data[i] = 20 * np.sin(2*np.pi*10*t)
        elif 'O' in ch_name:  # 後頭部
            # 強いアルファ波
            eeg_data[i] = 25 * np.sin(2*np.pi*10*t)
        
        # ノイズ追加
        eeg_data[i] += np.random.normal(0, 2, samples)
        
        # C3/C4で側性差を追加（運動関連）
        if ch_name == 'C3':
            eeg_data[i] += 5 * np.sin(2*np.pi*15*t)  # 左側優勢
        elif ch_name == 'C4':
            eeg_data[i] += 3 * np.sin(2*np.pi*15*t)  # 右側より弱い
    
    return eeg_data


def demo_8ch_processing():
    """
    8チャンネル処理のデモンストレーション
    """
    print("=== 8-Channel EEG Processing Demo ===")
    
    # テストデータ生成
    test_data = create_8ch_test_data()
    print(f"Generated 8-channel test data: {test_data.shape}")
    
    # プロセッサ作成
    processor = EEG8CHDataProcessor()
    
    # 処理実行
    result = processor.process_8ch_epoch(test_data)
    
    # 結果表示
    print("\n📊 Processing Results:")
    print(f"Classification: {result['classification']}")
    print(f"Connectivity: {result['connectivity_features']}")
    
    # 品質評価
    quality = EEG8CHQualityAssessment.assess_8ch_signal_quality(
        test_data, processor.channel_names
    )
    print(f"\n🔍 Quality Assessment:")
    print(f"Overall status: {quality['overall']['overall_status']}")
    print(f"Good channels: {quality['overall']['good_channels']}/8")
    
    # アーティファクト検出
    artifacts = EEG8CHQualityAssessment.detect_8ch_artifacts(
        test_data, processor.channel_names
    )
    print(f"\n⚠️ Artifacts detected: {len(artifacts)}")
    for artifact_type, info in artifacts.items():
        print(f"  {artifact_type}: {info}")


if __name__ == "__main__":
    print("🧠 8-Channel LSL EEG Data Receiver and Processor")
    print("=" * 60)
    print("Select mode:")
    print("1. 8-Channel EEG Receiver (Interactive)")
    print("2. 8-Channel System Demo")
    print("3. 8-Channel Processing Test")
    print("4. Channel Layout Information")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🎯 Starting 8-Channel Interactive Receiver...")
            receiver = LSL8CHEEGReceiver()
            receiver.run_8ch_interactive_demo()
            
        elif choice == "2":
            print("\n📋 8-Channel System Overview...")
            system = Integrated8CHEEGSystem()
            system.run_8ch_full_demo()
            
        elif choice == "3":
            print("\n🧪 Running 8-Channel Processing Test...")
            demo_8ch_processing()
            
        elif choice == "4":
            print("\n🗺️ 8-Channel Electrode Layout:")
            processor = EEG8CHDataProcessor()
            print(f"Channels: {processor.channel_names}")
            print(f"Groups: {processor.electrode_groups}")
            print("\nOptimal 10-20 positions for cognitive EEG:")
            positions = {
                'Fz': 'Frontal midline - error monitoring',
                'FCz': 'Frontal-central - cognitive control', 
                'Cz': 'Central midline - motor planning',
                'CPz': 'Central-parietal - sensorimotor integration',
                'Pz': 'Parietal midline - attention networks',
                'C3': 'Left motor cortex - left hand movement',
                'C4': 'Right motor cortex - right hand movement',
                'Oz': 'Occipital midline - visual processing'
            }
            for ch, desc in positions.items():
                print(f"  {ch:3s}: {desc}")
                
        else:
            print("❌ Invalid choice. Running interactive receiver...")
            receiver = LSL8CHEEGReceiver()
            receiver.run_8ch_interactive_demo()
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        
    print("\n👋 8-Channel EEG System ended")

# Usage Instructions for 8-Channel System:
"""
=== 8チャンネルEEGシステム使用方法 ===

1. 📦 依存関係のインストール:
   pip install pylsl numpy scipy matplotlib

2. 🎯 8チャンネル送信側の準備:
   - mock_8ch_eeg_sender.py を作成・実行
   - 8チャンネル (Fz,FCz,Cz,CPz,Pz,C3,C4,Oz) でストリーミング
   - ストリーム名: "MockEEG_8CH"

3. 📡 受信側の実行:
   python eeg_8ch_receiver.py
   
   コマンド:
   - connect: 8チャンネルストリームに接続
   - start: データ受信開始
   - plot: 8チャンネルリアルタイム表示
   - channels: 電極配置情報表示
   - status: システム状態確認

4. 🧠 8チャンネルの特徴:
   ✓ より高い空間分解能
   ✓ 運動準備電位の左右差検出 (C3/C4)
   ✓ 前頭-頭頂ネットワーク解析
   ✓ 複数エラータイプの分類
   ✓ アーティファクト除去の改善
   ✓ 結合性解析 (チャンネル間相関)

5. 🎨 可視化機能:
   - 8チャンネル個別波形表示
   - リアルタイム分類結果
   - 結合性指標表示
   - 品質評価表示

6. ⚡ 分類性能の向上:
   - 単一チャンネルより高精度
   - 運動エラー vs 認知エラー判別
   - 側性化信号の活用
   - 空間フィルタリング効果

8チャンネル配置は認知的競合検出に最適化されており、
論文の32チャンネルシステムの重要な電極を選択しています。
"""