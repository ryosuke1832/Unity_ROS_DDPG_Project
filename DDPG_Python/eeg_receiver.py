#!/usr/bin/env python3
"""
LSL EEG Data Receiver and Processor
LSLからEEGデータを受信し、論文と同様の処理を行うシステム

このスクリプトは：
1. LSLからEEGデータを受信
2. 論文で使用されている前処理を適用（2-50Hzフィルタリング）
3. 1.2秒のエポックを抽出
4. CNNによる分類（模擬）
5. リアルタイム可視化

Requirements:
- pylsl (pip install pylsl)
- numpy (pip install numpy)
- scipy (pip install scipy)
- matplotlib (pip install matplotlib)
- mne (pip install mne) - オプション：高度な前処理用
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

class EEGDataProcessor:
    def __init__(self, sampling_rate=250, epoch_length=1.2):
        """
        EEGデータ処理システムの初期化
        
        Args:
            sampling_rate: サンプリング周波数（Hz）
            epoch_length: エポック長（秒）
        """
        self.sampling_rate = sampling_rate
        self.epoch_length = epoch_length
        self.epoch_samples = int(sampling_rate * epoch_length)
        
        # 論文に基づくフィルター設計（2-50Hz）
        self.lowcut = 2.0
        self.highcut = 50.0
        self.filter_order = 4
        
        # バターワースフィルターの係数を計算
        nyquist = sampling_rate / 2
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        self.b, self.a = butter(self.filter_order, [low, high], btype='band')
        
        # データバッファ
        self.data_buffer = deque(maxlen=self.epoch_samples * 3)  # 3エポック分のバッファ
        self.processed_epochs = queue.Queue()
        
        print(f"EEG Processor initialized:")
        print(f"  Sampling rate: {sampling_rate} Hz")
        print(f"  Epoch length: {epoch_length} s ({self.epoch_samples} samples)")
        print(f"  Bandpass filter: {self.lowcut}-{self.highcut} Hz")
        
    def apply_preprocessing(self, data):
        """
        論文に基づく前処理を適用
        
        Args:
            data: EEGデータ (channels x samples)
            
        Returns:
            preprocessed_data: 前処理済みデータ
        """
        if data.shape[1] < self.filter_order * 3:
            return data  # データが短すぎる場合はそのまま返す
            
        # 2-50Hzバンドパスフィルター
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            try:
                filtered_data[ch] = filtfilt(self.b, self.a, data[ch])
            except Exception as e:
                print(f"Filtering error for channel {ch}: {e}")
                filtered_data[ch] = data[ch]  # フィルタリングに失敗した場合は元データ
                
        return filtered_data
        
    def extract_epoch(self, trigger_idx=None):
        """
        エポックを抽出
        
        Args:
            trigger_idx: トリガーのインデックス（Noneの場合は最新のデータ）
            
        Returns:
            epoch_data: エポックデータ (channels x epoch_samples)
        """
        if len(self.data_buffer) < self.epoch_samples:
            return None
            
        # バッファからデータを取得
        buffer_array = np.array(self.data_buffer)  # (samples, channels)
        buffer_array = buffer_array.T  # (channels, samples)
        
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
                padded_data = np.zeros((epoch_data.shape[0], self.epoch_samples))
                padded_data[:, :epoch_data.shape[1]] = epoch_data
                epoch_data = padded_data
                
        return epoch_data
        
    def compute_power_spectral_density(self, epoch_data):
        """
        パワースペクトル密度を計算（論文で使用されている手法）
        
        Args:
            epoch_data: エポックデータ (channels x samples)
            
        Returns:
            psd_features: PSD特徴量
        """
        psd_features = {}
        
        # 周波数帯域の定義（論文に基づく）
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 13),
            'beta': (13, 30)
        }
        
        for ch in range(epoch_data.shape[0]):
            channel_psd = {}
            
            # ウェルチ法でPSDを計算
            nperseg = min(256, epoch_data.shape[1])
            noverlap = nperseg // 2  # オーバーラップは通常npersegの半分
            
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
                
            psd_features[f'ch_{ch}'] = channel_psd
            
        return psd_features
        
    def classify_cognitive_conflict(self, epoch_data):
        """
        認知的競合を分類（模擬CNN分類器）
        論文のDeepConvNetに基づく3クラス分類
        
        Args:
            epoch_data: エポックデータ (channels x samples)
            
        Returns:
            classification: {"class": str, "confidence": float, "reward": int}
        """
        # 簡単な特徴量抽出（実際のCNNの代わり）
        
        # 前頭部チャンネル（Fz周辺）に注目
        frontal_channels = [4, 8, 9, 13]  # Fz, FC1, FC2, Cz approximation
        
        # 250-400ms区間の平均振幅を計算（ErrPの典型的な時間窓）
        start_idx = int(0.25 * self.sampling_rate)  # 250ms
        end_idx = int(0.4 * self.sampling_rate)     # 400ms
        
        if end_idx > epoch_data.shape[1]:
            end_idx = epoch_data.shape[1]
            
        frontal_amplitude = np.mean(epoch_data[frontal_channels, start_idx:end_idx])
        
        # 分類ルール（簡単化された版）
        if frontal_amplitude < -10:
            # 強い負の成分 = over-grip error
            class_name = "over_grip"
            confidence = min(0.95, abs(frontal_amplitude) / 15)
            reward = -100
        elif frontal_amplitude < -5:
            # 中程度の負の成分 = under-grip error  
            class_name = "under_grip"
            confidence = min(0.9, abs(frontal_amplitude) / 10)
            reward = 50
        else:
            # 正常または成功
            class_name = "success"
            confidence = 0.8
            reward = 100
            
        return {
            "class": class_name,
            "confidence": confidence,
            "reward": reward,
            "frontal_amplitude": frontal_amplitude
        }
        
    def process_epoch(self, epoch_data):
        """
        エポック全体の処理パイプライン
        """
        # 前処理
        preprocessed = self.apply_preprocessing(epoch_data)
        
        # PSD特徴量計算
        psd_features = self.compute_power_spectral_density(preprocessed)
        
        # 認知的競合分類
        classification = self.classify_cognitive_conflict(preprocessed)
        
        return {
            "preprocessed_data": preprocessed,
            "psd_features": psd_features,
            "classification": classification,
            "timestamp": time.time()
        }


class LSLEEGReceiver:
    def __init__(self, stream_name=" X.on-102807-0109"):
        """
        LSL EEG受信システムの初期化
        
        Args:
            stream_name: 受信するストリーム名
        """
        self.stream_name = stream_name
        self.inlet = None
        self.processor = None
        
        # 受信状態
        self.is_receiving = False
        self.received_samples = 0
        
        # リアルタイム表示用
        self.display_buffer = deque(maxlen=1000)  # 4秒分の表示バッファ
        
    def connect_to_stream(self):
        """
        LSLストリームに接続
        """
        print(f"Looking for stream '{self.stream_name}'...")
        
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
                    print(f"  - {stream.name()} ({stream.type()})")
                print("Make sure the sender is running and streaming has started.")
                return False
                
            # ストリームに接続
            self.inlet = StreamInlet(target_stream)
            
            # ストリーム情報を取得
            info = self.inlet.info()
            self.n_channels = info.channel_count()
            self.sampling_rate = int(info.nominal_srate())
            
            # プロセッサを初期化
            self.processor = EEGDataProcessor(self.sampling_rate)
            
            print(f"Connected to stream:")
            print(f"  Name: {info.name()}")
            print(f"  Channels: {self.n_channels}")
            print(f"  Sampling rate: {self.sampling_rate} Hz")
            
            return True
            
        except Exception as e:
            print(f"Error connecting to stream: {e}")
            print("Make sure:")
            print("1. The sender script is running")
            print("2. 'start' command was executed in sender")
            print("3. Both scripts are on the same network")
            return False
        
    def start_receiving(self):
        """
        データ受信を開始
        """
        if self.inlet is None:
            print("Not connected to any stream!")
            return
            
        self.is_receiving = True
        print("Starting data reception...")
        
        # 受信ループ
        while self.is_receiving:
            try:
                # データを受信（タイムアウト1秒）
                sample, timestamp = self.inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    self.received_samples += 1
                    
                    # データバッファに追加
                    self.processor.data_buffer.append(sample)
                    self.display_buffer.append(sample)
                    
                    # 1.2秒分のデータが蓄積されたらエポック処理
                    if len(self.processor.data_buffer) >= self.processor.epoch_samples:
                        if self.received_samples % (self.sampling_rate * 1.2) == 0:  # 1.2秒ごと
                            self._process_latest_epoch()
                            
            except Exception as e:
                print(f"Reception error: {e}")
                
    def _process_latest_epoch(self):
        """
        最新のエポックを処理
        """
        # エポックを抽出
        epoch_data = self.processor.extract_epoch()
        
        if epoch_data is not None:
            # 処理実行
            result = self.processor.process_epoch(epoch_data)
            
            # 結果を表示
            classification = result["classification"]
            print(f"\n=== Epoch Analysis ===")
            print(f"Class: {classification['class']}")
            print(f"Confidence: {classification['confidence']:.3f}")
            print(f"Reward: {classification['reward']}")
            print(f"Frontal amplitude: {classification['frontal_amplitude']:.2f}")
            print(f"Samples processed: {self.received_samples}")
            
    def stop_receiving(self):
        """
        データ受信を停止
        """
        self.is_receiving = False
        print("Data reception stopped.")
        
    def start_realtime_plot(self):
        """
        リアルタイム可視化を開始
        """
        if self.processor is None:
            print("Processor not initialized!")
            return
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle('Real-time EEG Data')
        
        # プロット初期化
        time_window = 4.0  # 4秒間の表示
        n_samples = int(time_window * self.sampling_rate)
        
        lines1 = []
        for ch in range(min(8, self.n_channels)):  # 最初の8チャンネルのみ表示
            line, = ax1.plot([], [], label=f'Ch {ch+1}')
            lines1.append(line)
            
        ax1.set_xlim(0, time_window)
        ax1.set_ylim(-50, 50)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (µV)')
        ax1.set_title('Raw EEG Channels (1-8)')
        ax1.legend()
        ax1.grid(True)
        
        # 分類結果の表示
        classification_text = ax2.text(0.5, 0.5, 'Waiting for classification...', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Classification Results')
        ax2.axis('off')
        
        def update_plot(frame):
            if len(self.display_buffer) > 0:
                # データバッファからプロット用データを準備
                buffer_data = np.array(list(self.display_buffer))
                if buffer_data.shape[0] > n_samples:
                    buffer_data = buffer_data[-n_samples:]
                    
                time_axis = np.linspace(0, time_window, buffer_data.shape[0])
                
                # 各チャンネルをプロット
                for ch, line in enumerate(lines1):
                    if ch < buffer_data.shape[1]:
                        # チャンネル間のオフセットを追加
                        y_data = buffer_data[:, ch] + ch * 20
                        line.set_data(time_axis, y_data)
                        
                # 最新の分類結果を表示
                if hasattr(self, 'latest_classification'):
                    cls = self.latest_classification
                    color_map = {
                        'success': 'green',
                        'over_grip': 'red', 
                        'under_grip': 'orange'
                    }
                    
                    text_content = f"Class: {cls['class']}\n"
                    text_content += f"Confidence: {cls['confidence']:.3f}\n"
                    text_content += f"Reward: {cls['reward']}\n"
                    text_content += f"Samples: {self.received_samples}"
                    
                    classification_text.set_text(text_content)
                    classification_text.set_color(color_map.get(cls['class'], 'black'))
                    
            return lines1 + [classification_text]
            
        # アニメーション開始
        ani = FuncAnimation(fig, update_plot, interval=50, blit=False)
        plt.tight_layout()
        plt.show()
        
        return ani
        
    def run_interactive_demo(self):
        """
        インタラクティブなデモを実行
        """
        print("\n=== LSL EEG Receiver Demo ===")
        print("Commands:")
        print("  'connect' - Connect to EEG stream")
        print("  'start' - Start data reception")
        print("  'stop' - Stop data reception")
        print("  'plot' - Start real-time plotting")
        print("  'status' - Show current status")
        print("  'quit' - Exit")
        
        plotting_thread = None
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'connect':
                    self.connect_to_stream()
                    
                elif command == 'start':
                    if self.inlet is not None:
                        # 受信を別スレッドで開始
                        reception_thread = threading.Thread(target=self.start_receiving)
                        reception_thread.daemon = True
                        reception_thread.start()
                    else:
                        print("Not connected to any stream. Use 'connect' first.")
                        
                elif command == 'stop':
                    self.stop_receiving()
                    
                elif command == 'plot':
                    if self.processor is not None:
                        print("Starting real-time plot... (Close plot window to continue)")
                        self.start_realtime_plot()
                    else:
                        print("Processor not initialized. Use 'connect' first.")
                        
                elif command == 'status':
                    self._print_status()
                    
                elif command == 'quit':
                    self.stop_receiving()
                    print("Exiting...")
                    break
                    
                else:
                    print("Unknown command.")
                    
            except KeyboardInterrupt:
                self.stop_receiving()
                print("\nExiting...")
                break
                
    def _process_latest_epoch(self):
        """
        最新のエポックを処理（更新版）
        """
        # エポックを抽出
        epoch_data = self.processor.extract_epoch()
        
        if epoch_data is not None:
            # 処理実行
            result = self.processor.process_epoch(epoch_data)
            
            # 結果を保存（プロット用）
            self.latest_classification = result["classification"]
            
            # 結果を表示
            classification = result["classification"]
            print(f"\n=== Epoch Analysis ===")
            print(f"Class: {classification['class']}")
            print(f"Confidence: {classification['confidence']:.3f}")
            print(f"Reward: {classification['reward']}")
            print(f"Frontal amplitude: {classification['frontal_amplitude']:.2f}")
            print(f"Samples processed: {self.received_samples}")
            
            # 強化学習システムに送信する場合のシミュレーション
            self._send_to_reinforcement_learning(classification)
            
    def _send_to_reinforcement_learning(self, classification):
        """
        強化学習システムに分類結果を送信（シミュレーション）
        """
        # 実際の実装では、ここでDDPGエージェントに報酬を送信
        reward_mapping = {
            "success": 100,
            "over_grip": -100, 
            "under_grip": 50
        }
        
        reward = reward_mapping.get(classification["class"], 0)
        
        # ここで実際のDDPGシステムと通信
        print(f"→ Sending reward {reward} to RL agent")
        
    def _print_status(self):
        """
        現在の状態を表示
        """
        print(f"\n=== System Status ===")
        print(f"Connected: {self.inlet is not None}")
        print(f"Receiving: {self.is_receiving}")
        print(f"Samples received: {self.received_samples}")
        if self.processor:
            print(f"Buffer size: {len(self.processor.data_buffer)}")
            print(f"Epoch samples needed: {self.processor.epoch_samples}")
        
        
class IntegratedEEGSystem:
    """
    送信側と受信側を統合したテストシステム
    """
    def __init__(self):
        self.sender = None
        self.receiver = None
        
    def run_full_demo(self):
        """
        送信と受信の完全なデモを実行
        """
        print("\n=== Integrated EEG System Demo ===")
        print("This demo simulates the complete pipeline from the research paper:")
        print("1. Mock EEG data generation with ErrP signals")
        print("2. LSL streaming")
        print("3. Real-time processing and classification")
        print("4. Reinforcement learning integration")
        
        # 送信側を別プロセスまたはスレッドで開始することを推奨
        print("\nTo run the full demo:")
        print("1. Run the sender script in one terminal:")
        print("   python mock_eeg_sender.py")
        print("2. Run the receiver script in another terminal:")
        print("   python eeg_receiver.py")
        print("3. Use 'connect' and 'start' in receiver")
        print("4. Use different conditions in sender (success, over, under)")
        print("5. Observe real-time classification results")


if __name__ == "__main__":
    print("Select mode:")
    print("1. EEG Receiver only")
    print("2. Integrated system info")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        receiver = LSLEEGReceiver()
        receiver.run_interactive_demo()
    elif choice == "2":
        system = IntegratedEEGSystem()
        system.run_full_demo()
    else:
        print("Invalid choice")
        
# Usage Instructions:
"""
=== 使用方法 ===

1. 依存関係のインストール:
   pip install pylsl numpy scipy matplotlib

2. 送信側の実行（別ターミナル）:
   python mock_eeg_sender.py
   
   コマンド:
   - start: ストリーミング開始
   - success/over/under: 把持条件を変更
   - marker X: マーカー送信

3. 受信側の実行:
   python eeg_receiver.py
   
   コマンド:
   - connect: ストリームに接続
   - start: データ受信開始
   - plot: リアルタイム可視化
   - status: 状態確認

4. 期待される結果:
   - リアルタイムEEGデータ表示
   - 自動的なErrP分類
   - 強化学習用の報酬値生成
   - 論文と同様の処理パイプライン

5. 統合テスト:
   - 送信側で条件を変更
   - 受信側で分類結果を確認
   - 異なる把持条件での信号パターンを観察
"""