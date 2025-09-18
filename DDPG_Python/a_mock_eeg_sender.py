#!/usr/bin/env python3
"""
LSL Mock EEG Data Sender
LSLを使って模擬EEGデータを送信するシステム

このスクリプトは実際のEEGデバイスなしで、
論文で使用されているような32チャンネルEEGデータを模擬的に生成し、
LSL経由で送信します。

Requirements:
- pylsl (pip install pylsl)
- numpy (pip install numpy)
- scipy (pip install scipy)
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock
from scipy import signal
import threading
import random

class MockEEGSender:
    def __init__(self, 
                 n_channels=32, 
                 sampling_rate=250, 
                 stream_name="MockEEG",
                 channel_format='float32'):
        """
        模擬EEGデータ送信システムの初期化
        
        Args:
            n_channels: チャンネル数（論文では32チャンネル）
            sampling_rate: サンプリング周波数（論文では250Hz）
            stream_name: LSLストリーム名
            channel_format: データ形式
        """
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.stream_name = stream_name
        
        # LSLストリーム情報を作成
        info = StreamInfo(
            name=stream_name,
            type='EEG',
            channel_count=n_channels,
            nominal_srate=sampling_rate,
            channel_format=channel_format,
            source_id='mock_eeg_12345'
        )
        
        # チャンネル情報を設定（10-20システムに基づく）
        self._setup_channel_info(info)
        
        # LSLアウトレットを作成
        self.outlet = StreamOutlet(info)
        
        # シミュレーション用のパラメータ
        self.is_running = False
        self.current_condition = "normal"  # normal, success, over_grip, under_grip
        self.time_counter = 0
        
        print(f"Mock EEG stream '{stream_name}' created with {n_channels} channels at {sampling_rate} Hz")
        print("Waiting for receivers...")
        
    def _setup_channel_info(self, info):
        """
        チャンネル情報をセットアップ（10-20電極配置システム）
        """
        # 32チャンネルの典型的な電極名
        channel_names = [
            'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5',
            'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
            'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
            'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
        ]
        
        channels = info.desc().append_child("channels")
        for i, name in enumerate(channel_names):
            ch = channels.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
            
    def _generate_background_eeg(self, duration_samples):
        """
        背景となるEEG信号を生成（アルファ、ベータ、ガンマ波など）
        """
        t = np.arange(duration_samples) / self.sampling_rate
        
        # 基本的な脳波成分を生成
        alpha_freq = 10  # アルファ波（8-12Hz）
        beta_freq = 20   # ベータ波（13-30Hz）
        theta_freq = 6   # シータ波（4-8Hz）
        delta_freq = 2   # デルタ波（0.5-4Hz）
        
        eeg_data = np.zeros((self.n_channels, duration_samples))
        
        for ch in range(self.n_channels):
            # 各チャンネルで異なる位相とアンプリチュードを持つ
            phase_shift = np.random.uniform(0, 2*np.pi)
            amplitude_factor = np.random.uniform(0.5, 1.5)
            
            # 複数の周波数成分を合成
            alpha_component = amplitude_factor * 10 * np.sin(2*np.pi*alpha_freq*t + phase_shift)
            beta_component = amplitude_factor * 5 * np.sin(2*np.pi*beta_freq*t + phase_shift + np.pi/4)
            theta_component = amplitude_factor * 15 * np.sin(2*np.pi*theta_freq*t + phase_shift + np.pi/2)
            delta_component = amplitude_factor * 20 * np.sin(2*np.pi*delta_freq*t + phase_shift + np.pi/3)
            
            # ノイズを追加
            noise = np.random.normal(0, 2, duration_samples)
            
            eeg_data[ch] = alpha_component + beta_component + theta_component + delta_component + noise
            
        return eeg_data
        
    def _generate_errp_signal(self, duration_samples, errp_type="success"):
        """
        Error-related Potential (ErrP)信号を生成
        論文で使用されている3つの条件に対応
        
        Args:
            duration_samples: サンプル数
            errp_type: "success", "over_grip", "under_grip"
        """
        t = np.arange(duration_samples) / self.sampling_rate
        
        # 背景のEEG信号を取得
        eeg_data = self._generate_background_eeg(duration_samples)
        
        # ErrPは主に前頭部中央（Fz, FCz, Cz）で観測される
        frontal_channels = [4, 8, 9, 13]  # Fz, FC1, FC2, Cz approximation
        
        if errp_type == "success":
            # 成功時：小さなP300様の成分
            for ch in frontal_channels:
                p300_latency = 0.3  # 300ms
                p300_amplitude = 5
                if p300_latency < t[-1]:
                    p300_idx = int(p300_latency * self.sampling_rate)
                    # ガウシアンパルスでP300を模擬
                    gaussian_pulse = p300_amplitude * signal.gaussian(50, std=10)
                    start_idx = max(0, p300_idx - 25)
                    end_idx = min(duration_samples, p300_idx + 25)
                    pulse_len = end_idx - start_idx
                    eeg_data[ch, start_idx:end_idx] += gaussian_pulse[:pulse_len]
                    
        elif errp_type == "over_grip":
            # 過度な把持：強いErrP（負の成分）
            for ch in frontal_channels:
                errp_latency = 0.25  # 250ms
                errp_amplitude = -15  # 負の成分
                if errp_latency < t[-1]:
                    errp_idx = int(errp_latency * self.sampling_rate)
                    # ErrPの典型的な形状
                    errp_pulse = errp_amplitude * signal.gaussian(40, std=8)
                    start_idx = max(0, errp_idx - 20)
                    end_idx = min(duration_samples, errp_idx + 20)
                    pulse_len = end_idx - start_idx
                    eeg_data[ch, start_idx:end_idx] += errp_pulse[:pulse_len]
                    
        elif errp_type == "under_grip":
            # 不十分な把持：中程度のErrP
            for ch in frontal_channels:
                errp_latency = 0.28  # 280ms
                errp_amplitude = -8  # 中程度の負の成分
                if errp_latency < t[-1]:
                    errp_idx = int(errp_latency * self.sampling_rate)
                    errp_pulse = errp_amplitude * signal.gaussian(35, std=7)
                    start_idx = max(0, errp_idx - 17)
                    end_idx = min(duration_samples, errp_idx + 18)
                    pulse_len = end_idx - start_idx
                    eeg_data[ch, start_idx:end_idx] += errp_pulse[:pulse_len]
        
        return eeg_data
        
    def set_condition(self, condition):
        """
        現在の条件を設定
        Args:
            condition: "normal", "success", "over_grip", "under_grip"
        """
        self.current_condition = condition
        print(f"Condition changed to: {condition}")
        
    def send_marker(self, marker_value):
        """
        イベントマーカーを送信（別のLSLストリームとして）
        """
        # マーカー用の別ストリームを作成（必要に応じて）
        marker_info = StreamInfo(
            name="MockEEG_Markers",
            type='Markers',
            channel_count=1,
            nominal_srate=0,  # irregular rate
            channel_format='string'
        )
        marker_outlet = StreamOutlet(marker_info)
        marker_outlet.push_sample([str(marker_value)], local_clock())
        print(f"Marker sent: {marker_value}")
        
    def start_streaming(self):
        """
        EEGデータのストリーミングを開始
        """
        self.is_running = True
        print("Starting EEG data streaming...")
        
        # 論文に基づく設定: 1.2秒のエポック長
        epoch_duration = 1.2  
        samples_per_chunk = int(self.sampling_rate * epoch_duration)
        
        while self.is_running:
            # 現在の条件に基づいてEEGデータを生成
            if self.current_condition == "normal":
                eeg_chunk = self._generate_background_eeg(samples_per_chunk)
            else:
                eeg_chunk = self._generate_errp_signal(samples_per_chunk, self.current_condition)
                
            # チャンク単位でデータを送信
            chunk_size = int(self.sampling_rate * 0.04)  # 40ms chunks
            
            for i in range(0, samples_per_chunk, chunk_size):
                if not self.is_running:
                    break
                    
                end_idx = min(i + chunk_size, samples_per_chunk)
                chunk = eeg_chunk[:, i:end_idx].T  # (samples, channels)
                
                # 各サンプルを個別に送信
                for sample in chunk:
                    self.outlet.push_sample(sample.tolist(), local_clock())
                    
                # リアルタイムに合わせて待機
                time.sleep(chunk_size / self.sampling_rate)
                
            self.time_counter += epoch_duration
            
    def stop_streaming(self):
        """
        ストリーミングを停止
        """
        self.is_running = False
        print("EEG streaming stopped.")


class MockEEGController:
    """
    模擬EEGシステムを制御するための簡単なコントローラー
    """
    def __init__(self):
        self.sender = MockEEGSender()
        
    def run_interactive_demo(self):
        """
        インタラクティブなデモを実行
        """
        print("\n=== Mock EEG System Demo ===")
        print("Commands:")
        print("  'start' - Start streaming")
        print("  'stop' - Stop streaming") 
        print("  'normal' - Set normal condition")
        print("  'success' - Set success condition (success grip)")
        print("  'over' - Set over-grip condition")
        print("  'under' - Set under-grip condition") 
        print("  'marker X' - Send marker X")
        print("  'quit' - Exit")
        print("\nThis simulates the 32-channel EEG system used in the paper.")
        print("The data includes realistic ErrP signals for different grip conditions.")
        
        # ストリーミングを別スレッドで開始
        streaming_thread = None
        
        while True:
            try:
                command = input("\nEnter command: ").strip().lower()
                
                if command == 'start':
                    if streaming_thread is None or not streaming_thread.is_alive():
                        streaming_thread = threading.Thread(target=self.sender.start_streaming)
                        streaming_thread.daemon = True
                        streaming_thread.start()
                    else:
                        print("Streaming is already running.")
                        
                elif command == 'stop':
                    self.sender.stop_streaming()
                    
                elif command == 'normal':
                    self.sender.set_condition('normal')
                    
                elif command == 'success':
                    self.sender.set_condition('success')
                    
                elif command == 'over':
                    self.sender.set_condition('over_grip')
                    
                elif command == 'under':
                    self.sender.set_condition('under_grip')
                    
                elif command.startswith('marker '):
                    marker_value = command.split(' ', 1)[1]
                    self.sender.send_marker(marker_value)
                    
                elif command == 'quit':
                    self.sender.stop_streaming()
                    print("Exiting...")
                    break
                    
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                self.sender.stop_streaming()
                print("\nExiting...")
                break


if __name__ == "__main__":
    controller = MockEEGController()
    controller.run_interactive_demo()