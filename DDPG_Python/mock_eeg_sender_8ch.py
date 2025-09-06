#!/usr/bin/env python3
"""
LSL Mock EEG Data Sender (8チャンネル版)
LSLを使って模擬EEGデータ（8チャンネル）を送信するシステム

このスクリプトは実際のEEGデバイスなしで、
8チャンネルEEGデータを模擬的に生成し、
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

class MockEEGSender8Ch:
    def __init__(self, 
                 n_channels=8, 
                 sampling_rate=250, 
                 stream_name="MockEEG_8CH",
                 channel_format='float32'):
        """
        模擬EEGデータ送信システムの初期化（8チャンネル版）
        
        Args:
            n_channels: チャンネル数（8チャンネル固定）
            sampling_rate: サンプリング周波数（250Hz）
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
            source_id='mock_eeg_8ch_12345'
        )
        
        # チャンネル情報を設定（8チャンネル用電極配置）
        self._setup_channel_info(info)
        
        # LSLアウトレットを作成
        self.outlet = StreamOutlet(info)
        
        # シミュレーション用のパラメータ
        self.is_running = False
        self.current_condition = "normal"  # normal, success, over_grip, under_grip
        self.time_counter = 0
        
        print(f"Mock EEG 8CH stream '{stream_name}' created with {n_channels} channels at {sampling_rate} Hz")
        print("Waiting for receivers...")
        
    def _setup_channel_info(self, info):
        """
        チャンネル情報をセットアップ（8チャンネル用10-20電極配置システム）
        前頭部・中央部・頭頂部の主要電極を選択
        """
        # 8チャンネルの主要電極名（認知機能・運動制御に重要な部位）
        channel_names = [
            'Fz',   # 前頭部中央（注意・実行機能）
            'F3',   # 左前頭部（言語・運動企画）
            'F4',   # 右前頭部（空間認知・運動企画）
            'Cz',   # 中央部（運動制御）
            'C3',   # 左中央部（右手運動制御）
            'C4',   # 右中央部（左手運動制御）
            'Pz',   # 頭頂部中央（感覚統合）
            'Oz'    # 後頭部中央（視覚処理）
        ]
        
        channels = info.desc().append_child("channels")
        for i, name in enumerate(channel_names):
            ch = channels.append_child("channel")
            ch.append_child_value("label", name)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
            
        print(f"📍 8チャンネル電極配置: {', '.join(channel_names)}")
        print(f"   主要部位: 前頭部(Fz,F3,F4), 中央部(Cz,C3,C4), 頭頂部(Pz), 後頭部(Oz)")
            
    def _generate_background_eeg(self, duration_samples):
        """
        背景となるEEG信号を生成（アルファ、ベータ、ガンマ波など）
        8チャンネル用に最適化
        """
        t = np.arange(duration_samples) / self.sampling_rate
        
        # 基本的な脳波成分を生成
        alpha_freq = 10  # アルファ波（8-12Hz）
        beta_freq = 20   # ベータ波（13-30Hz）
        theta_freq = 6   # シータ波（4-8Hz）
        delta_freq = 2   # デルタ波（0.5-4Hz）
        
        eeg_data = np.zeros((self.n_channels, duration_samples))
        
        # チャンネルごとの特性を定義
        channel_characteristics = {
            0: {'name': 'Fz', 'type': 'frontal', 'amplitude_factor': 1.0},    # 前頭部中央
            1: {'name': 'F3', 'type': 'frontal', 'amplitude_factor': 0.9},    # 左前頭部
            2: {'name': 'F4', 'type': 'frontal', 'amplitude_factor': 0.9},    # 右前頭部
            3: {'name': 'Cz', 'type': 'central', 'amplitude_factor': 1.2},    # 中央部
            4: {'name': 'C3', 'type': 'central', 'amplitude_factor': 1.1},    # 左中央部
            5: {'name': 'C4', 'type': 'central', 'amplitude_factor': 1.1},    # 右中央部
            6: {'name': 'Pz', 'type': 'parietal', 'amplitude_factor': 0.8},   # 頭頂部
            7: {'name': 'Oz', 'type': 'occipital', 'amplitude_factor': 1.3}   # 後頭部（アルファ波強い）
        }
        
        for ch in range(self.n_channels):
            ch_info = channel_characteristics[ch]
            
            # 各チャンネルで異なる位相とアンプリチュードを持つ
            phase_shift = np.random.uniform(0, 2*np.pi)
            base_amplitude = ch_info['amplitude_factor']
            
            # 部位による周波数成分の調整
            if ch_info['type'] == 'frontal':
                # 前頭部：ベータ波とシータ波が強い
                alpha_component = base_amplitude * 8 * np.sin(2*np.pi*alpha_freq*t + phase_shift)
                beta_component = base_amplitude * 12 * np.sin(2*np.pi*beta_freq*t + phase_shift + np.pi/4)
                theta_component = base_amplitude * 18 * np.sin(2*np.pi*theta_freq*t + phase_shift + np.pi/2)
                delta_component = base_amplitude * 15 * np.sin(2*np.pi*delta_freq*t + phase_shift + np.pi/3)
            elif ch_info['type'] == 'central':
                # 中央部：ミューリズム（8-12Hz）が特徴
                alpha_component = base_amplitude * 15 * np.sin(2*np.pi*alpha_freq*t + phase_shift)
                beta_component = base_amplitude * 10 * np.sin(2*np.pi*beta_freq*t + phase_shift + np.pi/4)
                theta_component = base_amplitude * 12 * np.sin(2*np.pi*theta_freq*t + phase_shift + np.pi/2)
                delta_component = base_amplitude * 18 * np.sin(2*np.pi*delta_freq*t + phase_shift + np.pi/3)
            elif ch_info['type'] == 'parietal':
                # 頭頂部：アルファ波が中程度
                alpha_component = base_amplitude * 12 * np.sin(2*np.pi*alpha_freq*t + phase_shift)
                beta_component = base_amplitude * 8 * np.sin(2*np.pi*beta_freq*t + phase_shift + np.pi/4)
                theta_component = base_amplitude * 10 * np.sin(2*np.pi*theta_freq*t + phase_shift + np.pi/2)
                delta_component = base_amplitude * 16 * np.sin(2*np.pi*delta_freq*t + phase_shift + np.pi/3)
            else:  # occipital
                # 後頭部：アルファ波が最も強い
                alpha_component = base_amplitude * 20 * np.sin(2*np.pi*alpha_freq*t + phase_shift)
                beta_component = base_amplitude * 6 * np.sin(2*np.pi*beta_freq*t + phase_shift + np.pi/4)
                theta_component = base_amplitude * 8 * np.sin(2*np.pi*theta_freq*t + phase_shift + np.pi/2)
                delta_component = base_amplitude * 12 * np.sin(2*np.pi*delta_freq*t + phase_shift + np.pi/3)
            
            # ノイズを追加
            noise = np.random.normal(0, 2, duration_samples)
            
            eeg_data[ch] = alpha_component + beta_component + theta_component + delta_component + noise
            
        return eeg_data
        
    def _generate_errp_signal(self, duration_samples, errp_type="success"):
        """
        Error-related Potential (ErrP)信号を生成（8チャンネル版）
        論文で使用されている3つの条件に対応
        
        Args:
            duration_samples: サンプル数
            errp_type: "success", "over_grip", "under_grip"
        """
        t = np.arange(duration_samples) / self.sampling_rate
        
        # 背景のEEG信号を取得
        eeg_data = self._generate_background_eeg(duration_samples)
        
        # ErrPは主に前頭部中央（Fz）と中央部（Cz）で観測される
        # 8チャンネル版では：Fz(ch0), Cz(ch3)が主要
        frontal_channels = [0, 1, 2]  # Fz, F3, F4
        central_channels = [3, 4, 5]  # Cz, C3, C4
        
        if errp_type == "success":
            # 成功時：小さなP300様の成分（前頭部・中央部）
            for ch in frontal_channels + central_channels:
                p300_latency = 0.3  # 300ms
                p300_amplitude = 5 if ch in frontal_channels else 3  # 前頭部で強く
                if p300_latency < t[-1]:
                    p300_idx = int(p300_latency * self.sampling_rate)
                    # ガウシアンパルスでP300を模擬
                    gaussian_pulse = p300_amplitude * signal.gaussian(50, std=10)
                    start_idx = max(0, p300_idx - 25)
                    end_idx = min(duration_samples, p300_idx + 25)
                    pulse_len = end_idx - start_idx
                    eeg_data[ch, start_idx:end_idx] += gaussian_pulse[:pulse_len]
                    
        elif errp_type == "over_grip":
            # 過度な把持：強いErrP（負の成分）- 前頭部で特に強い
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
            
            # 中央部でも軽度のErrP
            for ch in central_channels:
                errp_latency = 0.27  # 270ms
                errp_amplitude = -8  # 軽度の負の成分
                if errp_latency < t[-1]:
                    errp_idx = int(errp_latency * self.sampling_rate)
                    errp_pulse = errp_amplitude * signal.gaussian(35, std=7)
                    start_idx = max(0, errp_idx - 17)
                    end_idx = min(duration_samples, errp_idx + 18)
                    pulse_len = end_idx - start_idx
                    eeg_data[ch, start_idx:end_idx] += errp_pulse[:pulse_len]
                    
        elif errp_type == "under_grip":
            # 不十分な把持：中程度のErrP - 前頭部・中央部両方
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
            
            # 中央部でも同様のErrP
            for ch in central_channels:
                errp_latency = 0.30  # 300ms
                errp_amplitude = -6  # やや軽度の負の成分
                if errp_latency < t[-1]:
                    errp_idx = int(errp_latency * self.sampling_rate)
                    errp_pulse = errp_amplitude * signal.gaussian(30, std=6)
                    start_idx = max(0, errp_idx - 15)
                    end_idx = min(duration_samples, errp_idx + 15)
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
        print(f"🎯 Condition changed to: {condition}")
        if condition != "normal":
            print(f"   ErrP信号を生成中（主要部位: Fz, F3, F4, Cz, C3, C4）")
        
    def send_marker(self, marker_value):
        """
        イベントマーカーを送信（別のLSLストリームとして）
        """
        # マーカー用の別ストリームを作成
        marker_info = StreamInfo(
            name="MockEEG_8CH_Markers",
            type='Markers',
            channel_count=1,
            nominal_srate=0,  # irregular rate
            channel_format='string'
        )
        marker_outlet = StreamOutlet(marker_info)
        marker_outlet.push_sample([str(marker_value)], local_clock())
        print(f"📌 Marker sent: {marker_value}")
        
    def start_streaming(self):
        """
        EEGデータのストリーミングを開始
        """
        self.is_running = True
        print("🚀 Starting EEG 8CH data streaming...")
        print(f"   エポック長: 1.2秒 ({int(1.2 * self.sampling_rate)}サンプル)")
        print(f"   チャンク送信: 40ms間隔")
        
        # 論文に基づく設定: 1.2秒のエポック長
        epoch_duration = 1.2  
        samples_per_chunk = int(self.sampling_rate * epoch_duration)
        
        chunk_count = 0
        
        while self.is_running:
            # 現在の条件に基づいてEEGデータを生成
            if self.current_condition == "normal":
                eeg_chunk = self._generate_background_eeg(samples_per_chunk)
            else:
                eeg_chunk = self._generate_errp_signal(samples_per_chunk, self.current_condition)
                
            # チャンク単位でデータを送信
            chunk_size = int(self.sampling_rate * 0.04)  # 40ms chunks = 10サンプル
            
            for i in range(0, samples_per_chunk, chunk_size):
                if not self.is_running:
                    break
                    
                end_idx = min(i + chunk_size, samples_per_chunk)
                chunk = eeg_chunk[:, i:end_idx].T  # (samples, channels) = (10, 8)
                
                # 各サンプルを個別に送信
                for sample in chunk:
                    self.outlet.push_sample(sample.tolist(), local_clock())
                    
                # リアルタイムに合わせて待機
                time.sleep(chunk_size / self.sampling_rate)
                
            chunk_count += 1
            self.time_counter += epoch_duration
            
            # 進捗表示（10エポックごと）
            if chunk_count % 10 == 0:
                print(f"📊 {chunk_count}エポック送信完了 (条件: {self.current_condition}, "
                      f"経過時間: {self.time_counter:.1f}秒)")
            
    def stop_streaming(self):
        """
        ストリーミングを停止
        """
        self.is_running = False
        print("🛑 EEG 8CH streaming stopped.")


class MockEEGController8Ch:
    """
    模擬EEG 8チャンネルシステムを制御するための簡単なコントローラー
    """
    def __init__(self):
        self.sender = MockEEGSender8Ch()
        
    def run_interactive_demo(self):
        """
        インタラクティブなデモを実行
        """
        print("\n" + "="*60)
        print("🧠 Mock EEG 8Channel System Demo")
        print("="*60)
        print("📍 電極配置: Fz, F3, F4, Cz, C3, C4, Pz, Oz")
        print("⚡ サンプリング: 250Hz, エポック長: 1.2秒")
        print("\n💡 利用可能コマンド:")
        print("  'start'   - ストリーミング開始")
        print("  'stop'    - ストリーミング停止") 
        print("  'normal'  - 通常状態（背景EEG）")
        print("  'success' - 成功条件（P300様成分）")
        print("  'over'    - 過剰把持条件（強いErrP）")
        print("  'under'   - 不足把持条件（中程度ErrP）") 
        print("  'marker X'- マーカーX送信")
        print("  'status'  - 現在の状態表示")
        print("  'quit'    - 終了")
        print("\n🎯 このシステムは把持動作に関連するErrP信号をシミュレート")
        print("   前頭部（Fz,F3,F4）と中央部（Cz,C3,C4）で主要な信号変化")
        
        # ストリーミングを別スレッドで開始
        streaming_thread = None
        
        while True:
            try:
                command = input("\n🎮 コマンド入力: ").strip().lower()
                
                if command == 'start':
                    if streaming_thread is None or not streaming_thread.is_alive():
                        streaming_thread = threading.Thread(target=self.sender.start_streaming)
                        streaming_thread.daemon = True
                        streaming_thread.start()
                        print("✅ ストリーミング開始")
                    else:
                        print("⚠️ ストリーミングは既に実行中です")
                        
                elif command == 'stop':
                    self.sender.stop_streaming()
                    print("⏹️ ストリーミング停止要求送信")
                    
                elif command == 'normal':
                    self.sender.set_condition('normal')
                    print("📊 背景EEG生成中（通常状態）")
                    
                elif command == 'success':
                    self.sender.set_condition('success')
                    print("✅ 成功把持信号生成中（P300様成分）")
                    
                elif command == 'over':
                    self.sender.set_condition('over_grip')
                    print("🔴 過剰把持信号生成中（強いErrP - 前頭部・中央部）")
                    
                elif command == 'under':
                    self.sender.set_condition('under_grip')
                    print("🟡 不足把持信号生成中（中程度ErrP - 前頭部・中央部）")
                    
                elif command.startswith('marker '):
                    marker_value = command.split(' ', 1)[1]
                    self.sender.send_marker(marker_value)
                    print(f"📌 マーカー送信完了: {marker_value}")
                    
                elif command == 'status':
                    self._show_status(streaming_thread)
                    
                elif command == 'quit':
                    self.sender.stop_streaming()
                    print("👋 システム終了中...")
                    if streaming_thread and streaming_thread.is_alive():
                        streaming_thread.join(timeout=2.0)
                    print("✅ 終了完了")
                    break
                    
                else:
                    print("❓ 不明なコマンドです。'quit'で終了")
                    
            except KeyboardInterrupt:
                self.sender.stop_streaming()
                print("\n\n🛑 Ctrl+C で強制終了")
                break
            except Exception as e:
                print(f"⚠️ エラー発生: {e}")
    
    def _show_status(self, streaming_thread):
        """現在の状態を表示"""
        print("\n📊 システム状態:")
        print(f"   ストリーミング: {'🟢 実行中' if streaming_thread and streaming_thread.is_alive() else '🔴 停止中'}")
        print(f"   現在の条件: {self.sender.current_condition}")
        print(f"   チャンネル数: {self.sender.n_channels}")
        print(f"   サンプリング: {self.sender.sampling_rate}Hz")
        print(f"   ストリーム名: {self.sender.stream_name}")
        print(f"   経過時間: {self.sender.time_counter:.1f}秒")
        
        # チャンネル情報
        channel_names = ['Fz', 'F3', 'F4', 'Cz', 'C3', 'C4', 'Pz', 'Oz']
        print(f"   電極配置: {', '.join(channel_names)}")


if __name__ == "__main__":
    print("🧠 Mock EEG 8Channel Data Sender")
    print("LSLを使用した8チャンネルEEGデータ送信システム")
    
    controller = MockEEGController8Ch()
    controller.run_interactive_demo()