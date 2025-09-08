#!/usr/bin/env python3
"""
LSL-TCP同期エピソード収集システム

機能:
1. LSL（EEG）データをeeg_receiver.pyで受信・eeg_neuroadaptation_preprocessor.pyで前処理
2. TCP（Unity）データをunity_tcp_interface.pyで受信
3. EPISODE_ENDトリガー受信時に、直前のJSONデータを採用
4. トリガー時刻から3.2秒さかのぼって1.2秒分のLSLデータを抽出
5. episode_idはJSONの'episode'フィールドの値を使用
6. CSVファイルでの保存機能

依存関係:
- eeg_receiver.py (同一ディレクトリ)
- eeg_neuroadaptation_preprocessor.py (同一ディレクトリ)  
- unity_tcp_interface.py (同一ディレクトリ)
"""

import numpy as np
import pandas as pd
import time
import threading
import queue
import json
import os
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# 同一ディレクトリのモジュールをインポート
from eeg_receiver import LSLEEGReceiver, EEGDataProcessor
from eeg_neuroadaptation_preprocessor import NeuroadaptationEEGPreprocessor
from unity_tcp_interface import EEGTCPInterface

@dataclass
class Episode:
    """エピソードデータクラス"""
    episode_id: int
    trigger_timestamp: float
    lsl_data: np.ndarray  # (samples, channels) - 1.2秒分の前処理済みEEGデータ
    lsl_timestamps: np.ndarray  # LSLタイムスタンプ配列
    tcp_data: Dict[str, Any]  # 直前のTCPデータ
    tcp_timestamp: float
    sync_latency: float  # 同期遅延（ミリ秒）
    preprocessing_info: Dict[str, Any]  # 前処理情報
    
class LSLTCPEpisodeCollector:
    """LSL-TCP同期エピソード収集システム"""
    
    def __init__(self, 
                 lsl_stream_name='MockEEG',
                 tcp_host='127.0.0.1',
                 tcp_port=12345,
                 sampling_rate=250,
                 lookback_seconds=3.2,
                 episode_duration=1.2,
                 max_buffer_seconds=10.0,
                 save_to_csv=True,
                 enable_realtime_processing=False):
        """
        初期化
        
        Args:
            lsl_stream_name: LSLストリーム名
            tcp_host: TCPホスト
            tcp_port: TCPポート
            sampling_rate: サンプリング周波数
            lookback_seconds: トリガーからさかのぼる時間
            episode_duration: エピソード長（秒）
            max_buffer_seconds: 最大バッファ時間
        """
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.sampling_rate = sampling_rate
        self.lookback_seconds = lookback_seconds
        self.episode_duration = episode_duration
        self.save_to_csv = save_to_csv
        self.enable_realtime_processing = enable_realtime_processing
                
        # サンプル数計算
        self.lookback_samples = int(lookback_seconds * sampling_rate)  # 800サンプル
        self.episode_samples = int(episode_duration * sampling_rate)   # 300サンプル
        self.max_buffer_samples = int(max_buffer_seconds * sampling_rate)  # 2500サンプル
        
        # LSL受信システム
        self.eeg_receiver = LSLEEGReceiver(stream_name=lsl_stream_name)
        self.eeg_preprocessor = NeuroadaptationEEGPreprocessor(
            sampling_rate=sampling_rate,
            enable_asr=True,
            enable_ica=False  # リアルタイム用に高速化
        )
        
        # TCP通信システム
        # 受信バッファがあふれて未処理データが失われないようバッファサイズを拡大
        self.tcp_interface = EEGTCPInterface(host=tcp_host,
                                            port=tcp_port,
                                            max_buffer_size=10000)
        
        # データバッファ
        self.lsl_data_buffer = deque(maxlen=self.max_buffer_samples)
        self.lsl_timestamp_buffer = deque(maxlen=self.max_buffer_samples)
        self.tcp_data_buffer = deque(maxlen=10000)  # 最新1000件のTCPデータ
        
        # エピソード管理
        self.episodes = []
        self.episode_counter = 0  # 参考用カウンター
        self.trigger_queue = queue.Queue()
        
        # 実行制御
        self.is_running = False
        self.threads = []
        self.buffer_lock = threading.Lock()
        
        # セッション情報
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"DDPG_Python/logs/episodes_{self.session_id}"
        
        # 統計情報
        self.stats = {
            'total_triggers': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'avg_sync_latency_ms': 0.0,
            'start_time': None
        }
        
        print(f"🧠 LSL-TCP同期エピソード収集システム初期化完了")
        print(f"   セッションID: {self.session_id}")
        print(f"   ルックバック: {lookback_seconds}秒 ({self.lookback_samples}サンプル)")
        print(f"   エピソード長: {episode_duration}秒 ({self.episode_samples}サンプル)")
        print(f"   出力ディレクトリ: {self.output_dir}")
    
    def start_collection(self):
        """データ収集開始"""
        print(f"🚀 データ収集開始")
        
        # 出力ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # LSL接続
        if not self.eeg_receiver.connect_to_stream():
            print(f"❌ LSL接続失敗")
            return False
        
        # TCP接続
        if not self.tcp_interface.start_server():
            print(f"❌ TCP接続失敗")
            return False
        
        # # TCPメッセージコールバック設定
        # self.tcp_interface.add_message_callback(self._on_tcp_message_received)
        
        # 実行フラグ設定
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # スレッド開始
        self._start_threads()
        
        print(f"✅ データ収集開始完了")
        print(f"💡 トリガー待機中... (EPISODE_ENDでエピソード収集)")
        
        return True
    
    def _start_threads(self):
        """各種スレッドを開始"""
        # LSLデータ受信スレッド
        lsl_thread = threading.Thread(target=self._lsl_data_thread, daemon=True)
        lsl_thread.start()
        self.threads.append(lsl_thread)
        
        # TCP監視スレッド（unity_tcp_interfaceの受信データを監視）
        tcp_monitor_thread = threading.Thread(target=self._tcp_monitor_thread, daemon=True)
        tcp_monitor_thread.start()
        self.threads.append(tcp_monitor_thread)
        
        # エピソード処理スレッド
        episode_thread = threading.Thread(target=self._episode_processing_thread, daemon=True)
        episode_thread.start()
        self.threads.append(episode_thread)
        
        print(f"🔄 バックグラウンドスレッド開始完了")
    
    def _tcp_monitor_thread(self):
        """TCP受信データ監視スレッド（修正版）"""
        print(f"📡 TCP監視スレッド開始")

        while self.is_running:
            try:
                # 新着メッセージをキューから取り出して処理
                message_data = self.tcp_interface.received_data.popleft()
                print(f"📡 新着メッセージ処理: {str(message_data)[:50]}...")
                self._process_tcp_message(message_data)

            except IndexError:
                # 受信キューが空の場合は少し待機
                time.sleep(0.1)

            except Exception as e:
                if self.is_running:
                    print(f"⚠️ TCP監視エラー: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

        print(f"📡 TCP監視スレッド終了")


    def _process_tcp_message(self, message_data):
        """TCPメッセージの処理（強化版）"""
        print(f"🔍 TCP処理開始: {type(message_data)} = {str(message_data)[:100]}")
        tcp_timestamp = time.time()
        
        # メッセージの内容をチェック
        message_content = None
        
        if isinstance(message_data, str):
            message_content = message_data.strip()
            print(f"  → 文字列メッセージ: '{message_content}'")
        elif isinstance(message_data, dict):
            # 辞書の中の様々なキーをチェック
            for key in ['content', 'message', 'text']:
                if key in message_data:
                    message_content = str(message_data[key]).strip()
                    print(f"  → 辞書[{key}]: '{message_content}'")
                    break
            
            if message_content is None and self._is_robot_state_data(message_data):
                print(f"  → ロボット状態データ")
            elif message_content is None:
                print(f"  → 不明な辞書データ: {list(message_data.keys())}")
        
        # EPISODE_ENDトリガーの厳密チェック
        if message_content is not None and message_content == "EPISODE_END":
            print(f"🎯 EPISODE_ENDトリガー検出!")
            print(f"   受信時刻: {tcp_timestamp}")
            
            # 直前のJSONデータを検索
            previous_json_data = self._get_previous_json_data()
            if previous_json_data:
                robot_episode_id = previous_json_data.get('episode', 'unknown')
                print(f"📋 直前のJSONデータを採用: episode={robot_episode_id}")
                print(f"   データ: {previous_json_data}")
                
                trigger_info = {
                    'tcp_data': previous_json_data,
                    'tcp_timestamp': tcp_timestamp,
                    'trigger_timestamp': tcp_timestamp,
                    'trigger_type': 'EPISODE_END'
                }
                
                try:
                    print(f"📥 トリガーキューに追加中...")
                    self.trigger_queue.put(trigger_info, timeout=1.0)
                    self.stats['total_triggers'] += 1
                    print(f"✅ トリガー情報をキューに追加完了: エピソード{robot_episode_id}")
                    print(f"   キューサイズ: {self.trigger_queue.qsize()}")
                except Exception as e:
                    print(f"❌ トリガーキュー追加エラー: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️ 直前のJSONデータが見つかりません")
                print(f"   TCPバッファサイズ: {len(self.tcp_data_buffer)}")
                
                # デバッグ: 最新のデータをチェック
                recent_entries = list(self.tcp_data_buffer)[-10:]
                for i, entry in enumerate(recent_entries):
                    data = entry['data']
                    if isinstance(data, dict) and 'episode' in data:
                        print(f"    バッファ[{i}]: episode={data.get('episode')}")
                    else:
                        print(f"    バッファ[{i}]: {str(data)[:50]}")
            
            return  # EPISODE_END処理完了
        
        # ロボット状態データの処理
        if isinstance(message_data, dict) and self._is_robot_state_data(message_data):
            # TCPデータをバッファに追加
            tcp_entry = {
                'data': message_data,
                'timestamp': tcp_timestamp
            }
            self.tcp_data_buffer.append(tcp_entry)
            
            robot_episode = message_data.get('episode', 'unknown')
            grip_force = message_data.get('grip_force', 'unknown')
            print(f"📋 ロボット状態データ受信: episode={robot_episode}, grip_force={grip_force}")
        
        # その他のメッセージもバッファに追加
        tcp_entry = {
            'data': message_data if isinstance(message_data, dict) else {'type': 'text_message', 'content': str(message_data)},
            'timestamp': tcp_timestamp
        }
        self.tcp_data_buffer.append(tcp_entry)


    def _lsl_data_thread(self):
        """LSLデータ受信スレッド"""
        print(f"📡 LSLデータ受信開始")
        
        while self.is_running:
            try:
                # LSLからサンプル取得
                sample, timestamp = self.eeg_receiver.inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    with self.buffer_lock:
                        # 32チャンネルに正規化
                        if len(sample) >= 32:
                            normalized_sample = sample[:32]
                        else:
                            normalized_sample = sample + [0.0] * (32 - len(sample))
                        
                        # バッファに追加
                        self.lsl_data_buffer.append(normalized_sample)
                        self.lsl_timestamp_buffer.append(timestamp)
                
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ LSLデータ受信エラー: {e}")
                time.sleep(0.001)
        
        print(f"📡 LSLデータ受信終了")
    
    def _on_tcp_message_received(self, message_data: Dict[str, Any]):
        """TCP メッセージ受信時のコールバック"""
        tcp_timestamp = time.time()
        
        # 文字列メッセージの場合（EPISODE_ENDなど）
        if isinstance(message_data, str):
            message_str = message_data.strip()
            if message_str == "EPISODE_END":
                print(f"🎯 EPISODE_ENDトリガー検出")
                
                # 直前のJSONデータを検索
                previous_json_data = self._get_previous_json_data()
                if previous_json_data:
                    robot_episode_id = previous_json_data.get('episode', 'unknown')
                    print(f"📋 直前のJSONデータを採用: episode={robot_episode_id}")
                    
                    trigger_info = {
                        'tcp_data': previous_json_data,  # 直前のJSONデータを使用
                        'tcp_timestamp': tcp_timestamp,  # EPISODE_END受信時刻
                        'trigger_timestamp': tcp_timestamp,
                        'trigger_type': 'EPISODE_END'
                    }
                    self.trigger_queue.put(trigger_info)
                    self.stats['total_triggers'] += 1
                else:
                    print(f"⚠️ 直前のJSONデータが見つかりません")
            return
        
        # 辞書型メッセージの場合（ロボット状態データ）
        if isinstance(message_data, dict):
            # TCPデータをバッファに追加
            tcp_entry = {
                'data': message_data,
                'timestamp': tcp_timestamp
            }
            self.tcp_data_buffer.append(tcp_entry)
            
            # ロボット状態データかをチェック
            if self._is_robot_state_data(message_data):
                robot_episode = message_data.get('episode', 'unknown')
                grip_force = message_data.get('grip_force', 'unknown')
                print(f"📋 ロボット状態データ受信: episode={robot_episode}, grip_force={grip_force}")
    
    def _get_previous_json_data(self) -> Optional[Dict[str, Any]]:
        """直前のJSONデータ（ロボット状態データ）を取得"""
        # TCPバッファを逆順で検索
        for tcp_entry in reversed(list(self.tcp_data_buffer)):
            data = tcp_entry['data']
            
            # JSONデータ（辞書型）でロボット状態データの場合
            if (isinstance(data, dict) and self._is_robot_state_data(data)):
                return data
        
        return None
    
    def _is_robot_state_data(self, data: Dict[str, Any]) -> bool:
        """ロボット状態データかを判定"""
        # 必要なキーが含まれているかチェック
        required_keys = ['episode', 'position', 'velocity', 'grip_force']
        return all(key in data for key in required_keys)
    
    def _episode_processing_thread(self):
        """エピソード処理スレッド"""
        print(f"⚡ エピソード処理スレッド開始")
        
        while self.is_running:
            try:
                # トリガー待機（タイムアウト1秒）
                trigger_info = self.trigger_queue.get(timeout=1.0)
                
                # エピソード生成を試行
                episode = self._create_episode(trigger_info)
                
                if episode:
                    self.episodes.append(episode)
                    self.stats['successful_episodes'] += 1
                    
                    # CSVファイルに保存
                    self._save_episode_to_csv(episode)
                    
                    print(f"✅ エピソード{episode.episode_id}保存完了 "
                          f"(同期遅延: {episode.sync_latency:.1f}ms)")
                else:
                    self.stats['failed_episodes'] += 1
                    print(f"❌ エピソード作成失敗")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ エピソード処理エラー: {e}")
        
        print(f"⚡ エピソード処理スレッド終了")
    
    def _create_episode(self, trigger_info: Dict[str, Any]) -> Optional[Episode]:
        """トリガー情報からエピソードを作成"""
        trigger_timestamp = trigger_info['trigger_timestamp']
        tcp_data = trigger_info['tcp_data']
        tcp_timestamp = trigger_info['tcp_timestamp']
        
        with self.buffer_lock:
            # 十分なLSLデータがあるかチェック
            if len(self.lsl_data_buffer) < self.episode_samples:
                print(f"⚠️ LSLデータ不足: {len(self.lsl_data_buffer)}/{self.episode_samples}")
                return None
            
            # トリガー時刻に最も近いLSLタイムスタンプを検索
            timestamps = list(self.lsl_timestamp_buffer)
            time_diffs = [abs(ts - trigger_timestamp) for ts in timestamps]
            
            if not time_diffs:
                return None
            
            # 最も近いタイムスタンプのインデックス
            closest_idx = time_diffs.index(min(time_diffs))
            sync_latency = min(time_diffs) * 1000  # ミリ秒に変換
            
            # 3.2秒さかのぼって1.2秒分のデータを抽出
            lookback_start_idx = max(0, closest_idx - self.lookback_samples)
            episode_start_idx = lookback_start_idx
            episode_end_idx = min(len(self.lsl_data_buffer), 
                                episode_start_idx + self.episode_samples)
            
            # データが不足している場合
            if episode_end_idx - episode_start_idx < self.episode_samples:
                print(f"⚠️ エピソードデータ不足: {episode_end_idx - episode_start_idx}/{self.episode_samples}")
                return None
            
            # EEGデータ抽出
            raw_eeg_data = np.array([
                self.lsl_data_buffer[i] 
                for i in range(episode_start_idx, episode_end_idx)
            ])
            
            # タイムスタンプ抽出
            eeg_timestamps = np.array([
                self.lsl_timestamp_buffer[i]
                for i in range(episode_start_idx, episode_end_idx)
            ])
        
        # 前処理実行
        try:
            preprocessing_result = self.eeg_preprocessor.preprocess_epoch(raw_eeg_data)
            processed_eeg = preprocessing_result['processed_epoch']
            preprocessing_info = {
                'processing_time_ms': preprocessing_result['processing_time_ms'],
                'quality_metrics': preprocessing_result['quality_metrics'],
                'rejected_channels': preprocessing_result.get('rejected_channels', [])
            }
        except Exception as e:
            print(f"⚠️ 前処理エラー: {e}")
            # 前処理失敗時は生データを使用
            processed_eeg = raw_eeg_data
            preprocessing_info = {'error': str(e)}
        
        # episode_idは送信されたJSONデータの'episode'フィールドから取得
        json_episode_id = tcp_data.get('episode', self.episode_counter)
        
        # エピソード作成
        episode = Episode(
            episode_id=json_episode_id,  # JSONから取得したepisode番号を使用
            trigger_timestamp=trigger_timestamp,
            lsl_data=processed_eeg,
            lsl_timestamps=eeg_timestamps,
            tcp_data=tcp_data,
            tcp_timestamp=tcp_timestamp,
            sync_latency=sync_latency,
            preprocessing_info=preprocessing_info
        )
        
        # EPISODE_ENDトリガーの場合は詳細情報を表示
        if trigger_info.get('trigger_type') == 'EPISODE_END':
            print(f"📝 EPISODE_ENDトリガーでエピソード作成:")
            print(f"   ロボットエピソード番号: {json_episode_id}")
            print(f"   把持力: {tcp_data.get('grip_force', 'unknown')}N")
            print(f"   位置: {tcp_data.get('position', 'unknown')}")
            print(f"   接触状態: {tcp_data.get('contact', 'unknown')}")
        
        # カウンターは参考用として保持
        self.episode_counter += 1
        
        return episode
    
    def _save_episode_to_csv(self, episode: Episode):
        """エピソードをCSVファイルに保存"""
        try:
            # エピソード基本情報のCSV
            episode_info_file = os.path.join(self.output_dir, f"episode_{episode.episode_id:04d}_info.csv")
            info_data = {
                'episode_id': [episode.episode_id],
                'trigger_timestamp': [episode.trigger_timestamp],
                'tcp_timestamp': [episode.tcp_timestamp],
                'sync_latency_ms': [episode.sync_latency],
                'tcp_message_type': [episode.tcp_data.get('type', 'robot_state')],
                'robot_episode': [episode.tcp_data.get('episode', 'unknown')],
                'robot_position_x': [episode.tcp_data.get('position', [0,0,0])[0] if episode.tcp_data.get('position') else 0],
                'robot_position_y': [episode.tcp_data.get('position', [0,0,0])[1] if episode.tcp_data.get('position') else 0],
                'robot_position_z': [episode.tcp_data.get('position', [0,0,0])[2] if episode.tcp_data.get('position') else 0],
                'robot_velocity_x': [episode.tcp_data.get('velocity', [0,0,0])[0] if episode.tcp_data.get('velocity') else 0],
                'robot_velocity_y': [episode.tcp_data.get('velocity', [0,0,0])[1] if episode.tcp_data.get('velocity') else 0],
                'robot_velocity_z': [episode.tcp_data.get('velocity', [0,0,0])[2] if episode.tcp_data.get('velocity') else 0],
                'grip_force': [episode.tcp_data.get('grip_force', 0)],
                'actual_grip_force': [episode.tcp_data.get('actual_grip_force', 0)],
                'tcp_grip_force': [episode.tcp_data.get('tcp_grip_force', 0)],
                'contact': [episode.tcp_data.get('contact', False)],
                'contact_force': [episode.tcp_data.get('contact_force', 0)],
                'broken': [episode.tcp_data.get('broken', False)],
                'deformation': [episode.tcp_data.get('deformation', 0)],
                'tcp_data_json': [json.dumps(episode.tcp_data)],
                'preprocessing_time_ms': [episode.preprocessing_info.get('processing_time_ms', 0)],
                'rejected_channels': [str(episode.preprocessing_info.get('rejected_channels', []))]
            }
            pd.DataFrame(info_data).to_csv(episode_info_file, index=False)
            
            # LSLデータのCSV
            eeg_data_file = os.path.join(self.output_dir, f"episode_{episode.episode_id:04d}_eeg.csv")
            eeg_df = pd.DataFrame(episode.lsl_data)
            eeg_df.columns = [f'ch_{i:02d}' for i in range(episode.lsl_data.shape[1])]
            eeg_df['timestamp'] = episode.lsl_timestamps
            eeg_df['sample_index'] = range(len(eeg_df))
            eeg_df.to_csv(eeg_data_file, index=False)
            
            # 統合サマリーCSV（全エピソード）
            summary_file = os.path.join(self.output_dir, "episodes_summary.csv")
            summary_data = {
                'episode_id': episode.episode_id,
                'trigger_timestamp': episode.trigger_timestamp,
                'tcp_timestamp': episode.tcp_timestamp,
                'sync_latency_ms': episode.sync_latency,
                'tcp_message_type': episode.tcp_data.get('type', 'robot_state'),
                'robot_episode': episode.tcp_data.get('episode', 'unknown'),
                'grip_force': episode.tcp_data.get('grip_force', 0),
                'actual_grip_force': episode.tcp_data.get('actual_grip_force', 0),
                'contact': episode.tcp_data.get('contact', False),
                'contact_force': episode.tcp_data.get('contact_force', 0),
                'broken': episode.tcp_data.get('broken', False),
                'eeg_samples': episode.lsl_data.shape[0],
                'eeg_channels': episode.lsl_data.shape[1],
                'processing_time_ms': episode.preprocessing_info.get('processing_time_ms', 0)
            }
            
            # ファイルが存在しない場合はヘッダー付きで作成
            if not os.path.exists(summary_file):
                pd.DataFrame([summary_data]).to_csv(summary_file, index=False)
            else:
                pd.DataFrame([summary_data]).to_csv(summary_file, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"⚠️ CSV保存エラー: {e}")
    
    def stop_collection(self):
        """データ収集停止"""
        print(f"🛑 データ収集停止中...")
        
        self.is_running = False
        
        # TCP接続停止
        self.tcp_interface.stop_server()
        
        # 最終統計表示
        self._print_final_statistics()
        
        print(f"🛑 データ収集停止完了")
    
    def _print_final_statistics(self):
        """最終統計情報の表示"""
        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
        else:
            total_time = 0
        
        if self.stats['successful_episodes'] > 0:
            avg_latency = sum(ep.sync_latency for ep in self.episodes) / len(self.episodes)
        else:
            avg_latency = 0
        
        print(f"\n📊 データ収集統計:")
        print(f"   セッション時間     : {total_time:.1f}秒")
        print(f"   総トリガー数       : {self.stats['total_triggers']}")
        print(f"   成功エピソード数   : {self.stats['successful_episodes']}")
        print(f"   失敗エピソード数   : {self.stats['failed_episodes']}")
        if self.stats['total_triggers'] > 0:
            success_rate = self.stats['successful_episodes'] / self.stats['total_triggers'] * 100
            print(f"   成功率             : {success_rate:.1f}%")
        print(f"   平均同期遅延       : {avg_latency:.1f}ms")
        print(f"   出力ディレクトリ   : {self.output_dir}")
    
    def run_demo(self):
        """デモ実行（単体動作テスト用）"""
        print(f"🚀 LSL-TCP同期エピソード収集デモ開始")
        
        # データ収集開始
        if not self.start_collection():
            print(f"❌ システム開始失敗")
            return
        
        try:
            print(f"\n💡 デモ実行中:")
            print(f"   1. LSLデータ受信中（{self.lsl_stream_name}）")
            print(f"   2. TCP待機中（{self.tcp_host}:{self.tcp_port}）")
            print(f"   3. EPISODE_ENDでエピソード収集")
            print(f"   4. Ctrl+C で終了")
            print(f"\n📝 Unity側でTCPメッセージを送信してください:")
            print(f"   1. ロボット状態データ（10回程度）:")
            print(f"      {{\"episode\": 1, \"grip_force\": 10.5, \"position\": [0,0,0], ...}}")
            print(f"   2. エピソード終了トリガー:")
            print(f"      \"EPISODE_END\"")
            print(f"   → 直前のロボット状態データとLSLデータを組み合わせてエピソード保存")
            
            # メインループ
            while self.is_running:
                time.sleep(1.0)
                
                # 5秒ごとに状態表示
                if int(time.time()) % 5 == 0:
                    lsl_buffer_size = len(self.lsl_data_buffer)
                    tcp_buffer_size = len(self.tcp_data_buffer)
                    print(f"💻 状態: LSL={lsl_buffer_size}サンプル, "
                          f"TCP={tcp_buffer_size}メッセージ, "
                          f"エピソード={self.stats['successful_episodes']}件")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ デモ停止（Ctrl+C）")
        finally:
            self.stop_collection()


if __name__ == '__main__':
    print("🧠 LSL-TCP同期エピソード収集システム")
    print("=" * 60)
    print("選択してください:")
    print("1. CSV保存モード（デフォルト）")
    print("2. リアルタイム処理モード（DDPG学習等）")
    print("3. ハイブリッドモード（CSV保存+リアルタイム処理）")
    
    choice = input("選択 (1-3): ").strip()
    
    if choice == "2":
        # リアルタイム処理モードの例
        def on_episode_created(episode: Episode):
            """エピソード作成時のコールバック例（DDPG学習用）"""
            print(f"🤖 DDPG学習用: エピソード{episode.episode_id}を受信")
            print(f"   EEGデータ形状: {episode.lsl_data.shape}")
            print(f"   ロボット状態: grip_force={episode.tcp_data.get('grip_force')}, "
                  f"broken={episode.tcp_data.get('broken')}")
            
            # ここでDDPG学習システムにデータを送信
            # ddpg_system.process_episode(episode.lsl_data, episode.tcp_data)
        
        collector = LSLTCPEpisodeCollector(
            save_to_csv=False,  # CSV保存無効
            enable_realtime_processing=True  # リアルタイム処理有効
        )
        collector.add_episode_callback(on_episode_created)
        
    elif choice == "3":
        # ハイブリッドモードの例
        def on_episode_created(episode: Episode):
            """エピソード作成時のコールバック例"""
            print(f"🔄 ハイブリッド処理: エピソード{episode.episode_id}")
            # DDPG学習とCSV保存を同時実行
        
        collector = LSLTCPEpisodeCollector(
            save_to_csv=True,   # CSV保存有効
            enable_realtime_processing=True  # リアルタイム処理有効
        )
        collector.add_episode_callback(on_episode_created)
        
    else:
        # デフォルト：CSV保存モード
        collector = LSLTCPEpisodeCollector(
            save_to_csv=True,   # CSV保存有効
            enable_realtime_processing=False  # リアルタイム処理無効
        )
    
    # システム実行
    collector.run_demo()