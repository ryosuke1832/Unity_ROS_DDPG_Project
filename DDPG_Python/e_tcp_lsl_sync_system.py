#!/usr/bin/env python3
"""
LSL-TCP同期エピソード収集システム（状態橋渡し機能強化版）

変更要約（2025-09-21）:
- TCPメッセージ到着時刻をトリガーに、前0.3秒・後0.9秒のLSLを収集
- 未来0.9秒のLSLが到着するまでブロッキング待機
- robot_state受信を即トリガー（EPISODE_ENDは後方互換）
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
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from pylsl import local_clock


# 同一ディレクトリのモジュールをインポート
from b_eeg_receiver import LSLEEGReceiver, EEGDataProcessor
from d_eeg_neuroadaptation_preprocessor import NeuroadaptationEEGPreprocessor

# 修正版EEGTCPInterfaceをインポート
from c_unity_tcp_interface import EEGTCPInterface

@dataclass
class Episode:
    """エピソードデータクラス（拡張版）"""
    episode_id: int
    trigger_timestamp: float
    lsl_data: np.ndarray  # (samples, channels)
    lsl_timestamps: np.ndarray
    tcp_data: Dict[str, Any]
    tcp_timestamp: float
    sync_latency: float  # ms
    preprocessing_info: Dict[str, Any]
    state_source: str = "unknown"
    state_age_ms: float = 0.0
    quality_score: float = 1.0

class StateShareManager:
    """状態共有管理クラス（新追加）"""
    def __init__(self):
        self.subscribers = []
        self.latest_robot_state = None
        self.latest_robot_state_timestamp = None
        self.state_history = deque(maxlen=100)
        self.update_count = 0
        self.stats = {
            'total_updates': 0,
            'subscriber_notifications': 0,
            'notification_errors': 0,
            'history_size': 0
        }

    def subscribe(self, callback: Callable[[Dict[str, Any]], None], subscriber_name: str = "unknown"):
        self.subscribers.append({
            'callback': callback,
            'name': subscriber_name,
            'registered_at': time.time(),
            'notification_count': 0,
            'error_count': 0
        })
        print(f"📡 状態共有購読登録: {subscriber_name} (総購読者数: {len(self.subscribers)})")

    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        self.subscribers = [sub for sub in self.subscribers if sub['callback'] != callback]

    def update_robot_state(self, state_data: Dict[str, Any], source: str = "unknown"):
        try:
            current_time = time.time()
            self.latest_robot_state = state_data.copy()
            self.latest_robot_state_timestamp = current_time
            self.update_count += 1
            self.stats['total_updates'] += 1

            history_entry = {
                'timestamp': current_time,
                'source': source,
                'data': state_data.copy(),
                'episode': state_data.get('episode', 'unknown'),
                'grip_force': state_data.get('grip_force', 'unknown')
            }
            self.state_history.append(history_entry)
            self.stats['history_size'] = len(self.state_history)

            for subscriber in self.subscribers:
                try:
                    subscriber['callback'](state_data)
                    subscriber['notification_count'] += 1
                    self.stats['subscriber_notifications'] += 1
                except Exception as e:
                    print(f"⚠️ 状態通知エラー [{subscriber['name']}]: {e}")
                    subscriber['error_count'] += 1
                    self.stats['notification_errors'] += 1

            if len(self.subscribers) > 0:
                episode = state_data.get('episode', 'unknown')
                force = state_data.get('grip_force', 'unknown')
                print(f"🔗 状態共有通知: ep={episode}, force={force} → {len(self.subscribers)}件の購読者")

        except Exception as e:
            print(f"❌ 状態共有更新エラー: {e}")

    def get_latest_state(self) -> Optional[Dict[str, Any]]:
        if self.latest_robot_state and self.latest_robot_state_timestamp:
            age_ms = (time.time() - self.latest_robot_state_timestamp) * 1000
            return {
                'data': self.latest_robot_state.copy(),
                'timestamp': self.latest_robot_state_timestamp,
                'age_ms': age_ms,
                'update_count': self.update_count
            }
        return None

    def get_state_history(self, max_count: int = 10) -> List[Dict[str, Any]]:
        history_list = list(self.state_history)
        return history_list[-max_count:] if max_count else history_list

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            'subscribers_count': len(self.subscribers),
            'subscribers_info': [
                {
                    'name': sub['name'],
                    'notifications': sub['notification_count'],
                    'errors': sub['error_count']
                }
                for sub in self.subscribers
            ]
        }

class LSLTCPEpisodeCollector:
    """LSL-TCP同期エピソード収集システム（状態橋渡し機能強化版）"""

    def __init__(self, 
                 lsl_stream_name='MockEEG',
                 tcp_host='127.0.0.1',
                 tcp_port=12345,
                 sampling_rate=250,
                 # ↓ 旧パラメータ（互換のため受けるが未使用）
                 lookback_seconds=3.2,
                 episode_duration=1.2,
                 max_buffer_seconds=10.0,
                 save_to_csv=True,
                 enable_realtime_processing=False,
                 enable_state_sharing=True,
                 # ↓ 新パラメータ
                 pre_trigger_seconds: float = 0.3,
                 post_trigger_seconds: float = 0.9,
                 trigger_on_robot_state: bool = True):
        """
        Args:
            pre_trigger_seconds: トリガー前に取る秒数（デフォルト0.3）
            post_trigger_seconds: トリガー後に取る秒数（デフォルト0.9）
            trigger_on_robot_state: robot_state受信を即トリガーにする
        """
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.sampling_rate = sampling_rate

        # ✅ 新しい窓設定
        self.pre_trigger_seconds = float(pre_trigger_seconds)
        self.post_trigger_seconds = float(post_trigger_seconds)
        self.pre_trigger_samples = int(round(self.pre_trigger_seconds * sampling_rate))
        self.post_trigger_samples = int(round(self.post_trigger_seconds * sampling_rate))
        self.episode_samples = self.pre_trigger_samples + self.post_trigger_samples

        # 旧: lookback/episode_duration は非推奨（未使用）
        self.max_buffer_samples = int(max_buffer_seconds * sampling_rate)
        self.save_to_csv = save_to_csv

        # LSL受信システム
        self.eeg_receiver = LSLEEGReceiver(stream_name=lsl_stream_name)
        self.eeg_preprocessor = NeuroadaptationEEGPreprocessor(
            sampling_rate=sampling_rate,
            enable_asr=True,
            enable_ica=False
        )

        self.tcp_interface = EEGTCPInterface(
            host=tcp_host,
            port=tcp_port,
            max_buffer_size=10000,
            auto_reply=False
        )

        self.trigger_on_robot_state = trigger_on_robot_state
        self.state_share_manager = StateShareManager() if enable_state_sharing else None

        # データバッファ
        self.lsl_data_buffer = deque(maxlen=self.max_buffer_samples)
        self.lsl_timestamp_buffer = deque(maxlen=self.max_buffer_samples)
        self.tcp_data_buffer = deque(maxlen=10000)

        # エピソード管理
        self.episodes = []
        self.episode_counter = 0
        self.trigger_queue = queue.Queue()

        # 実行制御
        self.is_running = False
        self.threads = []
        self.buffer_lock = threading.Lock()

        # セッション情報
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"DDPG_Python/logs/episodes_{self.session_id}"

        self.stats = {
            'total_triggers': 0,
            'successful_episodes': 0,
            'failed_episodes': 0,
            'avg_sync_latency_ms': 0.0,
            'robot_state_updates': 0,
            'state_sharing_notifications': 0,
            'tcp_connection_count': 0,
            'lsl_samples_received': 0,
            'start_time': None
        }

        print("🧠 LSL-TCP同期エピソード収集システム初期化完了（状態橋渡し機能強化版）")
        print(f"   セッションID: {self.session_id}")
        print(f"   収集窓: 前{self.pre_trigger_seconds:.3f}s / 後{self.post_trigger_seconds:.3f}s "
              f"(合計{(self.pre_trigger_seconds+self.post_trigger_seconds):.3f}s, {self.episode_samples}samples@{self.sampling_rate}Hz)")
        print(f"   出力ディレクトリ: {self.output_dir}")
        print(f"   TCP自動応答: 無効")
        print(f"   状態共有機能: {'有効' if enable_state_sharing else '無効'}")
        if not trigger_on_robot_state:
            print("   トリガー: EPISODE_END（互換）")
        else:
            print("   トリガー: robot_state到着時刻")

    def add_state_subscriber(self, callback: Callable[[Dict[str, Any]], None], subscriber_name: str = "unknown"):
        if self.state_share_manager:
            self.state_share_manager.subscribe(callback, subscriber_name)
        else:
            print(f"⚠️ 状態共有機能が無効のため、購読登録できません: {subscriber_name}")

    def get_latest_robot_state(self) -> Optional[Dict[str, Any]]:
        if self.state_share_manager:
            return self.state_share_manager.get_latest_state()
        return None

    def start_collection(self):
        print("🚀 データ収集開始（状態橋渡し機能強化版）")
        os.makedirs(self.output_dir, exist_ok=True)

        self.tcp_interface.enable_debug_mode(enable_all_logs=False)

        if not self.eeg_receiver.connect_to_stream():
            print("❌ LSL接続失敗")
            return False

        if not self.tcp_interface.start_server():
            print("❌ TCP接続失敗")
            return False

        if self.state_share_manager:
            self.tcp_interface.add_state_update_callback(self._handle_robot_state_update)
            print("🔗 状態共有コールバック設定完了")

        self.is_running = True
        self.stats['start_time'] = time.time()
        self._start_threads()

        print("✅ データ収集開始完了")
        print("💡 robot_state受信でエピソード切り出し（前0.3s/後0.9s）")
        return True

    def _handle_robot_state_update(self, state_data: Dict[str, Any]):
        if self.state_share_manager and self._is_robot_state_data(state_data):
            self.stats['robot_state_updates'] += 1
            self.state_share_manager.update_robot_state(state_data, source="tcp_collector")
            self.stats['state_sharing_notifications'] += 1

    def _start_threads(self):
        lsl_thread = threading.Thread(target=self._lsl_data_thread, daemon=True)
        lsl_thread.start(); self.threads.append(lsl_thread)

        tcp_monitor_thread = threading.Thread(target=self._tcp_monitor_thread, daemon=True)
        tcp_monitor_thread.start(); self.threads.append(tcp_monitor_thread)

        episode_thread = threading.Thread(target=self._episode_processing_thread, daemon=True)
        episode_thread.start(); self.threads.append(episode_thread)

        stats_thread = threading.Thread(target=self._stats_monitoring_thread, daemon=True)
        stats_thread.start(); self.threads.append(stats_thread)

        print("🔄 バックグラウンドスレッド開始完了（状態橋渡し機能強化版）")

    def _stats_monitoring_thread(self):
        last_print_time = time.time()
        print_interval = 60
        while self.is_running:
            try:
                current_time = time.time()
                if current_time - last_print_time >= print_interval:
                    self._print_stats_summary()
                    last_print_time = current_time
                time.sleep(10)
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ 統計監視エラー: {e}")
                time.sleep(10)
        print("📊 統計監視スレッド終了")

    def _print_stats_summary(self):
        if not self.is_running:
            return
        uptime = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        print(f"\n📊 Collector統計 (稼働時間: {uptime:.0f}秒):")
        print(f"   成功エピソード: {self.stats['successful_episodes']}")
        print(f"   LSLサンプル受信: {self.stats['lsl_samples_received']}")
        print(f"   ロボット状態更新: {self.stats['robot_state_updates']}")
        if self.state_share_manager:
            share_stats = self.state_share_manager.get_stats()
            print(f"   状態共有通知: {share_stats['subscriber_notifications']}")
            print(f"   購読者数: {share_stats['subscribers_count']}")
        print(f"   TCP接続: {'接続中' if self.tcp_interface.is_connected else '未接続'}")

    def _tcp_monitor_thread(self):
        print("📡 TCP監視スレッド開始（状態橋渡し機能強化版）")
        consecutive_empty_count = 0
        max_consecutive_empty = 100
        while self.is_running:
            try:
                if len(self.tcp_interface.received_data) > 0:
                    message_data = self.tcp_interface.received_data.popleft()
                    consecutive_empty_count = 0
                    self._process_tcp_message(message_data)
                else:
                    consecutive_empty_count += 1
                    if consecutive_empty_count == max_consecutive_empty:
                        print(f"⚠️ TCP受信キューが{max_consecutive_empty * 0.1:.1f}秒間空です")
                        consecutive_empty_count = 0
                    time.sleep(0.1)
            except IndexError:
                time.sleep(0.1)
            except Exception as e:
                if self.is_running:
                    print(f"❌ TCP監視エラー: {e}")
                time.sleep(0.1)
        print("📡 TCP監視スレッド終了")

    def _enqueue_trigger(self, tcp_data: Dict[str, Any], tcp_ts_wall: float, trigger_lsl_ts: float, trigger_note: str):
        trigger_info = {
            'tcp_data': tcp_data,
            'tcp_ts_wall': tcp_ts_wall,     # 壁時計
            'trigger_lsl_ts': trigger_lsl_ts,  # ★ LSL基準
            'trigger_type': trigger_note,
            'trigger_content': trigger_note
        }
        try:
            self.trigger_queue.put(trigger_info, timeout=1.0)
            self.stats['total_triggers'] += 1
            print(f"✅ トリガー情報キューイング完了: ep={tcp_data.get('episode','unknown')} ({trigger_note})")
        except Exception as e:
            print(f"❌ トリガーキューイングエラー: {e}")

    def _process_tcp_message(self, message_data):
        tcp_timestamp = time.time()
        message_content = None
        message_type = "unknown"
        tcp_ts_wall = time.time()       # ログ/人間用
        tcp_ts_lsl  = local_clock()     # ★ LSL基準（これをトリガーに使う）

        if isinstance(message_data, str):
            message_content = message_data.strip()
            message_type = "direct_string"
        elif isinstance(message_data, dict):
            msg_type = message_data.get('type', '')
            if msg_type == 'text_message':
                message_content = message_data.get('content', '').strip()
                message_type = "text_message"
            elif self._is_robot_state_data(message_data):
                message_type = "robot_state"
            else:
                message_type = f"json_{msg_type}" if msg_type else "json_unknown"
                for key in ['content', 'message', 'text', 'command']:
                    if key in message_data:
                        potential_content = str(message_data[key]).strip()
                        if potential_content:
                            message_content = potential_content
                            break

        # ★ 互換: EPISODE_END でも引き続きトリガー可能
        episode_end_patterns = ["EPISODE_END", "EPISODE_COMPLETE", "END_EPISODE"]
        if (message_content and any(p in message_content.upper() for p in episode_end_patterns)):
            print(f"🎯 エピソード終了トリガー検出! [{message_type}]: '{message_content}'")
            previous_json_data = self._get_previous_json_data()
            if previous_json_data:
                self._enqueue_trigger(previous_json_data, tcp_ts_wall, tcp_ts_lsl, "EPISODE_END")
            else:
                print("⚠️ 直前のロボット状態が見つかりません")
                self._debug_tcp_buffer()
            return

        # ★ robot_state は即トリガー（本要件）
        if message_type == "robot_state":
            tcp_entry = {'data': message_data, 'timestamp': tcp_timestamp, 'type': 'robot_state'}
            self.tcp_data_buffer.append(tcp_entry)

            robot_episode = message_data.get('episode', 'unknown')
            grip_force = message_data.get('grip_force', 'unknown')
            print(f"📋 ロボット状態受信: episode={robot_episode}, force={grip_force}")

            if self.trigger_on_robot_state:
                self._enqueue_trigger(message_data, tcp_ts_wall, tcp_ts_lsl, "ROBOT_STATE")

        # その他のメッセージもバッファに追加
        if message_type != "robot_state":
            tcp_entry = {
                'data': message_data if isinstance(message_data, dict) else {
                    'type': 'text_message', 
                    'content': str(message_data),
                    'original_type': message_type
                },
                'timestamp': tcp_timestamp,
                'type': message_type
            }
            self.tcp_data_buffer.append(tcp_entry)

    def _debug_tcp_buffer(self):
        print("🔍 TCPバッファデバッグ (最新10件):")
        recent_entries = list(self.tcp_data_buffer)[-10:]
        for i, entry in enumerate(recent_entries):
            data = entry['data']
            entry_type = entry.get('type', 'unknown')
            timestamp = entry['timestamp']
            age = time.time() - timestamp
            if isinstance(data, dict):
                if 'episode' in data:
                    episode = data.get('episode', 'N/A')
                    force = data.get('grip_force', 'N/A')
                    print(f"  [{i}] {entry_type}: episode={episode}, force={force} ({age:.1f}s前)")
                else:
                    keys = list(data.keys())[:3]
                    print(f"  [{i}] {entry_type}: keys={keys} ({age:.1f}s前)")
            else:
                content = str(data)[:30]
                print(f"  [{i}] {entry_type}: '{content}...' ({age:.1f}s前)")

    def _lsl_data_thread(self):
        print("📡 LSLデータ受信開始")
        last_sample_time = time.time()
        while self.is_running:
            try:
                sample, timestamp = self.eeg_receiver.inlet.pull_sample(timeout=1.0)
                if sample is not None:
                    self.stats['lsl_samples_received'] += 1
                    last_sample_time = time.time()
                    with self.buffer_lock:
                        if len(sample) >= 32:
                            normalized_sample = sample[:32]
                        else:
                            normalized_sample = sample + [0.0] * (32 - len(sample))
                        self.lsl_data_buffer.append(normalized_sample)
                        self.lsl_timestamp_buffer.append(timestamp)
                else:
                    current_time = time.time()
                    if current_time - last_sample_time > 10:
                        print(f"⚠️ LSLサンプル受信停止: {current_time - last_sample_time:.1f}秒間未受信")
                        last_sample_time = current_time
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ LSLデータ受信エラー: {e}")
                time.sleep(0.001)
        print("📡 LSLデータ受信終了")

    def _get_previous_json_data(self) -> Optional[Dict[str, Any]]:
        for tcp_entry in reversed(list(self.tcp_data_buffer)):
            data = tcp_entry['data']
            entry_type = tcp_entry.get('type', 'unknown')
            if (entry_type == 'robot_state' and isinstance(data, dict) and self._is_robot_state_data(data)):
                return data
        return None

    def _is_robot_state_data(self, data: Dict[str, Any]) -> bool:
        if not isinstance(data, dict):
            return False
        required_patterns = [
            ['episode', 'position', 'velocity', 'grip_force'],
            ['episode', 'grip_force', 'contact'],
            ['robot_episode', 'force', 'position'],
            ['episode', 'grip_force', 'broken']
        ]
        for pattern in required_patterns:
            if all(key in data for key in pattern):
                return True
        essential_keys = ['episode', 'grip_force']
        if all(key in data for key in essential_keys):
            return True
        return False

    def _episode_processing_thread(self):
        print("⚡ エピソード処理スレッド開始")
        timed_out = False
        while self.is_running:
            try:
                trigger_info = self.trigger_queue.get(timeout=1.0)
                # 未来0.9sが揃うまで待機
                t_trigger = trigger_info['trigger_lsl_ts']           # ★ LSL基準
                t_need    = t_trigger + self.post_trigger_seconds   
                waited = 0.0
                while self.is_running:
                    with self.buffer_lock:
                        latest_ts = self.lsl_timestamp_buffer[-1] if len(self.lsl_timestamp_buffer) > 0 else None
                    if latest_ts is not None and latest_ts >= t_need:
                        break
                    time.sleep(0.01)
                    waited += 0.01
                    if waited >= (self.post_trigger_seconds + 2.0):
                        print("⚠️ 未来分のLSLが十分に揃いませんでした（タイムアウト）")
                        timed_out = True
                        break
                if timed_out:
                    self.stats['failed_episodes'] += 1
                    continue 

                episode = self._create_episode(trigger_info)
                if episode:
                    self.episodes.append(episode)
                    self.stats['successful_episodes'] += 1
                    if self.save_to_csv:
                        self._save_episode_to_csv(episode)
                    print(f"✅ エピソード{episode.episode_id}保存完了 "
                          f"(同期遅延: {episode.sync_latency:.1f}ms, 品質: {episode.quality_score:.2f})")
                else:
                    self.stats['failed_episodes'] += 1
                    print("❌ エピソード作成失敗")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ エピソード処理エラー: {e}")
        print("⚡ エピソード処理スレッド終了")

    def _create_episode(self, trigger_info: Dict[str, Any]) -> Optional[Episode]:
        t_trigger = trigger_info['trigger_lsl_ts']           # ★
        tcp_ts_wall = trigger_info['tcp_ts_wall']  
        tcp_data    = trigger_info['tcp_data']   

        # 切り出し区間（時刻ベース）
        t_start = t_trigger - self.pre_trigger_seconds
        t_end   = t_trigger + self.post_trigger_seconds

        with self.buffer_lock:
            if len(self.lsl_timestamp_buffer) == 0:
                print("⚠️ LSLデータが空です")
                return None

            timestamps = np.array(self.lsl_timestamp_buffer)
            # インデックスを時刻で決定
            # start_idx: 最初に t >= t_start となるインデックス
            # end_idx  : 最後に t <= t_end   となるインデックス
            start_idx = np.searchsorted(timestamps, t_start, side='left')
            end_idx = np.searchsorted(timestamps, t_end, side='right') - 1

            if start_idx < 0: start_idx = 0
            if end_idx >= len(timestamps): end_idx = len(timestamps) - 1

            if end_idx < start_idx:
                print(f"⚠️ 時間窓が不正: start_idx={start_idx}, end_idx={end_idx}")
                return None

            # 実データ抽出
            eeg_indices = range(start_idx, end_idx + 1)
            raw_eeg_data = np.array([self.lsl_data_buffer[i] for i in eeg_indices])
            eeg_timestamps = timestamps[start_idx:end_idx + 1]

        # 同期遅延はトリガーに最も近いサンプルとの差
        if len(eeg_timestamps) == 0:
            print("⚠️ 指定時間窓にサンプルがありません")
            return None
        sync_latency = float(np.min(np.abs(eeg_timestamps - t_trigger)) * 1000.0)

        # 前処理
        try:
            preprocessing_result = self.eeg_preprocessor.preprocess_epoch(raw_eeg_data)
            processed_eeg = preprocessing_result['processed_epoch']
            preprocessing_info = {
                'processing_time_ms': preprocessing_result['processing_time_ms'],
                'quality_metrics': preprocessing_result['quality_metrics'],
                'rejected_channels': preprocessing_result.get('rejected_channels', [])
            }
            quality_score = self._calculate_episode_quality(
                preprocessing_result['quality_metrics'],
                sync_latency,
                tcp_data
            )
        except Exception as e:
            print(f"⚠️ 前処理エラー: {e}")
            processed_eeg = raw_eeg_data
            preprocessing_info = {'error': str(e)}
            quality_score = 0.5

        json_episode_id = tcp_data.get('episode', self.episode_counter)
        state_source = "tcp_direct"
        processing_wall = time.time()
        state_age_ms = max(0.0, (processing_wall - tcp_ts_wall) * 1000.0)

        episode = Episode(
            episode_id=json_episode_id,
            trigger_timestamp=t_trigger,
            lsl_data=processed_eeg,
            lsl_timestamps=eeg_timestamps,
            tcp_data=tcp_data,  
            tcp_timestamp=tcp_ts_wall,
            sync_latency=sync_latency,
            preprocessing_info=preprocessing_info,
            state_source=state_source,
            state_age_ms=state_age_ms,
            quality_score=quality_score
        )

        if trigger_info.get('trigger_type'):
            print("📝 エピソード作成詳細:")
            print(f"   ロボットエピソード番号: {json_episode_id}")
            print(f"   把持力: {tcp_data.get('grip_force', 'unknown')}N")
            print(f"   品質スコア: {quality_score:.3f}")
            print(f"   同期遅延: {sync_latency:.1f}ms")
            print(f"   切出窓: [{t_start:.3f}s 〜 {t_end:.3f}s]  "
                  f"({len(eeg_timestamps)} samples)")

        self.episode_counter += 1
        return episode

    def _calculate_episode_quality(self, quality_metrics: Dict[str, Any], sync_latency: float, tcp_data: Dict[str, Any]) -> float:
        try:
            quality_factors = []
            snr_db = quality_metrics.get('snr_db', 0)
            snr_quality = min(1.0, max(0.0, snr_db / 40.0))
            quality_factors.append(snr_quality)

            artifact_ratio = quality_metrics.get('artifact_ratio', 1.0)
            artifact_quality = max(0.0, 1.0 - artifact_ratio)
            quality_factors.append(artifact_quality)

            sync_quality = max(0.0, 1.0 - min(1.0, sync_latency / 1000.0))
            quality_factors.append(sync_quality)

            tcp_completeness = 1.0
            required_fields = ['episode', 'grip_force', 'contact']
            for field in required_fields:
                if field not in tcp_data:
                    tcp_completeness -= 0.2
            tcp_completeness = max(0.0, tcp_completeness)
            quality_factors.append(tcp_completeness)

            weights = [0.3, 0.2, 0.3, 0.2]
            overall_quality = sum(w * q for w, q in zip(weights, quality_factors))
            return min(1.0, max(0.0, overall_quality))
        except Exception as e:
            print(f"⚠️ 品質スコア計算エラー: {e}")
            return 0.5

    def _save_episode_to_csv(self, episode: Episode):
        try:
            episode_info_file = os.path.join(self.output_dir, f"episode_{episode.episode_id:04d}_info.csv")
            info_data = {
                'episode_id': [episode.episode_id],
                'trigger_timestamp': [episode.trigger_timestamp],
                'tcp_timestamp': [episode.tcp_timestamp],
                'sync_latency_ms': [episode.sync_latency],
                'state_source': [episode.state_source],
                'state_age_ms': [episode.state_age_ms],
                'quality_score': [episode.quality_score],
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
                'rejected_channels': [str(episode.preprocessing_info.get('rejected_channels', []))],
                'snr_db': [episode.preprocessing_info.get('quality_metrics', {}).get('snr_db', 0)],
                'artifact_ratio': [episode.preprocessing_info.get('quality_metrics', {}).get('artifact_ratio', 0)]
            }
            pd.DataFrame(info_data).to_csv(episode_info_file, index=False)

            eeg_data_file = os.path.join(self.output_dir, f"episode_{episode.episode_id:04d}_eeg.csv")
            eeg_df = pd.DataFrame(episode.lsl_data)
            eeg_df.columns = [f'ch_{i:02d}' for i in range(episode.lsl_data.shape[1])]
            eeg_df['timestamp'] = episode.lsl_timestamps
            eeg_df['sample_index'] = range(len(eeg_df))
            eeg_df.to_csv(eeg_data_file, index=False)

            summary_file = os.path.join(self.output_dir, "episodes_summary.csv")
            summary_data = {
                'episode_id': episode.episode_id,
                'trigger_timestamp': episode.trigger_timestamp,
                'tcp_timestamp': episode.tcp_timestamp,
                'sync_latency_ms': episode.sync_latency,
                'quality_score': episode.quality_score,
                'state_source': episode.state_source,
                'tcp_message_type': episode.tcp_data.get('type', 'robot_state'),
                'robot_episode': episode.tcp_data.get('episode', 'unknown'),
                'grip_force': episode.tcp_data.get('grip_force', 0),
                'actual_grip_force': episode.tcp_data.get('actual_grip_force', 0),
                'contact': episode.tcp_data.get('contact', False),
                'contact_force': episode.tcp_data.get('contact_force', 0),
                'broken': episode.tcp_data.get('broken', False),
                'eeg_samples': episode.lsl_data.shape[0],
                'eeg_channels': episode.lsl_data.shape[1],
                'processing_time_ms': episode.preprocessing_info.get('processing_time_ms', 0),
                'snr_db': episode.preprocessing_info.get('quality_metrics', {}).get('snr_db', 0),
                'artifact_ratio': episode.preprocessing_info.get('quality_metrics', {}).get('artifact_ratio', 0)
            }
            if not os.path.exists(summary_file):
                pd.DataFrame([summary_data]).to_csv(summary_file, index=False)
            else:
                pd.DataFrame([summary_data]).to_csv(summary_file, mode='a', header=False, index=False)
        except Exception as e:
            print(f"⚠️ CSV保存エラー: {e}")

    def stop_collection(self):
        print("🛑 データ収集停止中...")
        self.is_running = False
        self.tcp_interface.stop_server()
        self._print_final_statistics()
        print("🛑 データ収集停止完了")

    def _print_final_statistics(self):
        total_time = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        if self.stats['successful_episodes'] > 0:
            avg_latency = sum(ep.sync_latency for ep in self.episodes) / len(self.episodes)
            avg_quality = sum(ep.quality_score for ep in self.episodes) / len(self.episodes)
        else:
            avg_latency = 0
            avg_quality = 0

        print("\n📊 データ収集統計（状態橋渡し機能強化版）:")
        print(f"   セッション時間       : {total_time:.1f}秒")
        print(f"   総トリガー数         : {self.stats['total_triggers']}")
        print(f"   成功エピソード数     : {self.stats['successful_episodes']}")
        print(f"   失敗エピソード数     : {self.stats['failed_episodes']}")
        print(f"   LSLサンプル受信数    : {self.stats['lsl_samples_received']}")
        print(f"   ロボット状態更新数   : {self.stats['robot_state_updates']}")
        if self.stats['total_triggers'] > 0:
            success_rate = self.stats['successful_episodes'] / self.stats['total_triggers'] * 100
            print(f"   成功率               : {success_rate:.1f}%")
        print(f"   平均同期遅延         : {avg_latency:.1f}ms")
        print(f"   平均品質スコア       : {avg_quality:.3f}")
        print(f"   出力ディレクトリ     : {self.output_dir}")
        print("   TCP自動応答          : 無効（学習側制御）")

        if self.state_share_manager:
            share_stats = self.state_share_manager.get_stats()
            print(f"   状態共有通知数       : {share_stats['subscriber_notifications']}")
            print(f"   最終購読者数         : {share_stats['subscribers_count']}")
            if share_stats['subscribers_count'] > 0:
                print("   購読者詳細:")
                for sub_info in share_stats['subscribers_info']:
                    print(f"     {sub_info['name']}: {sub_info['notifications']}通知, {sub_info['errors']}エラー")

        if len(self.episodes) > 0:
            quality_scores = [ep.quality_score for ep in self.episodes]
            high_quality = sum(1 for q in quality_scores if q >= 0.8)
            medium_quality = sum(1 for q in quality_scores if 0.5 <= q < 0.8)
            low_quality = sum(1 for q in quality_scores if q < 0.5)
            print("\n   品質分布:")
            print(f"     高品質(≥0.8): {high_quality}件 ({high_quality/len(self.episodes)*100:.1f}%)")
            print(f"     中品質(0.5-0.8): {medium_quality}件 ({medium_quality/len(self.episodes)*100:.1f}%)")
            print(f"     低品質(<0.5): {low_quality}件 ({low_quality/len(self.episodes)*100:.1f}%)")

if __name__ == '__main__':
    print("🧠 LSL-TCP同期エピソード収集システム（状態橋渡し機能強化版）")
    print("=" * 70)
    print("強化内容:")
    print("1. robot_state到着をトリガーに前0.3s/後0.9sのLSLを切り出し")
    print("2. 未来0.9sが溜まるまで待機してから切り出し")
    print("3. エピソード品質評価・統計表示は従来通り")
    print("=" * 70)

    collector = LSLTCPEpisodeCollector(
        save_to_csv=True,
        enable_realtime_processing=False,
        enable_state_sharing=True,   # 状態共有機能有効
        pre_trigger_seconds=0.3,
        post_trigger_seconds=0.9,
        trigger_on_robot_state=True  # ← これが肝
    )

    if collector.start_collection():
        try:
            print("\n💡 システム実行中:")
            print(f"   1. LSL受信中: {collector.lsl_stream_name}")
            print(f"   2. TCP待機中: {collector.tcp_host}:{collector.tcp_port}")
            print("   3. 状態共有機能: 有効")
            print("   4. 学習側制御: 把持力は学習システムが決定")
            print("   5. 切出窓: 前0.3s / 後0.9s（到着時刻基準）")
            print("   6. Ctrl+C で終了")
            while collector.is_running:
                time.sleep(5)
                if collector.stats['successful_episodes'] > 0:
                    print(f"📊 進捗: {collector.stats['successful_episodes']}エピソード収集済み "
                          f"(状態更新: {collector.stats['robot_state_updates']}件)")
        except KeyboardInterrupt:
            print("\n⏹️ 停止要求")
        finally:
            collector.stop_collection()
    else:
        print("❌ システム開始失敗")
