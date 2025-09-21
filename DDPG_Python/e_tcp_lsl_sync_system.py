#!/usr/bin/env python3
"""
LSL-TCP同期エピソード収集システム（状態橋渡し機能強化版）

修正点：
1. 状態更新コールバック機能追加
2. 外部システムとの状態共有機能実装  
3. エラーハンドリング・ログ機能強化
4. TCP接続状態の詳細監視
5. Unity互換性向上
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
    lsl_data: np.ndarray  # (samples, channels) - 1.2秒分の前処理済みEEGデータ
    lsl_timestamps: np.ndarray  # LSLタイムスタンプ配列
    tcp_data: Dict[str, Any]  # 直前のTCPデータ
    tcp_timestamp: float
    sync_latency: float  # 同期遅延（ミリ秒）
    preprocessing_info: Dict[str, Any]  # 前処理情報
    
    # 新追加：状態共有用の追加情報
    state_source: str = "unknown"  # 状態データの取得元
    state_age_ms: float = 0.0  # 状態データの経過時間
    quality_score: float = 1.0  # エピソード品質スコア

class StateShareManager:
    """状態共有管理クラス（新追加）"""
    
    def __init__(self):
        self.subscribers = []  # 状態更新の購読者リスト
        self.latest_robot_state = None
        self.latest_robot_state_timestamp = None
        self.state_history = deque(maxlen=100)  # 直近100件の状態履歴
        self.update_count = 0
        
        # 統計情報
        self.stats = {
            'total_updates': 0,
            'subscriber_notifications': 0,
            'notification_errors': 0,
            'history_size': 0
        }
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None], subscriber_name: str = "unknown"):
        """状態更新の購読登録"""
        self.subscribers.append({
            'callback': callback,
            'name': subscriber_name,
            'registered_at': time.time(),
            'notification_count': 0,
            'error_count': 0
        })
        print(f"📡 状態共有購読登録: {subscriber_name} (総購読者数: {len(self.subscribers)})")
    
    def unsubscribe(self, callback: Callable[[Dict[str, Any]], None]):
        """購読解除"""
        self.subscribers = [sub for sub in self.subscribers if sub['callback'] != callback]
    
    def update_robot_state(self, state_data: Dict[str, Any], source: str = "unknown"):
        """ロボット状態の更新と通知"""
        try:
            current_time = time.time()
            
            # 状態データを保存
            self.latest_robot_state = state_data.copy()
            self.latest_robot_state_timestamp = current_time
            self.update_count += 1
            self.stats['total_updates'] += 1
            
            # 履歴に追加
            history_entry = {
                'timestamp': current_time,
                'source': source,
                'data': state_data.copy(),
                'episode': state_data.get('episode', 'unknown'),
                'grip_force': state_data.get('grip_force', 'unknown')
            }
            self.state_history.append(history_entry)
            self.stats['history_size'] = len(self.state_history)
            
            # 購読者に通知
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
        """最新の状態データを取得"""
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
        """状態履歴を取得"""
        history_list = list(self.state_history)
        return history_list[-max_count:] if max_count else history_list
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
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
                 lookback_seconds=3.2,
                 episode_duration=1.2,
                 max_buffer_seconds=10.0,
                 save_to_csv=True,
                 enable_realtime_processing=False,
                 enable_state_sharing=True):
        """
        初期化（状態橋渡し機能強化版）
        
        Args:
            lsl_stream_name: LSLストリーム名
            tcp_host: TCPホスト
            tcp_port: TCPポート
            sampling_rate: サンプリング周波数
            lookback_seconds: トリガーからさかのぼる時間
            episode_duration: エピソード長（秒）
            max_buffer_seconds: 最大バッファ時間
            enable_state_sharing: 状態共有機能の有効化
        """
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.sampling_rate = sampling_rate
        self.lookback_seconds = lookback_seconds
        self.episode_duration = episode_duration
        self.save_to_csv = save_to_csv
        self.enable_realtime_processing = enable_realtime_processing
        self.enable_state_sharing = enable_state_sharing
                
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
        
        # ★ 修正: 修正版EEGTCPInterfaceを使用（auto_reply=False、デバッグ有効）
        self.tcp_interface = EEGTCPInterface(
            host=tcp_host,
            port=tcp_port,
            max_buffer_size=10000,
            auto_reply=False  # 自動応答を無効化
        )
        
        # ★ 新機能: 状態共有管理システム
        self.state_share_manager = StateShareManager() if enable_state_sharing else None
        
        # データバッファ
        self.lsl_data_buffer = deque(maxlen=self.max_buffer_samples)
        self.lsl_timestamp_buffer = deque(maxlen=self.max_buffer_samples)
        self.tcp_data_buffer = deque(maxlen=10000)  # 最新10000件のTCPデータ
        
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
        
        # ★ 修正: 統計情報拡張
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
        
        print(f"🧠 LSL-TCP同期エピソード収集システム初期化完了（状態橋渡し機能強化版）")
        print(f"   セッションID: {self.session_id}")
        print(f"   ルックバック: {lookback_seconds}秒 ({self.lookback_samples}サンプル)")
        print(f"   エピソード長: {episode_duration}秒 ({self.episode_samples}サンプル)")
        print(f"   出力ディレクトリ: {self.output_dir}")
        print(f"   TCP自動応答: 無効")
        print(f"   状態共有機能: {'有効' if enable_state_sharing else '無効'}")
    
    def add_state_subscriber(self, callback: Callable[[Dict[str, Any]], None], subscriber_name: str = "unknown"):
        """状態更新の購読者を追加（外部システム用）"""
        if self.state_share_manager:
            self.state_share_manager.subscribe(callback, subscriber_name)
        else:
            print(f"⚠️ 状態共有機能が無効のため、購読登録できません: {subscriber_name}")
    
    def get_latest_robot_state(self) -> Optional[Dict[str, Any]]:
        """最新のロボット状態を取得（外部システム用）"""
        if self.state_share_manager:
            return self.state_share_manager.get_latest_state()
        return None
    
    def start_collection(self):
        """データ収集開始（状態橋渡し機能強化版）"""
        print(f"🚀 データ収集開始（状態橋渡し機能強化版）")
        
        # 出力ディレクトリ作成
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ★ 修正: TCP接続にデバッグモード有効化
        self.tcp_interface.enable_debug_mode(enable_all_logs=False)
        
        # LSL接続
        if not self.eeg_receiver.connect_to_stream():
            print(f"❌ LSL接続失敗")
            return False
        
        # TCP接続
        if not self.tcp_interface.start_server():
            print(f"❌ TCP接続失敗")
            return False
        
        # ★ 新機能: 状態共有のための内部コールバック設定
        if self.state_share_manager:
            self.tcp_interface.add_state_update_callback(self._handle_robot_state_update)
            print(f"🔗 状態共有コールバック設定完了")
        
        # 実行フラグ設定
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # スレッド開始
        self._start_threads()
        
        print(f"✅ データ収集開始完了")
        print(f"💡 トリガー待機中... (EPISODE_ENDでエピソード収集)")
        if self.state_share_manager:
            print(f"🔗 状態共有準備完了 - 外部システムからadd_state_subscriber()で購読可能")
        
        return True
    
    def _handle_robot_state_update(self, state_data: Dict[str, Any]):
        """ロボット状態更新ハンドラ（状態共有機能用）"""
        if self.state_share_manager and self._is_robot_state_data(state_data):
            self.stats['robot_state_updates'] += 1
            self.state_share_manager.update_robot_state(state_data, source="tcp_collector")
            self.stats['state_sharing_notifications'] += 1
    
    def _start_threads(self):
        """各種スレッドを開始"""
        # LSLデータ受信スレッド
        lsl_thread = threading.Thread(target=self._lsl_data_thread, daemon=True)
        lsl_thread.start()
        self.threads.append(lsl_thread)
        
        # TCP監視スレッド
        tcp_monitor_thread = threading.Thread(target=self._tcp_monitor_thread, daemon=True)
        tcp_monitor_thread.start()
        self.threads.append(tcp_monitor_thread)
        
        # エピソード処理スレッド
        episode_thread = threading.Thread(target=self._episode_processing_thread, daemon=True)
        episode_thread.start()
        self.threads.append(episode_thread)
        
        # ★ 新機能: 統計監視スレッド
        stats_thread = threading.Thread(target=self._stats_monitoring_thread, daemon=True)
        stats_thread.start()
        self.threads.append(stats_thread)
        
        print(f"🔄 バックグラウンドスレッド開始完了（状態橋渡し機能強化版）")
    
    def _stats_monitoring_thread(self):
        """統計監視スレッド（新機能）"""
        last_print_time = time.time()
        print_interval = 60  # 60秒ごと
        
        while self.is_running:
            try:
                current_time = time.time()
                
                if current_time - last_print_time >= print_interval:
                    self._print_stats_summary()
                    last_print_time = current_time
                
                time.sleep(10)  # 10秒間隔でチェック
                
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ 統計監視エラー: {e}")
                time.sleep(10)
        
        print(f"📊 統計監視スレッド終了")
    
    def _print_stats_summary(self):
        """統計サマリー表示"""
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
        """TCP受信データ監視スレッド（強化版）"""
        print(f"📡 TCP監視スレッド開始（状態橋渡し機能強化版）")

        consecutive_empty_count = 0
        max_consecutive_empty = 100  # 10秒間（0.1秒 × 100回）
        
        while self.is_running:
            try:
                # 新着メッセージをキューから取り出して処理
                if len(self.tcp_interface.received_data) > 0:
                    message_data = self.tcp_interface.received_data.popleft()
                    consecutive_empty_count = 0  # リセット
                    self._process_tcp_message(message_data)
                else:
                    consecutive_empty_count += 1
                    
                    # 長時間メッセージがない場合の警告
                    if consecutive_empty_count == max_consecutive_empty:
                        print(f"⚠️ TCP受信キューが{max_consecutive_empty * 0.1:.1f}秒間空です")
                        consecutive_empty_count = 0  # 重複防止でリセット
                    
                    time.sleep(0.1)

            except IndexError:
                # 受信キューが空の場合
                time.sleep(0.1)

            except Exception as e:
                if self.is_running:
                    print(f"❌ TCP監視エラー: {e}")
                    import traceback
                    traceback.print_exc()
                time.sleep(0.1)

        print(f"📡 TCP監視スレッド終了")

    def _process_tcp_message(self, message_data):
        """TCPメッセージの処理（強化版）"""
        tcp_timestamp = time.time()
        
        # ★ 修正: より詳細なメッセージ分類
        message_content = None
        message_type = "unknown"
        
        if isinstance(message_data, str):
            message_content = message_data.strip()
            message_type = "direct_string"
        elif isinstance(message_data, dict):
            # 辞書の内容を詳細に分析
            msg_type = message_data.get('type', '')
            
            if msg_type == 'text_message':
                message_content = message_data.get('content', '').strip()
                message_type = "text_message"
            elif self._is_robot_state_data(message_data):
                message_type = "robot_state"
            else:
                # その他のJSON
                message_type = f"json_{msg_type}" if msg_type else "json_unknown"
                
                # contentやmessageフィールドをチェック
                for key in ['content', 'message', 'text', 'command']:
                    if key in message_data:
                        potential_content = str(message_data[key]).strip()
                        if potential_content:
                            message_content = potential_content
                            break
        
        # EPISODE_ENDトリガーの検出（複数パターン対応）
        episode_end_patterns = ["EPISODE_END", "EPISODE_COMPLETE", "END_EPISODE"]
        
        if (message_content and 
            any(pattern in message_content.upper() for pattern in episode_end_patterns)):
            
            print(f"🎯 エピソード終了トリガー検出! [{message_type}]: '{message_content}'")
            print(f"   受信時刻: {tcp_timestamp}")
            
            # 直前のロボット状態データを検索
            previous_json_data = self._get_previous_json_data()
            if previous_json_data:
                robot_episode_id = previous_json_data.get('episode', 'unknown')
                print(f"📋 直前のロボット状態採用: episode={robot_episode_id}")
                
                trigger_info = {
                    'tcp_data': previous_json_data,
                    'tcp_timestamp': tcp_timestamp,
                    'trigger_timestamp': tcp_timestamp,
                    'trigger_type': 'EPISODE_END',
                    'trigger_content': message_content
                }
                
                try:
                    self.trigger_queue.put(trigger_info, timeout=1.0)
                    self.stats['total_triggers'] += 1
                    print(f"✅ トリガー情報キューイング完了: エピソード{robot_episode_id}")
                except Exception as e:
                    print(f"❌ トリガーキューイングエラー: {e}")
            else:
                print(f"⚠️ 直前のロボット状態が見つかりません")
                self._debug_tcp_buffer()
            
            return
        
        # ロボット状態データの処理
        if message_type == "robot_state":
            tcp_entry = {
                'data': message_data,
                'timestamp': tcp_timestamp,
                'type': 'robot_state'
            }
            self.tcp_data_buffer.append(tcp_entry)
            
            robot_episode = message_data.get('episode', 'unknown')
            grip_force = message_data.get('grip_force', 'unknown')
            print(f"📋 ロボット状態受信: episode={robot_episode}, force={grip_force}")
        
        # その他のメッセージもバッファに追加
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
        """TCPバッファのデバッグ表示"""
        print(f"🔍 TCPバッファデバッグ (最新10件):")
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
                    keys = list(data.keys())[:3]  # 最初の3つのキー
                    print(f"  [{i}] {entry_type}: keys={keys} ({age:.1f}s前)")
            else:
                content = str(data)[:30]
                print(f"  [{i}] {entry_type}: '{content}...' ({age:.1f}s前)")

    def _lsl_data_thread(self):
        """LSLデータ受信スレッド（統計追加）"""
        print(f"📡 LSLデータ受信開始")
        
        last_sample_time = time.time()
        
        while self.is_running:
            try:
                # LSLからサンプル取得
                sample, timestamp = self.eeg_receiver.inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    self.stats['lsl_samples_received'] += 1
                    last_sample_time = time.time()
                    
                    with self.buffer_lock:
                        # 32チャンネルに正規化
                        if len(sample) >= 32:
                            normalized_sample = sample[:32]
                        else:
                            normalized_sample = sample + [0.0] * (32 - len(sample))
                        
                        # バッファに追加
                        self.lsl_data_buffer.append(normalized_sample)
                        self.lsl_timestamp_buffer.append(timestamp)
                else:
                    # タイムアウト時のチェック
                    current_time = time.time()
                    if current_time - last_sample_time > 10:  # 10秒間サンプルなし
                        print(f"⚠️ LSLサンプル受信停止: {current_time - last_sample_time:.1f}秒間未受信")
                        last_sample_time = current_time  # 重複防止
                
            except Exception as e:
                if self.is_running:
                    print(f"⚠️ LSLデータ受信エラー: {e}")
                time.sleep(0.001)
        
        print(f"📡 LSLデータ受信終了")
    
    def _get_previous_json_data(self) -> Optional[Dict[str, Any]]:
        """直前のロボット状態データを取得（改良版）"""
        # TCPバッファを逆順で検索（最新から）
        for tcp_entry in reversed(list(self.tcp_data_buffer)):
            data = tcp_entry['data']
            entry_type = tcp_entry.get('type', 'unknown')
            
            # ロボット状態データの判定
            if (entry_type == 'robot_state' and 
                isinstance(data, dict) and 
                self._is_robot_state_data(data)):
                return data
        
        return None
    
    def _is_robot_state_data(self, data: Dict[str, Any]) -> bool:
        """ロボット状態データかを判定（改良版）"""
        if not isinstance(data, dict):
            return False
        
        # 必須フィールドのパターン
        required_patterns = [
            ['episode', 'position', 'velocity', 'grip_force'],  # 標準パターン
            ['episode', 'grip_force', 'contact'],               # 最小パターン
            ['robot_episode', 'force', 'position'],             # 代替パターン
            ['episode', 'grip_force', 'broken']                 # 簡易パターン
        ]
        
        for pattern in required_patterns:
            if all(key in data for key in pattern):
                return True
        
        # 部分的なマッチも許容（episode + grip_forceは必須）
        essential_keys = ['episode', 'grip_force']
        if all(key in data for key in essential_keys):
            return True
        
        return False
    
    def _episode_processing_thread(self):
        """エピソード処理スレッド（品質評価追加）"""
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
                    if self.save_to_csv:
                        self._save_episode_to_csv(episode)
                    
                    print(f"✅ エピソード{episode.episode_id}保存完了 "
                          f"(同期遅延: {episode.sync_latency:.1f}ms, "
                          f"品質: {episode.quality_score:.2f})")
                else:
                    self.stats['failed_episodes'] += 1
                    print(f"❌ エピソード作成失敗")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ エピソード処理エラー: {e}")
        
        print(f"⚡ エピソード処理スレッド終了")
    
    def _create_episode(self, trigger_info: Dict[str, Any]) -> Optional[Episode]:
        """トリガー情報からエピソードを作成（品質評価追加）"""
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
            
            # ★ 新機能: エピソード品質スコア計算
            quality_score = self._calculate_episode_quality(
                preprocessing_result['quality_metrics'],
                sync_latency,
                tcp_data
            )
            
        except Exception as e:
            print(f"⚠️ 前処理エラー: {e}")
            # 前処理失敗時は生データを使用
            processed_eeg = raw_eeg_data
            preprocessing_info = {'error': str(e)}
            quality_score = 0.5  # 中間的な品質スコア
        
        # episode_idは送信されたJSONデータの'episode'フィールドから取得
        json_episode_id = tcp_data.get('episode', self.episode_counter)
        
        # ★ 新機能: 状態ソース情報
        state_source = "tcp_direct"
        state_age_ms = (tcp_timestamp - trigger_timestamp) * 1000
        
        # エピソード作成（拡張版）
        episode = Episode(
            episode_id=json_episode_id,
            trigger_timestamp=trigger_timestamp,
            lsl_data=processed_eeg,
            lsl_timestamps=eeg_timestamps,
            tcp_data=tcp_data,
            tcp_timestamp=tcp_timestamp,
            sync_latency=sync_latency,
            preprocessing_info=preprocessing_info,
            state_source=state_source,
            state_age_ms=state_age_ms,
            quality_score=quality_score
        )
        
        # エピソード詳細情報表示
        if trigger_info.get('trigger_type') == 'EPISODE_END':
            print(f"📝 エピソード作成詳細:")
            print(f"   ロボットエピソード番号: {json_episode_id}")
            print(f"   把持力: {tcp_data.get('grip_force', 'unknown')}N")
            print(f"   品質スコア: {quality_score:.3f}")
            print(f"   同期遅延: {sync_latency:.1f}ms")
        
        self.episode_counter += 1
        return episode
    
    def _calculate_episode_quality(self, quality_metrics: Dict[str, Any], sync_latency: float, tcp_data: Dict[str, Any]) -> float:
        """エピソード品質スコアを計算（新機能）"""
        try:
            quality_factors = []
            
            # SNR品質 (0-1)
            snr_db = quality_metrics.get('snr_db', 0)
            snr_quality = min(1.0, max(0.0, snr_db / 40.0))  # 40dBを最大として正規化
            quality_factors.append(snr_quality)
            
            # アーティファクト率品質 (0-1)
            artifact_ratio = quality_metrics.get('artifact_ratio', 1.0)
            artifact_quality = max(0.0, 1.0 - artifact_ratio)
            quality_factors.append(artifact_quality)
            
            # 同期品質 (0-1)
            sync_quality = max(0.0, 1.0 - min(1.0, sync_latency / 1000.0))  # 1秒以上で0
            quality_factors.append(sync_quality)
            
            # データ完全性品質 (0-1)
            tcp_completeness = 1.0
            required_fields = ['episode', 'grip_force', 'contact']
            for field in required_fields:
                if field not in tcp_data:
                    tcp_completeness -= 0.2
            tcp_completeness = max(0.0, tcp_completeness)
            quality_factors.append(tcp_completeness)
            
            # 総合品質スコア（重み付き平均）
            weights = [0.3, 0.2, 0.3, 0.2]  # SNR、アーティファクト、同期、データ完全性
            overall_quality = sum(w * q for w, q in zip(weights, quality_factors))
            
            return min(1.0, max(0.0, overall_quality))
            
        except Exception as e:
            print(f"⚠️ 品質スコア計算エラー: {e}")
            return 0.5  # デフォルト値
    
    def _save_episode_to_csv(self, episode: Episode):
        """エピソードをCSVファイルに保存（拡張版）"""
        try:
            # エピソード基本情報のCSV（拡張版）
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
            
            # LSLデータのCSV
            eeg_data_file = os.path.join(self.output_dir, f"episode_{episode.episode_id:04d}_eeg.csv")
            eeg_df = pd.DataFrame(episode.lsl_data)
            eeg_df.columns = [f'ch_{i:02d}' for i in range(episode.lsl_data.shape[1])]
            eeg_df['timestamp'] = episode.lsl_timestamps
            eeg_df['sample_index'] = range(len(eeg_df))
            eeg_df.to_csv(eeg_data_file, index=False)
            
            # 統合サマリーCSV（全エピソード）- 拡張版
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
            
            # ファイルが存在しない場合はヘッダー付きで作成
            if not os.path.exists(summary_file):
                pd.DataFrame([summary_data]).to_csv(summary_file, index=False)
            else:
                pd.DataFrame([summary_data]).to_csv(summary_file, mode='a', header=False, index=False)
            
        except Exception as e:
            print(f"⚠️ CSV保存エラー: {e}")
    
    def stop_collection(self):
        """データ収集停止（状態橋渡し機能強化版）"""
        print(f"🛑 データ収集停止中...")
        
        self.is_running = False
        
        # TCP接続停止
        self.tcp_interface.stop_server()
        
        # 最終統計表示
        self._print_final_statistics()
        
        print(f"🛑 データ収集停止完了")
    
    def _print_final_statistics(self):
        """最終統計情報の表示（拡張版）"""
        if self.stats['start_time']:
            total_time = time.time() - self.stats['start_time']
        else:
            total_time = 0
        
        if self.stats['successful_episodes'] > 0:
            avg_latency = sum(ep.sync_latency for ep in self.episodes) / len(self.episodes)
            avg_quality = sum(ep.quality_score for ep in self.episodes) / len(self.episodes)
        else:
            avg_latency = 0
            avg_quality = 0
        
        print(f"\n📊 データ収集統計（状態橋渡し機能強化版）:")
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
        print(f"   TCP自動応答          : 無効（学習側制御）")
        
        # 状態共有統計
        if self.state_share_manager:
            share_stats = self.state_share_manager.get_stats()
            print(f"   状態共有通知数       : {share_stats['subscriber_notifications']}")
            print(f"   最終購読者数         : {share_stats['subscribers_count']}")
            
            if share_stats['subscribers_count'] > 0:
                print(f"   購読者詳細:")
                for sub_info in share_stats['subscribers_info']:
                    print(f"     {sub_info['name']}: {sub_info['notifications']}通知, {sub_info['errors']}エラー")
        
        # 品質分布統計
        if len(self.episodes) > 0:
            quality_scores = [ep.quality_score for ep in self.episodes]
            high_quality = sum(1 for q in quality_scores if q >= 0.8)
            medium_quality = sum(1 for q in quality_scores if 0.5 <= q < 0.8)
            low_quality = sum(1 for q in quality_scores if q < 0.5)
            
            print(f"\n   品質分布:")
            print(f"     高品質(≥0.8): {high_quality}件 ({high_quality/len(self.episodes)*100:.1f}%)")
            print(f"     中品質(0.5-0.8): {medium_quality}件 ({medium_quality/len(self.episodes)*100:.1f}%)")
            print(f"     低品質(<0.5): {low_quality}件 ({low_quality/len(self.episodes)*100:.1f}%)")

if __name__ == '__main__':
    print("🧠 LSL-TCP同期エピソード収集システム（状態橋渡し機能強化版）")
    print("=" * 70)
    print("強化内容:")
    print("1. 状態共有管理システム追加")
    print("2. 外部システムとの状態橋渡し機能実装")
    print("3. エピソード品質評価機能追加")
    print("4. 詳細統計・監視機能強化")
    print("5. Unity互換性向上")
    print("=" * 70)
    
    collector = LSLTCPEpisodeCollector(
        save_to_csv=True,
        enable_realtime_processing=False,
        enable_state_sharing=True  # 状態共有機能有効
    )
    
    # システム実行
    if collector.start_collection():
        try:
            print(f"\n💡 システム実行中（状態橋渡し機能強化版）:")
            print(f"   1. LSL受信中: MockEEG")
            print(f"   2. TCP待機中: 127.0.0.1:12345")
            print(f"   3. 状態共有機能: 有効")
            print(f"   4. 学習側制御: 把持力は学習システムが決定")
            print(f"   5. 品質評価: リアルタイム実行")
            print(f"   6. Ctrl+C で終了")
            
            while collector.is_running:
                time.sleep(5)
                if collector.stats['successful_episodes'] > 0:
                    print(f"📊 進捗: {collector.stats['successful_episodes']}エピソード収集済み "
                          f"(状態更新: {collector.stats['robot_state_updates']}件)")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ 停止要求")
        finally:
            collector.stop_collection()
    else:
        print(f"❌ システム開始失敗")