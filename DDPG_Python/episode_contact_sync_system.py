#!/usr/bin/env python3
"""
エピソード初回接触同期システム（メモリ最適化版）
基本設定（1.2秒エポック）で、各エピソードの初回接触時のデータのみを収集
🔥 修正：フィードバック値をA2CUnitySystemと同じようにランダム生成
📉 最適化：接触時の前後数秒のデータのみ保持してメモリ使用量を削減

仕様:
- エポック長: 1.2秒 (300サンプル)
- 32チャンネル EEG
- 各エピソードの初回接触（contact=True）のみ記録
- 同一エピソード内の重複を自動除外
- フィードバック値：ランダム生成（2.0-30.0N）
- メモリ最適化：接触時前後のデータのみ保持
"""

import numpy as np
import time
import threading
import json
import random  # 🔥 追加：ランダム値生成用
from collections import deque
import csv
import os
from datetime import datetime

# 既存システムを拡張
from systems.tcp_lsl_sync_system import TCPLSLSynchronizer

class EpisodeContactSynchronizer(TCPLSLSynchronizer):
    """エピソード初回接触同期システム（メモリ最適化版）"""
    
    def __init__(self, *args, **kwargs):
        # 🔥 ランダムフィードバック値の範囲を追加
        self.min_feedback_value = kwargs.pop('min_feedback_value', 2.0)
        self.max_feedback_value = kwargs.pop('max_feedback_value', 30.0)
        
        # 📉 メモリ最適化設定
        self.contact_buffer_duration = kwargs.pop('contact_buffer_duration', 3.0)  # 接触時前後の保持秒数
        
        super().__init__(*args, **kwargs)
        
        # 論文準拠の基本設定
        self.epoch_duration = 1.2  # 秒
        self.sampling_rate = 250  # Hz
        self.epoch_samples = int(self.epoch_duration * self.sampling_rate)  # 300サンプル
        self.n_channels = 32
        
        # エピソード管理
        self.current_episodes = {}  # episode_number -> episode_info
        self.processed_episodes = set()  # 既に処理済みのエピソード番号
        
        # 📉 最適化されたLSL連続データバッファ
        # 接触前のデータ保持用（接触検出まで常時保持）
        contact_buffer_samples = int(self.contact_buffer_duration * self.sampling_rate)  # 3秒分 = 750サンプル
        self.lsl_continuous_buffer = deque(maxlen=contact_buffer_samples)  
        self.lsl_timestamp_buffer = deque(maxlen=contact_buffer_samples)
        
        # 📉 接触時エピソード専用データストレージ（一時的）
        self.episode_contact_data = {}  # episode_num -> {'buffer': deque, 'timestamps': deque, 'contact_time': float}
        
        print(f"🎯 エピソード初回接触同期システム初期化（メモリ最適化版）:")
        print(f"   エポック長: {self.epoch_duration}秒 ({self.epoch_samples}サンプル)")
        print(f"   対象: 各エピソードの初回接触時のみ")
        print(f"   除外: 同一エピソード内の重複データ")
        print(f"   📉 接触時バッファ: {self.contact_buffer_duration}秒 ({contact_buffer_samples}サンプル)")
        print(f"   🎲 フィードバック値範囲: [{self.min_feedback_value:.1f}, {self.max_feedback_value:.1f}]")

    def tcp_receiver_thread(self):
        """TCP受信スレッド（エピソード・接触判定付き）"""
        print("🚀 TCP受信スレッド開始（エピソード管理付き）")
        
        while self.is_running:
            try:
                if self.unity_tcp.received_data:
                    tcp_data = self.unity_tcp.received_data.popleft()
                    receive_time = time.time()
                    
                    # エピソード・接触情報を解析
                    print(f"[DBG] raw TCP: {tcp_data}")
                    episode_info = self._parse_episode_contact_info(tcp_data)
                    
                    if episode_info and self._should_process_event(episode_info):
                        # 🔒 重要：処理決定時に即座に処理済みマーク
                        episode_num = episode_info['episode']
                        self._update_episode_state(episode_info)
                        
                        # 📉 初回接触検出時：現在のバッファを該当エピソード専用に移動
                        self._archive_contact_buffer_for_episode(episode_num, receive_time)
                        
                        # TCPイベントをバッファに追加
                        tcp_event = {
                            'timestamp': tcp_data.get('timestamp', receive_time),
                            'system_time': receive_time,
                            'data': tcp_data,
                            'type': tcp_data.get('type', 'unknown'),
                            'episode_info': episode_info
                        }
                        
                        self.tcp_event_buffer.append(tcp_event)
                        self.sync_stats['total_tcp_events'] += 1
                        
                        print(f"📥 Episode {episode_info['episode']} 初回接触確定: "
                              f"接触={episode_info['contact']}, "
                              f"把持力={episode_info.get('grip_force', 'N/A')}")
                    
                    # 🔒 処理対象外の場合も軽くログ出力（デバッグ用）
                    elif episode_info:
                        episode_num = episode_info['episode']
                        if episode_num in self.processed_episodes and episode_num % 10 == 0:
                            print(f"⏭️ Episode {episode_num}: スキップ（処理済み）")
                
                time.sleep(0.001)
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ TCP受信エラー: {e}")

    def _archive_contact_buffer_for_episode(self, episode_num: int, contact_time: float):
        """📉 接触検出時：現在のLSLバッファを該当エピソード専用に移動"""
        if episode_num not in self.episode_contact_data:
            # 接触前のデータをエピソード専用ストレージに移動
            episode_buffer = deque(self.lsl_continuous_buffer)  # 現在のバッファをコピー
            timestamp_buffer = deque(self.lsl_timestamp_buffer)  # タイムスタンプもコピー
            
            # エピソード専用データを作成
            post_contact_samples = int(self.contact_buffer_duration * self.sampling_rate)  # 接触後の追加保持サンプル数
            post_contact_buffer = deque(maxlen=post_contact_samples)
            post_contact_timestamps = deque(maxlen=post_contact_samples)
            
            self.episode_contact_data[episode_num] = {
                'pre_contact_buffer': episode_buffer,
                'pre_contact_timestamps': timestamp_buffer,
                'post_contact_buffer': post_contact_buffer,
                'post_contact_timestamps': post_contact_timestamps,
                'contact_time': contact_time,
                'samples_needed': post_contact_samples
            }
            
            print(f"📦 Episode {episode_num}: 接触前データ保存完了 ({len(episode_buffer)}サンプル)")

    def _parse_episode_contact_info(self, tcp_data: dict) -> dict:
        """TCPデータからエピソード・接触情報を抽出"""
        try:
            episode_info = {}
            
            # エピソード番号を取得
            episode_num = None
            if 'episode' in tcp_data:
                episode_num = tcp_data['episode']
            elif 'episode_number' in tcp_data:
                episode_num = tcp_data['episode_number']
            
            if episode_num is None:
                return None
                
            episode_info['episode'] = int(episode_num)
            
            # 接触状態を取得
            contact = False
            if 'contact' in tcp_data:
                contact = tcp_data['contact']
            elif 'hasContact' in tcp_data:
                contact = tcp_data['hasContact']
            elif 'has_contact' in tcp_data:
                contact = tcp_data['has_contact']
            
            # Booleanに変換
            if isinstance(contact, str):
                contact = contact.lower() in ['true', '1', 'yes']
            elif isinstance(contact, (int, float)):
                contact = bool(contact)
            
            episode_info['contact'] = contact
            
            # エピソードアクティブ状態
            active = tcp_data.get('active', True)
            if isinstance(active, str):
                active = active.lower() in ['true', '1', 'yes']
            episode_info['active'] = bool(active)
            
            # 追加情報
            episode_info['grip_force'] = tcp_data.get('grip_force', tcp_data.get('currentGripForce', 0))
            episode_info['timestamp'] = tcp_data.get('timestamp', time.time())
            
            return episode_info
            
        except Exception as e:
            print(f"⚠️ エピソード情報解析エラー: {e}, データ: {str(tcp_data)[:100]}")
            return None

    def _should_process_event(self, episode_info: dict) -> bool:
        """🔒 厳格な重複チェック：イベントを処理すべきかを判定"""


        episode_num = episode_info['episode']
        contact = episode_info['contact']
        active = episode_info['active']

        print(f"[DBG] should_process? episode={episode_num}, contact={contact}, "
            f"active={active}, already={episode_num in self.processed_episodes}")
        
        # 非アクティブなエピソードは無視
        if not active:
            return False
        
        # 接触していない場合は無視
        if not contact:
            return False
        
        # 🔒 最重要チェック：既に処理済みのエピソードは絶対に無視
        if episode_num in self.processed_episodes:
            if episode_num % 5 == 0:  # 5回に1回だけログ出力（スパム防止）
                print(f"🚫 Episode {episode_num}: 既に処理済み - スキップ")
            return False
        
        # 🔒 二重チェック：現在のエピソード状態確認
        if episode_num in self.current_episodes:
            prev_info = self.current_episodes[episode_num]
            # 以前に接触が記録されている場合は無視（初回ではない）
            if prev_info.get('contact', False):
                print(f"🚫 Episode {episode_num}: 既に接触記録あり - スキップ")
                return False
        
        # 🔒 最終確認：処理済みに即座に追加して重複防止
        print(f"✅ Episode {episode_num}: 初回接触検出 - 処理開始")
        return True

    def _update_episode_state(self, episode_info: dict):
        """🔒 エピソード状態を更新（即座に処理済みマーキング）"""
        episode_num = episode_info['episode']
        
        # 🔒 重要：即座に処理済みに追加（重複防止）
        self.processed_episodes.add(episode_num)
        
        # エピソード情報を更新
        self.current_episodes[episode_num] = episode_info
        
        # 確認ログ
        print(f"🔒 Episode {episode_num}: 処理済みマーク完了（重複防止）")

    def lsl_receiver_thread(self):
        """📉 LSL連続データ受信スレッド（メモリ最適化版）"""
        print("🚀 LSL連続受信スレッド開始（メモリ最適化版）")
        
        if not self.lsl_inlet:
            print("❌ LSL inlet が初期化されていません")
            return
            
        while self.is_running:
            try:
                sample, lsl_timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    # 📉 メイン連続バッファに追加（接触前データとして保持）
                    self.lsl_continuous_buffer.append(np.array(sample))
                    self.lsl_timestamp_buffer.append(lsl_timestamp)
                    
                    # 📉 すでに接触を検出したエピソードの接触後データを追加
                    self._update_post_contact_data(sample, lsl_timestamp)
                    
                    self.sync_stats['total_lsl_events'] += 1
                    
                    # 1秒ごとにデバッグ出力
                    if self.sync_stats['total_lsl_events'] % 250 == 0:
                        buffer_duration = len(self.lsl_continuous_buffer) / self.sampling_rate
                        active_episodes = len(self.episode_contact_data)
                        print(f"🧠 LSL連続受信: {self.sync_stats['total_lsl_events']} samples, "
                              f"メインバッファ: {buffer_duration:.1f}秒, "
                              f"処理済みエピソード数: {len(self.processed_episodes)}, "
                              f"アクティブエピソード: {active_episodes}")
                        
            except Exception as e:
                if self.is_running:
                    print(f"❌ LSL受信エラー: {e}")
                    time.sleep(1.0)

    def _update_post_contact_data(self, sample: np.ndarray, lsl_timestamp: float):
        """📉 接触後データの更新（必要な分のみ）"""
        completed_episodes = []
        
        for episode_num, episode_data in self.episode_contact_data.items():
            if 'post_contact_buffer' in episode_data and not episode_data.get('completed', False):
                # 接触後データが必要な場合のみ追加
                if len(episode_data['post_contact_buffer']) < episode_data['samples_needed']:
                    episode_data['post_contact_buffer'].append(np.array(sample))
                    episode_data['post_contact_timestamps'].append(lsl_timestamp)
                else:
                    # 📉 必要なサンプル数に達したら完了マーク
                    episode_data['completed'] = True
                    completed_episodes.append(episode_num)
        
        # 完了ログは1回のみ出力
        for episode_num in completed_episodes:
            print(f"📦 Episode {episode_num}: 接触後データ収集完了、メモリから削除準備")

    def _extract_epoch_around_time(self, target_time: float, episode_num: int = None):
        """📉 指定時刻周辺の1.2秒エポックを抽出（エピソード専用データを使用）"""
        
        # エピソード専用データが利用可能な場合はそれを使用
        if episode_num and episode_num in self.episode_contact_data:
            return self._extract_epoch_from_episode_data(target_time, episode_num)
        
        # フォールバック：メインバッファから抽出
        if len(self.lsl_continuous_buffer) < self.epoch_samples:
            return None, None, float('inf')
        
        # タイムスタンプバッファから最も近い時刻を検索
        timestamps = list(self.lsl_timestamp_buffer)
        time_diffs = [abs(ts + self.calculate_time_offset() - target_time) for ts in timestamps]
        
        if not time_diffs:
            return None, None, float('inf')
        
        min_diff_idx = time_diffs.index(min(time_diffs))
        sync_latency = min(time_diffs)
        
        # 1.2秒エポック範囲を計算（接触時刻を中心に前後0.6秒）
        half_epoch = self.epoch_samples // 2  # 150サンプル
        start_idx = max(0, min_diff_idx - half_epoch)
        end_idx = min(len(self.lsl_continuous_buffer), start_idx + self.epoch_samples)
        
        # 実際のエポックサイズを調整
        if end_idx - start_idx < self.epoch_samples:
            start_idx = max(0, end_idx - self.epoch_samples)
        
        # エポックデータを抽出
        epoch_samples = []
        epoch_timestamps = []
        
        for i in range(start_idx, end_idx):
            if i < len(self.lsl_continuous_buffer):
                epoch_samples.append(self.lsl_continuous_buffer[i])
                epoch_timestamps.append(self.lsl_timestamp_buffer[i])
        
        # データ不足の場合はゼロパディング
        while len(epoch_samples) < self.epoch_samples:
            epoch_samples.append(np.zeros(self.n_channels))
            epoch_timestamps.append(target_time)
        
        epoch_data = np.array(epoch_samples)  # (300, 32)
        
        return epoch_data, epoch_timestamps, sync_latency

    def _extract_epoch_from_episode_data(self, target_time: float, episode_num: int):
        """📉 エピソード専用データから1.2秒エポックを抽出"""
        episode_data = self.episode_contact_data[episode_num]
        
        # 接触前後のデータを結合
        combined_buffer = list(episode_data['pre_contact_buffer']) + list(episode_data['post_contact_buffer'])
        combined_timestamps = list(episode_data['pre_contact_timestamps']) + list(episode_data['post_contact_timestamps'])
        
        if len(combined_buffer) < self.epoch_samples:
            print(f"⚠️ Episode {episode_num}: エピソード専用データ不足 ({len(combined_buffer)}/{self.epoch_samples})")
            return None, None, float('inf')
        
        # タイムスタンプから最適な中心点を検索
        time_diffs = [abs(ts + self.calculate_time_offset() - target_time) for ts in combined_timestamps]
        
        if not time_diffs:
            return None, None, float('inf')
        
        min_diff_idx = time_diffs.index(min(time_diffs))
        sync_latency = min(time_diffs)
        
        # 1.2秒エポック範囲を計算
        half_epoch = self.epoch_samples // 2
        start_idx = max(0, min_diff_idx - half_epoch)
        end_idx = min(len(combined_buffer), start_idx + self.epoch_samples)
        
        if end_idx - start_idx < self.epoch_samples:
            start_idx = max(0, end_idx - self.epoch_samples)
        
        # エポックデータを抽出
        epoch_samples = combined_buffer[start_idx:end_idx]
        epoch_timestamps = combined_timestamps[start_idx:end_idx]
        
        # データ不足の場合はゼロパディング
        while len(epoch_samples) < self.epoch_samples:
            epoch_samples.append(np.zeros(self.n_channels))
            epoch_timestamps.append(target_time)
        
        epoch_data = np.array(epoch_samples)
        
        # 📉 エポック抽出完了後、エピソードデータをクリーンアップ
        self._cleanup_episode_data(episode_num)
        
        return epoch_data, epoch_timestamps, sync_latency

    def _cleanup_episode_data(self, episode_num: int):
        """📉 エピソード専用データのクリーンアップ（メモリ解放）"""
        if episode_num in self.episode_contact_data:
            episode_data = self.episode_contact_data[episode_num]
            pre_count = len(episode_data.get('pre_contact_buffer', []))
            post_count = len(episode_data.get('post_contact_buffer', []))
            total_samples = pre_count + post_count
            
            # エピソードデータを削除
            del self.episode_contact_data[episode_num]
            
            print(f"🗑️ Episode {episode_num}: メモリクリーンアップ完了 "
                  f"(解放: 接触前{pre_count} + 接触後{post_count} = {total_samples}サンプル)")

    def _generate_random_feedback(self, episode_info: dict) -> float:
        """
        🔥 A2CUnitySystemと同じパターンでランダムフィードバック値を生成
        """
        return random.uniform(self.min_feedback_value, self.max_feedback_value)

    def _create_synchronized_event(self, tcp_event: dict, lsl_event: dict = None):
        """エピソード初回接触エポック同期イベントを作成"""
        try:
            tcp_timestamp = tcp_event['system_time']
            episode_info = tcp_event['episode_info']
            episode_num = episode_info['episode']
            
            # 📉 エピソード専用データからエポック抽出
            epoch_data, epoch_timestamps, sync_latency = self._extract_epoch_around_time(
                tcp_timestamp, episode_num
            )
            
            if epoch_data is None:
                print(f"⚠️ Episode {episode_info['episode']}: エポックデータ不足")
                return None
            
            # 🔥 修正：固定された把持力ではなく、ランダムフィードバック値を生成
            feedback_value = self._generate_random_feedback(episode_info)
            
            # エピソード初回接触同期イベント作成
            episode_sync_event = {
                'episode_number': episode_info['episode'],
                'contact_timestamp': tcp_timestamp,
                'epoch_data': epoch_data,  # (300, 32) 
                'epoch_timestamps': epoch_timestamps,
                'episode_info': episode_info,
                'tcp_data': tcp_event['data'],
                'feedback_value': feedback_value,  # 🔥 修正：ランダム値を使用
                'sync_latency': sync_latency,
                'epoch_quality': self._assess_epoch_quality(epoch_data)
            }
            
            return episode_sync_event
            
        except Exception as e:
            print(f"❌ エピソード同期イベント作成エラー: {e}")
            return None

    def _assess_epoch_quality(self, epoch_data: np.ndarray) -> dict:
        """エポック品質評価"""
        if epoch_data is None or epoch_data.size == 0:
            return {'quality': 'poor', 'zero_ratio': 1.0}
        
        # ゼロサンプルの割合
        zero_count = np.sum(epoch_data == 0)
        total_count = epoch_data.size
        zero_ratio = zero_count / total_count
        
        # 信号強度評価
        mean_amplitude = np.mean(np.abs(epoch_data))
        
        # 品質判定
        if zero_ratio > 0.5:
            quality = 'poor'
        elif zero_ratio > 0.1:
            quality = 'fair'
        elif mean_amplitude < 1.0:
            quality = 'low_signal'
        else:
            quality = 'good'
        
        return {
            'quality': quality,
            'zero_ratio': zero_ratio,
            'mean_amplitude': mean_amplitude,
            'epoch_shape': epoch_data.shape
        }

    def _save_episode_epoch_to_csv(self, episode_sync_event: dict):
        """エピソードエポックをCSV形式で保存"""
        try:
            csv_filename = f"episode_contact_epochs_{self.session_id}.csv"
            
            # ファイルが存在しない場合、ヘッダーを追加
            file_exists = os.path.exists(csv_filename)
            
            with open(csv_filename, 'a', newline='', encoding='utf-8') as f:
                if not file_exists:
                    # CSVヘッダー
                    header = [
                        'episode_number', 'contact_timestamp', 'feedback_value', 
                        'sync_latency_ms', 'epoch_quality', 'zero_ratio',
                        'mean_amplitude', 'session_id'
                    ]
                    
                    # EEGチャンネルのヘッダーを追加（300サンプル × 32チャンネル = 9600列）
                    for sample_idx in range(self.epoch_samples):
                        for ch in range(self.n_channels):
                            header.append(f'eeg_s{sample_idx:03d}_ch{ch:02d}')
                    
                    writer = csv.writer(f)
                    writer.writerow(header)
                
                # データ行を構築
                quality_info = episode_sync_event.get('epoch_quality', {})
                row = [
                    episode_sync_event['episode_number'],
                    episode_sync_event['contact_timestamp'],
                    episode_sync_event['feedback_value'],
                    episode_sync_event['sync_latency'] * 1000,  # ms変換
                    quality_info.get('quality', 'unknown'),
                    quality_info.get('zero_ratio', 0.0),
                    quality_info.get('mean_amplitude', 0.0),
                    self.session_id
                ]
                
                # EEGエポックデータをフラット化して追加（9600列）
                epoch_flat = episode_sync_event['epoch_data'].flatten()  # (300,32) -> (9600,)
                row.extend(epoch_flat.tolist())
                
                writer = csv.writer(f)
                writer.writerow(row)
                
        except Exception as e:
            print(f"❌ エピソードCSV保存エラー: {e}")

    def _process_synchronization(self):
        """エピソード初回接触同期処理"""
        if not self.tcp_event_buffer:
            return
            
        # 最新のTCPイベント（初回接触）を取得
        tcp_event = self.tcp_event_buffer.pop()
        print(f"[DBG] pop tcp_event: episode={tcp_event['episode_info']['episode']}, "
            f"type={tcp_event['data'].get('type')}, buffer_len={len(self.tcp_event_buffer)}")

        
        # エピソード初回接触同期イベントを作成
        episode_sync_event = self._create_synchronized_event(tcp_event)
        
        if episode_sync_event:
            self.synchronized_events.append(episode_sync_event)
            self.sync_stats['successful_syncs'] += 1
            
            # エピソードCSVに保存
            self._save_episode_epoch_to_csv(episode_sync_event)
            
            # 統計更新
            latency_ms = episode_sync_event['sync_latency'] * 1000
            if latency_ms > self.sync_stats['max_latency_ms']:
                self.sync_stats['max_latency_ms'] = latency_ms
            if latency_ms < self.sync_stats['min_latency_ms']:
                self.sync_stats['min_latency_ms'] = latency_ms
            
            total_syncs = self.sync_stats['successful_syncs']
            current_avg = self.sync_stats['avg_latency_ms']
            new_avg = (current_avg * (total_syncs - 1) + latency_ms) / total_syncs
            self.sync_stats['avg_latency_ms'] = new_avg
            
            # デバッグ出力
            quality = episode_sync_event.get('epoch_quality', {})
            episode_info = episode_sync_event['episode_info']

            print(f"🎯 Episode {episode_info['episode']} 初回接触エポック保存成功:")
            print(f"   遅延: {latency_ms:.1f}ms, ランダムフィードバック: {episode_sync_event['feedback_value']:.3f}N")
            print(f"   品質: {quality.get('quality', 'unknown')}, ゼロ率: {quality.get('zero_ratio', 0.0):.3f}")
            
            # 🔥 重要：Unityにランダムフィードバック値を送信
            self._send_feedback_to_unity(episode_sync_event['feedback_value'], episode_info)
        else:
            self.sync_stats['failed_syncs'] += 1

    def _send_feedback_to_unity(self, feedback_value: float, episode_info: dict):
        """
        🔥 重要：A2CUnitySystemと同じパターンでUnityにフィードバック値を送信
        """
        message = {
            'type': 'grip_force_command',
            'target_force': float(feedback_value),
            'timestamp': time.time(),
            'episode_number': episode_info['episode'],
            'session_id': self.session_id
        }
        
        # UnityTCPInterfaceではなく、親クラスのunity_tcpを使用
        if hasattr(self, 'unity_tcp') and self.unity_tcp:
            try:
                message_json = json.dumps(message)
                if hasattr(self.unity_tcp, 'send_message'):
                    success = self.unity_tcp.send_message(message_json)
                elif hasattr(self.unity_tcp, 'send_data'):
                    success = self.unity_tcp.send_data(message_json)
                else:
                    print(f"❌ Unity TCP送信メソッドが見つかりません")
                    return
                    
                if success:
                    print(f"📤 Unity応答送信成功: {feedback_value:.3f}N")
                else:
                    print(f"❌ Unity応答送信失敗")
            except Exception as e:
                print(f"❌ Unity送信エラー: {e}")
        else:
            print(f"❌ Unity TCP接続が初期化されていません")

    def _cleanup_old_episode_data(self):
        """📉 古いエピソードデータの定期クリーンアップ"""
        current_time = time.time()
        cleanup_threshold = 60.0  # 60秒以上古いデータを削除
        
        episodes_to_remove = []
        for episode_num, episode_data in self.episode_contact_data.items():
            contact_time = episode_data.get('contact_time', current_time)
            
            # 完了済み、または60秒以上経過したデータを削除対象に
            if episode_data.get('completed', False) or (current_time - contact_time) > cleanup_threshold:
                episodes_to_remove.append(episode_num)
        
        # クリーンアップ実行
        for episode_num in episodes_to_remove:
            if episode_num in self.episode_contact_data:
                episode_data = self.episode_contact_data[episode_num]
                pre_count = len(episode_data.get('pre_contact_buffer', []))
                post_count = len(episode_data.get('post_contact_buffer', []))
                
                del self.episode_contact_data[episode_num]
                print(f"🧹 Episode {episode_num}: 自動クリーンアップ (解放: {pre_count + post_count}サンプル)")

    def _print_final_statistics(self):
        """最終統計表示（エピソード版）"""
        stats = self.sync_stats
        print(f"\n{'='*60}")
        print(f"📊 エピソード初回接触同期システム 最終統計")
        print(f"{'='*60}")
        print(f"処理済みエピソード数   : {len(self.processed_episodes)}")
        print(f"収集エポック数         : {stats['successful_syncs']:,}")
        print(f"総LSLサンプル数        : {stats['total_lsl_events']:,}")
        
        # 🔍 重複チェック結果の表示
        expected_epochs = len(self.processed_episodes)
        actual_epochs = stats['successful_syncs']
        if actual_epochs > expected_epochs:
            excess_epochs = actual_epochs - expected_epochs
            print(f"⚠️  重複検出             : {excess_epochs:,}エポック（{excess_epochs/expected_epochs*100:.1f}%重複）")
            print(f"❌ データサイズ異常     : 期待{expected_epochs}エポック → 実際{actual_epochs:,}エポック")
        else:
            print(f"✅ 重複チェック正常     : 期待{expected_epochs} = 実際{actual_epochs}エポック")
        
        print(f"平均同期遅延           : {stats['avg_latency_ms']:.2f}ms")
        print(f"最大同期遅延           : {stats['max_latency_ms']:.2f}ms")
        print(f"最小同期遅延           : {stats['min_latency_ms']:.2f}ms")
        print(f"フィードバック値範囲   : [{self.min_feedback_value:.1f}, {self.max_feedback_value:.1f}]N")
        print(f"📉 メモリ最適化効果    : 接触時前後{self.contact_buffer_duration}秒のみ保持")
        print(f"セッションID           : {self.session_id}")
        print(f"CSVファイル            : episode_contact_epochs_{self.session_id}.csv")
        
        # 📊 ファイルサイズ推定
        epoch_size_mb = (300 * 32 * 8) / 1024 / 1024  # 0.07MB per epoch
        expected_size_mb = expected_epochs * epoch_size_mb
        actual_size_mb = actual_epochs * epoch_size_mb
        
        print(f"🗂️ ファイルサイズ推定   : 期待{expected_size_mb:.1f}MB → 実際{actual_size_mb:.0f}MB")
        
        print(f"{'='*60}")
        
        if self.processed_episodes:
            episodes_list = sorted(list(self.processed_episodes))
            print(f"処理済みエピソード: {episodes_list}")
        
        # 📉 メモリ使用状況の報告
        active_episode_count = len(self.episode_contact_data)
        main_buffer_samples = len(self.lsl_continuous_buffer)
        print(f"📊 メモリ使用状況:")
        print(f"   メインLSLバッファ   : {main_buffer_samples:,}サンプル")
        print(f"   アクティブエピソード : {active_episode_count}")

    def run_episode_collection_session(self, duration_seconds=600, target_episodes=50):
        """📉 エピソード収集セッションの実行（メモリ最適化版）"""
        if not self.start_synchronization_system():
            return
            
        print(f"🎯 エピソード初回接触データ収集開始")
        print(f"⏱️ 収集時間: {duration_seconds}秒 ({duration_seconds//60}分)")
        print(f"🎯 目標エピソード数: {target_episodes}")
        print(f"📉 メモリ最適化: 接触時前後{self.contact_buffer_duration}秒のみ保持")
        
        start_time = time.time()
        last_cleanup_time = start_time
        cleanup_interval = 30.0  # 30秒ごとにクリーンアップ
        
        try:
            while self.is_running:
                elapsed = time.time() - start_time
                current_time = time.time()
                
                # 📉 定期的なメモリクリーンアップ
                if current_time - last_cleanup_time > cleanup_interval:
                    self._cleanup_old_episode_data()
                    last_cleanup_time = current_time
                
                # 進捗表示
                if elapsed % 30 == 0 and elapsed > 0:
                    episodes_processed = len(self.processed_episodes)
                    progress_pct = (episodes_processed / target_episodes) * 100 if target_episodes > 0 else 0
                    remaining_time = duration_seconds - elapsed
                    
                    # 📉 メモリ使用状況も表示
                    active_episodes = len(self.episode_contact_data)
                    main_buffer_size = len(self.lsl_continuous_buffer)
                    
                    print(f"📈 進捗: {elapsed:.0f}秒経過 | "
                          f"処理済み: {episodes_processed}/{target_episodes} ({progress_pct:.1f}%) | "
                          f"残り: {remaining_time:.0f}秒 | "
                          f"📉 メモリ: メイン{main_buffer_size}, アクティブ{active_episodes}")
                
                # 終了条件チェック
                if elapsed >= duration_seconds:
                    print(f"⏰ 制限時間に達しました（{duration_seconds}秒）")
                    break
                    
                if len(self.processed_episodes) >= target_episodes:
                    print(f"🎯 目標エピソード数に達しました（{target_episodes}エピソード）")
                    break
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print(f"\n⚡ ユーザーによる中断")
        finally:
            print(f"🔚 データ収集終了...")
            
            # 📉 最終クリーンアップ
            for episode_num in list(self.episode_contact_data.keys()):
                self._cleanup_episode_data(episode_num)
            
            self.stop_synchronization_system()
            self._print_final_statistics()

# 📉 使用例とメモリ最適化の効果
if __name__ == "__main__":
    print("🎯 エピソード初回接触同期システム（メモリ最適化版）")
    
    # メモリ最適化設定
    sync_system = EpisodeContactSynchronizer(
        tcp_host='127.0.0.1',
        tcp_port=12345,
        lsl_stream_name='MockEEG',
        max_sync_events=100,
        sync_tolerance_ms=50,
        min_feedback_value=2.0,
        max_feedback_value=30.0,
        contact_buffer_duration=3.0  # 📉 接触時前後3秒のみ保持
    )
    
    print(f"📉 メモリ最適化効果:")
    print(f"   従来版: 3.6秒 × 250Hz × 32ch = 28,800データポイントを常時保持")
    print(f"   最適化版: 3.0秒 × 250Hz × 32ch × エピソード数のみ = 大幅削減")
    print(f"   推定削減率: 70-90% (エピソード数による)")
    
    # エピソード収集セッション実行
    sync_system.run_episode_collection_session(
        duration_seconds=300,  # 5分間
        target_episodes=20     # 20エピソード
    )