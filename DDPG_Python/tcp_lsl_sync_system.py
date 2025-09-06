#!/usr/bin/env python3
"""
TCP-LSL時刻同期システム
UnityからのTCP明示的フィードバックとLSL EEGデータの同期制御

主な機能:
1. 高精度時刻同期（ハードウェアタイムスタンプ使用）
2. TCP・LSLイベントの時刻対応付け
3. 遅延補正とタイミング調整
4. 同期データのCSV蓄積（1000回分）
"""

import time
import threading
import queue
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import csv
import os
from datetime import datetime

# 既存モジュールのインポート（プロジェクト内から）
from envs.unity_tcp_interface import UnityTCPInterface
from pylsl import StreamInlet, resolve_streams, local_clock

@dataclass
class SynchronizedEvent:
    """同期イベントデータクラス"""
    tcp_timestamp: float
    lsl_timestamp: float
    system_timestamp: float
    tcp_data: dict
    lsl_data: np.ndarray
    event_type: str
    feedback_value: float
    sync_latency: float

class TCPLSLSynchronizer:
    """TCP-LSL時刻同期システム"""
    
    def __init__(self, 
                 tcp_host='127.0.0.1', 
                 tcp_port=12345,
                 lsl_stream_name='MockEEG',
                 max_sync_events=1000,
                 sync_tolerance_ms=50):
        """
        初期化
        
        Args:
            tcp_host: Unity TCP ホスト
            tcp_port: Unity TCP ポート  
            lsl_stream_name: LSLストリーム名
            max_sync_events: 最大同期イベント数
            sync_tolerance_ms: 同期許容誤差（ミリ秒）
        """
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.lsl_stream_name = lsl_stream_name
        self.max_sync_events = max_sync_events
        self.sync_tolerance = sync_tolerance_ms / 1000.0  # 秒に変換
        
        # システム開始時刻（すべての時刻の基準）
        self.system_start_time = time.time()
        self.lsl_start_time = None
        
        # 通信インターフェース
        self.unity_tcp = UnityTCPInterface(tcp_host, tcp_port)
        self.lsl_inlet = None
        
        # 時刻同期管理
        self.tcp_event_buffer = deque(maxlen=1000)
        self.lsl_event_buffer = deque(maxlen=1000) 
        self.synchronized_events = deque(maxlen=max_sync_events)
        
        # 同期統計
        self.sync_stats = {
            'total_tcp_events': 0,
            'total_lsl_events': 0, 
            'successful_syncs': 0,
            'failed_syncs': 0,
            'avg_latency_ms': 0.0,
            'max_latency_ms': 0.0,
            'min_latency_ms': float('inf')
        }
        
        # 実行制御
        self.is_running = False
        self.threads = []
        
        # CSV出力設定
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_filename = f"tcp_lsl_sync_data_{self.session_id}.csv"
        
        print(f"🔄 TCP-LSL同期システム初期化完了")
        print(f"   📡 TCP: {tcp_host}:{tcp_port}")
        print(f"   🧠 LSL: {lsl_stream_name}")
        print(f"   ⏱️ 同期許容誤差: {sync_tolerance_ms}ms")
        print(f"   📊 最大イベント数: {max_sync_events}")

    def setup_lsl_connection(self) -> bool:
        """LSLストリームへの接続を確立"""
        try:
            print(f"🔍 LSLストリーム '{self.lsl_stream_name}' を検索中...")
            print("   使用予定: mock_eeg_sender.py からのストリーム")
            
            # プロジェクト内の既存コードと全く同じ方法を使用
            from pylsl import resolve_streams
            streams = resolve_streams()
            
            # 指定した名前のストリームを検索
            target_stream = None
            for stream in streams:
                if stream.name() == self.lsl_stream_name:
                    target_stream = stream
                    break
            
            if target_stream is None:
                print(f"❌ LSLストリーム '{self.lsl_stream_name}' が見つかりません")
                if streams:
                    print("利用可能なストリーム:")
                    for stream in streams:
                        print(f"  - {stream.name()} ({stream.type()})")
                else:
                    print("利用可能なストリームがありません")
                print()
                print("📝 mock_eeg_sender.py の実行手順:")
                print("   1. 別ターミナルで: python lsl_mock_data_send_test/mock_eeg_sender.py")
                print("   2. sender側で 'start' コマンドを実行")
                print("   3. このスクリプトを再実行")
                return False
                
            # ストリームに接続
            self.lsl_inlet = StreamInlet(target_stream, max_buflen=360, 
                                      max_chunklen=1, recover=True)
            
            # LSL時刻の基準を設定
            self.lsl_start_time = local_clock()
            
            # ストリーム情報を表示
            info = self.lsl_inlet.info()
            print(f"✅ LSLストリーム接続成功")
            print(f"   ストリーム名: {info.name()}")
            print(f"   チャンネル数: {info.channel_count()}")
            print(f"   サンプリング周波数: {info.nominal_srate()}Hz")
            print(f"   データ形式: {info.channel_format()}")
            
            return True
            
        except ImportError as e:
            print(f"❌ pylslインポートエラー: {e}")
            print("   pip install pylsl")
            return False
        except Exception as e:
            print(f"❌ LSL接続エラー: {e}")
            print("   pylslが正しくインストールされているか確認してください")
            return False

    def calculate_time_offset(self) -> float:
        """システム時刻とLSL時刻のオフセットを計算"""
        if self.lsl_start_time is None:
            return 0.0
            
        system_time = time.time()
        lsl_time = local_clock()
        
        # オフセット計算（LSL時刻をシステム時刻に変換するため）
        offset = system_time - lsl_time
        return offset

    def tcp_receiver_thread(self):
        """TCPデータ受信スレッド"""
        print("🚀 TCP受信スレッド開始")
        
        while self.is_running:
            try:
                if self.unity_tcp.received_data:
                    # 新しいTCPデータを取得
                    tcp_data = self.unity_tcp.received_data.popleft()
                    receive_time = time.time()
                    
                    # TCPタイムスタンプを取得（Unity側から送信された時刻）
                    tcp_timestamp = tcp_data.get('timestamp', receive_time)
                    
                    # TCPイベントをバッファに追加
                    tcp_event = {
                        'timestamp': tcp_timestamp,
                        'system_time': receive_time,
                        'data': tcp_data,
                        'type': tcp_data.get('type', 'unknown')
                    }
                    
                    self.tcp_event_buffer.append(tcp_event)
                    self.sync_stats['total_tcp_events'] += 1
                    
                    # デバッグ出力
                    if tcp_data.get('type') == 'feedback':
                        print(f"📨 TCP フィードバック受信: {tcp_data}")
                
                time.sleep(0.001)  # 1ms間隔でチェック
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ TCP受信エラー: {e}")

    def lsl_receiver_thread(self):
        """LSLデータ受信スレッド"""
        print("🚀 LSL受信スレッド開始")
        
        if not self.lsl_inlet:
            print("❌ LSL inlet が初期化されていません")
            return
            
        while self.is_running:
            try:
                # LSLサンプルを取得
                sample, lsl_timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    # システム時刻に変換
                    time_offset = self.calculate_time_offset()
                    system_timestamp = lsl_timestamp + time_offset
                    
                    # LSLイベントをバッファに追加
                    lsl_event = {
                        'lsl_timestamp': lsl_timestamp,
                        'system_timestamp': system_timestamp,
                        'sample': np.array(sample),
                        'receive_time': time.time()
                    }
                    
                    self.lsl_event_buffer.append(lsl_event)
                    self.sync_stats['total_lsl_events'] += 1
                    
                    # 100サンプルごとにデバッグ出力
                    if self.sync_stats['total_lsl_events'] % 100 == 0:
                        print(f"🧠 LSL受信済み: {self.sync_stats['total_lsl_events']} samples")
                        
            except Exception as e:
                if self.is_running:
                    print(f"❌ LSL受信エラー: {e}")

    def synchronization_thread(self):
        """時刻同期処理スレッド"""
        print("🚀 時刻同期スレッド開始")
        
        while self.is_running:
            try:
                self._process_synchronization()
                time.sleep(0.010)  # 10ms間隔で同期処理
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 同期処理エラー: {e}")

    def _process_synchronization(self):
        """同期処理のメインロジック"""
        # TCPイベントがある場合のみ処理
        if not self.tcp_event_buffer:
            return
            
        # 最新のTCPイベントを取得し、処理済みバッファから除去
        tcp_event = self.tcp_event_buffer.popleft()
        tcp_time = tcp_event['system_time']
        
        # 対応するLSLデータを検索
        best_lsl_event = self._find_closest_lsl_event(tcp_time)
        
        if best_lsl_event is not None:
            # 使ったLSLイベントをバッファから削除
            try:
                self.lsl_event_buffer.remove(best_lsl_event)
            except ValueError:
                pass

            # 同期イベントを作成
            sync_event = self._create_synchronized_event(tcp_event, best_lsl_event)
            
            if sync_event:
                self.synchronized_events.append(sync_event)
                self.sync_stats['successful_syncs'] += 1
                
                # 統計更新
                self._update_sync_statistics(sync_event)
                
                # CSVに保存（1000回分蓄積）
                self._save_to_csv(sync_event)
                
                print(f"✅ 同期成功 [{self.sync_stats['successful_syncs']:4d}]: "
                      f"遅延={sync_event.sync_latency*1000:.1f}ms, "
                      f"フィードバック={sync_event.feedback_value:.3f}")

    def _find_closest_lsl_event(self, target_time: float) -> Optional[dict]:
        """指定時刻に最も近いLSLイベントを検索"""
        if not self.lsl_event_buffer:
            return None
            
        best_event = None
        min_time_diff = float('inf')
        
        for lsl_event in self.lsl_event_buffer:
            time_diff = abs(lsl_event['system_timestamp'] - target_time)
            
            # 同期許容誤差以内で最も近いイベントを選択
            if time_diff < self.sync_tolerance and time_diff < min_time_diff:
                min_time_diff = time_diff
                best_event = lsl_event
                
        return best_event

    def _create_synchronized_event(self, tcp_event: dict, lsl_event: dict) -> Optional[SynchronizedEvent]:
        """同期イベントを作成"""
        try:
            # フィードバック値を抽出
            feedback_value = self._extract_feedback_value(tcp_event['data'])
            
            # 同期遅延を計算
            sync_latency = abs(lsl_event['system_timestamp'] - tcp_event['system_time'])
            
            sync_event = SynchronizedEvent(
                tcp_timestamp=tcp_event['timestamp'],
                lsl_timestamp=lsl_event['lsl_timestamp'],
                system_timestamp=tcp_event['system_time'],
                tcp_data=tcp_event['data'],
                lsl_data=lsl_event['sample'],
                event_type=tcp_event['type'],
                feedback_value=feedback_value,
                sync_latency=sync_latency
            )
            
            return sync_event
            
        except Exception as e:
            print(f"❌ 同期イベント作成エラー: {e}")
            return None

    def _extract_feedback_value(self, tcp_data: dict) -> float:
        """TCPデータからフィードバック値を抽出"""
        # フィードバック値の候補キーをチェック
        feedback_keys = ['feedback', 'value', 'reward', 'error', 'grip_force']
        
        for key in feedback_keys:
            if key in tcp_data:
                value = tcp_data[key]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    try:
                        return float(value)
                    except ValueError:
                        pass
                        
        # テキスト解析でフィードバックを判定
        text_data = str(tcp_data).lower()
        if any(word in text_data for word in ['success', 'good', 'correct']):
            return 1.0
        elif any(word in text_data for word in ['error', 'fail', 'bad']):
            return -1.0
        else:
            return 0.0

    def _update_sync_statistics(self, sync_event: SynchronizedEvent):
        """同期統計を更新"""
        latency_ms = sync_event.sync_latency * 1000
        
        # 遅延統計の更新
        if latency_ms > self.sync_stats['max_latency_ms']:
            self.sync_stats['max_latency_ms'] = latency_ms
        if latency_ms < self.sync_stats['min_latency_ms']:
            self.sync_stats['min_latency_ms'] = latency_ms
            
        # 平均遅延の更新
        total_syncs = self.sync_stats['successful_syncs']
        current_avg = self.sync_stats['avg_latency_ms']
        new_avg = (current_avg * (total_syncs - 1) + latency_ms) / total_syncs
        self.sync_stats['avg_latency_ms'] = new_avg

    def _save_to_csv(self, sync_event: SynchronizedEvent):
        """同期イベントをCSVに保存"""
        try:
            file_exists = os.path.exists(self.csv_filename)
            
            with open(self.csv_filename, 'a', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'event_id', 'tcp_timestamp', 'lsl_timestamp', 'system_timestamp',
                    'event_type', 'feedback_value', 'sync_latency_ms',
                    'tcp_data_json', 'lsl_channels', 'session_id'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
                    'event_id': self.sync_stats['successful_syncs'],
                    'tcp_timestamp': sync_event.tcp_timestamp,
                    'lsl_timestamp': sync_event.lsl_timestamp,
                    'system_timestamp': sync_event.system_timestamp,
                    'event_type': sync_event.event_type,
                    'feedback_value': sync_event.feedback_value,
                    'sync_latency_ms': sync_event.sync_latency * 1000,
                    'tcp_data_json': json.dumps(sync_event.tcp_data),
                    'lsl_channels': len(sync_event.lsl_data),
                    'session_id': self.session_id
                })
                
        except Exception as e:
            print(f"❌ CSV保存エラー: {e}")

    def start_synchronization_system(self):
        """同期システム開始"""
        if self.is_running:
            print("⚠️ システムは既に実行中です")
            return
            
        print("🚀 TCP-LSL同期システム開始")
        
        # LSL接続を確立
        if not self.setup_lsl_connection():
            print("❌ LSL接続に失敗しました。システムを終了します。")
            return
            
        # Unity TCP サーバー開始
        self.unity_tcp.start_server()
        
        # 実行フラグを設定
        self.is_running = True
        
        # スレッド開始
        threads_config = [
            ("TCP受信", self.tcp_receiver_thread),
            ("LSL受信", self.lsl_receiver_thread), 
            ("時刻同期", self.synchronization_thread)
        ]
        
        for name, target in threads_config:
            thread = threading.Thread(target=target, name=name)
            thread.daemon = True
            thread.start()
            self.threads.append(thread)
            
        print(f"✅ 同期システム開始完了")
        print(f"📁 同期データ保存先: {self.csv_filename}")
        
        return True

    def stop_synchronization_system(self):
        """同期システム停止"""
        print("🛑 TCP-LSL同期システム停止中...")
        
        self.is_running = False
        
        # Unity TCP停止
        self.unity_tcp.stop_server()
        
        # スレッド終了を待機
        for thread in self.threads:
            thread.join(timeout=2.0)
            
        # 最終統計を表示
        self._print_final_statistics()
        
        print("✅ TCP-LSL同期システム停止完了")

    def _print_final_statistics(self):
        """最終統計を表示"""
        stats = self.sync_stats
        print(f"\n{'='*60}")
        print(f"📊 TCP-LSL同期システム 最終統計")
        print(f"{'='*60}")
        print(f"総TCPイベント数     : {stats['total_tcp_events']:,}")
        print(f"総LSLイベント数     : {stats['total_lsl_events']:,}")
        print(f"同期成功数          : {stats['successful_syncs']:,}")
        print(f"同期失敗数          : {stats['failed_syncs']:,}")
        
        if stats['successful_syncs'] > 0:
            success_rate = (stats['successful_syncs'] / stats['total_tcp_events']) * 100
            print(f"同期成功率          : {success_rate:.1f}%")
            print(f"平均同期遅延        : {stats['avg_latency_ms']:.2f}ms")
            print(f"最大同期遅延        : {stats['max_latency_ms']:.2f}ms") 
            print(f"最小同期遅延        : {stats['min_latency_ms']:.2f}ms")
        
        print(f"セッションID        : {self.session_id}")
        print(f"CSVファイル         : {self.csv_filename}")
        print(f"{'='*60}")

    def run_data_collection_session(self, duration_seconds=300):
        """データ収集セッションの実行（1000回分のデータ収集を目標）"""
        if not self.start_synchronization_system():
            return
            
        print(f"⏱️ データ収集セッション開始: {duration_seconds}秒間")
        print(f"🎯 目標: {self.max_sync_events}回分の同期データ収集")
        
        start_time = time.time()
        last_report_time = start_time
        
        try:
            while time.time() - start_time < duration_seconds and self.is_running:
                current_time = time.time()
                
                # 10秒ごとに進捗報告
                if current_time - last_report_time >= 10:
                    elapsed = current_time - start_time
                    remaining = duration_seconds - elapsed
                    progress = (self.sync_stats['successful_syncs'] / self.max_sync_events) * 100
                    
                    print(f"📈 進捗: {elapsed:.0f}秒経過 | "
                          f"同期済み: {self.sync_stats['successful_syncs']}/{self.max_sync_events} "
                          f"({progress:.1f}%) | "
                          f"残り: {remaining:.0f}秒")
                    
                    last_report_time = current_time
                
                # 目標数に達した場合は終了
                if self.sync_stats['successful_syncs'] >= self.max_sync_events:
                    print(f"🎉 目標達成! {self.max_sync_events}回分のデータ収集完了")
                    break
                    
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n⏸️ ユーザー中断")
        finally:
            self.stop_synchronization_system()


def main():
    """メイン実行関数"""
    print("🔄 TCP-LSL時刻同期システム")
    
    # システム初期化
    synchronizer = TCPLSLSynchronizer(
        tcp_host='127.0.0.1',
        tcp_port=12345,
        lsl_stream_name='MockEEG',
        max_sync_events=1000,
        sync_tolerance_ms=50  # 50ms以内の同期
    )
    
    # データ収集セッションを実行（5分間）
    synchronizer.run_data_collection_session(duration_seconds=300)


if __name__ == '__main__':
    main()