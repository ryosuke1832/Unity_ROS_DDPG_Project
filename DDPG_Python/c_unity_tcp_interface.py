#!/usr/bin/env python3
"""
EEG系システム用統合TCP通信モジュール（包括的修正版）

修正点：
1. タイプ名正規化の統一（grip_force_request）
2. 詳細ログ追加とエラーハンドリング強化
3. 健全性チェック機能追加
4. 状態データ橋渡し機能強化
5. Unity側との互換性向上
"""

from collections import deque
import socket
import threading
import json
import time
from typing import Callable, Dict, Any, Optional, List
import random

class EEGTCPInterface:
    """
    EEG系システム用統合TCP通信インターフェース（包括的修正版）
    """
    
    def __init__(self, host='127.0.0.1', port=12345, max_buffer_size=1000, auto_reply=False):
        self.host = host
        self.port = port
        self.max_buffer_size = max_buffer_size
        self.auto_reply = auto_reply
        
        # サーバー管理
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.is_connected = False
        self.is_running = False
        
        # データバッファ
        self.received_data = deque(maxlen=max_buffer_size)
        self.sent_data = deque(maxlen=max_buffer_size)
        
        # 直近のロボット状態を保持
        self.last_robot_state = None
        self.last_robot_state_timestamp = None
        
        # コールバック関数
        self.message_callbacks = []
        self.connection_callbacks = []
        self.disconnect_callbacks = []
        self.state_update_callbacks = []  # 新追加：状態更新時のコールバック
        
        # 把持力設定
        self.min_grip_force = 2.0
        self.max_grip_force = 30.0
        self.default_grip_force = 10.0
        
        # 統計情報（詳細化）
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'connection_count': 0,
            'grip_force_requests': 0,
            'grip_force_responses': 0,
            'auto_responses': 0,
            'text_normalizations': 0,
            'json_requests': 0,  # 新追加
            'state_updates': 0,  # 新追加
            'callback_errors': 0,  # 新追加
            'send_errors': 0,  # 新追加
            'last_activity': None,
            'start_time': None
        }
        
        # スレッド管理
        self.threads = []
        
        # デバッグ設定
        self.debug_mode = False
        self.log_all_messages = False
        
        print(f"🔌 EEG TCP インターフェース初期化（包括的修正版）: {host}:{port}")
        print(f"   把持力範囲: {self.min_grip_force:.1f} - {self.max_grip_force:.1f} N")
        print(f"   自動応答: {'有効' if auto_reply else '無効'}")
        print(f"   タイプ正規化: REQUEST_GRIP_FORCE → grip_force_request (統一)")
    
    def enable_debug_mode(self, enable_all_logs=False):
        """デバッグモード有効化"""
        self.debug_mode = True
        self.log_all_messages = enable_all_logs
        print(f"🐛 デバッグモード有効化: 全メッセージログ={'有効' if enable_all_logs else '無効'}")
    
    def _debug_log(self, message: str):
        """デバッグログ出力"""
        if self.debug_mode:
            print(f"[DEBUG] {message}")
    
    def _is_robot_state_data(self, data: Dict[str, Any]) -> bool:
        """ロボット状態データかを判定（拡張版）"""
        if not isinstance(data, dict):
            return False
        
        # 複数のパターンに対応
        required_patterns = [
            ['episode', 'position', 'velocity', 'grip_force'],  # 標準パターン
            ['episode', 'grip_force', 'contact'],  # 簡易パターン
            ['robot_episode', 'force', 'position']  # 代替パターン
        ]
        
        for pattern in required_patterns:
            if all(key in data for key in pattern):
                return True
        
        # 部分的なマッチも許容
        essential_keys = ['episode', 'grip_force']
        if all(key in data for key in essential_keys):
            self._debug_log(f"ロボット状態データ（部分マッチ）: {list(data.keys())}")
            return True
        
        return False
    
    def _update_robot_state(self, data: Dict[str, Any]):
        """ロボット状態の更新と通知"""
        try:
            self.last_robot_state = data.copy()
            self.last_robot_state_timestamp = time.time()
            self.stats['state_updates'] += 1
            
            # 状態更新コールバック実行
            for callback in self.state_update_callbacks:
                try:
                    callback(data)
                except Exception as e:
                    print(f"⚠️ 状態更新コールバックエラー: {e}")
                    self.stats['callback_errors'] += 1
            
            if self.debug_mode:
                episode = data.get('episode', 'unknown')
                grip_force = data.get('grip_force', 'unknown')
                print(f"🔄 ロボット状態更新: ep={episode}, force={grip_force}")
                
        except Exception as e:
            print(f"❌ ロボット状態更新エラー: {e}")
    
    def set_grip_force_range(self, min_force: float, max_force: float):
        """把持力の範囲を設定"""
        self.min_grip_force = min_force
        self.max_grip_force = max_force
        print(f"🎛️ 把持力範囲更新: {min_force:.1f} - {max_force:.1f} N")
    
    def add_message_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """メッセージ受信時のコールバック関数を追加"""
        self.message_callbacks.append(callback)
        print(f"📞 メッセージコールバック追加: {callback.__name__}")
    
    def add_state_update_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """状態更新時のコールバック関数を追加（新機能）"""
        self.state_update_callbacks.append(callback)
        print(f"📊 状態更新コールバック追加: {callback.__name__}")
    
    def add_connection_callback(self, callback: Callable[[str, int], None]):
        """接続時のコールバック関数を追加"""
        self.connection_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable[[], None]):
        """切断時のコールバック関数を追加"""
        self.disconnect_callbacks.append(callback)
    
    def start_server(self) -> bool:
        """TCPサーバーを開始"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            print(f"🟢 EEG TCP サーバー開始: {self.host}:{self.port}")
            print(f"   デバッグモード: {'有効' if self.debug_mode else '無効'}")
            
            # 接続待機スレッド開始
            accept_thread = threading.Thread(target=self._accept_connections, daemon=True)
            accept_thread.start()
            self.threads.append(accept_thread)
            
            return True
            
        except Exception as e:
            print(f"❌ サーバー開始エラー: {e}")
            return False
    
    def _accept_connections(self):
        """クライアント接続を受け入れる"""
        print(f"👂 Unity接続待機中...")
        
        while self.is_running:
            try:
                self.client_socket, self.client_address = self.server_socket.accept()
                self.is_connected = True
                self.stats['connection_count'] += 1
                self.stats['last_activity'] = time.time()
                
                print(f"✅ Unity クライアント接続: {self.client_address}")
                
                # 接続コールバック実行
                for callback in self.connection_callbacks:
                    try:
                        callback(self.client_address[0], self.client_address[1])
                    except Exception as e:
                        print(f"⚠️ 接続コールバックエラー: {e}")
                        self.stats['callback_errors'] += 1
                
                # 受信スレッド開始
                receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
                receive_thread.start()
                self.threads.append(receive_thread)
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 接続受け入れエラー: {e}")
                break
    
    def _receive_loop(self):
        """データ受信ループ"""
        print(f"🔄 データ受信開始: {self.client_address}")
        
        data_buffer = ""
        
        while self.is_running and self.is_connected:
            try:
                # データ受信
                data = self.client_socket.recv(1024)
                if not data:
                    print(f"📡 Unity クライアントが接続を閉じました")
                    break
                
                # デコードしてバッファに追加
                received_str = data.decode('utf-8')
                data_buffer += received_str
                
                # 行単位で分割処理
                while '\n' in data_buffer:
                    line, data_buffer = data_buffer.split('\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    # メッセージ処理
                    self._process_received_message(line)
                
            except Exception as e:
                if self.is_running and self.is_connected:
                    print(f"❌ データ受信エラー: {e}")
                break
        
        self._handle_disconnect()
    
    def _process_received_message(self, message_str: str):
        """受信メッセージの処理（包括的修正版）"""
        try:
            self.stats['messages_received'] += 1
            self.stats['last_activity'] = time.time()
            
            if self.log_all_messages:
                print(f"📥 受信: {message_str}")
            
            # JSONとして解析を試行
            try:
                message_data = json.loads(message_str)
                self._debug_log(f"JSON解析成功: {message_data}")
                
                # 受信データをバッファに追加
                self.received_data.append(message_data)
                
                # ロボット状態データの処理
                if self._is_robot_state_data(message_data):
                    self._update_robot_state(message_data)
                
                # 自動応答処理（条件チェック強化）
                if self.auto_reply:
                    self._handle_grip_force_request(message_data)
                
                # コールバック実行（常に実行）
                self._execute_message_callbacks(message_data)
                        
            except json.JSONDecodeError:
                # JSON以外のメッセージ（テキストコマンド等）
                self._debug_log(f"テキストメッセージ: {message_str}")
                
                message_upper = message_str.strip().upper()
                
                # ★ 修正1: REQUEST_GRIP_FORCE の正規化統一
                if message_upper == "REQUEST_GRIP_FORCE":
                    self.stats['text_normalizations'] += 1
                    
                    normalized = {
                        'type': 'grip_force_request',  # ★ 統一: grip_force_request
                        'timestamp': time.time(),
                        'source': 'text',
                        'original_text': message_str
                    }
                    
                    # コンテキスト情報追加
                    if self.last_robot_state:
                        normalized['episode'] = self.last_robot_state.get('episode')
                        normalized['context'] = self.last_robot_state.copy()
                        normalized['context_age_ms'] = (time.time() - self.last_robot_state_timestamp) * 1000
                    
                    print(f"🧩 正規化: REQUEST_GRIP_FORCE → grip_force_request (episode={normalized.get('episode')})")
                    
                    # バッファに追加
                    self.received_data.append(normalized)
                    
                    # コールバック実行
                    self._execute_message_callbacks(normalized)
                    
                    return
                
                # 自動応答処理（テキストコマンド）
                auto_handled = False
                if self.auto_reply:
                    auto_handled = self._handle_unity_text_commands(message_str)
                
                # テキストメッセージもバッファに追加
                text_data = {
                    'type': 'text_message',
                    'content': message_str,
                    'timestamp': time.time()
                }
                self.received_data.append(text_data)
                
                # コールバック実行（自動処理されなかった場合、または自動応答無効時）
                if not auto_handled or not self.auto_reply:
                    self._execute_message_callbacks(text_data)
                
        except Exception as e:
            print(f"❌ メッセージ処理エラー: {e}")
            self.stats['callback_errors'] += 1
    
    def _execute_message_callbacks(self, message_data: Dict[str, Any]):
        """メッセージコールバックの実行（エラーハンドリング強化）"""
        for callback in self.message_callbacks:
            try:
                callback(message_data)
            except Exception as e:
                print(f"⚠️ メッセージコールバックエラー [{callback.__name__}]: {e}")
                self.stats['callback_errors'] += 1
    
    def _handle_grip_force_request(self, message_data: Dict[str, Any]):
        """把持力リクエストの処理（自動応答制御付き）"""
        if not self.auto_reply:
            return
        
        message_type = message_data.get('type', '').lower()
        
        # ★ 修正2: 複数パターンの把持力リクエスト検出
        grip_request_types = [
            'grip_force_request',
            'request_grip_force', 
            'grip_request',
            'force_request'
        ]
        
        if (message_type in grip_request_types or
            'grip' in message_type or 'force' in message_type):
            
            self.stats['grip_force_requests'] += 1
            self.stats['auto_responses'] += 1
            self.stats['json_requests'] += 1
            
            print(f"🎯 自動把持力リクエスト検出: {message_data}")
            
            # 把持力を生成
            grip_force = self._generate_grip_force(message_data)
            
            # 応答送信
            success = self._send_grip_force_response(grip_force, message_data)
            
            if success:
                self.stats['grip_force_responses'] += 1
                print(f"✅ 自動把持力応答送信成功: {grip_force:.2f}N")
            else:
                print(f"❌ 自動把持力応答送信失敗")
                self.stats['send_errors'] += 1
    
    def _handle_unity_text_commands(self, message_str: str) -> bool:
        """Unity側のテキストコマンドへの対応（修正版）"""
        if not self.auto_reply:
            return False
        
        message_upper = message_str.strip().upper()
        
        # ★ 修正3: 把持力リクエスト関連のテキストコマンド（完全一致）
        if message_upper == "REQUEST_GRIP_FORCE":
            print(f"🎯 自動テキスト把持力リクエスト検出: {message_str}")
            
            # コンテキスト付きで把持力生成
            context = {'type': 'text_request', 'source': 'text_command'}
            if self.last_robot_state:
                context.update(self.last_robot_state)
            
            grip_force = self._generate_grip_force(context)
            success = self._send_grip_force_response(grip_force, context)
            
            if success:
                self.stats['grip_force_responses'] += 1
                self.stats['auto_responses'] += 1
                print(f"✅ 自動テキスト把持力応答送信: {grip_force:.2f}N")
            else:
                self.stats['send_errors'] += 1
            
            return True
        
        # 接続確認メッセージ
        if message_upper in ['PING', 'CONNECT', 'HELLO', 'TEST']:
            print(f"🔔 接続確認メッセージ: {message_str}")
            
            response = {
                'type': 'pong',
                'message': 'EEG TCP Interface Ready (Fixed Version)',
                'timestamp': time.time(),
                'stats': self.get_stats_summary()
            }
            self.send_message(response)
            return True
        
        return False
    
    def _generate_grip_force(self, message_data: Dict[str, Any]) -> float:
        """把持力を生成（改良版）"""
        # メッセージに特定の要求があるかチェック
        requested_force = message_data.get('requested_force') or message_data.get('target_force')
        if requested_force is not None:
            try:
                force = float(requested_force)
                return max(self.min_grip_force, min(self.max_grip_force, force))
            except ValueError:
                pass
        
        # エピソード情報に基づく生成
        episode = message_data.get('episode', 0)
        if episode and episode > 0:
            # エピソード番号に基づいてある程度の規則性
            base_force = self.min_grip_force + (episode % 10) * (self.max_grip_force - self.min_grip_force) / 10
            noise = random.uniform(-2.0, 2.0)
            grip_force = base_force + noise
        else:
            # 適度なランダム（中央値周辺に偏重）
            center = (self.min_grip_force + self.max_grip_force) / 2
            range_half = (self.max_grip_force - self.min_grip_force) / 4
            grip_force = random.gauss(center, range_half)
        
        # 範囲内にクランプ
        return max(self.min_grip_force, min(self.max_grip_force, grip_force))
    
    def _send_grip_force_response(self, grip_force: float, original_message: Dict[str, Any]) -> bool:
        """把持力応答を送信（Unity互換性向上）"""
        
        # ★ 修正4: Unity側との互換性を考慮した複数形式
        response = {
            'type': 'grip_force_command',
            'target_force': round(grip_force, 2),      # Python標準
            'targetForce': round(grip_force, 2),       # Unity C# キャメルケース
            'force': round(grip_force, 2),             # 簡易形式
            'timestamp': time.time(),
            'session_id': f"eeg_tcp_auto_{int(time.time())}"
        }
        
        # 元メッセージの情報を引き継ぎ
        if 'episode' in original_message:
            response['episode_number'] = original_message['episode']
            response['episodeNumber'] = original_message['episode']  # キャメルケース
        if 'request_id' in original_message:
            response['request_id'] = original_message['request_id']
            response['requestId'] = original_message['request_id']   # キャメルケース
        
        # 送信元情報
        response['source'] = 'auto_reply' if self.auto_reply else 'manual'
        response['port'] = self.port
        
        return self.send_message(response)
    
    def send_message(self, message_data: Dict[str, Any]) -> bool:
        """メッセージを送信（エラーハンドリング強化）"""
        if not self.is_connected or not self.client_socket:
            self._debug_log("送信失敗: 接続なし")
            self.stats['send_errors'] += 1
            return False
        
        try:
            # タイムスタンプ追加
            if 'timestamp' not in message_data:
                message_data['timestamp'] = time.time()
            
            # JSON形式で送信
            json_message = json.dumps(message_data, ensure_ascii=False) + '\n'
            self.client_socket.send(json_message.encode('utf-8'))
            
            # 送信履歴に追加
            self.sent_data.append(message_data.copy())
            self.stats['messages_sent'] += 1
            self.stats['last_activity'] = time.time()
            
            if self.debug_mode or message_data.get('type') == 'grip_force_command':
                msg_type = message_data.get('type', 'unknown')
                target_force = message_data.get('target_force', 'N/A')
                print(f"📤 送信: {msg_type} - force={target_force}")
            
            return True
            
        except Exception as e:
            print(f"❌ メッセージ送信エラー: {e}")
            self.stats['send_errors'] += 1
            self._handle_disconnect()
            return False
    
    def get_stats_summary(self) -> Dict[str, Any]:
        """統計情報のサマリーを取得"""
        return {
            'messages_received': self.stats['messages_received'],
            'messages_sent': self.stats['messages_sent'],
            'grip_force_requests': self.stats['grip_force_requests'],
            'grip_force_responses': self.stats['grip_force_responses'],
            'state_updates': self.stats['state_updates'],
            'errors': self.stats['callback_errors'] + self.stats['send_errors'],
            'uptime_seconds': time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        }
    
    def get_last_robot_state(self) -> Optional[Dict[str, Any]]:
        """最新のロボット状態を取得"""
        if self.last_robot_state and self.last_robot_state_timestamp:
            age_ms = (time.time() - self.last_robot_state_timestamp) * 1000
            return {
                'data': self.last_robot_state.copy(),
                'timestamp': self.last_robot_state_timestamp,
                'age_ms': age_ms
            }
        return None
    
    def _handle_disconnect(self):
        """接続切断の処理"""
        print(f"🔌 接続切断処理開始")
        self.is_connected = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        
        # 切断コールバック実行
        for callback in self.disconnect_callbacks:
            try:
                callback()
            except Exception as e:
                print(f"⚠️ 切断コールバックエラー: {e}")
                self.stats['callback_errors'] += 1
        
        print(f"📡 Unity クライアント切断完了")
    
    def stop_server(self):
        """サーバーを停止"""
        print(f"🛑 EEG TCP サーバー停止中...")
        
        self.is_running = False
        self.is_connected = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # 統計表示
        self._print_statistics()
        
        print("🛑 EEG TCP サーバー停止完了")
    
    def _print_statistics(self):
        """統計情報の表示（詳細化）"""
        print(f"\n📊 EEG TCP インターフェース統計（包括的修正版）:")
        print(f"   接続回数               : {self.stats['connection_count']}")
        print(f"   受信メッセージ数       : {self.stats['messages_received']}")
        print(f"   送信メッセージ数       : {self.stats['messages_sent']}")
        print(f"   把持力リクエスト数     : {self.stats['grip_force_requests']}")
        print(f"   把持力応答数           : {self.stats['grip_force_responses']}")
        print(f"   JSONリクエスト数       : {self.stats['json_requests']}")
        print(f"   自動応答数             : {self.stats['auto_responses']}")
        print(f"   テキスト正規化数       : {self.stats['text_normalizations']}")
        print(f"   ロボット状態更新数     : {self.stats['state_updates']}")
        print(f"   コールバックエラー数   : {self.stats['callback_errors']}")
        print(f"   送信エラー数           : {self.stats['send_errors']}")
        
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            print(f"   稼働時間               : {uptime:.1f}秒")
        
        # 最新状態情報
        if self.last_robot_state:
            age_ms = (time.time() - self.last_robot_state_timestamp) * 1000
            episode = self.last_robot_state.get('episode', 'unknown')
            print(f"   最新ロボット状態       : Episode {episode} ({age_ms:.1f}ms前)")
    
    def run_demo(self):
        """デモ実行（包括的修正版）"""
        print(f"🚀 EEG TCP インターフェース デモ開始（包括的修正版）")
        
        # デバッグモード有効化
        self.enable_debug_mode(enable_all_logs=True)
        
        if not self.start_server():
            print(f"❌ サーバー開始失敗")
            return
        
        try:
            print(f"💡 Unity側での推奨操作:")
            print(f"   1. テキスト 'REQUEST_GRIP_FORCE' 送信 → grip_force_request に正規化")
            print(f"   2. JSON {{'type':'grip_force_request'}} 送信")
            print(f"   3. ロボット状態JSON送信 → 状態更新として記録")
            print(f"   自動応答: {'有効' if self.auto_reply else '無効'}")
            print(f"   Ctrl+C で終了")
            
            # メインループ
            while self.is_running:
                time.sleep(1.0)
                
                # 定期的な状態表示
                if int(time.time()) % 10 == 0:
                    if self.is_connected:
                        requests = self.stats['grip_force_requests']
                        responses = self.stats['grip_force_responses']
                        print(f"🔗 接続中: リクエスト{requests}/応答{responses} - 待機中...")
                    else:
                        print(f"⏳ Unity接続待機中...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ デモ停止")
        finally:
            self.stop_server()


# デモ実行用のコールバック関数
def on_message_received(message_data):
    """メッセージ受信時のデモ用処理"""
    msg_type = message_data.get('type', 'unknown')
    print(f"🔔 デモコールバック: {msg_type}")
    
    if msg_type == 'grip_force_request':
        print(f"   → 把持力リクエスト検出！")
        episode = message_data.get('episode', 'N/A')
        source = message_data.get('source', 'unknown')
        print(f"   → Episode: {episode}, Source: {source}")

def on_state_update(state_data):
    """状態更新時のデモ用処理"""
    episode = state_data.get('episode', 'unknown')
    grip_force = state_data.get('grip_force', 'unknown')
    print(f"📊 状態更新デモ: Episode {episode}, Force {grip_force}")

def on_client_connected(host, port):
    """クライアント接続時のデモ用処理"""
    print(f"🎉 新しいUnityクライアント: {host}:{port}")

def on_client_disconnected():
    """クライアント切断時のデモ用処理"""
    print(f"👋 Unityクライアント切断")


if __name__ == '__main__':
    print("🔧 EEG TCP インターフェース（包括的修正版）")
    print("=" * 60)
    print("修正内容:")
    print("1. タイプ正規化統一: REQUEST_GRIP_FORCE → grip_force_request")
    print("2. Unity互換性向上: target_force + targetForce 両対応")
    print("3. エラーハンドリング強化")
    print("4. 詳細ログ・統計機能追加")
    print("5. 状態橋渡し機能強化")
    print("=" * 60)
    
    print("\n自動応答モードを選択してください:")
    print("1. 自動応答有効（旧動作・テスト用）")
    print("2. 自動応答無効（学習用・推奨）")
    
    choice = input("選択 (1-2): ").strip()
    auto_reply = (choice == "1")
    
    interface = EEGTCPInterface(host='127.0.0.1', port=12345, auto_reply=auto_reply)
    
    # デモ用コールバック設定
    interface.add_message_callback(on_message_received)
    interface.add_state_update_callback(on_state_update)
    interface.add_connection_callback(on_client_connected)
    interface.add_disconnect_callback(on_client_disconnected)
    
    # 把持力範囲設定
    interface.set_grip_force_range(min_force=5.0, max_force=25.0)
    
    # デモ実行
    interface.run_demo()