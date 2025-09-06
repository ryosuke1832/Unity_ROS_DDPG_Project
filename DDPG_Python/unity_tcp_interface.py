#!/usr/bin/env python3
"""
EEG系システム用統合TCP通信モジュール
lsl_classification.pyとeeg_ddpg_rl_system.pyで使用するTCP通信を統一

機能:
- Unity との双方向TCP通信
- JSON メッセージ送受信
- コールバック機能（メッセージ受信時の処理）
- EEG分類器・強化学習システム向けの特化機能
- a2cClient.SendGripForceRequest()への自動応答
"""

from collections import deque
import socket
import threading
import json
import time
from typing import Callable, Dict, Any
import random

class EEGTCPInterface:
    """
    EEG系システム用統合TCP通信インターフェース
    Unity の a2cClient.SendGripForceRequest() に自動応答
    """
    
    def __init__(self, host='127.0.0.1', port=12345, max_buffer_size=1000):
        self.host = host
        self.port = port
        self.max_buffer_size = max_buffer_size
        
        # サーバー管理
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.is_connected = False
        self.is_running = False
        
        # データバッファ
        self.received_data = deque(maxlen=max_buffer_size)
        self.sent_data = deque(maxlen=max_buffer_size)  # 送信履歴
        
        # コールバック関数（受信時の処理）
        self.message_callbacks = []
        self.connection_callbacks = []
        self.disconnect_callbacks = []
        
        # 把持力設定
        self.min_grip_force = 2.0   # 最小把持力 (N)
        self.max_grip_force = 30.0  # 最大把持力 (N)
        self.default_grip_force = 10.0  # デフォルト把持力 (N)
        
        # 統計情報
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'connection_count': 0,
            'grip_force_requests': 0,
            'grip_force_responses': 0,
            'last_activity': None,
            'start_time': None
        }
        
        # スレッド管理
        self.threads = []
        
        print(f"🔌 EEG TCP インターフェース初期化: {host}:{port}")
        print(f"   把持力範囲: {self.min_grip_force:.1f} - {self.max_grip_force:.1f} N")
    
    def set_grip_force_range(self, min_force: float, max_force: float):
        """把持力の範囲を設定"""
        self.min_grip_force = min_force
        self.max_grip_force = max_force
        print(f"🎛️ 把持力範囲更新: {min_force:.1f} - {max_force:.1f} N")
    
    def add_message_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """メッセージ受信時のコールバック関数を追加"""
        self.message_callbacks.append(callback)
        print(f"📞 メッセージコールバック追加: {callback.__name__}")
    
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
        """受信メッセージの処理"""
        try:
            self.stats['messages_received'] += 1
            self.stats['last_activity'] = time.time()
            
            print(f"📥 受信: {message_str}")
            
            # JSONとして解析を試行
            try:
                message_data = json.loads(message_str)
                print(f"📋 JSON解析成功: {message_data}")
                
                # 受信データをバッファに追加
                self.received_data.append(message_data)
                
                # 把持力リクエストの処理
                self._handle_grip_force_request(message_data)
                
                # コールバック実行
                for callback in self.message_callbacks:
                    try:
                        callback(message_data)
                    except Exception as e:
                        print(f"⚠️ メッセージコールバックエラー: {e}")
                        
            except json.JSONDecodeError:
                # JSON以外のメッセージ（テキストコマンド等）
                print(f"📝 テキストメッセージ: {message_str}")
                
                # Unity側の特定メッセージへの対応
                if self._handle_unity_text_commands(message_str):
                    return
                
                # テキストメッセージもバッファに追加
                text_data = {
                    'type': 'text_message',
                    'content': message_str,
                    'timestamp': time.time()
                }
                self.received_data.append(text_data)
                
        except Exception as e:
            print(f"❌ メッセージ処理エラー: {e}")
    
    def _handle_grip_force_request(self, message_data: Dict[str, Any]):
        """把持力リクエストの処理"""
        message_type = message_data.get('type', '').lower()
        
        # 把持力リクエストの検出（複数のパターンに対応）
        if (message_type in ['grip_force_request', 'request_grip_force', 'grip_request'] or
            'grip' in message_type or 'force' in message_type):
            
            self.stats['grip_force_requests'] += 1
            print(f"🎯 把持力リクエスト検出: {message_data}")
            
            # 把持力を生成（ランダムまたはロジックベース）
            grip_force = self._generate_grip_force(message_data)
            
            # 応答送信
            success = self._send_grip_force_response(grip_force, message_data)
            
            if success:
                self.stats['grip_force_responses'] += 1
                print(f"✅ 把持力応答送信成功: {grip_force:.2f}N")
            else:
                print(f"❌ 把持力応答送信失敗")
    
    def _handle_unity_text_commands(self, message_str: str) -> bool:
        """Unity側のテキストコマンドへの対応"""
        message_lower = message_str.lower()
        
        # 把持力リクエスト関連のテキストコマンド
        if any(keyword in message_lower for keyword in ['grip', 'force', 'request', 'command']):
            print(f"🎯 テキスト把持力リクエスト検出: {message_str}")
            
            # デフォルト把持力で応答
            grip_force = self._generate_grip_force({})
            success = self._send_grip_force_response(grip_force, {'type': 'text_request'})
            
            if success:
                self.stats['grip_force_responses'] += 1
                print(f"✅ テキスト把持力応答送信: {grip_force:.2f}N")
            
            return True
        
        # 接続確認メッセージ
        if any(keyword in message_lower for keyword in ['ping', 'connect', 'hello', 'test']):
            print(f"🔔 接続確認メッセージ: {message_str}")
            
            response = {
                'type': 'pong',
                'message': 'EEG TCP Interface Ready',
                'timestamp': time.time()
            }
            self.send_message(response)
            return True
        
        return False
    
    def _generate_grip_force(self, message_data: Dict[str, Any]) -> float:
        """把持力を生成（ランダムまたはロジックベース）"""
        # メッセージに特定の要求があるかチェック
        requested_force = message_data.get('requested_force')
        if requested_force is not None:
            try:
                force = float(requested_force)
                return max(self.min_grip_force, min(self.max_grip_force, force))
            except ValueError:
                pass
        
        # エピソード情報に基づく生成
        episode = message_data.get('episode', 0)
        if episode > 0:
            # エピソード番号に基づいてある程度の規則性を持たせる
            base_force = self.min_grip_force + (episode % 10) * (self.max_grip_force - self.min_grip_force) / 10
            noise = random.uniform(-2.0, 2.0)  # ±2Nのノイズ
            grip_force = base_force + noise
        else:
            # 完全ランダム
            grip_force = random.uniform(self.min_grip_force, self.max_grip_force)
        
        # 範囲内にクランプ
        return max(self.min_grip_force, min(self.max_grip_force, grip_force))
    
    def _send_grip_force_response(self, grip_force: float, original_message: Dict[str, Any]) -> bool:
        """把持力応答を送信"""
        response = {
            'type': 'grip_force_command',
            'target_force': round(grip_force, 2),
            'timestamp': time.time(),
            'session_id': f"eeg_tcp_{int(time.time())}"
        }
        
        # 元メッセージの情報を引き継ぎ
        if 'episode' in original_message:
            response['episode_number'] = original_message['episode']
        if 'request_id' in original_message:
            response['request_id'] = original_message['request_id']
        
        return self.send_message(response)
    
    def send_message(self, message_data: Dict[str, Any]) -> bool:
        """メッセージを送信"""
        if not self.is_connected or not self.client_socket:
            print(f"⚠️ 送信失敗: 接続なし")
            return False
        
        try:
            # タイムスタンプ追加
            if 'timestamp' not in message_data:
                message_data['timestamp'] = time.time()
            
            # JSON形式で送信
            json_message = json.dumps(message_data) + '\n'
            self.client_socket.send(json_message.encode('utf-8'))
            
            # 送信履歴に追加
            self.sent_data.append(message_data.copy())
            self.stats['messages_sent'] += 1
            self.stats['last_activity'] = time.time()
            
            print(f"📤 送信: {message_data.get('type', 'unknown')} - {str(message_data)[:80]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ メッセージ送信エラー: {e}")
            self._handle_disconnect()
            return False
    
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
        """統計情報の表示"""
        print(f"\n📊 EEG TCP インターフェース統計:")
        print(f"   接続回数            : {self.stats['connection_count']}")
        print(f"   受信メッセージ数    : {self.stats['messages_received']}")
        print(f"   送信メッセージ数    : {self.stats['messages_sent']}")
        print(f"   把持力リクエスト数  : {self.stats['grip_force_requests']}")
        print(f"   把持力応答数        : {self.stats['grip_force_responses']}")
        
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            print(f"   稼働時間            : {uptime:.1f}秒")
    
    def run_demo(self):
        """デモ実行（テスト用）"""
        print(f"🚀 EEG TCP インターフェース デモ開始")
        
        if not self.start_server():
            print(f"❌ サーバー開始失敗")
            return
        
        try:
            print(f"💡 Unity側で a2cClient.SendGripForceRequest() を実行してください")
            print(f"   自動で把持力応答が送信されます")
            print(f"   Ctrl+C で終了")
            
            # メインループ
            while self.is_running:
                time.sleep(1.0)
                
                # 定期的な状態表示
                if int(time.time()) % 10 == 0:  # 10秒ごと
                    if self.is_connected:
                        print(f"🔗 接続中: {self.client_address} - リクエスト待機中...")
                    else:
                        print(f"⏳ Unity接続待機中...")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ デモ停止")
        finally:
            self.stop_server()


# カスタムコールバック関数の例
def on_message_received(message_data):
    """メッセージ受信時のカスタム処理"""
    print(f"🔔 カスタムコールバック: {message_data.get('type', 'unknown')}")

def on_client_connected(host, port):
    """クライアント接続時のカスタム処理"""
    print(f"🎉 新しいUnityクライアント: {host}:{port}")

def on_client_disconnected():
    """クライアント切断時のカスタム処理"""
    print(f"👋 Unityクライアント切断")


if __name__ == '__main__':
    # EEG TCP インターフェース作成
    interface = EEGTCPInterface(host='127.0.0.1', port=12345)
    
    # カスタムコールバック設定（オプション）
    interface.add_message_callback(on_message_received)
    interface.add_connection_callback(on_client_connected)
    interface.add_disconnect_callback(on_client_disconnected)
    
    # 把持力範囲設定（オプション）
    interface.set_grip_force_range(min_force=5.0, max_force=25.0)
    
    # デモ実行
    interface.run_demo()