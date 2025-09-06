#!/usr/bin/env python3
"""
EEG系システム用統合TCP通信モジュール
lsl_classification.pyとeeg_ddpg_rl_system.pyで使用するTCP通信を統一

機能:
- Unity との双方向TCP通信
- JSON メッセージ送受信
- コールバック機能（メッセージ受信時の処理）
- EEG分類器・強化学習システム向けの特化機能
"""

from collections import deque
import socket
import threading
import json
import time
import random

class EEGTCPInterface:
    """
    EEG系システム用統合TCP通信インターフェース
    lsl_classification.py と eeg_ddpg_rl_system.py の共通TCP機能を提供
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
        
        # 統計情報
        self.stats = {
            'messages_received': 0,
            'messages_sent': 0,
            'connection_count': 0,
            'last_activity': None,
            'start_time': None
        }
        
        # スレッド管理
        self.threads = []
        
        print(f"🔌 EEG TCP インターフェース初期化: {host}:{port}")
    
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
        print(f"👂 接続待機中...")
        
        while self.is_running:
            try:
                self.client_socket, self.client_address = self.server_socket.accept()
                self.is_connected = True
                self.stats['connection_count'] += 1
                self.stats['last_activity'] = time.time()
                
                print(f"✅ クライアント接続: {self.client_address}")
                
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
                    print(f"📡 クライアントが接続を閉じました")
                    break
                
                # デコードしてバッファに追加
                received_str = data.decode('utf-8')
                data_buffer += received_str
                
                # 行単位またはJSON単位で分割処理
                while '\n' in data_buffer or self._has_complete_json(data_buffer):
                    if '\n' in data_buffer:
                        line, data_buffer = data_buffer.split('\n', 1)
                    else:
                        # JSON終端で分割
                        line, data_buffer = self._extract_json(data_buffer)
                    
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
    
    def send_message(self, message_dict):
        """メッセージを送信"""
        if not self.is_connected or not self.client_socket:
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
            self._disconnect()
            return False

    def handle_grip_force_requests(self, min_force=0.1, max_force=30.0):
        """Unityからの把持力リクエストに応答しランダムな把持力を送信"""
        while self.is_running:
            if self.received_data:
                data = self.received_data.popleft()
                if isinstance(data, dict) and data.get('type') == 'request_grip_force':
                    grip_force = random.uniform(min_force, max_force)
                    response = {
                        'type': 'grip_force_command',
                        'target_force': grip_force
                    }
                    if self.send_message(response):
                        print(f"📤 把持力送信: {grip_force:.2f}N")
            time.sleep(0.01)

    def _disconnect(self):
        """接続を切断"""
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

    def stop_server(self):
        """サーバーを停止"""
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
        print("🛑 Unity TCP サーバー停止")


if __name__ == '__main__':
    interface = UnityTCPInterface()
    interface.start_server()
    try:
        interface.handle_grip_force_requests()
    except KeyboardInterrupt:
        print('🛑 デモ停止')
    finally:
        interface.stop_server()

