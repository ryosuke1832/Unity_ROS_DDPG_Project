
from collections import deque
import socket
import threading
import json
import time




# Unity TCP通信インターフェース（既存と同じ）
class UnityTCPInterface:
    """Unity との TCP 通信インターフェース"""
    
    def __init__(self, host='localhost', port=12345):
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.is_connected = False
        self.is_running = False
        self.received_data = deque(maxlen=100)
        
    def start_server(self):
        """サーバーを開始"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.is_running = True
            
            print(f"🟢 Unity TCP サーバー開始: {self.host}:{self.port}")
            
            # 接続待機スレッド
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
        except Exception as e:
            print(f"❌ サーバー開始エラー: {e}")
    
    def _accept_connections(self):
        """クライアント接続を受け入れる"""
        while self.is_running:
            try:
                self.client_socket, client_address = self.server_socket.accept()
                self.is_connected = True
                print(f"✅ Unity クライアント接続: {client_address}")
                
                # 受信スレッド開始
                receive_thread = threading.Thread(target=self._receive_loop)
                receive_thread.daemon = True
                receive_thread.start()
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 接続受け入れエラー: {e}")
                break
    
    def _receive_loop(self):
        """データ受信ループ"""
        buffer = ""
        while self.is_running and self.is_connected:
            try:
                data = self.client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    if line.strip():
                        message = line.strip()
                        try:
                            parsed_data = json.loads(message)
                            self.received_data.append(parsed_data)
                        except json.JSONDecodeError:
                            text_message = {
                                'type': message.lower(),
                                'raw_message': message,
                                'timestamp': time.time()
                            }
                            self.received_data.append(text_message)
                            print(f"📥 テキスト受信: {message}")
                        
            except Exception as e:
                print(f"❌ データ受信エラー: {e}")
                break
        
        self._disconnect()
    
    def send_message(self, message_dict):
        """メッセージを送信"""
        if not self.is_connected or not self.client_socket:
            return False
        
        try:
            json_message = json.dumps(message_dict) + '\n'
            self.client_socket.send(json_message.encode('utf-8'))
            return True
        except Exception as e:
            print(f"❌ メッセージ送信エラー: {e}")
            self._disconnect()
            return False
    
    def _disconnect(self):
        """接続を切断"""
        self.is_connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        print("🔌 Unity クライアント切断")
    
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
