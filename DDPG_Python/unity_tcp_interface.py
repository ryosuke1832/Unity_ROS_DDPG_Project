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
from typing import Callable, Dict, Any, Optional
from datetime import datetime
from random_number_generator import generate_random_integer, generate_random_grip_force

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
    
    def _has_complete_json(self, data_buffer: str) -> bool:
        """完全なJSONが含まれているかチェック"""
        brace_count = 0
        for char in data_buffer:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return True
        return False
    
    def _extract_json(self, data_buffer: str) -> tuple:
        """データバッファから完全なJSONを抽出"""
        brace_count = 0
        end_pos = -1
        
        for i, char in enumerate(data_buffer):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_pos = i + 1
                    break
        
        if end_pos > 0:
            json_str = data_buffer[:end_pos]
            remaining = data_buffer[end_pos:]
            return json_str, remaining
        else:
            return data_buffer, ""
    
    def _process_received_message(self, message: str):
        """受信メッセージの処理"""
        try:
            # JSON解析試行
            try:
                parsed_data = json.loads(message)
                message_type = 'json'
            except json.JSONDecodeError:
                # JSONでない場合はテキストメッセージとして処理
                parsed_data = {
                    'type': 'text_message',
                    'content': message,
                    'timestamp': time.time()
                }
                message_type = 'text'
            
            # タイムスタンプ追加（存在しない場合）
            if 'timestamp' not in parsed_data:
                parsed_data['timestamp'] = time.time()
            
            # バッファに追加
            self.received_data.append(parsed_data)
            self.stats['messages_received'] += 1
            self.stats['last_activity'] = time.time()
            
            print(f"📥 受信 ({message_type}): {str(parsed_data)[:1000]}...")
            
            # コールバック実行
            for callback in self.message_callbacks:
                try:
                    callback(parsed_data)
                except Exception as e:
                    print(f"⚠️ メッセージコールバックエラー: {e}")
                    
        except Exception as e:
            print(f"❌ メッセージ処理エラー: {e}")
    
    def send_message(self, message_data: Dict[str, Any]) -> bool:
        """メッセージを送信"""
        if not self.is_connected or not self.client_socket:
            print(f"⚠️ 送信失敗: 接続されていません")
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
    
    def send_text_message(self, text: str) -> bool:
        """テキストメッセージを送信"""
        message_data = {
            'type': 'text_message',
            'content': text,
            'timestamp': time.time()
        }
        return self.send_message(message_data)
    
    def send_grip_force_command(self, target_force: float, episode_id: Optional[int] = None, 
                               duration: float = 1.0, session_id: Optional[str] = None) -> bool:
        """把持力コマンド送信（eeg_ddpg_rl_system.py用）"""
        message_data = {
            'type': 'grip_force_command',
            'target_force': float(target_force),
            'duration': duration,
            'timestamp': time.time()
        }
        
        if episode_id is not None:
            message_data['episode_number'] = int(episode_id)
        
        if session_id is not None:
            message_data['session_id'] = session_id
        
        return self.send_message(message_data)
    
    def send_classification_result(self, class_name: str, class_id: int, confidence: float,
                                  trigger_data: str = "", method: str = "eeg_classifier") -> bool:
        """分類結果送信（lsl_classification.py用）"""
        message_data = {
            'type': 'classification_result',
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence,
            'trigger_data': trigger_data,
            'method': method,
            'timestamp': time.time()
        }
        return self.send_message(message_data)
    
    def send_random_grip_force(self, episode_id: Optional[int] = None, 
                              min_force: int = 2, max_force: int = 30,
                              session_id: Optional[str] = None) -> bool:
        """
        ランダムな整数把持力を生成してTCP送信
        
        Args:
            episode_id: エピソードID
            min_force: 最小把持力（整数）
            max_force: 最大把持力（整数）
            session_id: セッションID
        
        Returns:
            bool: 送信成功可否
        """
        # ランダム整数生成
        random_grip_force = generate_random_integer(min_force, max_force)
        
        print(f"🎲 ランダム把持力生成: {random_grip_force}N (範囲: {min_force}-{max_force})")
        
        # TCP送信（episode_contact_sync_system.pyと同じ形式）
        message_data = {
            'type': 'grip_force_command',
            'target_force': float(random_grip_force),  # Unityは浮動小数点数を期待
            'timestamp': time.time(),
            'generation_method': 'random_integer',
            'force_range': {'min': min_force, 'max': max_force}
        }
        
        if episode_id is not None:
            message_data['episode_number'] = int(episode_id)
        
        if session_id is not None:
            message_data['session_id'] = session_id
        
        success = self.send_message(message_data)
        
        if success:
            print(f"📤 ランダム把持力送信成功: {random_grip_force}N → Unity")
        else:
            print(f"❌ ランダム把持力送信失敗")
        
        return success
    
    def send_random_float_grip_force(self, episode_id: Optional[int] = None, 
                                    min_force: float = 2.0, max_force: float = 30.0,
                                    session_id: Optional[str] = None) -> bool:
        """
        ランダムな浮動小数点把持力を生成してTCP送信
        
        Args:
            episode_id: エピソードID
            min_force: 最小把持力（浮動小数点）
            max_force: 最大把持力（浮動小数点）
            session_id: セッションID
        
        Returns:
            bool: 送信成功可否
        """
        # ランダム浮動小数点数生成
        random_grip_force = generate_random_grip_force(min_force, max_force)
        
        print(f"🎲 ランダム把持力生成: {random_grip_force:.2f}N (範囲: {min_force:.1f}-{max_force:.1f})")
        
        # TCP送信
        message_data = {
            'type': 'grip_force_command',
            'target_force': float(random_grip_force),
            'timestamp': time.time(),
            'generation_method': 'random_float',
            'force_range': {'min': min_force, 'max': max_force}
        }
        
        if episode_id is not None:
            message_data['episode_number'] = int(episode_id)
        
        if session_id is not None:
            message_data['session_id'] = session_id
        
        success = self.send_message(message_data)
        
        if success:
            print(f"📤 ランダム把持力送信成功: {random_grip_force:.2f}N → Unity")
        else:
            print(f"❌ ランダム把持力送信失敗")
        
        return success
    
    def auto_respond_with_random_grip_force(self, enable: bool = True, 
                                           use_integer: bool = True,
                                           min_force: float = 2, max_force: float = 30):
        """
        受信メッセージに対してランダム把持力で自動応答する機能の有効/無効
        
        Args:
            enable: 自動応答の有効/無効
            use_integer: 整数使用（True）か浮動小数点使用（False）
            min_force: 最小把持力
            max_force: 最大把持力
        """
        def auto_responder(message):
            # 把持力要求またはエピソードデータを受信した場合
            if (message.get('type') in ['grip_force_request', 'episode_data'] or 
                'episode' in message or 'grip_force' in message):
                
                episode_id = None
                # エピソードID抽出
                for key in ['episode', 'episode_number', 'episode_id']:
                    if key in message:
                        episode_id = message[key]
                        break
                
                print(f"🤖 自動応答トリガー: {message.get('type', 'unknown')} メッセージ")
                
                # ランダム把持力送信
                if use_integer:
                    self.send_random_grip_force(
                        episode_id=episode_id,
                        min_force=int(min_force),
                        max_force=int(max_force)
                    )
                else:
                    self.send_random_float_grip_force(
                        episode_id=episode_id,
                        min_force=float(min_force),
                        max_force=float(max_force)
                    )
        
        if enable:
            self.add_message_callback(auto_responder)
            value_type = "整数" if use_integer else "浮動小数点"
            print(f"🤖 自動ランダム応答有効: {value_type} ({min_force}-{max_force})")
        else:
            # 既存のコールバックから削除（実装簡略化のため、ここでは警告のみ）
            print(f"⚠️ 自動応答無効化: 新しいインスタンスを作成してください")
    
    def send_periodic_random_grip_force(self, interval_seconds: float = 5.0,
                                       count: int = 10, use_integer: bool = True,
                                       min_force: float = 2, max_force: float = 30):
        """
        定期的にランダム把持力を送信（テスト用）
        
        Args:
            interval_seconds: 送信間隔（秒）
            count: 送信回数
            use_integer: 整数使用フラグ
            min_force: 最小把持力
            max_force: 最大把持力
        """
        def periodic_sender():
            print(f"🔄 定期送信開始: {count}回, {interval_seconds}秒間隔")
            
            for i in range(count):
                if not self.is_connected:
                    print(f"⚠️ 定期送信中断: 接続切断 ({i+1}/{count})")
                    break
                
                episode_id = i + 1
                
                if use_integer:
                    self.send_random_grip_force(
                        episode_id=episode_id,
                        min_force=int(min_force),
                        max_force=int(max_force),
                        session_id=f"periodic_test_{datetime.now().strftime('%H%M%S')}"
                    )
                else:
                    self.send_random_float_grip_force(
                        episode_id=episode_id,
                        min_force=float(min_force),
                        max_force=float(max_force),
                        session_id=f"periodic_test_{datetime.now().strftime('%H%M%S')}"
                    )
                
                if i < count - 1:  # 最後以外は待機
                    time.sleep(interval_seconds)
            
            print(f"✅ 定期送信完了: {count}回送信")
        
        # 別スレッドで実行
        sender_thread = threading.Thread(target=periodic_sender, daemon=True)
        sender_thread.start()
        self.threads.append(sender_thread)
        
        return sender_thread
    
    def _handle_disconnect(self):
        """切断処理"""
        print(f"🔌 クライアント切断: {self.client_address}")
        
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
        
        # スレッド終了待機
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        self._print_final_stats()
        print(f"✅ EEG TCP サーバー停止完了")
    
    def get_latest_message(self, message_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """最新のメッセージを取得"""
        if not self.received_data:
            return None
        
        if message_type is None:
            return self.received_data[-1]
        
        # 指定された型の最新メッセージを検索
        for message in reversed(self.received_data):
            if message.get('type') == message_type:
                return message
        
        return None
    
    def get_messages_since(self, timestamp: float) -> list:
        """指定時刻以降のメッセージを取得"""
        messages = []
        for message in self.received_data:
            if message.get('timestamp', 0) >= timestamp:
                messages.append(message)
        return messages
    
    def get_message_count(self, message_type: Optional[str] = None) -> int:
        """メッセージ数を取得"""
        if message_type is None:
            return len(self.received_data)
        
        count = 0
        for message in self.received_data:
            if message.get('type') == message_type:
                count += 1
        return count
    
    def clear_buffers(self):
        """バッファをクリア"""
        self.received_data.clear()
        self.sent_data.clear()
        print(f"🧹 バッファクリア完了")
    
    def get_stats(self) -> Dict[str, Any]:
        """統計情報を取得"""
        current_time = time.time()
        uptime = current_time - self.stats['start_time'] if self.stats['start_time'] else 0
        
        return {
            'uptime_seconds': uptime,
            'connection_count': self.stats['connection_count'],
            'messages_received': self.stats['messages_received'],
            'messages_sent': self.stats['messages_sent'],
            'is_connected': self.is_connected,
            'buffer_usage': {
                'received': len(self.received_data),
                'sent': len(self.sent_data),
                'max_size': self.max_buffer_size
            },
            'last_activity': self.stats['last_activity'],
            'current_client': self.client_address
        }
    
    def _print_final_stats(self):
        """最終統計を表示"""
        stats = self.get_stats()
        print(f"\n📊 EEG TCP インターフェース 最終統計:")
        print(f"   稼働時間: {stats['uptime_seconds']:.1f}秒")
        print(f"   接続回数: {stats['connection_count']}")
        print(f"   受信メッセージ: {stats['messages_received']}")
        print(f"   送信メッセージ: {stats['messages_sent']}")
        print(f"   バッファ使用量: 受信{stats['buffer_usage']['received']}/送信{stats['buffer_usage']['sent']}")


class EEGTCPClient:
    """
    EEG TCP クライアント（テスト・デバッグ用）
    """
    
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
        self.socket = None
        self.is_connected = False
    
    def connect(self) -> bool:
        """サーバーに接続"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.is_connected = True
            print(f"✅ サーバーに接続: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ 接続エラー: {e}")
            return False
    
    def send_json(self, data: Dict[str, Any]) -> bool:
        """JSONデータを送信"""
        if not self.is_connected:
            return False
        
        try:
            json_message = json.dumps(data) + '\n'
            self.socket.send(json_message.encode('utf-8'))
            print(f"📤 送信: {data}")
            return True
        except Exception as e:
            print(f"❌ 送信エラー: {e}")
            return False
    
    def send_test_grip_data(self, grip_force: float, episode_id: int = 1):
        """テスト用把持データ送信"""
        test_data = {
            'episode': episode_id,
            'grip_force': grip_force,
            'timestamp': time.time(),
            'test_mode': True
        }
        return self.send_json(test_data)
    
    def send_test_trigger(self, trigger_type: str = "contact"):
        """テスト用トリガー送信"""
        trigger_data = {
            'type': 'trigger',
            'trigger_type': trigger_type,
            'timestamp': time.time()
        }
        return self.send_json(trigger_data)
    
    def disconnect(self):
        """接続を切断"""
        if self.socket:
            self.socket.close()
            self.is_connected = False
            print(f"🔌 接続切断")


def demo_random_grip_force_system():
    """ランダム把持力送信システムのデモ（元のコードベース対応）"""
    print("🎲 ランダム把持力送信システム デモ")
    print("=" * 50)
    
    # TCP インターフェース作成
    tcp_interface = EEGTCPInterface(host='127.0.0.1', port=12345)
    
    # 接続時の自動応答設定
    def on_connection(host, port):
        print(f"🎉 新規接続: {host}:{port}")
        tcp_interface.send_text_message("Random Grip Force System Ready!")
        
        # 接続直後にサンプルランダム値を送信
        tcp_interface.send_random_grip_force(episode_id=0, min_force=2, max_force=30)
    
    # メッセージ受信時の自動応答
    def on_message(message):
        print(f"📨 受信: {message.get('type', 'unknown')}")
        
        # 元のコード要求に基づく整数ランダム生成＋送信
        if any(key in message for key in ['episode', 'grip_force', 'contact']):
            episode_id = message.get('episode', message.get('episode_number', 1))
            
            # generate_random_integer() を使用してランダム整数生成
            random_value = generate_random_integer(2, 30)
            print(f"🎲 ランダムな値: {random_value}")
            
            # TCP で送り返す
            tcp_interface.send_random_grip_force(
                episode_id=episode_id,
                min_force=2,
                max_force=30,
                session_id="demo_random_session"
            )
    
    def on_disconnect():
        print(f"👋 クライアント切断")
    
    # コールバック登録
    tcp_interface.add_connection_callback(on_connection)
    tcp_interface.add_message_callback(on_message)
    tcp_interface.add_disconnect_callback(on_disconnect)
    
    # サーバー開始
    if tcp_interface.start_server():
        print(f"💡 使用例:")
        print(f"   1. 別ターミナル: demo_random_client()")
        print(f"   2. 自動応答モード: auto_random_demo()")
        print(f"   3. 定期送信テスト: periodic_random_demo()")
        print(f"   4. Ctrl+C で終了")
        
        try:
            while True:
                time.sleep(1)
                
                # 10秒ごとに統計表示
                if int(time.time()) % 10 == 0:
                    stats = tcp_interface.get_stats()
                    if stats['messages_received'] > 0 or stats['messages_sent'] > 0:
                        print(f"📊 統計: 受信{stats['messages_received']}, 送信{stats['messages_sent']}")
                
        except KeyboardInterrupt:
            print(f"\n⏹️ 停止要求")
        finally:
            tcp_interface.stop_server()
    
    return tcp_interface


def demo_random_client():
    """ランダム把持力システム用テストクライアント"""
    print("📱 ランダム把持力テストクライアント")
    
    client = EEGTCPClient(host='127.0.0.1', port=12345)
    
    if client.connect():
        # 元のコードで期待されるメッセージ形式でテスト
        test_episodes = [
            {'episode': 1, 'grip_force': 10.5, 'contact': True, 'timestamp': time.time()},
            {'episode': 2, 'grip_force': 15.2, 'contact': True, 'timestamp': time.time()},
            {'episode': 3, 'grip_force': 8.7, 'contact': True, 'timestamp': time.time()},
            {'type': 'grip_force_request', 'episode_number': 4},
            {'type': 'episode_data', 'episode': 5, 'data': 'test_data'}
        ]
        
        print(f"🧪 {len(test_episodes)}個のテストメッセージを送信...")
        
        for i, episode_data in enumerate(test_episodes):
            print(f"\n📤 送信 {i+1}/{len(test_episodes)}: {episode_data}")
            client.send_json(episode_data)
            
            print(f"⏳ 応答待機中... (3秒)")
            time.sleep(3)
        
        print(f"✅ 全テスト完了")
        client.disconnect()
    else:
        print("❌ サーバーに接続できませんでした")


def auto_random_demo():
    """自動ランダム応答デモ"""
    print("🤖 自動ランダム応答デモ")
    
    tcp_interface = EEGTCPInterface(host='127.0.0.1', port=12346)  # 別ポート
    
    # 自動応答機能を有効化（整数ランダム値使用）
    tcp_interface.auto_respond_with_random_grip_force(
        enable=True,
        use_integer=True,  # 元のコード要求：整数
        min_force=2,
        max_force=30
    )
    
    tcp_interface.start_server()
    
    print(f"🤖 自動応答サーバー稼働中（ポート12346）")
    print(f"   任意のメッセージ → 自動でランダム整数把持力を送信")
    
    try:
        time.sleep(60)  # 1分間稼働
    except KeyboardInterrupt:
        pass
    finally:
        tcp_interface.stop_server()


def periodic_random_demo():
    """定期的ランダム送信デモ"""
    print("🔄 定期的ランダム送信デモ")
    
    tcp_interface = EEGTCPInterface(host='127.0.0.1', port=12347)  # 別ポート
    
    if tcp_interface.start_server():
        print(f"📡 定期送信サーバー開始（ポート12347）")
        
        # 接続待機
        print(f"⏳ クライアント接続待機中...")
        while not tcp_interface.is_connected:
            time.sleep(0.5)
        
        print(f"✅ クライアント接続確認、定期送信開始")
        
        # 定期的にランダム整数把持力を送信
        tcp_interface.send_periodic_random_grip_force(
            interval_seconds=2.0,  # 2秒間隔
            count=15,              # 15回送信
            use_integer=True,      # 整数使用
            min_force=2,
            max_force=30
        )
        
        # 送信完了まで待機
        time.sleep(35)  # 15回 × 2秒 + α
        
        tcp_interface.stop_server()


# メイン実行部分の更新
if __name__ == "__main__":
    import sys
    
    print("🎲 EEG TCP通信 + ランダム把持力システム")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'client':
            demo_random_client()
        elif sys.argv[1] == 'auto':
            auto_random_demo()
        elif sys.argv[1] == 'periodic':
            periodic_random_demo()
        else:
            print("❓ 使用方法:")
            print("   python eeg_tcp_module.py         # メインサーバー")
            print("   python eeg_tcp_module.py client  # テストクライアント")
            print("   python eeg_tcp_module.py auto    # 自動応答デモ")
            print("   python eeg_tcp_module.py periodic # 定期送信デモ")
    else:
        # デフォルト: ランダム把持力システム
        demo_random_grip_force_system()