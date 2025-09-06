#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCP通信でA2C演算結果をUnityに送信する改良版スクリプト
コマンドライン入力を削除し、値の代入による自動送信機能を追加
"""

import socket
import threading
import json
import time
import numpy as np
from datetime import datetime

class A2CTCPCommunicator:
    def __init__(self, host='localhost', port=12345):
        """
        A2C TCP通信クラスの初期化
        
        Args:
            host (str): サーバーのホスト名
            port (int): サーバーのポート番号
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.client_socket = None
        self.client_address = None
        self.is_running = False
        self.is_connected = False
        
        # A2C演算結果保存用変数
        self.recommended_force = 0.0
        self.calculated_reward = 0.0
        self.action_values = []
        self.state_values = []
        
        # フィードバック制御用変数
        self.auto_send_enabled = True
        self.send_interval = 1.0  # 1秒間隔で送信
        self.last_send_time = 0.0
        
        # 統計情報
        self.messages_sent = 0
        self.messages_received = 0
        
    def start_server(self):
        """
        TCPサーバーを開始
        """
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            self.is_running = True
            
            print(f"🟢 A2C TCP サーバー開始: {self.host}:{self.port}")
            print("Unity クライアントの接続を待機中...")
            
            # クライアント接続待機スレッド
            accept_thread = threading.Thread(target=self._accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            # 自動送信スレッド
            auto_send_thread = threading.Thread(target=self._auto_send_loop)
            auto_send_thread.daemon = True
            auto_send_thread.start()
            
        except Exception as e:
            print(f"❌ サーバー開始エラー: {e}")
            self.stop_server()
    
    def _accept_connections(self):
        """
        クライアント接続を受け入れる
        """
        while self.is_running:
            try:
                self.client_socket, self.client_address = self.server_socket.accept()
                self.is_connected = True
                print(f"✅ Unity クライアント接続: {self.client_address}")
                
                # メッセージ受信スレッド開始
                receive_thread = threading.Thread(target=self._receive_messages)
                receive_thread.daemon = True
                receive_thread.start()
                
            except Exception as e:
                if self.is_running:
                    print(f"❌ 接続受け入れエラー: {e}")
                break
    
    def _receive_messages(self):
        """
        Unity からのメッセージを受信
        """
        buffer = ""
        while self.is_running and self.is_connected:
            try:
                data = self.client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                lines = buffer.split('\n')
                buffer = lines[-1]  # 最後の不完全な行は保持
                
                for line in lines[:-1]:
                    if line.strip():
                        self._process_received_message(line.strip())
                        
            except Exception as e:
                print(f"❌ メッセージ受信エラー: {e}")
                break
        
        self._disconnect_client()
    
    def _process_received_message(self, message):
        """
        受信したメッセージを処理
        
        Args:
            message (str): 受信したJSONメッセージ
        """
        try:
            data = json.loads(message)
            self.messages_received += 1
            
            message_type = data.get('type', 'unknown')
            timestamp = data.get('timestamp', time.time())
            
            print(f"📥 受信メッセージ: {message_type}")
            
            if message_type == 'can_state':
                self._handle_can_state(data)
            elif message_type == 'ping':
                self._send_pong()
            elif message_type == 'episode_start':
                self._handle_episode_start(data)
            elif message_type == 'episode_end':
                self._handle_episode_end(data)
            else:
                print(f"⚠️ 未知のメッセージタイプ: {message_type}")
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON解析エラー: {e}")
            print(f"問題のメッセージ: {message}")
    
    def _handle_can_state(self, data):
        """
        缶の状態データを処理（ここでA2C演算を実行）
        
        Args:
            data (dict): 缶の状態データ
        """
        print(f"🥫 缶状態受信:")
        print(f"  - つぶれ状況: {data.get('is_crushed', False)}")
        print(f"  - 把持結果: {data.get('grasp_result', 'unknown')}")
        print(f"  - 現在の力: {data.get('current_force', 0.0):.2f}N")
        print(f"  - 累積力: {data.get('accumulated_force', 0.0):.2f}N")
        
        # A2C演算をシミュレート（実際のA2Cアルゴリズムに置き換え）
        self._simulate_a2c_calculation(data)
    
    def _simulate_a2c_calculation(self, state_data):
        """
        A2C演算をシミュレート（実際のA2Cアルゴリズムに置き換えてください）
        
        Args:
            state_data (dict): 状態データ
        """
        # 状態変数の抽出
        current_force = state_data.get('current_force', 0.0)
        accumulated_force = state_data.get('accumulated_force', 0.0)
        is_crushed = state_data.get('is_crushed', False)
        
        # 簡単なA2C演算シミュレーション
        # 実際のA2Cモデルを使用する場合は、ここを置き換えてください
        
        # 状態ベクトルの作成
        state_vector = np.array([
            current_force / 100.0,  # 正規化
            accumulated_force / 1000.0,  # 正規化
            1.0 if is_crushed else 0.0
        ])
        
        # アクション値の計算（例：推奨する力の値）
        if is_crushed:
            # つぶれている場合は力を弱める
            self.recommended_force = max(0.0, current_force * 0.5)
            self.calculated_reward = -10.0  # 負の報酬
        else:
            # 正常な場合は適度な力を推奨
            target_force = 15.0  # 目標値
            self.recommended_force = min(target_force, current_force + 2.0)
            self.calculated_reward = 1.0 - abs(current_force - target_force) / target_force
        
        # ノイズを追加（探索のため）
        noise = np.random.normal(0, 0.1)
        self.recommended_force += noise
        self.recommended_force = max(0.0, min(50.0, self.recommended_force))  # クランプ
        
        print(f"🧠 A2C演算結果:")
        print(f"  - 推奨力: {self.recommended_force:.2f}N")
        print(f"  - 報酬: {self.calculated_reward:.3f}")
    
    def _handle_episode_start(self, data):
        """
        エピソード開始を処理
        """
        print("🆕 新エピソード開始")
        self.reset_statistics()
    
    def _handle_episode_end(self, data):
        """
        エピソード終了を処理
        """
        print("🏁 エピソード終了")
        self.print_statistics()
    
    def _send_pong(self):
        """
        Ping に対する Pong を送信
        """
        pong_message = {
            'type': 'pong',
            'timestamp': time.time()
        }
        self.send_message(pong_message)
    
    def _auto_send_loop(self):
        """
        自動送信ループ（定期的にA2C結果を送信）
        """
        while self.is_running:
            current_time = time.time()
            
            if (self.auto_send_enabled and 
                self.is_connected and 
                current_time - self.last_send_time >= self.send_interval):
                
                self.send_a2c_result()
                self.last_send_time = current_time
            
            time.sleep(0.1)  # CPU使用率を抑える
    
    def send_a2c_result(self):
        """
        A2C演算結果をUnityに送信
        """
        if not self.is_connected:
            return False
        
        result_message = {
            'type': 'grip_force_command',
            'target_force': float(self.recommended_force),
            'calculated_reward': float(self.calculated_reward),
            'execution_mode': 'auto',
            'duration': 1.0,
            'timestamp': time.time(),
            'message': f'A2C推奨力: {self.recommended_force:.2f}N'
        }
        
        return self.send_message(result_message)
    
    def send_message(self, message_dict):
        """
        メッセージを送信
        
        Args:
            message_dict (dict): 送信するメッセージ
            
        Returns:
            bool: 送信成功可否
        """
        if not self.is_connected or not self.client_socket:
            return False
        
        try:
            json_message = json.dumps(message_dict) + '\n'
            self.client_socket.send(json_message.encode('utf-8'))
            self.messages_sent += 1
            
            print(f"📤 送信: {message_dict.get('type', 'unknown')} - "
                  f"力: {message_dict.get('target_force', 'N/A')}")
            
            return True
            
        except Exception as e:
            print(f"❌ メッセージ送信エラー: {e}")
            self._disconnect_client()
            return False
    
    def _disconnect_client(self):
        """
        クライアント接続を切断
        """
        self.is_connected = False
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
            self.client_socket = None
        print("🔌 Unity クライアント切断")
    
    def stop_server(self):
        """
        サーバーを停止
        """
        self.is_running = False
        self.is_connected = False
        
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        
        print("🛑 A2C TCP サーバー停止")
    
    def set_a2c_values(self, recommended_force=None, reward=None):
        """
        A2C演算結果を外部から設定
        
        Args:
            recommended_force (float): 推奨力
            reward (float): 計算された報酬
        """
        if recommended_force is not None:
            self.recommended_force = float(recommended_force)
        if reward is not None:
            self.calculated_reward = float(reward)
        
        print(f"🎯 A2C値更新: 力={self.recommended_force:.2f}N, 報酬={self.calculated_reward:.3f}")
    
    def enable_auto_send(self, enabled=True, interval=1.0):
        """
        自動送信機能の有効/無効設定
        
        Args:
            enabled (bool): 自動送信の有効/無効
            interval (float): 送信間隔（秒）
        """
        self.auto_send_enabled = enabled
        self.send_interval = interval
        
        status = "有効" if enabled else "無効"
        print(f"🔄 自動送信: {status} (間隔: {interval}秒)")
    
    def print_statistics(self):
        """
        統計情報を表示
        """
        print(f"📊 通信統計:")
        print(f"  - 送信メッセージ数: {self.messages_sent}")
        print(f"  - 受信メッセージ数: {self.messages_received}")
        print(f"  - 現在の推奨力: {self.recommended_force:.2f}N")
        print(f"  - 現在の報酬: {self.calculated_reward:.3f}")
    
    def reset_statistics(self):
        """
        統計情報をリセット
        """
        self.messages_sent = 0
        self.messages_received = 0


def main():
    """
    メイン実行関数
    """
    print("🚀 A2C TCP通信システム開始")
    
    # A2C通信クラスのインスタンス作成
    a2c_comm = A2CTCPCommunicator(host='localhost', port=12345)
    
    try:
        # サーバー開始
        a2c_comm.start_server()
        
        # 自動送信を有効化（1秒間隔）
        a2c_comm.enable_auto_send(enabled=True, interval=1.0)
        
        print("\n📝 使用可能なコマンド:")
        print("  force <値>  : 推奨力を設定 (例: force 15.5)")
        print("  reward <値> : 報酬を設定 (例: reward 0.8)")
        print("  auto on/off : 自動送信の切り替え")
        print("  stats       : 統計情報表示")
        print("  quit        : 終了")
        print()
        
        # メインループ（コマンドライン入力の代わりに値設定用）
        while True:
            try:
                command = input("A2C> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                elif command.startswith('force '):
                    try:
                        value = float(command.split(' ', 1)[1])
                        a2c_comm.set_a2c_values(recommended_force=value)
                    except (ValueError, IndexError):
                        print("❌ 使用方法: force <数値>")
                elif command.startswith('reward '):
                    try:
                        value = float(command.split(' ', 1)[1])
                        a2c_comm.set_a2c_values(reward=value)
                    except (ValueError, IndexError):
                        print("❌ 使用方法: reward <数値>")
                elif command == 'auto on':
                    a2c_comm.enable_auto_send(enabled=True)
                elif command == 'auto off':
                    a2c_comm.enable_auto_send(enabled=False)
                elif command == 'stats':
                    a2c_comm.print_statistics()
                elif command == 'send':
                    a2c_comm.send_a2c_result()
                elif command == '':
                    continue
                else:
                    print("❓ 不明なコマンド。利用可能: force, reward, auto, stats, quit")
                    
            except KeyboardInterrupt:
                break
            except EOFError:
                break
    
    except Exception as e:
        print(f"❌ エラー: {e}")
    
    finally:
        a2c_comm.stop_server()
        print("👋 A2C TCP通信システム終了")


if __name__ == "__main__":
    main()