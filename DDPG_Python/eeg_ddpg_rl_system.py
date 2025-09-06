#!/usr/bin/env python3
"""
EEG強化学習システム（DDPG版）
TCP側に把持力のフィードバックを送る機能を実装

機能:
1. LSLデータとTCPデータを受け取り、時刻合わせ
2. LSLデータを3.2秒さかのぼって1.2秒間切り出し
3. eeg_classifier_function.pyでOverGrip/UnderGrip/Correct分類
4. 分類結果とTCP把持力データから報酬計算
5. DDPGで適切な把持力を学習・送信
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import socket
import time
import threading
import json
from collections import deque, Counter
from datetime import datetime
import csv
import os
import pickle
from typing import Tuple, Dict, List, Optional

# 既存システムからインポート
try:
    from pylsl import StreamInlet, resolve_streams
    print("✅ pylsl インポート成功")
except ImportError as e:
    print(f"❌ pylsl インポートエラー: {e}")
    print("pip install pylsl を実行してください")
    import sys
    sys.exit(1)

# 既存の分類器関数をインポート
try:
    from eeg_classifier_function import classify_eeg_epoch
    print("✅ EEG分類器関数インポート成功")
except ImportError as e:
    print(f"⚠️ eeg_classifier_function.pyが見つかりません: {e}")
    print("分類機能は無効化されます")
    classify_eeg_epoch = None


class DDPGAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) エージェント
    状態: [現在の把持力, 分類結果信頼度, 前回の分類結果]
    行動: [次の把持力値]
    """
    
    def __init__(self, state_dim=3, action_dim=1, hidden_dim=128, lr_actor=1e-4, lr_critic=1e-3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ネットワーク設定
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        
        # Actor-Critic ネットワーク
        self.actor = self._build_actor().to(self.device)
        self.critic = self._build_critic().to(self.device)
        self.target_actor = self._build_actor().to(self.device)
        self.target_critic = self._build_critic().to(self.device)
        
        # オプティマイザー
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # ハイパーパラメータ
        self.gamma = 0.99  # 割引率
        self.tau = 0.005   # ターゲットネットワークのソフト更新率
        self.noise_std = 0.1  # 探索ノイズ
        
        # Experience Replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # ターゲットネットワーク初期化
        self._hard_update(self.target_actor, self.actor)
        self._hard_update(self.target_critic, self.critic)
        
        print(f"🤖 DDPGエージェント初期化完了")
        print(f"   状態次元: {state_dim}, 行動次元: {action_dim}")
        print(f"   デバイス: {self.device}")
    
    def _build_actor(self):
        """Actor ネットワーク"""
        return nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.action_dim),
            nn.Sigmoid()  # 把持力は0-1範囲
        )
    
    def _build_critic(self):
        """Critic ネットワーク"""
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
    
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """行動選択（把持力の決定）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # 探索ノイズ追加
        if add_noise:
            noise = np.random.normal(0, self.noise_std * noise_scale, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, 1)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """経験の保存"""
        transition = (state, action, reward, next_state, done)
        self.memory.append(transition)
    
    def update_networks(self):
        """ネットワークの更新"""
        if len(self.memory) < self.batch_size:
            return None, None
        
        # バッチサンプリング
        batch = list(self.memory)[-self.batch_size:]  # 最新のバッチを使用
        
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Critic更新
        self.critic.train()
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(torch.cat([next_states, next_actions], dim=1))
            target_q = rewards.unsqueeze(1) + (self.gamma * target_q * (1 - dones.unsqueeze(1)))
        
        current_q = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor更新
        self.actor.train()
        actor_actions = self.actor(states)
        actor_loss = -self.critic(torch.cat([states, actor_actions], dim=1)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ターゲットネットワークのソフト更新
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, target, source):
        """ソフトターゲット更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def _hard_update(self, target, source):
        """ハードターゲット更新"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save_models(self, filepath_prefix):
        """モデルの保存"""
        torch.save(self.actor.state_dict(), f"{filepath_prefix}_actor.pth")
        torch.save(self.critic.state_dict(), f"{filepath_prefix}_critic.pth")
    
    def load_models(self, filepath_prefix):
        """モデルの読み込み"""
        try:
            self.actor.load_state_dict(torch.load(f"{filepath_prefix}_actor.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(f"{filepath_prefix}_critic.pth", map_location=self.device))
            print(f"✅ DDPGモデル読み込み成功: {filepath_prefix}")
        except Exception as e:
            print(f"⚠️ DDPGモデル読み込み失敗: {e}")


class EEGReinforcementLearningSystem:
    """
    EEG強化学習システムのメインクラス
    LSL-TCP同期、EEG分類、DDPG学習を統合
    """
    
    def __init__(self,
                 tcp_host='127.0.0.1',
                 tcp_port=12345,
                 lsl_stream_name='MockEEG',
                 sampling_rate=250,
                 epoch_duration=1.2,
                 lookback_duration=3.2,
                 eeg_model_path='./models/best_eeg_classifier.pth'):
        
        # 設定
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.lsl_stream_name = lsl_stream_name
        self.sampling_rate = sampling_rate
        self.epoch_duration = epoch_duration
        self.lookback_duration = lookback_duration
        self.eeg_model_path = eeg_model_path
        
        # データバッファ
        self.epoch_samples = int(epoch_duration * sampling_rate)  # 300サンプル
        self.lookback_samples = int(lookback_duration * sampling_rate)  # 800サンプル
        self.eeg_buffer = deque(maxlen=self.lookback_samples)
        self.buffer_lock = threading.Lock()
        
        # DDPGエージェント
        self.ddpg_agent = DDPGAgent(state_dim=3, action_dim=1)
        
        # 状態管理
        self.running = False
        self.current_state = np.zeros(3)  # [現在の把持力, 分類信頼度, 前回分類結果]
        self.previous_action = None
        self.episode_count = 0
        
        # TCP接続管理
        self.current_client_socket = None
        self._last_predicted_grip_force = 10.0  # 既定値（N）
        
        # 統計
        self.episode_rewards = deque(maxlen=100)
        self.classification_history = deque(maxlen=1000)
        self.learning_stats = {
            'total_episodes': 0,
            'total_steps': 0,
            'avg_reward': 0.0,
            'actor_losses': deque(maxlen=100),
            'critic_losses': deque(maxlen=100)
        }
        
        # ログファイル
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"eeg_rl_log_{self.session_id}.csv"
        self._init_log_file()
        
        print(f"🧠 EEG強化学習システム初期化完了")
        print(f"   TCP: {tcp_host}:{tcp_port}")
        print(f"   LSL: {lsl_stream_name}")
        print(f"   エポック: {epoch_duration}秒 ({self.epoch_samples}サンプル)")
        print(f"   ルックバック: {lookback_duration}秒")
        print(f"   セッションID: {self.session_id}")
    
    def _init_log_file(self):
        """ログファイルの初期化"""
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'episode', 'step', 'tcp_grip_force', 'predicted_grip_force',
                'eeg_classification', 'classification_confidence', 'reward',
                'current_state', 'action_taken', 'actor_loss', 'critic_loss'
            ])
    
    def setup_connections(self):
        """LSLとTCP接続の設定"""
        # LSL接続
        if not self._setup_lsl_connection():
            return False
        
        # TCP接続
        if not self._setup_tcp_connection():
            return False
        
        return True
    
    def _setup_lsl_connection(self):
        """LSL接続の設定"""
        try:
            print(f"🔍 LSLストリーム検索中: {self.lsl_stream_name}")
            streams = resolve_streams()
            
            target_stream = None
            for stream in streams:
                if stream.name() == self.lsl_stream_name:
                    target_stream = stream
                    break
            
            if target_stream is None:
                print(f"❌ LSLストリーム未発見: {self.lsl_stream_name}")
                return False
            
            self.lsl_inlet = StreamInlet(target_stream)
            print(f"✅ LSL接続成功: {self.lsl_stream_name}")
            return True
            
        except Exception as e:
            print(f"❌ LSL接続エラー: {e}")
            return False
    
    def _setup_tcp_connection(self):
        """TCP接続の設定"""
        try:
            print(f"🔌 TCP接続設定中: {self.tcp_host}:{self.tcp_port}")
            
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.tcp_socket.bind((self.tcp_host, self.tcp_port))
            self.tcp_socket.listen(1)
            
            print(f"✅ TCP待機中: {self.tcp_host}:{self.tcp_port}")
            return True
            
        except Exception as e:
            print(f"❌ TCP設定エラー: {e}")
            return False
    
    def lsl_data_thread(self):
        """LSLデータ受信スレッド"""
        print(f"🔄 LSLデータ受信開始")
        
        while self.running:
            try:
                sample, timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    with self.buffer_lock:
                        # 32チャンネル対応
                        if len(sample) >= 32:
                            self.eeg_buffer.append(sample[:32])
                        else:
                            padded_sample = sample + [0.0] * (32 - len(sample))
                            self.eeg_buffer.append(padded_sample)
                
            except Exception as e:
                if self.running:
                    print(f"⚠️ LSLデータ受信エラー: {e}")
                time.sleep(0.001)
        
        print(f"🔄 LSLデータ受信終了")
    
    def tcp_processing_thread(self):
        """TCP処理スレッド"""
        print(f"📡 TCP処理開始")
        step_count = 0
        
        while self.running:
            try:
                # クライアント接続待機
                client_socket, client_address = self.tcp_socket.accept()
                print(f"📡 TCP接続受付: {client_address}")

                try:
                    import socket as _sock
                    client_socket.setsockopt(_sock.IPPROTO_TCP, _sock.TCP_NODELAY, 1)
                except Exception:
                    pass
                
                # 現在のクライアント接続を保存（フィードバック送信用）
                self.current_client_socket = client_socket
                
                # データバッファ
                data_buffer = ""
                
                while self.running:
                    try:
                        # データ受信
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        
                        # データをバッファに追加
                        received_str = data.decode('utf-8')
                        data_buffer += received_str
                        
                        print(f"📥 受信データ: '{received_str.strip()}'")
                        
                        # Unity固有のメッセージをチェック
                        if self._handle_unity_messages(received_str):
                            continue
                        
                        # 完全なJSONメッセージを抽出
                        while '\n' in data_buffer or '}' in data_buffer:
                            # 改行または}で区切って処理
                            if '\n' in data_buffer:
                                line, data_buffer = data_buffer.split('\n', 1)
                            else:
                                # JSONの終端を探す
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
                                    line = data_buffer[:end_pos]
                                    data_buffer = data_buffer[end_pos:]
                                else:
                                    break
                            
                            # 空行スキップ
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                # JSONデータ解析
                                tcp_data = json.loads(line)
                                print(f"✅ JSON解析成功: {tcp_data}")
                                
                                # 強化学習ステップ実行
                                self._process_rl_step(tcp_data, step_count)
                                step_count += 1
                                
                            except json.JSONDecodeError as e:
                                print(f"⚠️ JSONデコードエラー (行: '{line[:50]}...'): {e}")
                                continue
                            except Exception as e:
                                print(f"⚠️ ステップ処理エラー: {e}")
                                continue
                        
                    except Exception as e:
                        print(f"⚠️ TCP処理エラー: {e}")
                        break
                
                # 接続終了時のクリーンアップ
                self.current_client_socket = None
                client_socket.close()
                print(f"📡 TCP接続終了: {client_address}")
                
            except Exception as e:
                if self.running:
                    print(f"⚠️ TCP接続エラー: {e}")
                time.sleep(1.0)  # エラー時の待機
        
        print(f"📡 TCP処理終了")

    def _handle_unity_messages(self, message):
        message = message.strip()

        # 既存ハンドリング
        if message.startswith('RESULT_'):
            print(f"🎮 Unity結果メッセージ: {message}")
            return True
        elif message.startswith('EPISODE_'):
            print(f"🎮 Unityエピソードメッセージ: {message}")
            self.episode_count += 1
            return True
        elif message.startswith('RESET'):
            print(f"🎮 Unityリセットメッセージ: {message}")
            return True
        elif message.startswith('CONNECT') or message.startswith('PING'):
            print(f"🎮 Unity接続メッセージ: {message}")
            # PINGに対して、直近値で即レスしてもよい
            self._send_grip_force_feedback(self._last_predicted_grip_force)
            return True

        # “コマンド待機”系の曖昧な文面も拾う
        if ("REQUEST" in message or "COMMAND" in message or "WAIT" in message or "GRIP" in message):
            print(f"🎮 Unityコマンド要求らしきメッセージ: {message}")
            self._send_grip_force_feedback(self._last_predicted_grip_force)
            return True

        return False



    def _process_rl_step(self, tcp_data, step_count):
        """強化学習ステップの処理"""
        current_time = time.time()
        
        # 初期化（エラー回避）
        actor_loss = None
        critic_loss = None
        
        try:
            # 1. EEGデータ取得・分類
            eeg_epoch = self._extract_eeg_epoch()
            if eeg_epoch is None:
                print("⚠️ EEGデータが不十分：既定値で応答")
                # 直近値があればそれ、なければ既定値で即返信
                self._send_grip_force_feedback(getattr(self, "_last_predicted_grip_force", 10.0),
                                            tcp_data.get("episode_id") if isinstance(tcp_data, dict) else None)
                return
            
            classification_result = self._classify_eeg_data(eeg_epoch)
            
            # 2. 現在の把持力取得
            current_grip_force = self._extract_grip_force(tcp_data)

            # 2.5 エピソード番号抽出（送信用）
            episode_id = self._extract_episode_number(tcp_data)
            
            # 3. 状態更新
            new_state = self._update_state(current_grip_force, classification_result)
            
            # 4. 報酬計算
            reward = self._calculate_reward(classification_result, current_grip_force)
            
            # 5. DDPGエージェント更新（前のステップの経験があれば）
            if self.previous_action is not None and hasattr(self, 'previous_state'):
                self.ddpg_agent.store_transition(
                    self.previous_state,
                    self.previous_action,
                    reward,
                    new_state,
                    False  # エピソード終了フラグ
                )
                
                # ネットワーク更新
                try:
                    actor_loss, critic_loss = self.ddpg_agent.update_networks()
                    if actor_loss is not None:
                        self.learning_stats['actor_losses'].append(actor_loss)
                        self.learning_stats['critic_losses'].append(critic_loss)
                except Exception as e:
                    print(f"⚠️ ネットワーク更新エラー: {e}")
                    actor_loss = 0.0
                    critic_loss = 0.0
            
            # 6. 次の行動決定
            next_action = self.ddpg_agent.select_action(new_state)
            
            # 7. 把持力スケーリング（0-1 → 実際の把持力範囲）
            grip_force_min, grip_force_max = 5.0, 20.0  # 実際の把持力範囲
            predicted_grip_force = grip_force_min + (grip_force_max - grip_force_min) * next_action[0]
            
            print(f"🎯 DDPG行動決定: 正規化値={next_action[0]:.3f} → 把持力={predicted_grip_force:.2f}N")
            
            # 8. TCP送信（修正版を呼び出し）
            print(f"📤 TCP送信開始...")
            self._send_grip_force_feedback(predicted_grip_force, episode_id=episode_id)
            print(f"📤 TCP送信完了")
            
            # 9. 状態・統計更新
            self.previous_state = self.current_state.copy()
            self.previous_action = next_action
            self.current_state = new_state
            
            self.learning_stats['total_steps'] += 1
            
            # 10. ログ記録
            self._log_step(
                step_count, current_grip_force, predicted_grip_force,
                classification_result, reward, new_state, next_action,
                actor_loss, critic_loss
            )
            
            # 進捗表示（10ステップごと）
            if step_count % 10 == 0:
                self._print_progress(step_count, classification_result, reward)
                
        except Exception as e:
            print(f"⚠️ 強化学習ステップエラー: {e}")
            import traceback
            traceback.print_exc()

    def _extract_episode_number(self, tcp_data):
        """受信JSONからエピソード番号を抽出（複数キー対応）"""
        if not isinstance(tcp_data, dict):
            return None
        for k in ['episode', 'episode_number', 'episode_id']:
            if k in tcp_data:
                try:
                    return int(tcp_data[k])
                except Exception:
                    pass
        return None

    def _send_grip_force_feedback(self, grip_force, episode_id=None):
        """
        Unity へ把持力を送信（EpisodeContactSynchronizer と同じ形式）
        type: 'grip_force_command'
        target_force: <float>
        """
        payload = {
            "type": "grip_force_command",
            "target_force": float(grip_force),
            "timestamp": time.time(),
            "episode_number": int(episode_id) if episode_id is not None else int(self.episode_count),
            "session_id": self.session_id
        }

        line = json.dumps(payload) + "\n"
        print(f"🔄 送信(同一ソケット): {line.strip()}")

        try:
            if getattr(self, "current_client_socket", None):
                self.current_client_socket.sendall(line.encode("utf-8"))
                # 直近値を保持（WAIT/PING 即応で使う）
                self._last_predicted_grip_force = float(grip_force)
                print("✅ 同一ソケットへ送信成功 (grip_force_command)")
                return True
            else:
                print("⚠️ 現在のクライアント接続がありません（送信不可）")
        except Exception as e:
            print(f"⚠️ 同一ソケット送信失敗: {e}")

        # 保険：ファイルに残す
        self._save_feedback_to_file(payload)
        return False
  


    def _send_via_current_connection(self, feedback_json):
        """現在の接続経由で送信（JSON Lines を想定）"""
        try:
            if getattr(self, "current_client_socket", None):
                # ★ sendall + \n（feedback_json は末尾 \n 付きで渡す前提）
                self.current_client_socket.sendall(feedback_json.encode("utf-8"))
                print("📤 現在接続経由送信完了")
                return True
            else:
                print("   現在接続なし")
        except Exception as e:
            print(f"⚠️ 現在接続送信失敗: {e}")
        return False

    
    def _send_via_new_connection(self, feedback_data):
        """新しい接続で送信"""
        try:
            # Unity側のフィードバック受信ポートに送信
            feedback_port = self.tcp_port + 1  # 12346
            
            print(f"📡 新規接続試行: {self.tcp_host}:{feedback_port}")
            
            feedback_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            feedback_socket.settimeout(2.0)  # 2秒タイムアウト
            feedback_socket.connect((self.tcp_host, feedback_port))
            
            feedback_json = json.dumps(feedback_data) + '\n'
            feedback_socket.send(feedback_json.encode('utf-8'))
            feedback_socket.close()
            
            print(f"📤 新規接続送信完了: ポート{feedback_port}")
            return True
            
        except ConnectionRefusedError:
            print(f"💡 フィードバック受信側未準備: ポート{feedback_port}")
        except Exception as e:
            print(f"⚠️ 新規接続送信失敗: {e}")
        return False
    
    def _save_feedback_to_file(self, feedback_data):
        """フィードバックをファイルに保存"""
        try:
            feedback_file = f"grip_force_feedback_{self.session_id}.txt"
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()}: {json.dumps(feedback_data)}\n")
            print(f"💾 フィードバックファイル保存: {feedback_file}")
        except Exception as e:
            print(f"⚠️ ファイル保存エラー: {e}")
    
    def _extract_eeg_epoch(self):
        """3.2秒さかのぼって1.2秒間のEEGエポックを切り出し"""
        with self.buffer_lock:
            if len(self.eeg_buffer) < self.epoch_samples:
                return None
            
            # 3.2秒前（800サンプル前）から1.2秒間（300サンプル）を切り出し
            lookback_start = len(self.eeg_buffer) - int(3.2 * self.sampling_rate)
            epoch_start = max(0, lookback_start)
            epoch_end = epoch_start + self.epoch_samples
            
            if epoch_end > len(self.eeg_buffer):
                return None
            
            epoch_data = np.array(list(self.eeg_buffer)[epoch_start:epoch_end])
            return epoch_data  # shape: (300, 32)
    
    def _classify_eeg_data(self, eeg_epoch):
        """EEG分類の実行"""
    def _classify_eeg_data(self, eeg_epoch):
        """EEG分類の実行"""
        if classify_eeg_epoch is None:
            # 分類器が利用できない場合はダミーの結果を返す
            return {
                'class_name': 'Correct',
                'class_id': 0,
                'confidence': 0.5,
                'raw_probabilities': [0.5, 0.25, 0.25]
            }
        
        try:
            class_name, class_id, confidence = classify_eeg_epoch(eeg_epoch, self.eeg_model_path)
            
            result = {
                'class_name': class_name,
                'class_id': class_id,
                'confidence': confidence,
                'raw_probabilities': None  # classify_eeg_epochから取得できない場合
            }
            
            self.classification_history.append(result)
            return result
            
        except Exception as e:
            print(f"⚠️ EEG分類エラー: {e}")
            # エラー時のデフォルト結果
            return {
                'class_name': 'Correct',
                'class_id': 0,
                'confidence': 0.1,
                'raw_probabilities': [0.33, 0.33, 0.34]
            }
    
    def _extract_grip_force(self, tcp_data):
        """TCPデータから把持力を抽出"""
        # 複数の可能なキーを試行
        possible_keys = ['grip_force', 'gripForce', 'force', 'gripping_force', 'target_force']
        
        grip_force = None
        for key in possible_keys:
            if key in tcp_data:
                grip_force = tcp_data[key]
                break
        
        # デフォルト値
        if grip_force is None:
            grip_force = 10.0
            print(f"💡 把持力データなし、デフォルト値使用: {grip_force}")
        
        # 文字列→数値変換
        if isinstance(grip_force, str):
            try:
                grip_force = float(grip_force)
            except ValueError:
                grip_force = 10.0
        
        # 0-1範囲に正規化
        grip_force_min, grip_force_max = 5.0, 20.0
        normalized_grip_force = (grip_force - grip_force_min) / (grip_force_max - grip_force_min)
        return np.clip(normalized_grip_force, 0, 1)
    
    def _update_state(self, current_grip_force, classification_result):
        """状態の更新"""
        # 状態: [現在の把持力, 分類信頼度, 前回分類結果]
        new_state = np.array([
            current_grip_force,
            classification_result['confidence'],
            classification_result['class_id'] / 2.0  # 0-2 → 0-1に正規化
        ])
        return new_state
    
    def _calculate_reward(self, classification_result, current_grip_force):
        """報酬の計算"""
        class_name = classification_result['class_name']
        confidence = classification_result['confidence']
        
        # 基本報酬設定
        if class_name == 'Correct':
            base_reward = 100.0
        elif class_name == 'UnderGrip':
            base_reward = -50.0
        elif class_name == 'OverGrip':
            base_reward = -50.0
        else:
            base_reward = 0.0
        
        # 信頼度による重み付け
        confidence_weight = confidence
        
        # 把持力の適切性による追加報酬
        grip_force_reward = 0.0
        if class_name == 'Correct' and 0.3 <= current_grip_force <= 0.7:
            grip_force_reward = 20.0
        elif class_name == 'UnderGrip' and current_grip_force < 0.5:
            grip_force_reward = -10.0
        elif class_name == 'OverGrip' and current_grip_force > 0.5:
            grip_force_reward = -10.0
        
        total_reward = (base_reward * confidence_weight) + grip_force_reward
        return total_reward
    
    def _log_step(self, step, tcp_grip_force, predicted_grip_force, classification_result, 
                  reward, state, action, actor_loss, critic_loss):
        """ステップのログ記録"""
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.episode_count,
                    step,
                    f"{tcp_grip_force:.4f}" if tcp_grip_force is not None else "N/A",
                    f"{predicted_grip_force:.4f}" if predicted_grip_force is not None else "N/A",
                    classification_result.get('class_name', 'Unknown'),
                    f"{classification_result.get('confidence', 0.0):.4f}",
                    f"{reward:.4f}" if reward is not None else "N/A",
                    f"{list(state)}" if state is not None else "N/A",
                    f"{list(action)}" if action is not None else "N/A",
                    f"{actor_loss:.6f}" if actor_loss is not None else "N/A",
                    f"{critic_loss:.6f}" if critic_loss is not None else "N/A"
                ])
        except Exception as e:
            print(f"⚠️ ログ記録エラー: {e}")
    
    def _print_progress(self, step, classification_result, reward):
        """進捗表示"""
        try:
            avg_actor_loss = np.mean(self.learning_stats['actor_losses']) if self.learning_stats['actor_losses'] else 0
            avg_critic_loss = np.mean(self.learning_stats['critic_losses']) if self.learning_stats['critic_losses'] else 0
            
            class_name = classification_result.get('class_name', 'Unknown')
            confidence = classification_result.get('confidence', 0.0)
            
            print(f"📈 ステップ {step}: {class_name} "
                  f"(信頼度: {confidence:.3f}) | "
                  f"報酬: {reward:.1f} | "
                  f"損失: Actor={avg_actor_loss:.4f}, Critic={avg_critic_loss:.4f}")
        except Exception as e:
            print(f"⚠️ 進捗表示エラー: {e}")
    
    def run(self, duration_seconds=1800, max_episodes=None):
        """システムの実行"""
        if not self.setup_connections():
            return False
        
        print(f"🚀 EEG強化学習システム開始")
        print(f"⏱️ 実行時間: {duration_seconds}秒 ({duration_seconds//60}分)")
        if max_episodes:
            print(f"🎯 最大エピソード数: {max_episodes}")
        
        self.running = True
        start_time = time.time()
        
        try:
            # データ受信スレッド開始
            lsl_thread = threading.Thread(target=self.lsl_data_thread, daemon=True)
            tcp_thread = threading.Thread(target=self.tcp_processing_thread, daemon=True)
            
            lsl_thread.start()
            tcp_thread.start()
            
            print(f"✅ 全システム稼働中")
            print(f"   LSLデータ受信: 開始")
            print(f"   TCP処理: 開始")
            print(f"   DDPG学習: 準備完了")
            print(f"\n💡 使用方法:")
            print(f"   1. Unity等からTCP {self.tcp_port}にデータ送信")
            print(f"   2. 自動でEEG分類・DDPG学習実行")
            print(f"   3. 適切な把持力をTCP経由で送信")
            print(f"   4. Ctrl+Cで終了")
            
            # メインループ
            while True:
                elapsed = time.time() - start_time
                
                # 終了条件チェック
                if elapsed >= duration_seconds:
                    print(f"\n⏰ 制限時間到達（{duration_seconds}秒）")
                    break
                
                if max_episodes and self.episode_count >= max_episodes:
                    print(f"\n🎯 最大エピソード数到達（{max_episodes}）")
                    break
                
                # 進捗報告（30秒ごと）
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    self._print_session_progress(elapsed, duration_seconds)
                
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ ユーザー中断")
        except Exception as e:
            print(f"\n❌ 実行エラー: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            self._print_final_statistics()
            
        return True
    
    def _print_session_progress(self, elapsed, total_duration):
        """セッション進捗の表示"""
        remaining = total_duration - elapsed
        progress_pct = (elapsed / total_duration) * 100
        
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        memory_size = len(self.ddpg_agent.memory)
        
        print(f"📊 進捗: {elapsed:.0f}秒経過 ({progress_pct:.1f}%) | "
              f"ステップ: {self.learning_stats['total_steps']} | "
              f"平均報酬: {avg_reward:.1f} | "
              f"経験数: {memory_size} | "
              f"残り: {remaining:.0f}秒")
    
    def _print_final_statistics(self):
        """最終統計の表示"""
        print(f"\n{'='*70}")
        print(f"🧠 EEG強化学習システム 最終統計")
        print(f"{'='*70}")
        
        # 学習統計
        total_steps = self.learning_stats['total_steps']
        avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
        
        print(f"📈 学習統計:")
        print(f"   総ステップ数         : {total_steps}")
        print(f"   エピソード数         : {self.episode_count}")
        print(f"   平均報酬             : {avg_reward:.2f}")
        print(f"   経験蓄積数           : {len(self.ddpg_agent.memory)}")
        
        # 損失統計
        if self.learning_stats['actor_losses']:
            avg_actor_loss = np.mean(self.learning_stats['actor_losses'])
            avg_critic_loss = np.mean(self.learning_stats['critic_losses'])
            print(f"   平均Actor損失        : {avg_actor_loss:.4f}")
            print(f"   平均Critic損失       : {avg_critic_loss:.4f}")
        
        # 分類統計
        if self.classification_history:
            classifications = list(self.classification_history)
            class_counts = Counter([c['class_name'] for c in classifications])
            avg_confidence = np.mean([c['confidence'] for c in classifications])
            
            print(f"🧠 EEG分類統計:")
            print(f"   総分類回数           : {len(classifications)}")
            print(f"   平均信頼度           : {avg_confidence:.3f}")
            for class_name, count in class_counts.items():
                percentage = (count / len(classifications)) * 100
                print(f"   {class_name:12s}     : {count}回 ({percentage:.1f}%)")
        
        # ファイル出力
        print(f"📂 出力ファイル:")
        print(f"   学習ログ             : {self.log_file}")
        
        # モデル保存
        model_prefix = f"models/ddpg_eeg_rl_{self.session_id}"
        os.makedirs("models", exist_ok=True)
        self.ddpg_agent.save_models(model_prefix)
        print(f"   DDPGモデル           : {model_prefix}_*.pth")
        
        print(f"{'='*70}")
    
    def stop(self):
        """システム停止"""
        print(f"🛑 システム停止中...")
        self.running = False
        
        # TCP接続クローズ
        try:
            if hasattr(self, 'tcp_socket'):
                self.tcp_socket.close()
        except:
            pass
        
        print(f"✅ システム停止完了")


class EEGRLSystemConfig:
    """
    EEG強化学習システムの設定クラス
    パラメータの調整やプリセット設定を提供
    """
    
    @staticmethod
    def get_default_config():
        """デフォルト設定"""
        return {
            'tcp_host': '127.0.0.1',
            'tcp_port': 12345,
            'lsl_stream_name': 'MockEEG',
            'sampling_rate': 250,
            'epoch_duration': 1.2,
            'lookback_duration': 3.2,
            'eeg_model_path': './models/best_eeg_classifier.pth',
            'duration_seconds': 1800,  # 30分
            'max_episodes': None
        }
    
    @staticmethod
    def get_quick_test_config():
        """クイックテスト用設定"""
        config = EEGRLSystemConfig.get_default_config()
        config.update({
            'duration_seconds': 300,  # 5分
            'max_episodes': 50
        })
        return config
    
    @staticmethod
    def get_long_training_config():
        """長時間学習用設定"""
        config = EEGRLSystemConfig.get_default_config()
        config.update({
            'duration_seconds': 3600,  # 1時間
            'max_episodes': 500
        })
        return config


class TCPTestClient:
    """
    テスト用TCPクライアント
    システムの動作確認用
    """
    
    def __init__(self, host='127.0.0.1', port=12345):
        self.host = host
        self.port = port
    
    def send_test_data(self, num_messages=10, interval=2.0):
        """テストデータの送信"""
        print(f"🧪 TCPテストクライアント開始")
        print(f"   送信先: {self.host}:{self.port}")
        print(f"   メッセージ数: {num_messages}")
        print(f"   間隔: {interval}秒")
        
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((self.host, self.port))
            print(f"✅ TCP接続成功")
            
            for i in range(num_messages):
                # テストデータ作成
                test_data = {
                    'message_id': i,
                    'timestamp': time.time(),
                    'grip_force': 8.0 + (i % 5) * 2.0,  # 8-16の範囲で変化
                    'episode_id': i // 5,
                    'task_type': 'pick_and_place',
                    'object_type': 'aluminum_can'
                }
                
                # JSON送信
                json_message = json.dumps(test_data) + '\n'
                client_socket.send(json_message.encode('utf-8'))
                
                print(f"📤 送信 {i+1}/{num_messages}: 把持力={test_data['grip_force']:.1f}N")
                
                time.sleep(interval)
            
            print(f"✅ 全メッセージ送信完了")
            client_socket.close()
            
        except Exception as e:
            print(f"❌ TCPテストクライアントエラー: {e}")


def run_test_session():
    """テストセッションの実行"""
    print("🧪 EEG強化学習システム テストセッション")
    print("=" * 50)
    
    # システム初期化
    config = EEGRLSystemConfig.get_quick_test_config()
    config['duration_seconds'] = 120  # 2分間のテスト
    
    system = EEGReinforcementLearningSystem(
        tcp_host=config['tcp_host'],
        tcp_port=config['tcp_port'],
        lsl_stream_name=config['lsl_stream_name'],
        sampling_rate=config['sampling_rate'],
        epoch_duration=config['epoch_duration'],
        lookback_duration=config['lookback_duration'],
        eeg_model_path=config['eeg_model_path']
    )
    
    # システム開始（別スレッド）
    import threading
    system_thread = threading.Thread(
        target=lambda: system.run(duration_seconds=config['duration_seconds'])
    )
    system_thread.start()
    
    # 少し待ってからテストデータ送信
    time.sleep(3)
    
    # テストクライアント実行
    test_client = TCPTestClient(config['tcp_host'], config['tcp_port'])
    test_client.send_test_data(num_messages=20, interval=1.0)
    
    # システム終了待機
    system_thread.join()
    
    print("🧪 テストセッション完了")


def main():
    """メイン関数"""
    print("🧠 EEG強化学習システム（DDPG版）")
    print("=" * 70)
    
    # 設定選択メニュー
    print("\n実行モードを選択してください:")
    print("1. デフォルト設定（30分間）")
    print("2. クイックテスト（5分間）")
    print("3. 長時間学習（1時間）")
    print("4. カスタム設定")
    print("5. テストセッション（システム動作確認）")
    
    try:
        choice = input("選択 (1-5): ").strip()
        
        if choice == '5':
            # テストセッション実行
            run_test_session()
            return
        elif choice == '1':
            config = EEGRLSystemConfig.get_default_config()
            print("✅ デフォルト設定を使用")
        elif choice == '2':
            config = EEGRLSystemConfig.get_quick_test_config()
            print("✅ クイックテスト設定を使用")
        elif choice == '3':
            config = EEGRLSystemConfig.get_long_training_config()
            print("✅ 長時間学習設定を使用")
        elif choice == '4':
            config = EEGRLSystemConfig.get_default_config()
            print("カスタム設定を入力してください（Enterでデフォルト値）:")
            
            duration = input(f"実行時間（秒）[{config['duration_seconds']}]: ")
            if duration:
                config['duration_seconds'] = int(duration)
            
            max_episodes = input(f"最大エピソード数（未指定なら空白）[{config['max_episodes']}]: ")
            if max_episodes:
                config['max_episodes'] = int(max_episodes)
            
            tcp_port = input(f"TCPポート[{config['tcp_port']}]: ")
            if tcp_port:
                config['tcp_port'] = int(tcp_port)
                
            print("✅ カスタム設定を適用")
        else:
            print("❌ 無効な選択です。デフォルト設定を使用します。")
            config = EEGRLSystemConfig.get_default_config()
    
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️ 入力エラー。デフォルト設定を使用します。")
        config = EEGRLSystemConfig.get_default_config()
    
    # 設定確認
    print(f"\n📋 使用設定:")
    print(f"   TCP接続: {config['tcp_host']}:{config['tcp_port']}")
    print(f"   LSLストリーム: {config['lsl_stream_name']}")
    print(f"   実行時間: {config['duration_seconds']}秒 ({config['duration_seconds']//60}分)")
    print(f"   最大エピソード: {config['max_episodes'] or '無制限'}")
    print(f"   EEGモデル: {config['eeg_model_path']}")
    print(f"   エポック設定: {config['lookback_duration']}秒前から{config['epoch_duration']}秒間")
    
    # システム作成・実行
    try:
        system = EEGReinforcementLearningSystem(
            tcp_host=config['tcp_host'],
            tcp_port=config['tcp_port'],
            lsl_stream_name=config['lsl_stream_name'],
            sampling_rate=config['sampling_rate'],
            epoch_duration=config['epoch_duration'],
            lookback_duration=config['lookback_duration'],
            eeg_model_path=config['eeg_model_path']
        )
        
        print(f"\n🚀 システム実行開始...")
        print(f"💡 トラブルシューティング:")
        print(f"   - JSONエラー: データ形式を確認")
        print(f"   - TCP送信エラー: フィードバック受信側の準備を確認")
        print(f"   - EEGデータ不足: LSLストリームの動作を確認")
        
        success = system.run(
            duration_seconds=config['duration_seconds'],
            max_episodes=config['max_episodes']
        )
        
        if success:
            print(f"\n✅ 正常終了")
        else:
            print(f"\n❌ エラー終了")
            
    except Exception as e:
        print(f"\n❌ システム初期化エラー: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()