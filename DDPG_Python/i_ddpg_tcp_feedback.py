#!/usr/bin/env python3
"""
TypeA DDPG学習システム（修正版・即効パッチ適用済み）

修正点：
1. 停止条件を「収集済みエピソード数」に基づかせる
2. 実エピソード数でself.episode_countを更新
3. ループ終了時にフラグを落としてから保存
4. スレッドをデーモンにしない
5. 停止時の保存条件を「データの有無」で判定

これにより以下の問題を解決：
- 400エピソードで停止しない問題
- ログが保存されない問題
- デーモンスレッド由来のstdoutロック例外
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import json
import time
import threading
import queue
import argparse
from datetime import datetime
from collections import deque, namedtuple
from pathlib import Path
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

# 既存モジュールのインポート
from e_tcp_lsl_sync_system import LSLTCPEpisodeCollector, Episode
from c_unity_tcp_interface import EEGTCPInterface

# PyTorch設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎯 デバイス: {device}")

# DDPG用の経験バッファ
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    """DDPG Actorネットワーク"""
    def __init__(self, state_dim=4, action_dim=1, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # He初期化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # [-1, 1]の範囲
        return x

class Critic(nn.Module):
    """DDPG Criticネットワーク"""
    def __init__(self, state_dim=4, action_dim=1, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    
    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class OUNoise:
    """Ornstein-Uhlenbeck ノイズ"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """経験再生バッファ"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        import random
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class AnalysisUtils:
    """分析用ユーティリティ"""
    
    @staticmethod
    def calculate_auc(y_values):
        """AUC計算（台形則）"""
        return float(np.trapz(y_values, dx=1.0))
    
    @staticmethod
    def moving_average(x, window=100):
        """移動平均"""
        series = pd.Series(x)
        return series.rolling(window, min_periods=1).mean().values
    
    @staticmethod
    def detect_plateau(y_smooth, window=200, eps=1e-3):
        """plateau検出"""
        if len(y_smooth) < window:
            return None, None
        
        segment = y_smooth[-window:]
        gradient = np.abs(np.diff(segment)).mean()
        
        if gradient < eps:
            plateau_value = float(np.mean(segment))
            plateau_episode = len(y_smooth) - window
            return plateau_value, plateau_episode
        
        return None, None
    
    @staticmethod
    def find_time_to_threshold(y_values, threshold=0.70):
        """閾値到達時間"""
        indices = np.where(y_values >= threshold)[0]
        return int(indices[0]) if len(indices) > 0 else None

class TypeADDPGSystem:
    """TypeA DDPG学習システム（即効パッチ適用済み）"""
    
    def __init__(self, experiment_type="A_400", seed=42):
        """
        Args:
            experiment_type: "A_400" or "A_long"
            seed: ランダムシード
        """
        # シード設定
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        self.experiment_type = experiment_type
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"DDPG_Python/logs/typea_{experiment_type}_seed{seed}_{self.session_id}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # エピソード数設定
        if experiment_type == "A_400":
            self.target_episodes = 400
        elif experiment_type == "A_long":
            self.target_episodes = 5000
        else:
            self.target_episodes = 400
        
        # 対称正規化パラメータ（F中心11.5N、半幅3.5N）
        self.force_center = 11.5  # N
        self.force_halfwidth = 3.5  # N
        self.force_min = self.force_center - self.force_halfwidth  # 8N
        self.force_max = self.force_center + self.force_halfwidth  # 15N
        
        # 状態空間設計 [force_norm, contact, broken, prev_action]
        self.state_dim = 4
        self.action_dim = 1
        
        # DDPGネットワーク
        self.actor = Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(device)
        self.critic = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(device)
        
        # ターゲットネットワーク初期化
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # オプティマイザ
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # ハイパーパラメータ
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 64
        
        # 経験バッファ
        self.replay_buffer = ReplayBuffer(capacity=100000)
        
        # 探索ノイズ
        self.noise = OUNoise(self.action_dim, sigma=0.2)
        self.noise_decay = 0.995
        
        # データ収集システム
        self.episode_collector = None
        
        # TCP通信（修正: auto_reply=False で自動応答を無効化）
        self.tcp_interface = EEGTCPInterface(host='127.0.0.1', port=12346, auto_reply=False)
        
        # 実行制御
        self.is_running = False
        self.learning_thread = None
        
        # Pending方式用の状態管理
        self.pending_state = None
        self.pending_action = None
        self.episode_count = 0  # 表示・進捗用のカウンタ
        
        # 状態管理（修正: 直近のロボット状態を保持）
        self.last_tcp_data = None
        
        # 統計（エピソードごと）
        self.episode_data = []  # 各エピソードの詳細データ
        
        # 報酬パラメータ
        self.reward_params = {
            'success_reward': 10.0,
            'error_penalty_coeff': 1.0,
            'damage_penalty': 20.0,
            'contact_bonus': 2.0
        }
        
        print(f"🤖 TypeA DDPG学習システム初期化完了（リクエスト型修正済み）")
        print(f"   実験タイプ: {experiment_type}")
        print(f"   シード: {seed}")
        print(f"   目標エピソード数: {self.target_episodes}")
        print(f"   対称正規化: {self.force_min}-{self.force_max}N → [-1,1]")
        print(f"   出力ディレクトリ: {self.output_dir}")
    
    def _handle_tcp_state(self, message_data):
        """TCP状態メッセージ処理（ロボット状態を保持）"""
        try:
            # JSONのロボット状態（episode/grip_force等）が来たときに更新
            if isinstance(message_data, dict) and 'episode' in message_data and 'grip_force' in message_data:
                self.last_tcp_data = message_data
                print(f"📊 ロボット状態更新: ep={message_data.get('episode')}, force={message_data.get('grip_force'):.2f}N")
        except Exception as e:
            print(f"⚠️ TCP状態処理エラー: {e}")
    
    def normalize_force(self, force):
        """把持力の対称正規化 [8-15N] → [-1,1]"""
        return (force - self.force_center) / self.force_halfwidth
    
    def denormalize_action(self, action):
        """アクションの逆正規化 [-1,1] → [8-15N]"""
        return action * self.force_halfwidth + self.force_center
    
    def calculate_explicit_reward(self, tcp_data):
        """明示的報酬の計算"""
        actual_grip_force = tcp_data.get('grip_force', 0.0)
        contact = tcp_data.get('contact', False)
        broken = tcp_data.get('broken', False)
        
        reward = 0.0
        
        # 成功報酬（8-15N範囲内）
        if self.force_min <= actual_grip_force <= self.force_max:
            reward += self.reward_params['success_reward']
        
        # 把持力誤差ペナルティ
        force_error = abs(actual_grip_force - self.force_center)
        reward -= self.reward_params['error_penalty_coeff'] * force_error
        
        # 破損ペナルティ
        if broken:
            reward -= self.reward_params['damage_penalty']
        
        # 接触ボーナス
        if contact and not broken:
            reward += self.reward_params['contact_bonus']
        
        return reward
    
    def create_state(self, tcp_data, prev_action):
        """状態ベクトルの作成（対称正規化）"""
        grip_force = tcp_data.get('grip_force', self.force_center)
        force_norm = self.normalize_force(grip_force)
        
        contact = 1.0 if tcp_data.get('contact', False) else 0.0
        broken = 1.0 if tcp_data.get('broken', False) else 0.0
        
        state = np.array([
            force_norm,
            contact,
            broken,
            prev_action
        ], dtype=np.float32)
        
        return state
    
    def select_action(self, state, add_noise=True):
        """アクション選択"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if add_noise and self.is_running:
            noise_sample = self.noise.sample()
            action += noise_sample
        
        action = np.clip(action, -1.0, 1.0)
        return action
    
    def update_networks(self):
        """DDPGネットワークの更新"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Criticの更新
        next_action = self.actor_target(next_state)
        target_q = self.critic_target(next_state, next_action)
        target_q = reward + (self.gamma * target_q * (1 - done))
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()
        
        # Actorの更新
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # ターゲットネットワークのソフトアップデート
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        # ノイズ減衰
        self.noise.sigma *= self.noise_decay
        self.noise.sigma = max(self.noise.sigma, 0.01)
        
        return actor_loss.item(), critic_loss.item(), current_q.mean().item()
    
    def soft_update(self, target, source, tau):
        """ソフトアップデート"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def handle_grip_force_request(self, message_data):
        """Unityからの把持力リクエスト処理（JSON・テキスト両対応）"""
        try:
            # JSONリクエストの場合
            is_json_request = (isinstance(message_data, dict) and 
                             message_data.get('type') == 'grip_force_request')
            
            # テキストリクエストの場合
            is_text_request = (isinstance(message_data, dict) and 
                             message_data.get('type') == 'text_message' and 
                             message_data.get('content') == 'REQUEST_GRIP_FORCE')
            
            # いずれでもない場合は無視
            if not (is_json_request or is_text_request):
                return
            
            print(f"🎯 把持力リクエスト検出: {'JSON' if is_json_request else 'TEXT'}")
            
            # 状態作成用のTCPデータを決定
            if is_json_request:
                # JSONリクエストの場合はそのデータを使用
                tcp_data = message_data
            else:
                # テキストリクエストの場合は直近のロボット状態を使用
                if self.last_tcp_data is None:
                    print(f"⚠️ REQUEST_GRIP_FORCE を受信しましたが、直近のロボット状態がないため pending を設定できません")
                    # デフォルト状態で処理を続行
                    tcp_data = {
                        'grip_force': self.force_center,
                        'contact': False,
                        'broken': False,
                        'episode': 0
                    }
                else:
                    tcp_data = self.last_tcp_data
                    print(f"📊 直近状態を使用: ep={tcp_data.get('episode')}, force={tcp_data.get('grip_force', 0):.2f}N")
            
            # 前回のアクション値
            prev_action = 0.0 if self.pending_action is None else self.pending_action[0]
            
            # 状態作成
            state = self.create_state(tcp_data, prev_action)
            
            # アクション選択
            action = self.select_action(state, add_noise=True)
            
            # Pending状態を保存（K=1設計）
            self.pending_state = state
            self.pending_action = action
            
            # アクションを把持力に変換
            grip_force = self.denormalize_action(action[0])
            grip_force = np.clip(grip_force, 5.0, 25.0)  # 安全クランプ
            
            print(f"🤖 TypeA把持力決定: {grip_force:.2f}N (action: {action[0]:.3f}, noise_σ: {self.noise.sigma:.3f})")
            
            # TCP応答送信
            response = {
                'type': 'grip_force_command',
                'target_force': float(grip_force),
                'timestamp': time.time(),
                'session_id': f"typea_{self.experiment_type}_seed{self.seed}_{int(time.time())}"
            }
            self.tcp_interface.send_message(response)
            
        except Exception as e:
            print(f"❌ 把持力リクエスト処理エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def run_learning(self):
        """TypeA DDPG学習実行"""
        print(f"🚀 TypeA DDPG学習開始 ({self.experiment_type}, seed={self.seed})")
        
        # エピソード収集システム初期化（★ 修正: auto_reply=False で自動応答を無効化）
        self.episode_collector = LSLTCPEpisodeCollector(
            lsl_stream_name='MockEEG',
            tcp_host='127.0.0.1',
            tcp_port=12345,
            save_to_csv=True
        )
        
        # ★ 重要: episode_collectorの内部EEGTCPInterfaceも自動応答を無効化する必要
        # （LSLTCPEpisodeCollectorが修正済みであることを前提）
        
        if not self.episode_collector.start_collection():
            print("❌ エピソード収集開始失敗")
            return False
        
        # TCP通信開始とコールバック設定
        if not self.tcp_interface.start_server():
            print("❌ TCP通信開始失敗")
            return False
        
        # コールバック設定
        print("🔗 TCPコールバック設定中...")
        self.tcp_interface.add_message_callback(self._handle_tcp_state)  # ロボット状態保持用
        self.tcp_interface.add_message_callback(self.handle_grip_force_request)  # 把持力リクエスト処理用
        print("✅ TCPコールバック設定完了")
        print("📋 ポート設定:")
        print("   エピソード収集: 127.0.0.1:12345 (自動応答無効)")
        print("   学習システム: 127.0.0.1:12346 (自動応答無効)")
        print("💡 Unity接続先: 12346ポートに接続してください")
        
        # 学習ループ開始（★ daemon=False に変更）
        self.is_running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=False)
        self.learning_thread.start()
        
        print(f"✅ TypeA学習開始完了")
        return True
    
    def _learning_loop(self):
        """学習ループ（即効パッチ適用済み）"""
        print(f"🔄 TypeA学習ループ開始（Pending方式・即効パッチ適用済み）")
        
        last_episode_count = 0
        
        # ★ パッチ1: 停止条件を「収集済みエピソード数」に基づかせる
        while self.is_running:
            try:
                current_episode_count = len(self.episode_collector.episodes)
                
                # ★ 収集済みが目標に到達したら終了
                if current_episode_count >= self.target_episodes:
                    print(f"🎯 目標エピソード数到達: {current_episode_count}/{self.target_episodes}")
                    break
                
                if current_episode_count > last_episode_count:
                    # 新しいエピソードを処理
                    for i in range(last_episode_count, current_episode_count):
                        episode = self.episode_collector.episodes[i]
                        
                        print(f"🆕 エピソード {episode.episode_id} 受信 (pending: {'あり' if self.pending_state is not None else 'なし'})")
                        
                        # Pending状態がある場合のみ経験を追加（K=1設計）
                        if self.pending_state is not None and self.pending_action is not None:
                            # 報酬計算
                            reward = self.calculate_explicit_reward(episode.tcp_data)
                            
                            # 次状態作成
                            next_state = self.create_state(episode.tcp_data, self.pending_action[0])
                            
                            # 経験バッファに追加（done=True: K=1設計）
                            self.replay_buffer.push(
                                self.pending_state,
                                self.pending_action,
                                reward,
                                next_state,
                                True  # K=1なので毎回done=True
                            )
                            
                            # ネットワーク更新
                            if len(self.replay_buffer) >= self.batch_size:
                                actor_loss, critic_loss, avg_q = self.update_networks()
                            else:
                                actor_loss, critic_loss, avg_q = 0, 0, 0
                            
                            # エピソードデータ記録
                            self._record_episode_data(episode, reward, actor_loss, critic_loss, avg_q)
                            
                            print(f"✅ 学習ステップ実行: episode={len(self.episode_data)}, reward={reward:.2f}, buffer_size={len(self.replay_buffer)}")
                            
                            # 進捗表示
                            if len(self.episode_data) % 50 == 0:
                                self._print_learning_progress()
                            
                            # モデル保存
                            if len(self.episode_data) % 100 == 0:
                                self._save_model()
                        else:
                            # フォールバック: EPISODE_ENDフォールバック（推奨追加機能）
                            print(f"⚠️ Pending状態なしでEPISODE_END受信: episode={episode.episode_id}")
                            print(f"   直近状態: {self.last_tcp_data is not None}")
                            print(f"   この場合は学習ステップをスキップします")
                        
                        # Pending状態リセット
                        self.pending_state = None
                        self.pending_action = None
                    
                    last_episode_count = current_episode_count
                
                # ★ パッチ2: 表示・停止用のカウンタは「収集済み実数」に同期
                self.episode_count = current_episode_count
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 学習ループエラー: {e}")
                time.sleep(1.0)
        
        # ★ パッチ3: ループ終了時にフラグを落としてから保存
        print(f"✅ TypeA学習ループ完了: {self.episode_count}エピソード")
        self.is_running = False
        self._save_final_results()
    
    def _record_episode_data(self, episode, reward, actor_loss, critic_loss, avg_q):
        """エピソードデータの記録"""
        grip_force = episode.tcp_data.get('grip_force', 0.0)
        contact = episode.tcp_data.get('contact', False)
        broken = episode.tcp_data.get('broken', False)
        
        # 成功判定
        success = self.force_min <= grip_force <= self.force_max
        
        # 把持力誤差
        force_error = abs(grip_force - self.force_center)
        
        episode_info = {
            'episode': len(self.episode_data) + 1,  # 学習エピソード番号
            'reward': reward,
            'grip_force': grip_force,
            'success': success,
            'force_error': force_error,
            'contact': contact,
            'broken': broken,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'avg_q_value': avg_q,
            'noise_sigma': self.noise.sigma
        }
        
        self.episode_data.append(episode_info)
    
    def _print_learning_progress(self):
        """学習進捗表示"""
        if len(self.episode_data) == 0:
            return
        
        # 最新50エピソードの統計
        recent_data = self.episode_data[-50:]
        recent_rewards = [d['reward'] for d in recent_data]
        recent_successes = [d['success'] for d in recent_data]
        recent_errors = [d['force_error'] for d in recent_data]
        recent_damages = [d['broken'] for d in recent_data]
        
        success_rate = np.mean(recent_successes)
        avg_reward = np.mean(recent_rewards)
        avg_error = np.mean(recent_errors)
        damage_rate = np.mean(recent_damages)
        
        print(f"\n📊 TypeA進捗 (学習済み {len(self.episode_data)}/{self.target_episodes}):")
        print(f"   平均報酬（最新50）: {avg_reward:.2f}")
        print(f"   成功率（最新50）: {success_rate:.1%}")
        print(f"   平均力誤差（最新50）: {avg_error:.2f}N")
        print(f"   破損率（最新50）: {damage_rate:.1%}")
        print(f"   バッファサイズ: {len(self.replay_buffer)}")
        print(f"   探索ノイズσ: {self.noise.sigma:.3f}")
    
    def _save_model(self):
        """モデル保存"""
        try:
            model_path = os.path.join(self.output_dir, f'typea_model_ep{len(self.episode_data)}.pth')
            torch.save({
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'episode_count': len(self.episode_data),
                'experiment_type': self.experiment_type,
                'seed': self.seed
            }, model_path)
        except Exception as e:
            print(f"⚠️ モデル保存エラー: {e}")
    
    def _calculate_advanced_metrics(self):
        """高度な指標計算"""
        if len(self.episode_data) == 0:
            return {}
        
        # DataFrameに変換
        df = pd.DataFrame(self.episode_data)
        
        # 移動平均計算（成功率）
        success_ma = AnalysisUtils.moving_average(df['success'].values, window=100)
        
        # AUC計算
        auc_all = AnalysisUtils.calculate_auc(success_ma)
        auc_0_400 = AnalysisUtils.calculate_auc(success_ma[:400]) if len(success_ma) >= 400 else auc_all
        
        # plateau検出
        plateau_value, plateau_episode = AnalysisUtils.detect_plateau(success_ma, window=200, eps=1e-3)
        
        # time-to-70%
        time_to_70 = AnalysisUtils.find_time_to_threshold(success_ma, threshold=0.70)
        
        # 最終性能（最新100エピソード平均）
        final_success_rate = np.mean(df['success'].iloc[-100:]) if len(df) >= 100 else np.mean(df['success'])
        final_reward = np.mean(df['reward'].iloc[-100:]) if len(df) >= 100 else np.mean(df['reward'])
        final_force_error = np.mean(df['force_error'].iloc[-100:]) if len(df) >= 100 else np.mean(df['force_error'])
        final_damage_rate = np.mean(df['broken'].iloc[-100:]) if len(df) >= 100 else np.mean(df['broken'])
        
        return {
            'auc_all': auc_all,
            'auc_0_400': auc_0_400,
            'plateau_value': plateau_value,
            'plateau_episode': plateau_episode,
            'time_to_70': time_to_70,
            'final_success_rate': final_success_rate,
            'final_reward': final_reward,
            'final_force_error': final_force_error,
            'final_damage_rate': final_damage_rate,
            'success_moving_average': success_ma.tolist()
        }
    
    def _save_final_results(self):
        """最終結果保存"""
        print(f"💾 最終結果保存中...")
        
        # ★ パッチ適用: データが空でも最低限の統計は保存
        if len(self.episode_data) == 0:
            print(f"⚠️ 学習データがありませんが、基本情報を保存します")
            # 最低限の情報を保存
            basic_stats = {
                'experiment_type': self.experiment_type,
                'seed': self.seed,
                'total_episodes': self.episode_count,
                'target_episodes': self.target_episodes,
                'learning_episodes': 0,
                'message': 'No learning data collected'
            }
            json_path = os.path.join(self.output_dir, 'final_stats.json')
            with open(json_path, 'w') as f:
                json.dump(basic_stats, f, indent=2)
            print(f"📄 基本情報保存: {json_path}")
            return
        
        # DataFrame作成
        df = pd.DataFrame(self.episode_data)
        
        # 移動平均追加
        df['success_rate_ma100'] = AnalysisUtils.moving_average(df['success'].values, window=100)
        df['reward_ma50'] = AnalysisUtils.moving_average(df['reward'].values, window=50)
        
        # learning_results.csv保存
        csv_data = {
            'episode': df['episode'],
            'reward': df['reward'],
            'success_rate': df['success_rate_ma100'],  # 移動平均
            'force_error': df['force_error'],
            'damage_rate': df['broken'].astype(int)
        }
        results_df = pd.DataFrame(csv_data)
        csv_path = os.path.join(self.output_dir, 'learning_results.csv')
        results_df.to_csv(csv_path, index=False)
        
        # 高度な指標計算
        advanced_metrics = self._calculate_advanced_metrics()
        
        # final_stats.json保存
        final_stats = {
            'experiment_type': self.experiment_type,
            'seed': self.seed,
            'total_episodes': self.episode_count,
            'target_episodes': self.target_episodes,
            'learning_episodes': len(self.episode_data),
            **advanced_metrics,
            'reward_parameters': self.reward_params,
            'force_normalization': {
                'center': self.force_center,
                'halfwidth': self.force_halfwidth,
                'range': [self.force_min, self.force_max]
            }
        }
        
        json_path = os.path.join(self.output_dir, 'final_stats.json')
        with open(json_path, 'w') as f:
            json.dump(final_stats, f, indent=2)
        
        # master_auc.json保存（集計スクリプト用）
        master_auc = {
            'auc_0_400': advanced_metrics['auc_0_400'],
            'auc_all': advanced_metrics['auc_all'],
            'time_to_70': advanced_metrics['time_to_70'],
            'plateau_value': advanced_metrics['plateau_value'],
            'plateau_at_episode': advanced_metrics['plateau_episode'],
            'final_success_rate_at_400': advanced_metrics['success_moving_average'][399] if len(advanced_metrics['success_moving_average']) > 399 else None,
            'final_success_rate': advanced_metrics['final_success_rate']
        }
        
        master_auc_path = os.path.join(self.output_dir, 'master_auc.json')
        with open(master_auc_path, 'w') as f:
            json.dump(master_auc, f, indent=2)
        
        # 学習曲線プロット
        self._plot_learning_curves(df)
        
        print(f"✅ 最終結果保存完了:")
        print(f"   CSV: {csv_path}")
        print(f"   統計: {json_path}")
        print(f"   Master AUC: {master_auc_path}")
        print(f"   学習エピソード数: {len(self.episode_data)}/{self.episode_count}")
        print(f"   最終成功率: {advanced_metrics['final_success_rate']:.1%}")
        print(f"   AUC(0-400): {advanced_metrics['auc_0_400']:.2f}")
        print(f"   AUC(全域): {advanced_metrics['auc_all']:.2f}")
        if advanced_metrics['time_to_70'] is not None:
            print(f"   Time-to-70%: Episode {advanced_metrics['time_to_70']}")
        if advanced_metrics['plateau_value'] is not None:
            print(f"   Plateau: {advanced_metrics['plateau_value']:.3f} (Episode {advanced_metrics['plateau_episode']})")
    
    def _plot_learning_curves(self, df):
        """学習曲線のプロット"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'TypeA DDPG Learning Curves ({self.experiment_type}, seed={self.seed})')
            
            episodes = df['episode'].values
            
            # 成功率（移動平均）
            axes[0, 0].plot(episodes, df['success_rate_ma100'], 'g-', linewidth=2, label='Success Rate (MA100)')
            axes[0, 0].axhline(y=0.7, color='r', linestyle='--', alpha=0.7, label='70% threshold')
            axes[0, 0].set_title('Success Rate (8-15N)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Success Rate')
            axes[0, 0].set_ylim(0, 1)
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 報酬
            axes[0, 1].plot(episodes, df['reward'], alpha=0.3, label='Raw')
            axes[0, 1].plot(episodes, df['reward_ma50'], 'r-', linewidth=2, label='MA50')
            axes[0, 1].set_title('Episode Rewards')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Reward')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 把持力誤差
            axes[1, 0].plot(episodes, df['force_error'], alpha=0.3)
            force_error_ma = AnalysisUtils.moving_average(df['force_error'].values, 50)
            axes[1, 0].plot(episodes, force_error_ma, 'orange', linewidth=2)
            axes[1, 0].set_title('Grip Force Error |F - F*|')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Error (N)')
            axes[1, 0].grid(True)
            
            # Q値とLoss
            if 'avg_q_value' in df.columns and df['avg_q_value'].notna().any():
                q_values = df['avg_q_value'].dropna()
                q_episodes = df.loc[df['avg_q_value'].notna(), 'episode']
                axes[1, 1].plot(q_episodes, q_values, 'b-', alpha=0.7, label='Avg Q-value')
                axes[1, 1].set_ylabel('Q-value', color='b')
                axes[1, 1].tick_params(axis='y', labelcolor='b')
                
                # Actor Loss（右軸）
                if 'actor_loss' in df.columns and df['actor_loss'].notna().any():
                    ax2 = axes[1, 1].twinx()
                    actor_losses = df['actor_loss'].dropna()
                    loss_episodes = df.loc[df['actor_loss'].notna(), 'episode']
                    ax2.plot(loss_episodes, actor_losses, 'r-', alpha=0.7, label='Actor Loss')
                    ax2.set_ylabel('Actor Loss', color='r')
                    ax2.tick_params(axis='y', labelcolor='r')
            
            axes[1, 1].set_title('Q-values & Actor Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            plot_path = os.path.join(self.output_dir, 'learning_curves.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 学習曲線保存: {plot_path}")
            
        except Exception as e:
            print(f"⚠️ プロット保存エラー: {e}")
    
    def stop_learning(self):
        """学習停止（即効パッチ適用済み）"""
        print(f"🛑 TypeA学習停止中...")
        
        self.is_running = False
        
        if self.episode_collector:
            self.episode_collector.stop_collection()
        
        if self.tcp_interface:
            self.tcp_interface.stop_server()
        
        # ★ パッチ5: 停止時の保存条件を「データの有無」で判定
        if len(self.episode_data) > 0:
            self._save_final_results()
        
        # 学習スレッドの終了を待つ（daemon=Falseなので）
        if self.learning_thread and self.learning_thread.is_alive():
            print(f"⏳ 学習スレッド終了待機中...")
            self.learning_thread.join(timeout=10)  # 最大10秒待機
            if self.learning_thread.is_alive():
                print(f"⚠️ 学習スレッドが10秒以内に終了しませんでした")
            else:
                print(f"✅ 学習スレッド正常終了")
        
        print(f"✅ TypeA学習停止完了")

# 集計スクリプト
def aggregate_multiple_seeds(base_dir, experiment_type, seeds):
    """複数シードの結果を集計"""
    print(f"📊 複数シード結果集計: {experiment_type}, seeds={seeds}")
    
    all_results = []
    
    for seed in seeds:
        # シードディレクトリ検索
        pattern = f"typea_{experiment_type}_seed{seed}_*"
        matching_dirs = list(Path(base_dir).glob(pattern))
        
        if not matching_dirs:
            print(f"⚠️ Seed {seed}の結果が見つかりません: {pattern}")
            continue
        
        # 最新のディレクトリを使用
        latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)
        master_auc_path = latest_dir / "master_auc.json"
        
        if master_auc_path.exists():
            with open(master_auc_path, 'r') as f:
                seed_result = json.load(f)
                seed_result['seed'] = seed
                all_results.append(seed_result)
            print(f"   Seed {seed}: 読み込み完了")
        else:
            print(f"⚠️ Seed {seed}: master_auc.json が見つかりません")
    
    if len(all_results) == 0:
        print(f"❌ 有効な結果が見つかりません")
        return
    
    # 統計計算
    metrics = ['auc_0_400', 'auc_all', 'final_success_rate', 'time_to_70', 'plateau_value']
    aggregated = {}
    
    for metric in metrics:
        values = [r[metric] for r in all_results if r.get(metric) is not None]
        if values:
            aggregated[f'{metric}_mean'] = np.mean(values)
            aggregated[f'{metric}_std'] = np.std(values)
            aggregated[f'{metric}_ci95'] = 1.96 * np.std(values) / np.sqrt(len(values))
            aggregated[f'{metric}_values'] = values
    
    aggregated['n_seeds'] = len(all_results)
    aggregated['seeds'] = [r['seed'] for r in all_results]
    aggregated['experiment_type'] = experiment_type
    
    # 結果保存
    output_path = Path(base_dir) / f"aggregated_{experiment_type}.json"
    with open(output_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"✅ 集計結果保存: {output_path}")
    
    # サマリー表示
    print(f"\n📈 {experiment_type} 集計結果 (n={len(all_results)}):")
    for metric in ['auc_0_400', 'auc_all', 'final_success_rate']:
        if f'{metric}_mean' in aggregated:
            mean_val = aggregated[f'{metric}_mean']
            ci_val = aggregated[f'{metric}_ci95']
            print(f"   {metric}: {mean_val:.3f} ± {ci_val:.3f}")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="TypeA DDPG学習システム（即効パッチ適用済み）")
    parser.add_argument("--type", choices=["A_400", "A_long"], default="A_400",
                       help="実験タイプ")
    parser.add_argument("--seed", type=int, default=42,
                       help="ランダムシード")
    parser.add_argument("--multi-seed", nargs="+", type=int,
                       help="複数シード実行 (例: --multi-seed 1 2 3 4 5)")
    parser.add_argument("--aggregate", action="store_true",
                       help="既存結果の集計のみ実行")
    
    args = parser.parse_args()
    
    print(f"🤖 TypeA DDPG学習システム（即効パッチ適用済み）")
    print(f"=" * 60)
    print(f"修正内容:")
    print(f"  1. 停止条件を「収集済みエピソード数」に基づかせる")
    print(f"  2. 実エピソード数でepisode_countを更新")
    print(f"  3. ループ終了時にフラグを落としてから保存")
    print(f"  4. スレッドをデーモンにしない")
    print(f"  5. 停止時の保存条件を「データの有無」で判定")
    print(f"=" * 60)
    
    base_output_dir = "DDPG_Python/logs"
    
    if args.aggregate:
        # 集計のみ実行
        if args.multi_seed:
            aggregate_multiple_seeds(base_output_dir, args.type, args.multi_seed)
        else:
            print(f"❌ --aggregate には --multi-seed が必要です")
        return
    
    if args.multi_seed:
        # 複数シード実行
        print(f"🔄 複数シード実行: {args.type}, seeds={args.multi_seed}")
        
        for seed in args.multi_seed:
            print(f"\n🌱 Seed {seed} 開始...")
            
            system = TypeADDPGSystem(experiment_type=args.type, seed=seed)
            
            if system.run_learning():
                try:
                    # 学習完了まで待機
                    while system.is_running and system.episode_count < system.target_episodes:
                        time.sleep(10)
                        if system.episode_count % 100 == 0 and system.episode_count > 0:
                            print(f"   Seed {seed}: {system.episode_count}/{system.target_episodes}ep")
                    
                    if system.episode_count >= system.target_episodes:
                        print(f"✅ Seed {seed} 完了!")
                    
                except KeyboardInterrupt:
                    print(f"⏹️ Seed {seed} 中断")
                finally:
                    system.stop_learning()
            else:
                print(f"❌ Seed {seed} 開始失敗")
        
        # 自動集計
        print(f"\n📊 自動集計実行...")
        aggregate_multiple_seeds(base_output_dir, args.type, args.multi_seed)
        
    else:
        # 単一シード実行
        system = TypeADDPGSystem(experiment_type=args.type, seed=args.seed)
        
        if system.run_learning():
            try:
                print(f"\n💡 TypeA学習実行中:")
                print(f"   実験タイプ: {args.type}")
                print(f"   シード: {args.seed}")
                print(f"   目標エピソード数: {system.target_episodes}")
                print(f"   Ctrl+C で終了")
                
                # 進捗モニタリング
                last_progress_time = time.time()
                while system.is_running and system.episode_count < system.target_episodes:
                    time.sleep(5)
                    
                    # 30秒ごとに進捗表示
                    current_time = time.time()
                    if current_time - last_progress_time >= 30:
                        print(f"🔄 進捗: {system.episode_count}/{system.target_episodes}エピソード")
                        last_progress_time = current_time
                
                if system.episode_count >= system.target_episodes:
                    print(f"🎉 目標エピソード数達成！")
                
            except KeyboardInterrupt:
                print(f"\n⏹️ 学習中断")
            finally:
                system.stop_learning()
        else:
            print(f"❌ 学習開始失敗")

if __name__ == "__main__":
    main()