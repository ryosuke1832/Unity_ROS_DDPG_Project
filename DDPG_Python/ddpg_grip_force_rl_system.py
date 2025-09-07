#!/usr/bin/env python3
"""
DDPG把持力強化学習システム

パターン1: リアルタイム学習モード - LSL/TCP データを受信してリアルタイムで学習
パターン2: 長期学習モード - 事前に作成したエージェントで自己学習

機能:
1. 分類機（grip_force_classifier.py）でEEGデータを3クラス分類
2. DDPGでTCP GripForceを最適化
3. unity_tcp_interface.pyのリクエストに把持力を応答
4. tcp_lsl_sync_systemでデータ同期・収集
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import threading
import queue
import json
import os
from collections import deque, namedtuple
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pickle

# 既存システムの活用
from tcp_lsl_sync_system import LSLTCPEpisodeCollector
from unity_tcp_interface import EEGTCPInterface
from grip_force_classifier import RealtimeGripForceClassifier, load_csv_data

# PyTorch設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 デバイス: {device}")

# 経験バッファ用のナップルタイプ
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    """DDPG Actor ネットワーク（把持力出力）"""
    
    def __init__(self, state_dim=5, action_dim=1, hidden_dim=128):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # He初期化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # [-1, 1]
        return x


class Critic(nn.Module):
    """DDPG Critic ネットワーク（Q値出力）"""
    
    def __init__(self, state_dim=5, action_dim=1, hidden_dim=128):
        super(Critic, self).__init__()
        
        # State pathway
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # Combined pathway (state + action)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # He初期化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class ReplayBuffer:
    """経験再生バッファ"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """経験を追加"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """バッチサンプリング"""
        experiences = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = torch.FloatTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """DDPG エージェント"""
    
    def __init__(self, 
                 state_dim=5, 
                 action_dim=1, 
                 lr_actor=1e-4, 
                 lr_critic=1e-3,
                 gamma=0.99,
                 tau=0.001,
                 noise_std=0.2,
                 buffer_capacity=100000):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        
        # ネットワーク
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        # ターゲットネットワークの初期化
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 最適化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 経験バッファ
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 統計情報
        self.training_step = 0
        self.episode_count = 0
        
        print(f"🤖 DDPG エージェント初期化完了")
        print(f"   状態次元: {state_dim}, 行動次元: {action_dim}")
        print(f"   学習率: Actor={lr_actor}, Critic={lr_critic}")
    
    def get_action(self, state, add_noise=True):
        """行動選択（把持力出力）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise
        
        # [-1, 1] → [2, 30] に変換（把持力範囲）
        grip_force = self._convert_action_to_grip_force(action[0])
        return grip_force, action[0]
    
    def _convert_action_to_grip_force(self, action_value):
        """行動値を把持力に変換"""
        # action_value: [-1, 1] → grip_force: [2, 30]
        grip_force = 2.0 + (action_value + 1.0) * (30.0 - 2.0) / 2.0
        return np.clip(grip_force, 2.0, 30.0)
    
    def _convert_grip_force_to_action(self, grip_force):
        """把持力を行動値に変換"""
        # grip_force: [2, 30] → action_value: [-1, 1]
        action_value = 2.0 * (grip_force - 2.0) / (30.0 - 2.0) - 1.0
        return np.clip(action_value, -1.0, 1.0)
    
    def update(self, batch_size=64):
        """ネットワーク更新"""
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # バッチサンプリング
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Critic更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards.unsqueeze(1) + self.gamma * target_q * (~dones).unsqueeze(1)
        
        current_q = self.critic(states, actions.unsqueeze(1))
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ターゲットネットワーク更新
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        self.training_step += 1
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, target, source):
        """ソフトアップデート"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def save_model(self, filepath):
        """モデル保存"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count
        }, filepath)
        print(f"💾 DDPGモデル保存: {filepath}")
    
    def load_model(self, filepath):
        """モデル読み込み"""
        checkpoint = torch.load(filepath, map_location=device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint.get('training_step', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        
        # ターゲットネットワーク更新
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        print(f"📂 DDPGモデル読み込み: {filepath}")
        print(f"   学習ステップ: {self.training_step}, エピソード: {self.episode_count}")


class GripForceEnvironmentState:
    """把持力環境の状態管理"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """状態リセット"""
        self.eeg_classification = 1  # 0: UnderGrip, 1: Success, 2: OverGrip
        self.previous_grip_force = 10.0
        self.tcp_grip_force = 0.0
        self.contact = False
        self.episode_step = 0
        return self.get_state()
    
    def get_state(self):
        """現在の状態ベクトルを取得"""
        state = np.array([
            self.eeg_classification / 2.0,  # [0, 1] に正規化
            self.previous_grip_force / 30.0,  # [0, 1] に正規化
            self.tcp_grip_force / 30.0,  # [0, 1] に正規化
            1.0 if self.contact else 0.0,
            self.episode_step / 100.0  # エピソード進行度
        ])
        return state
    
    def update(self, eeg_class, grip_force, tcp_data):
        """状態更新"""
        self.eeg_classification = eeg_class
        self.previous_grip_force = grip_force
        self.tcp_grip_force = tcp_data.get('grip_force', 0.0)
        self.contact = tcp_data.get('contact', False)
        self.episode_step += 1
        
        return self.get_state()
    
    def calculate_reward(self, eeg_class, grip_force, tcp_data):
        """報酬計算"""
        reward = 0.0
        
        # EEG分類に基づく報酬
        if eeg_class == 1:  # Success
            reward += 10.0
        elif eeg_class == 0:  # UnderGrip
            reward -= 5.0
        elif eeg_class == 2:  # OverGrip
            reward -= 8.0
        
        # TCP把持力との差に基づく報酬
        tcp_force = tcp_data.get('grip_force', 10.0)
        force_diff = abs(grip_force - tcp_force)
        reward -= force_diff * 0.5  # 差が大きいほどペナルティ
        
        # 接触状態の報酬
        if tcp_data.get('contact', False):
            reward += 2.0
        
        # 破損ペナルティ
        if tcp_data.get('broken', False):
            reward -= 15.0
        
        return reward


class DDPGGripForceSystem:
    """DDPG把持力強化学習システム"""
    
    def __init__(self,
                 classifier_model_path='models/best_grip_force_classifier.pth',
                 lsl_stream_name='MockEEG',
                 tcp_host='127.0.0.1',
                 tcp_port=12345):
        
        self.classifier_model_path = classifier_model_path
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        
        # 分類器初期化
        self.classifier = None
        self.init_classifier()
        
        # DDPGエージェント
        self.agent = DDPGAgent(state_dim=5, action_dim=1)
        
        # 環境状態
        self.env_state = GripForceEnvironmentState()
        
        # データ収集システム
        self.data_collector = None
        self.tcp_interface = None
        
        # 学習モード
        self.learning_mode = None  # 'realtime' or 'self_training'
        self.is_running = False
        
        # 統計情報
        self.stats = {
            'total_episodes': 0,
            'total_rewards': [],
            'classification_accuracy': [],
            'grip_force_history': [],
            'start_time': None
        }
        
        # モデル保存パス
        self.model_save_dir = "models/ddpg"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print(f"🚀 DDPG把持力強化学習システム初期化完了")
    
    def init_classifier(self):
        """分類器の初期化"""
        try:
            if os.path.exists(self.classifier_model_path):
                self.classifier = RealtimeGripForceClassifier(
                    model_path=self.classifier_model_path,
                    lsl_stream_name=self.lsl_stream_name,
                    tcp_host=self.tcp_host,
                    tcp_port=self.tcp_port
                )
                print(f"✅ 分類器読み込み成功: {self.classifier_model_path}")
            else:
                print(f"⚠️ 分類器モデルが見つかりません: {self.classifier_model_path}")
                print(f"   grip_force_classifier.py で事前に学習してください")
        except Exception as e:
            print(f"❌ 分類器初期化エラー: {e}")
    
    def start_realtime_learning_mode(self):
        """パターン1: リアルタイム学習モード"""
        print(f"🔴 パターン1: リアルタイム学習モード開始")
        
        if not self.classifier:
            print(f"❌ 分類器が利用できません")
            return False
        
        self.learning_mode = 'realtime'
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # データ収集システム初期化
        self.data_collector = LSLTCPEpisodeCollector(
            lsl_stream_name=self.lsl_stream_name,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port,
            save_to_csv=True
        )
        
        # TCP応答システム初期化
        self.tcp_interface = EEGTCPInterface(
            host=self.tcp_host,
            port=self.tcp_port + 1  # 別ポートで応答
        )
        
        # コールバック設定
        self.tcp_interface.add_message_callback(self._on_grip_force_request)
        
        # システム開始
        if not self.data_collector.start_collection():
            print(f"❌ データ収集開始失敗")
            return False
        
        if not self.tcp_interface.start_server():
            print(f"❌ TCP応答サーバー開始失敗")
            return False
        
        # 分類器開始
        if not self.classifier.start_classification():
            print(f"❌ 分類器開始失敗")
            return False
        
        # リアルタイム学習スレッド
        learning_thread = threading.Thread(target=self._realtime_learning_loop, daemon=True)
        learning_thread.start()
        
        print(f"✅ リアルタイム学習モード開始完了")
        return True
    
    def start_self_training_mode(self, pretrained_model_path=None):
        """パターン2: 長期学習モード（自己学習）"""
        print(f"🔵 パターン2: 長期学習モード開始")
        
        self.learning_mode = 'self_training'
        self.is_running = True
        self.stats['start_time'] = time.time()
        
        # 事前学習モデル読み込み
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            self.agent.load_model(pretrained_model_path)
            print(f"📂 事前学習モデル読み込み: {pretrained_model_path}")
        
        # 自己学習スレッド
        self_training_thread = threading.Thread(target=self._self_training_loop, daemon=True)
        self_training_thread.start()
        
        print(f"✅ 長期学習モード開始完了")
        return True
    
    def _on_grip_force_request(self, message_data):
        """把持力リクエストへの応答"""
        try:
            # 現在の状態を取得
            current_state = self.env_state.get_state()
            
            # DDPGエージェントから把持力を取得
            grip_force, action_value = self.agent.get_action(current_state, add_noise=False)
            
            # 応答送信
            response = {
                'type': 'grip_force_command',
                'target_force': round(grip_force, 2),
                'timestamp': time.time(),
                'agent_action': round(action_value, 3),
                'learning_mode': self.learning_mode
            }
            
            self.tcp_interface.send_message(response)
            
            # 統計更新
            self.stats['grip_force_history'].append(grip_force)
            
            print(f"🎯 把持力応答: {grip_force:.2f}N (行動値: {action_value:.3f})")
            
        except Exception as e:
            print(f"❌ 把持力応答エラー: {e}")
    
    def _realtime_learning_loop(self):
        """リアルタイム学習ループ"""
        print(f"🔄 リアルタイム学習ループ開始")
        
        while self.is_running:
            try:
                # エピソード収集待機
                if self.data_collector.episodes:
                    # 新しいエピソードを処理
                    episode = self.data_collector.episodes.pop(0)
                    self._process_episode_for_learning(episode)
                
                time.sleep(0.1)  # 100ms間隔
                
            except Exception as e:
                print(f"⚠️ リアルタイム学習エラー: {e}")
                time.sleep(1.0)
        
        print(f"🔄 リアルタイム学習ループ終了")
    
    def _process_episode_for_learning(self, episode):
        """エピソードを学習に使用"""
        try:
            # EEG分類
            eeg_data = episode.lsl_data  # (300, 32)
            eeg_class, confidence = self._classify_eeg_data(eeg_data)
            
            # 状態更新
            tcp_data = episode.tcp_data
            grip_force = tcp_data.get('grip_force', 10.0)
            
            # 前の状態
            prev_state = self.env_state.get_state()
            
            # 状態更新
            current_state = self.env_state.update(eeg_class, grip_force, tcp_data)
            
            # 報酬計算
            reward = self.env_state.calculate_reward(eeg_class, grip_force, tcp_data)
            
            # 行動値変換
            action_value = self.agent._convert_grip_force_to_action(grip_force)
            
            # 経験をバッファに追加
            done = tcp_data.get('broken', False)
            self.agent.replay_buffer.push(
                prev_state, action_value, reward, current_state, done
            )
            
            # 学習実行
            if len(self.agent.replay_buffer) > 64:
                critic_loss, actor_loss = self.agent.update()
                
                if critic_loss is not None:
                    print(f"📈 学習: EP={episode.episode_id}, "
                          f"分類={['Under', 'Success', 'Over'][eeg_class]}, "
                          f"報酬={reward:.2f}, "
                          f"Critic損失={critic_loss:.4f}")
            
            # 統計更新
            self.stats['total_episodes'] += 1
            self.stats['total_rewards'].append(reward)
            self.stats['classification_accuracy'].append(1.0 if eeg_class == 1 else 0.0)
            
            # 定期的なモデル保存
            if self.stats['total_episodes'] % 50 == 0:
                self._save_checkpoint()
                
        except Exception as e:
            print(f"⚠️ エピソード学習エラー: {e}")
    
    def _classify_eeg_data(self, eeg_data):
        """EEGデータの分類"""
        try:
            if self.classifier and hasattr(self.classifier, 'classify_epoch'):
                return self.classifier.classify_epoch(eeg_data)
            else:
                # ダミー分類（テスト用）
                return random.randint(0, 2), 0.33
        except Exception as e:
            print(f"⚠️ EEG分類エラー: {e}")
            return 1, 0.5  # デフォルトはSuccess
    
    def _self_training_loop(self):
        """自己学習ループ（シミュレーション環境）"""
        print(f"🔄 自己学習ループ開始")
        
        episode_count = 0
        
        while self.is_running:
            try:
                episode_count += 1
                episode_reward = 0.0
                episode_steps = 0
                
                # エピソード初期化
                state = self.env_state.reset()
                
                # エピソード実行
                for step in range(100):  # 最大100ステップ
                    # 行動選択
                    grip_force, action_value = self.agent.get_action(state, add_noise=True)
                    
                    # シミュレーション環境でのステップ
                    next_state, reward, done = self._simulate_environment_step(
                        state, grip_force, action_value
                    )
                    
                    # 経験バッファに追加
                    self.agent.replay_buffer.push(
                        state, action_value, reward, next_state, done
                    )
                    
                    episode_reward += reward
                    episode_steps += 1
                    state = next_state
                    
                    # 学習実行
                    if len(self.agent.replay_buffer) > 64:
                        critic_loss, actor_loss = self.agent.update()
                    
                    if done:
                        break
                
                # エピソード完了
                self.stats['total_episodes'] += 1
                self.stats['total_rewards'].append(episode_reward)
                
                if episode_count % 10 == 0:
                    avg_reward = np.mean(self.stats['total_rewards'][-10:])
                    print(f"🎮 自己学習 EP={episode_count}, "
                          f"報酬={episode_reward:.2f}, "
                          f"平均報酬={avg_reward:.2f}, "
                          f"ステップ={episode_steps}")
                
                # 定期的なモデル保存
                if episode_count % 100 == 0:
                    self._save_checkpoint()
                
                time.sleep(0.1)  # 短い休憩
                
            except Exception as e:
                print(f"⚠️ 自己学習エラー: {e}")
                time.sleep(1.0)
        
        print(f"🔄 自己学習ループ終了")
    
    def _simulate_environment_step(self, state, grip_force, action_value):
        """シミュレーション環境でのステップ実行"""
        # ダミーのシミュレーション（実際の環境では物理シミュレーション）
        
        # EEG分類をシミュレート
        target_force = 12.0  # 目標把持力
        force_error = abs(grip_force - target_force)
        
        if force_error < 2.0:
            eeg_class = 1  # Success
        elif grip_force < target_force:
            eeg_class = 0  # UnderGrip
        else:
            eeg_class = 2  # OverGrip
        
        # ダミーTCPデータ
        tcp_data = {
            'grip_force': grip_force + random.uniform(-1.0, 1.0),
            'contact': random.random() > 0.3,
            'broken': force_error > 8.0
        }
        
        # 状態更新
        next_state = self.env_state.update(eeg_class, grip_force, tcp_data)
        
        # 報酬計算
        reward = self.env_state.calculate_reward(eeg_class, grip_force, tcp_data)
        
        # 終了条件
        done = tcp_data['broken'] or self.env_state.episode_step > 50
        
        return next_state, reward, done
    
    def _save_checkpoint(self):
        """チェックポイント保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # DDPGモデル保存
        model_path = os.path.join(self.model_save_dir, f"ddpg_agent_{timestamp}.pth")
        self.agent.save_model(model_path)
        
        # 統計情報保存
        stats_path = os.path.join(self.model_save_dir, f"training_stats_{timestamp}.pkl")
        with open(stats_path, 'wb') as f:
            pickle.dump(self.stats, f)
        
        print(f"💾 チェックポイント保存: {model_path}")
    
    def stop_learning(self):
        """学習停止"""
        print(f"⏹️ DDPG学習システム停止中...")
        
        self.is_running = False
        
        # データ収集システム停止
        if self.data_collector:
            self.data_collector.stop_collection()
        
        # TCP応答システム停止
        if self.tcp_interface:
            self.tcp_interface.stop_server()
        
        # 分類器停止
        if self.classifier:
            self.classifier.stop_classification()
        
        # 最終チェックポイント保存
        self._save_checkpoint()
        
        # 統計表示
        self._print_final_statistics()
        
        print(f"✅ DDPG学習システム停止完了")
    
    def _print_final_statistics(self):
        """最終統計表示"""
        print(f"\n📊 DDPG学習システム統計:")
        print(f"=" * 60)
        print(f"   学習モード: {self.learning_mode}")
        print(f"   総エピソード数: {self.stats['total_episodes']}")
        
        if self.stats['total_rewards']:
            print(f"   平均報酬: {np.mean(self.stats['total_rewards']):.3f}")
            print(f"   最高報酬: {np.max(self.stats['total_rewards']):.3f}")
            print(f"   最新10エピソード平均: {np.mean(self.stats['total_rewards'][-10:]):.3f}")
        
        if self.stats['classification_accuracy']:
            accuracy = np.mean(self.stats['classification_accuracy']) * 100
            print(f"   EEG分類精度: {accuracy:.1f}%")
        
        if self.stats['grip_force_history']:
            print(f"   平均把持力: {np.mean(self.stats['grip_force_history']):.2f}N")
            print(f"   把持力範囲: {np.min(self.stats['grip_force_history']):.2f} - {np.max(self.stats['grip_force_history']):.2f}N")
        
        if self.stats['start_time']:
            training_time = time.time() - self.stats['start_time']
            print(f"   学習時間: {training_time:.1f}秒")
        
        print(f"   DDPGエージェント学習ステップ: {self.agent.training_step}")
    
    def test_agent(self, num_episodes=10):
        """エージェントのテスト実行"""
        print(f"🧪 DDPG エージェントテスト開始 ({num_episodes}エピソード)")
        
        test_rewards = []
        
        for episode in range(num_episodes):
            state = self.env_state.reset()
            episode_reward = 0.0
            
            for step in range(50):
                # ノイズなしで行動選択
                grip_force, action_value = self.agent.get_action(state, add_noise=False)
                
                # シミュレーションステップ
                next_state, reward, done = self._simulate_environment_step(
                    state, grip_force, action_value
                )
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            test_rewards.append(episode_reward)
            print(f"   テストエピソード {episode+1}: 報酬={episode_reward:.2f}")
        
        avg_test_reward = np.mean(test_rewards)
        print(f"✅ テスト完了 - 平均報酬: {avg_test_reward:.3f}")
        
        return avg_test_reward


def main():
    """メイン実行関数"""
    print(f"🚀 DDPG把持力強化学習システム")
    print(f"=" * 60)
    print(f"パターン1: リアルタイム学習モード（LSL/TCP連携）")
    print(f"パターン2: 長期学習モード（自己学習）")
    print(f"")
    
    # システム初期化
    ddpg_system = DDPGGripForceSystem(
        classifier_model_path='models/best_grip_force_classifier.pth',
        lsl_stream_name='MockEEG',
        tcp_host='127.0.0.1',
        tcp_port=12345
    )
    
    # モード選択
    print(f"学習モードを選択してください:")
    print(f"1. リアルタイム学習モード（LSL/TCP連携）")
    print(f"2. 長期学習モード（自己学習）")
    print(f"3. エージェントテスト")
    print(f"4. 既存モデル読み込み + リアルタイム学習")
    print(f"5. 既存モデル読み込み + 長期学習")
    
    choice = input(f"選択 (1-5): ").strip()
    
    try:
        if choice == "1":
            # パターン1: リアルタイム学習
            print(f"\n🔴 リアルタイム学習モード開始")
            
            if ddpg_system.start_realtime_learning_mode():
                print(f"💡 システム稼働中...")
                print(f"   LSL/TCPデータを受信して学習実行")
                print(f"   Unity からの把持力リクエストに自動応答")
                print(f"   Ctrl+C で終了")
                
                try:
                    while ddpg_system.is_running:
                        time.sleep(1.0)
                        
                        # 定期的な統計表示
                        if ddpg_system.stats['total_episodes'] > 0 and ddpg_system.stats['total_episodes'] % 20 == 0:
                            recent_rewards = ddpg_system.stats['total_rewards'][-10:]
                            if recent_rewards:
                                avg_reward = np.mean(recent_rewards)
                                print(f"📈 最新10エピソード平均報酬: {avg_reward:.3f}")
                        
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    ddpg_system.stop_learning()
            else:
                print(f"❌ リアルタイム学習モード開始失敗")
        
        elif choice == "2":
            # パターン2: 長期学習（自己学習）
            print(f"\n🔵 長期学習モード開始")
            
            if ddpg_system.start_self_training_mode():
                print(f"💡 自己学習実行中...")
                print(f"   シミュレーション環境でエージェント学習")
                print(f"   Ctrl+C で終了")
                
                try:
                    while ddpg_system.is_running:
                        time.sleep(5.0)
                        
                        # 定期的な進捗表示
                        if ddpg_system.stats['total_episodes'] > 0:
                            recent_rewards = ddpg_system.stats['total_rewards'][-20:]
                            if recent_rewards:
                                avg_reward = np.mean(recent_rewards)
                                print(f"📈 最新20エピソード平均報酬: {avg_reward:.3f} "
                                      f"(総エピソード: {ddpg_system.stats['total_episodes']})")
                        
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    ddpg_system.stop_learning()
            else:
                print(f"❌ 長期学習モード開始失敗")
        
        elif choice == "3":
            # エージェントテスト
            print(f"\n🧪 エージェントテスト")
            model_path = input(f"テスト用モデルパス (空でデフォルト): ").strip()
            
            if model_path and os.path.exists(model_path):
                ddpg_system.agent.load_model(model_path)
                print(f"📂 モデル読み込み: {model_path}")
            
            ddpg_system.test_agent(num_episodes=20)
        
        elif choice == "4":
            # 既存モデル + リアルタイム学習
            print(f"\n🔴 既存モデル読み込み + リアルタイム学習")
            model_path = input(f"既存DDPGモデルパス: ").strip()
            
            if not model_path or not os.path.exists(model_path):
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                return
            
            ddpg_system.agent.load_model(model_path)
            
            if ddpg_system.start_realtime_learning_mode():
                try:
                    while ddpg_system.is_running:
                        time.sleep(1.0)
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    ddpg_system.stop_learning()
        
        elif choice == "5":
            # 既存モデル + 長期学習
            print(f"\n🔵 既存モデル読み込み + 長期学習")
            model_path = input(f"既存DDPGモデルパス: ").strip()
            
            if not model_path or not os.path.exists(model_path):
                print(f"❌ モデルファイルが見つかりません: {model_path}")
                return
            
            if ddpg_system.start_self_training_mode(pretrained_model_path=model_path):
                try:
                    while ddpg_system.is_running:
                        time.sleep(5.0)
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    ddpg_system.stop_learning()
        
        else:
            print(f"❌ 無効な選択です")
    
    except Exception as e:
        print(f"❌ システムエラー: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n👋 DDPG把持力強化学習システム終了")


# 単体テスト用関数
def test_ddpg_components():
    """DDPGコンポーネントのテスト"""
    print(f"🧪 DDPGコンポーネントテスト")
    
    # エージェントテスト
    print(f"1. DDPGエージェントテスト")
    agent = DDPGAgent(state_dim=5, action_dim=1)
    
    # ダミー状態で行動選択
    dummy_state = np.random.randn(5)
    grip_force, action_value = agent.get_action(dummy_state)
    print(f"   ダミー状態: {dummy_state}")
    print(f"   出力把持力: {grip_force:.2f}N")
    print(f"   行動値: {action_value:.3f}")
    
    # 経験バッファテスト
    print(f"2. 経験バッファテスト")
    for i in range(10):
        state = np.random.randn(5)
        action = np.random.randn()
        reward = np.random.randn()
        next_state = np.random.randn(5)
        done = random.random() > 0.8
        
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    print(f"   バッファサイズ: {len(agent.replay_buffer)}")
    
    # 学習テスト
    if len(agent.replay_buffer) >= 8:
        print(f"3. 学習テスト")
        critic_loss, actor_loss = agent.update(batch_size=8)
        print(f"   Critic損失: {critic_loss:.4f}")
        print(f"   Actor損失: {actor_loss:.4f}")
    
    # 環境状態テスト
    print(f"4. 環境状態テスト")
    env_state = GripForceEnvironmentState()
    state = env_state.reset()
    print(f"   初期状態: {state}")
    
    # 状態更新テスト
    dummy_tcp_data = {
        'grip_force': 12.5,
        'contact': True,
        'broken': False
    }
    new_state = env_state.update(1, 11.0, dummy_tcp_data)
    reward = env_state.calculate_reward(1, 11.0, dummy_tcp_data)
    print(f"   更新後状態: {new_state}")
    print(f"   計算報酬: {reward:.2f}")
    
    print(f"✅ DDPGコンポーネントテスト完了")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_ddpg_components()
    else:
        main()