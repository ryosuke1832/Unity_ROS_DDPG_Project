#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDPG リアルタイムフィードバック学習システム

既存モジュールを統合:
1. e_tcp_lsl_sync_system.py - LSL/TCPデータ受信
2. g_grip_force_realtime_classifier.py - EEG分類
3. DDPGエージェントによる把持力最適化学習
4. Unity側へのリアルタイムフィードバック送信

フロー:
Unity → TCP/LSLデータ → EEG分類 → DDPG学習 → 把持力フィードバック → Unity
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
from datetime import datetime
from collections import deque, namedtuple
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt

# 既存モジュールをインポート
from e_tcp_lsl_sync_system import LSLTCPEpisodeCollector, Episode
from g_grip_force_realtime_classifier import RealtimeGripForceClassifier
from c_unity_tcp_interface import EEGTCPInterface

# PyTorch設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 使用デバイス: {device}")

# 経験バッファ用のnamedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DDPGActor(nn.Module):
    """DDPG アクターネットワーク（把持力を出力）"""
    
    def __init__(self, state_dim=7, action_dim=1, hidden_dim=256, max_action=1.0):
        super(DDPGActor, self).__init__()
        
        self.max_action = max_action
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Layer Normalization (BatchNormの代替、単一サンプル対応)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout(x)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.tanh(self.fc3(x))  # [-1, 1]の範囲に正規化
        return x * self.max_action

class DDPGCritic(nn.Module):
    """DDPG クリティックネットワーク（Q値を出力）"""
    
    def __init__(self, state_dim=7, action_dim=1, hidden_dim=256):
        super(DDPGCritic, self).__init__()
        
        # State pathway
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)  # LayerNorm使用
        
        # Combined pathway (state + action)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)  # LayerNorm使用
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 重み初期化
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state, action):
        x = F.relu(self.ln1(self.fc1(state)))
        x = self.dropout(x)
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout(x)
        q_value = self.fc3(x)
        return q_value

class OUNoise:
    """Ornstein-Uhlenbeck ノイズ（探索用）"""
    
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
    
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """経験を追加"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """バッチサンプリング"""
        if len(self.buffer) < batch_size:
            return None
        
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    """DDPGエージェント"""
    
    def __init__(self, state_dim=7, action_dim=1, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.005):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # ネットワーク初期化
        self.actor = DDPGActor(state_dim, action_dim).to(device)
        self.actor_target = DDPGActor(state_dim, action_dim).to(device)
        self.critic = DDPGCritic(state_dim, action_dim).to(device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(device)
        
        # ターゲットネットワークの初期化
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)
        
        # オプティマイザ
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 経験再生バッファ
        self.memory = ReplayBuffer()
        
        # ノイズ
        self.noise = OUNoise(action_dim)
        
        # 学習統計
        self.actor_losses = []
        self.critic_losses = []
    
    def hard_update(self, target, source):
        """ハードアップデート（完全コピー）"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """ソフトアップデート（徐々に更新）"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """アクション選択"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        # 推論モードに設定（BatchNorm対応）
        self.actor.eval()
        
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        
        # 学習モードに戻す
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample() * noise_scale
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update(self, batch_size=64):
        """ネットワーク更新"""
        sample = self.memory.sample(batch_size)
        if sample is None:
            return
        
        # 学習モードに設定
        self.actor.train()
        self.critic.train()
        
        state, action, reward, next_state, done = sample
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        
        # Criticの更新
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_q = self.critic_target(next_state, next_action)
            target_q = reward + (self.gamma * target_q * (1 - done))
        
        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target_q)
        
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
        
        # ターゲットネットワーク更新
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        # 統計記録
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())

class GripForceEnvironment:
    """把持力環境（状態管理・報酬計算）"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """環境リセット"""
        self.episode_count = 0
        self.success_count = 0
        self.total_reward = 0.0
        self.previous_action = 0.0
    
    def create_state(self, classification_result, tcp_data, previous_action=0.0):
        """
        状態ベクトル作成
        
        Returns:
            state: [class_onehot(3), grip_force(1), contact(1), broken(1), prev_action(1)] = 7次元
        """
        # 分類結果をワンホットエンコード
        class_onehot = [0, 0, 0]
        if classification_result is not None:
            if isinstance(classification_result, dict):
                class_idx = classification_result.get('predicted_class_idx', 0)
            else:
                class_idx = classification_result
            
            if 0 <= class_idx <= 2:
                class_onehot[class_idx] = 1
        
        # TCP データから特徴量抽出
        grip_force = tcp_data.get('grip_force', 10.0) / 30.0  # 正規化 [0-30N] -> [0-1]
        contact = 1.0 if tcp_data.get('contact', False) else 0.0
        broken = 1.0 if tcp_data.get('broken', False) else 0.0
        
        # 状態ベクトル作成
        state = np.array(class_onehot + [grip_force, contact, broken, previous_action], dtype=np.float32)
        return state
    
    def calculate_reward(self, classification_result, tcp_data, action_value):
        """
        報酬計算
        
        Args:
            classification_result: EEG分類結果
            tcp_data: TCPデータ（ロボット状態）
            action_value: DDPGアクションの値
            
        Returns:
            reward: 計算された報酬値
        """
        reward = 0.0
        
        # 分類結果に基づく基本報酬
        if isinstance(classification_result, dict):
            predicted_class = classification_result.get('predicted_class', 'Success')
            confidence = classification_result.get('confidence', 0.5)
        else:
            class_names = ['UnderGrip', 'Success', 'OverGrip']
            predicted_class = class_names[classification_result] if 0 <= classification_result <= 2 else 'Success'
            confidence = 0.8
        
        # クラス別報酬
        if predicted_class == 'Success':
            reward += 10.0 * confidence  # 成功に対する高い報酬
            self.success_count += 1
        elif predicted_class == 'UnderGrip':
            reward -= 3.0 * confidence   # 軽い把持不足ペナルティ
        elif predicted_class == 'OverGrip':
            reward -= 8.0 * confidence   # 重い過度把持ペナルティ
        
        # 接触成功報酬
        if tcp_data.get('contact', False):
            reward += 5.0
            # 接触力に基づく追加報酬
            contact_force = tcp_data.get('contact_force', 0)
            if 0 < contact_force < 20:  # 適度な接触力
                reward += 2.0
        else:
            reward -= 2.0  # 接触失敗ペナルティ
        
        # 破損ペナルティ
        if tcp_data.get('broken', False):
            reward -= 20.0  # 重いペナルティ
        
        # 把持力の適切性を評価
        actual_grip_force = tcp_data.get('grip_force', 10.0)
        
        # 目標範囲（8-15N）への近さを報酬化
        target_min, target_max = 8.0, 15.0
        if target_min <= actual_grip_force <= target_max:
            reward += 3.0  # 適切範囲報酬
        else:
            # 範囲外の距離に応じたペナルティ
            if actual_grip_force < target_min:
                distance = target_min - actual_grip_force
            else:
                distance = actual_grip_force - target_max
            reward -= distance * 0.5
        
        # アクションのスムーズさ報酬（急激な変化を抑制）
        action_change = abs(action_value - self.previous_action)
        if action_change > 0.5:
            reward -= action_change * 0.5
        
        # 統計更新
        self.total_reward += reward
        self.previous_action = action_value
        
        return reward

class DDPGRealtimeFeedbackSystem:
    """DDPG リアルタイムフィードバックシステム統合クラス"""
    
    def __init__(self, 
                 model_path='models/improved_grip_force_classifier_*.pth',
                 lsl_stream_name='MockEEG',
                 tcp_host='127.0.0.1',
                 tcp_port=12345,
                 feedback_port=12346):
        
        self.model_path = model_path
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.feedback_port = feedback_port
        
        # サブシステム初期化
        self.init_data_collector()
        self.init_classifier()
        self.init_feedback_interface()
        
        # DDPGエージェント
        self.agent = DDPGAgent(state_dim=7, action_dim=1, lr_actor=1e-4, lr_critic=1e-3)
        
        # 環境
        self.environment = GripForceEnvironment()
        
        # フィードバックループ管理
        self.is_running = False
        self.learning_thread = None
        
        # 学習統計
        self.stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_rewards': [],
            'episode_rewards': [],
            'classification_accuracy': [],
            'grip_force_history': [],
            'start_time': None,
            'learning_updates': 0
        }
        
        # 状態管理
        self.previous_state = None
        self.previous_action = None
        self.current_episode_reward = 0.0
        
        # モデル保存
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = f"models/ddpg_realtime_{self.session_id}"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print(f"🚀 DDPG リアルタイムフィードバックシステム初期化完了")
        print(f"   セッションID: {self.session_id}")
        print(f"   モデル保存先: {self.model_save_dir}")
    
    def init_data_collector(self):
        """データ収集システム初期化"""
        self.data_collector = LSLTCPEpisodeCollector(
            lsl_stream_name=self.lsl_stream_name,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port,
            save_to_csv=False  # リアルタイム学習では無効
        )
        print(f"✅ データ収集システム初期化完了")
    
    def init_classifier(self):
        """分類機初期化"""
        self.classifier = RealtimeGripForceClassifier(
            model_path=self.model_path,
            lsl_stream_name=self.lsl_stream_name,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port
        )
        
        if not self.classifier.load_model():
            print(f"⚠️ 分類機読み込み失敗 - デモモードで続行")
            self.classifier = None
        else:
            print(f"✅ 分類機初期化完了")
    
    def init_feedback_interface(self):
        """フィードバック通信インターフェース初期化"""
        self.feedback_interface = EEGTCPInterface(
            host=self.tcp_host,
            port=self.feedback_port
        )
        
        # 把持力リクエストコールバック設定
        self.feedback_interface.add_message_callback(self.handle_grip_force_request)
        
        print(f"✅ フィードバック通信初期化完了 (Port: {self.feedback_port})")
    
    def handle_grip_force_request(self, message_data):
        """Unity側からの把持力リクエスト処理"""
        try:
            if message_data.get('type') == 'grip_force_request':
                # 現在の状態から最適な把持力を計算
                if self.previous_state is not None:
                    # DDPGエージェントからアクション取得（ノイズなし、推論用）
                    action = self.agent.select_action(self.previous_state, add_noise=False, noise_scale=0.0)
                    
                    # アクションを把持力に変換 [-1,1] -> [5,25]N
                    grip_force = self.action_to_grip_force(action[0])
                    
                    print(f"🎯 DDPG把持力フィードバック: {grip_force:.2f}N (action: {action[0]:.3f})")
                    
                else:
                    # 状態がない場合はデフォルト値
                    grip_force = 12.0
                    print(f"⚠️ 初期状態 - デフォルト把持力: {grip_force}N")
                
                # フィードバック送信
                response = {
                    'type': 'grip_force_command',
                    'target_force': float(grip_force),
                    'timestamp': time.time(),
                    'session_id': f"ddpg_rt_{self.session_id}",
                    'learning_episode': self.stats['total_episodes']
                }
                self.feedback_interface.send_message(response)
                
                # 統計更新
                self.stats['grip_force_history'].append(grip_force)
                
        except Exception as e:
            print(f"❌ 把持力リクエスト処理エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def action_to_grip_force(self, action_value):
        """DDPGアクション値を把持力に変換"""
        # [-1, 1] -> [5, 25]N の範囲でマッピング
        min_force, max_force = 5.0, 25.0
        grip_force = (action_value + 1.0) / 2.0 * (max_force - min_force) + min_force
        return np.clip(grip_force, min_force, max_force)
    
    def grip_force_to_action(self, grip_force):
        """把持力をDDPGアクション値に変換"""
        # [5, 25]N -> [-1, 1] の範囲でマッピング
        min_force, max_force = 5.0, 25.0
        action_value = 2.0 * (grip_force - min_force) / (max_force - min_force) - 1.0
        return np.clip(action_value, -1.0, 1.0)
    
    def classify_episode_data(self, episode):
        """エピソードデータをEEG分類"""
        try:
            if self.classifier:
                return self.classifier.classify_episode(episode)
            else:
                # デモ用のランダム分類
                classes = ['UnderGrip', 'Success', 'OverGrip']
                random_class = np.random.choice(classes, p=[0.3, 0.5, 0.2])
                class_idx = classes.index(random_class)
                return {
                    'predicted_class': random_class,
                    'predicted_class_idx': class_idx,
                    'confidence': np.random.uniform(0.6, 0.9)
                }
        except Exception as e:
            print(f"⚠️ EEG分類エラー: {e}")
            return {
                'predicted_class': 'Success',
                'predicted_class_idx': 1,
                'confidence': 0.5
            }
    
    def start_learning(self):
        """リアルタイム学習開始"""
        print(f"🔴 DDPGリアルタイム学習開始")
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.environment.reset()
        
        # データ収集開始
        if not self.data_collector.start_collection():
            print(f"❌ データ収集開始失敗")
            return False
        
        # フィードバック通信開始
        if not self.feedback_interface.start_server():
            print(f"❌ フィードバック通信開始失敗")
            return False
        
        # 学習ループスレッド開始
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        print(f"✅ DDPGリアルタイム学習開始完了")
        print(f"💡 Unity側で以下を実行してください:")
        print(f"   1. ロボット状態データ送信 (TCP Port {self.tcp_port})")
        print(f"   2. EPISODE_END トリガー送信")
        print(f"   3. 把持力リクエスト送信 (TCP Port {self.feedback_port})")
        return True
    
    def _learning_loop(self):
        """学習ループ（バックグラウンド実行）"""
        print(f"🔄 学習ループ開始")
        
        while self.is_running:
            try:
                # 新しいエピソードをチェック
                if len(self.data_collector.episodes) > self.stats['total_episodes']:
                    # 最新エピソード取得
                    latest_episode = self.data_collector.episodes[self.stats['total_episodes']]
                    
                    print(f"🆕 新エピソード受信: Episode {latest_episode.episode_id}")
                    
                    # EEG分類実行
                    classification_result = self.classify_episode_data(latest_episode)
                    
                    print(f"🧠 EEG分類結果: {classification_result['predicted_class']} "
                          f"(信頼度: {classification_result['confidence']:.3f})")
                    
                    # 現在の状態作成
                    current_state = self.environment.create_state(
                        classification_result, 
                        latest_episode.tcp_data, 
                        self.previous_action[0] if self.previous_action is not None else 0.0
                    )
                    
                    # 前エピソードの経験を処理
                    if (self.previous_state is not None and 
                        self.previous_action is not None):
                        
                        # 報酬計算
                        reward = self.environment.calculate_reward(
                            classification_result, 
                            latest_episode.tcp_data, 
                            self.previous_action[0]
                        )
                        
                        self.current_episode_reward += reward
                        
                        # エピソード終了判定
                        done = latest_episode.tcp_data.get('broken', False)
                        
                        # 経験バッファに追加
                        self.agent.memory.push(
                            self.previous_state, 
                            self.previous_action, 
                            reward, 
                            current_state, 
                            done
                        )
                        
                        print(f"📈 報酬: {reward:.2f}, 累積報酬: {self.current_episode_reward:.2f}")
                        
                        # ネットワーク更新（十分な経験がある場合）
                        if len(self.agent.memory) >= 64:
                            self.agent.update(batch_size=64)
                            self.stats['learning_updates'] += 1
                            
                            if self.stats['learning_updates'] % 10 == 0:
                                print(f"🎓 学習更新: {self.stats['learning_updates']}回")
                        
                        # エピソード終了処理
                        if done or self.stats['total_episodes'] % 10 == 0:
                            self.stats['episode_rewards'].append(self.current_episode_reward)
                            if classification_result['predicted_class'] == 'Success':
                                self.stats['successful_episodes'] += 1
                            
                            print(f"📊 エピソード{self.stats['total_episodes']}完了: "
                                  f"累積報酬={self.current_episode_reward:.2f}")
                            
                            self.current_episode_reward = 0.0
                            self.agent.noise.reset()  # ノイズリセット
                    
                    # 次のアクション選択（探索用ノイズ付き）
                    current_action = self.agent.select_action(current_state, add_noise=True, noise_scale=0.2)
                    
                    # 状態更新
                    self.previous_state = current_state
                    self.previous_action = current_action
                    self.stats['total_episodes'] += 1
                    
                    # 定期的なモデル保存
                    if self.stats['total_episodes'] % 50 == 0:
                        self.save_model()
                        self.plot_learning_curves()
                
                time.sleep(0.1)  # 100ms待機
                
            except Exception as e:
                print(f"❌ 学習ループエラー: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(1.0)
        
        print(f"🔄 学習ループ終了")
    
    def save_model(self, filepath=None):
        """モデル保存"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_save_dir, f"ddpg_realtime_{timestamp}.pth")
        
        try:
            torch.save({
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic_state_dict': self.agent.critic.state_dict(),
                'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer': self.agent.critic_optimizer.state_dict(),
                'stats': self.stats,
                'episodes': self.stats['total_episodes'],
                'session_id': self.session_id
            }, filepath)
            
            print(f"💾 モデル保存完了: {filepath}")
            
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
    
    def plot_learning_curves(self):
        """学習カーブをプロット"""
        try:
            if len(self.stats['episode_rewards']) < 2:
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'DDPG リアルタイム学習進捗 - Episode {self.stats["total_episodes"]}')
            
            # エピソード報酬
            axes[0, 0].plot(self.stats['episode_rewards'])
            axes[0, 0].set_title('エピソード報酬')
            axes[0, 0].set_xlabel('エピソード')
            axes[0, 0].set_ylabel('累積報酬')
            axes[0, 0].grid(True)
            
            # 移動平均報酬
            if len(self.stats['episode_rewards']) > 10:
                window = 10
                moving_avg = np.convolve(self.stats['episode_rewards'], 
                                       np.ones(window)/window, mode='valid')
                axes[0, 1].plot(moving_avg)
                axes[0, 1].set_title(f'移動平均報酬 (窓幅={window})')
                axes[0, 1].set_xlabel('エピソード')
                axes[0, 1].set_ylabel('平均報酬')
                axes[0, 1].grid(True)
            
            # 学習ロス
            if self.agent.actor_losses:
                axes[1, 0].plot(self.agent.actor_losses, label='Actor Loss', alpha=0.7)
                axes[1, 0].plot(self.agent.critic_losses, label='Critic Loss', alpha=0.7)
                axes[1, 0].set_title('学習ロス')
                axes[1, 0].set_xlabel('更新回数')
                axes[1, 0].set_ylabel('ロス')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # 成功率
            success_rate = self.stats['successful_episodes'] / max(self.stats['total_episodes'], 1) * 100
            axes[1, 1].bar(['成功率'], [success_rate])
            axes[1, 1].set_title(f'成功率: {success_rate:.1f}%')
            axes[1, 1].set_ylabel('成功率 (%)')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存
            plot_path = os.path.join(self.model_save_dir, f"learning_progress_{self.stats['total_episodes']:04d}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"⚠️ プロット保存エラー: {e}")
    
    def stop_learning(self):
        """学習停止"""
        print(f"⏹️ DDPGリアルタイム学習停止中...")
        
        self.is_running = False
        
        # 各サブシステム停止
        if self.data_collector:
            self.data_collector.stop_collection()
        
        if self.feedback_interface:
            self.feedback_interface.stop_server()
        
        # 最終モデル保存
        self.save_model()
        self.plot_learning_curves()
        
        # 最終統計表示
        self.print_final_statistics()
        
        print(f"✅ DDPGリアルタイム学習停止完了")
    
    def print_final_statistics(self):
        """最終統計表示"""
        print(f"\n📊 DDPGリアルタイム学習統計:")
        print(f"   総エピソード数: {self.stats['total_episodes']}")
        print(f"   成功エピソード数: {self.stats['successful_episodes']}")
        
        if self.stats['total_episodes'] > 0:
            success_rate = self.stats['successful_episodes'] / self.stats['total_episodes'] * 100
            print(f"   成功率: {success_rate:.1f}%")
        
        if self.stats['episode_rewards']:
            avg_reward = np.mean(self.stats['episode_rewards'][-50:])  # 最新50エピソードの平均
            print(f"   平均報酬（最新50）: {avg_reward:.2f}")
        
        if self.stats['grip_force_history']:
            avg_grip_force = np.mean(self.stats['grip_force_history'][-100:])
            print(f"   平均把持力（最新100）: {avg_grip_force:.2f}N")
        
        print(f"   学習更新回数: {self.stats['learning_updates']}")
        print(f"   経験バッファサイズ: {len(self.agent.memory)}")
        
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            print(f"   稼働時間: {uptime:.1f}秒")
        
        print(f"   モデル保存先: {self.model_save_dir}")
    
    def run_demo(self):
        """デモ実行"""
        print(f"🚀 DDPGリアルタイムフィードバックシステム デモ実行")
        
        if self.start_learning():
            try:
                print(f"\n💡 システム稼働中...")
                print(f"   📡 LSLデータ受信: {self.lsl_stream_name}")
                print(f"   📡 TCP受信ポート: {self.tcp_port}")
                print(f"   📡 フィードバック送信ポート: {self.feedback_port}")
                print(f"   🎓 DDPGリアルタイム学習実行中")
                print(f"   Ctrl+C で終了")
                
                while self.is_running:
                    time.sleep(5.0)
                    
                    # 定期的な進捗表示
                    print(f"📈 進捗: エピソード{self.stats['total_episodes']}, "
                          f"学習更新{self.stats['learning_updates']}回, "
                          f"成功{self.stats['successful_episodes']}件")
                    
            except KeyboardInterrupt:
                print(f"\n⏹️ ユーザー停止")
            finally:
                self.stop_learning()
        else:
            print(f"❌ システム開始失敗")

def main():
    """メイン実行関数"""
    print(f"🧠 DDPG リアルタイムフィードバックシステム")
    print(f"=" * 70)
    print(f"既存モジュール統合:")
    print(f"  📡 e_tcp_lsl_sync_system.py - LSL/TCPデータ受信")
    print(f"  🧠 g_grip_force_realtime_classifier.py - EEG分類")
    print(f"  🤖 DDPGエージェント - 把持力最適化学習")
    print(f"  📤 Unity フィードバック送信")
    print(f"=" * 70)
    
    # システム初期化（設定可能）
    system = DDPGRealtimeFeedbackSystem(
        model_path='models/improved_grip_force_classifier_*.pth',
        lsl_stream_name='MockEEG',  # 必要に応じて変更
        tcp_host='127.0.0.1',
        tcp_port=12345,          # データ受信ポート
        feedback_port=12346      # フィードバック送信ポート
    )
    
    # デモ実行
    system.run_demo()

if __name__ == "__main__":
    main()