#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TypeB DDPG LSLフィードバック学習システム

i_ddpg_tcp_feedback.py をベースにして、LSLを使ったTypeBシステム
e_tcp_lsl_sync_system.py でLSLデータを受け取り、
h_ddpg_realtime_feedback_system.py の分類機で把持力クラス分けし、
DDPGで学習しながら把持力のフィードバックを送信

機能:
1. LSLリアルタイムEEGデータ受信
2. EEG分類による把持力予測
3. DDPGエージェントによる最適化学習
4. Unity側への把持力フィードバック送信
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")       
from matplotlib import pyplot as plt
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
from g_grip_force_realtime_classifier import RealtimeGripForceClassifier
from c_unity_tcp_interface import EEGTCPInterface

# PyTorch設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🎯 デバイス: {device}")

# DDPG用の経験バッファ
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    """DDPG Actorネットワーク（把持力出力）"""
    def __init__(self, state_dim=7, action_dim=1, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Layer Normalization for better stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # He初期化
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))  # [-1, 1]の範囲
        return x

class Critic(nn.Module):
    """DDPG Criticネットワーク（Q値出力）"""
    def __init__(self, state_dim=7, action_dim=1, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
    
    def forward(self, state, action):
        x = F.relu(self.ln1(self.fc1(state)))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.ln2(self.fc2(x)))
        q_value = self.fc3(x)
        return q_value

class ReplayBuffer:
    """経験バッファ"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = torch.FloatTensor([self.buffer[i].state for i in batch]).to(device)
        actions = torch.FloatTensor([self.buffer[i].action for i in batch]).to(device)
        rewards = torch.FloatTensor([self.buffer[i].reward for i in batch]).to(device)
        next_states = torch.FloatTensor([self.buffer[i].next_state for i in batch]).to(device)
        dones = torch.BoolTensor([self.buffer[i].done for i in batch]).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class OUNoise:
    """Ornstein-Uhlenbeck process noise"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class DDPGAgent:
    """DDPG エージェント"""
    def __init__(self, state_dim=7, action_dim=1, lr_actor=1e-4, lr_critic=1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # メインネットワーク
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        
        # ターゲットネットワーク
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
        # ターゲットネットワークの初期化
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)
        
        # オプティマイザー
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # ハイパーパラメータ
        self.gamma = 0.99
        self.tau = 0.005  # ソフトアップデート係数
        
        # 経験バッファ
        self.memory = ReplayBuffer(capacity=100000)
        
        # ノイズ
        self.noise = OUNoise(action_dim, sigma=0.2)
        
    def select_action(self, state, add_noise=True, noise_scale=1.0):
        """アクション選択"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        self.actor.train()
        
        if add_noise:
            action += noise_scale * self.noise.sample()
            action = np.clip(action, -1, 1)
        
        return action
    
    def update(self, batch_size=64):
        """ネットワークの更新"""
        if len(self.memory) < batch_size:
            return None, None
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Critic更新
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (self.gamma * target_q * (~dones))
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor更新
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ソフトアップデート
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
    
    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

class SystemHealthChecker:
    """システム健全性チェッククラス"""
    def __init__(self):
        self.checks = {}
        self.last_check_results = {}
        
    def register_check(self, name, check_func, critical=False):
        self.checks[name] = {'func': check_func, 'critical': critical}
    
    def run_all_checks(self):
        results = {}
        all_passed = True
        
        for name, check_info in self.checks.items():
            try:
                result = check_info['func']()
                results[name] = result
                self.last_check_results[name] = result
                
                if not result['success'] and check_info['critical']:
                    all_passed = False
                    
            except Exception as e:
                results[name] = {'success': False, 'details': f'チェック実行エラー: {e}'}
                if check_info['critical']:
                    all_passed = False
        
        return {'all_passed': all_passed, 'results': results}

class EEGLSLEnvironment:
    """EEG-LSLベースの環境クラス"""
    def __init__(self, classifier=None):
        self.classifier = classifier
        self.current_state = None
        self.previous_grip_force = 12.0
        self.episode_reward = 0.0
        
        # 報酬パラメータ
        self.reward_params = {
            'success_reward': 10.0,
            'error_penalty_coeff': 2.0,
            'damage_penalty': 15.0,
            'contact_bonus': 3.0,
            'classification_bonus': 2.0
        }
        
    def compute_state_from_eeg(self, eeg_data, tcp_data):
        """EEGデータとTCPデータから状態を計算"""
        try:
            state = np.zeros(7)  # 7次元状態ベクトル
            
            # EEG特徴量の抽出
            if eeg_data is not None and len(eeg_data) > 0:
                # 基本的な統計特徴量
                state[0] = np.mean(eeg_data)  # 平均
                state[1] = np.std(eeg_data)   # 標準偏差
                state[2] = np.max(eeg_data) - np.min(eeg_data)  # 範囲
                
                # 周波数特徴量（簡易FFT）
                fft = np.fft.fft(eeg_data.flatten()[:512])  # 最初の512サンプル
                power_spectrum = np.abs(fft[:256])  # ナイキスト周波数まで
                state[3] = np.mean(power_spectrum[:30])   # 低周波成分
                state[4] = np.mean(power_spectrum[30:100])  # 中周波成分
            
            # TCP情報から状態を追加
            if tcp_data and isinstance(tcp_data, dict):
                state[5] = tcp_data.get('grip_force', self.previous_grip_force) / 30.0  # 正規化
                state[6] = tcp_data.get('contact_pressure', 0.0) / 10.0  # 正規化
            
            # 分類機による把持力予測（利用可能な場合）
            if self.classifier:
                try:
                    predicted_class, confidence = self.classifier.classify_episode_data(eeg_data)
                    # 分類結果を状態に反映（オプション）
                    class_map = {'UnderGrip': -1, 'Success': 0, 'OverGrip': 1}
                    if predicted_class in class_map:
                        # 状態ベクトルの最後に分類情報を追加する場合
                        pass  # 現在は7次元固定のため、追加情報として利用
                except Exception as e:
                    print(f"⚠️ EEG分類エラー: {e}")
            
            self.current_state = state
            return state
            
        except Exception as e:
            print(f"❌ 状態計算エラー: {e}")
            return np.zeros(7)  # デフォルト状態
    
    def compute_reward(self, action, tcp_data, eeg_classification=None):
        """報酬計算"""
        reward = 0.0
        
        try:
            if tcp_data and isinstance(tcp_data, dict):
                # アクションから把持力に変換
                grip_force = self.action_to_grip_force(action[0])
                target_force = tcp_data.get('target_grip_force', 15.0)
                actual_force = tcp_data.get('grip_force', grip_force)
                
                # 把持成功報酬
                force_error = abs(actual_force - target_force)
                if force_error < 2.0:  # 許容誤差内
                    reward += self.reward_params['success_reward']
                    reward += self.reward_params['contact_bonus']
                else:
                    reward -= force_error * self.reward_params['error_penalty_coeff']
                
                # 破損ペナルティ
                if actual_force > 25.0:  # 過度な力
                    reward -= self.reward_params['damage_penalty']
                
                # 分類機による追加報酬
                if eeg_classification:
                    predicted_class, confidence = eeg_classification
                    if predicted_class == 'Success' and confidence > 0.8:
                        reward += self.reward_params['classification_bonus']
                
                self.episode_reward += reward
                
        except Exception as e:
            print(f"⚠️ 報酬計算エラー: {e}")
            reward = -1.0  # エラーペナルティ
        
        return reward
    
    def action_to_grip_force(self, action_value):
        """アクション値を把持力に変換 [-1,1] -> [5,25]N"""
        return 5.0 + (action_value + 1.0) * 10.0  # [5, 25]N

class TypeBDDPGLSLSystem:
    """TypeB DDPG LSLフィードバックシステム統合クラス"""
    
    def __init__(self, 
                 model_path='models/improved_grip_force_classifier_*.pth',
                 lsl_stream_name='MockEEG',
                 tcp_host='127.0.0.1',
                 tcp_port=12345,
                 feedback_port=12346,
                 experiment_type="B_400"):
        
        self.model_path = model_path
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.feedback_port = feedback_port
        self.experiment_type = experiment_type
        
        # コンポーネント初期化
        self.init_lsl_data_collector()
        self.init_eeg_classifier()
        self.init_feedback_interface()
        
        # DDPGエージェント
        self.agent = DDPGAgent(state_dim=7, action_dim=1, lr_actor=1e-4, lr_critic=1e-3)
        
        # 環境
        self.environment = EEGLSLEnvironment(classifier=self.classifier)
        
        # 実行制御
        self.is_running = False
        self.learning_thread = None
        self.data_processing_thread = None
        
        # 学習統計
        self.stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_rewards': [],
            'episode_rewards': [],
            'classification_accuracy': [],
            'grip_force_history': [],
            'eeg_data_count': 0,
            'learning_updates': 0,
            'start_time': None
        }
        
        # 状態管理
        self.previous_state = None
        self.previous_action = None
        self.current_episode_reward = 0.0
        
        # データキューイング
        self.eeg_data_queue = queue.Queue(maxsize=1000)
        self.tcp_data_queue = queue.Queue(maxsize=1000)
        
        # モデル保存
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = f"models/ddpg_lsl_typeb_{self.session_id}"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # エピソード管理
        self.target_episodes = 400 if "400" in experiment_type else 1000
        self.episode_count = 0
        
        # 健全性チェック
        self.health_checker = SystemHealthChecker()
        self._setup_health_checks()
        
        print(f"🚀 TypeB DDPG LSLフィードバックシステム初期化完了")
        print(f"   実験タイプ: {experiment_type}")
        print(f"   セッションID: {self.session_id}")
        print(f"   モデル保存先: {self.model_save_dir}")
        print(f"   目標エピソード数: {self.target_episodes}")
    
    def init_lsl_data_collector(self):
        """LSLデータ収集システム初期化"""
        self.data_collector = LSLTCPEpisodeCollector(
            lsl_stream_name=self.lsl_stream_name,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port,
            save_to_csv=False,  # リアルタイム学習のためファイル保存は無効
            enable_state_sharing=True,  # 状態共有を有効
            trigger_on_robot_state=True  # ロボット状態でトリガー
        )
        
        print(f"✅ LSLデータ収集システム初期化完了")
    
    def init_eeg_classifier(self):
        """EEG分類機初期化"""
        try:
            self.classifier = RealtimeGripForceClassifier(
                model_path=self.model_path,
                lsl_stream_name=self.lsl_stream_name,
                tcp_host=self.tcp_host,
                tcp_port=self.tcp_port
            )
            
            if not self.classifier.load_model():
                print(f"⚠️ EEG分類機読み込み失敗 - 基本機能で続行")
                self.classifier = None
            else:
                print(f"✅ EEG分類機初期化完了")
                
        except Exception as e:
            print(f"⚠️ EEG分類機初期化エラー: {e}")
            self.classifier = None
    
    def init_feedback_interface(self):
        """フィードバック通信インターフェース初期化"""
        self.feedback_interface = EEGTCPInterface(
            host=self.tcp_host,
            port=self.feedback_port,
            auto_reply=False
        )
        
        # 把持力リクエストコールバック設定
        self.feedback_interface.add_message_callback(self.handle_grip_force_request)
        
        print(f"✅ フィードバック通信初期化完了 (Port: {self.feedback_port})")
    

    def handle_grip_force_request(self, message_data):
        """Unity側からの把持力リクエスト処理"""
        try:
            # 現在の状態から最適な把持力を計算
            if self.previous_state is not None:
                # DDPGエージェントからアクション取得（推論用）
                action = self.agent.select_action(self.previous_state, add_noise=False, noise_scale=0.0)
                
                # アクションを把持力に変換
                grip_force = self.environment.action_to_grip_force(action[0])
                
                print(f"🎯 TypeB把持力フィードバック: {grip_force:.2f}N (action: {action[0]:.3f})")
                
            else:
                # 状態がない場合はデフォルト値
                grip_force = 12.0
                print(f"⚠️ 初期状態 - デフォルト把持力: {grip_force}N")
            
            # フィードバック送信
            response = {
                'type': 'grip_force_command',
                'target_force': float(grip_force),
                'timestamp': time.time(),
                'session_id': f"ddpg_lsl_typeb_{self.session_id}",
                'learning_episode': self.stats['total_episodes'],
                'system_type': 'TypeB_LSL'
            }
            
            self.feedback_interface.send_message(response)
            
            # 統計更新
            self.stats['grip_force_history'].append(grip_force)
            
        except Exception as e:
            print(f"❌ 把持力リクエスト処理エラー: {e}")
    
    def _data_processing_loop(self):
        """データ処理ループ（別スレッド）"""
        print(f"🔄 データ処理ループ開始")
        
        last_processed_episode = 0
        
        while self.is_running:
            try:
                # LSLTCPEpisodeCollectorから新しいエピソードを取得
                if hasattr(self.data_collector, 'episodes') and len(self.data_collector.episodes) > last_processed_episode:
                    # 新しいエピソードを処理
                    for i in range(last_processed_episode, len(self.data_collector.episodes)):
                        episode = self.data_collector.episodes[i]
                        
                        # TCPデータをキューに追加（シミュレーション）
                        if not self.tcp_data_queue.full():
                            self.tcp_data_queue.put(episode.tcp_data, timeout=0.1)
                        
                        # エピソードデータをキューに追加
                        episode_data = {
                            'lsl_data': episode.lsl_data,
                            'lsl_timestamps': episode.lsl_timestamps,
                            'episode_id': episode.episode_id,
                            'trigger_timestamp': episode.trigger_timestamp
                        }
                        
                        if not self.eeg_data_queue.full():
                            self.eeg_data_queue.put(episode_data, timeout=0.1)
                            self.stats['eeg_data_count'] += 1
                        else:
                            print(f"⚠️ EEGデータキューフル - データを破棄")
                    
                    last_processed_episode = len(self.data_collector.episodes)
                
                # EEGデータとTCPデータの処理
                if not self.eeg_data_queue.empty() and not self.tcp_data_queue.empty():
                    eeg_episode = self.eeg_data_queue.get(timeout=1.0)
                    tcp_data = self.tcp_data_queue.get(timeout=1.0)
                    
                    # 状態計算
                    state = self.environment.compute_state_from_eeg(
                        eeg_episode.get('lsl_data'), 
                        tcp_data
                    )
                    
                    # EEG分類（可能な場合）
                    eeg_classification = None
                    if self.classifier and eeg_episode.get('lsl_data') is not None:
                        try:
                            eeg_classification = self.classifier.classify_episode_data(
                                eeg_episode.get('lsl_data')
                            )
                        except Exception as e:
                            print(f"⚠️ EEG分類エラー: {e}")
                    
                    # DDPG学習ステップ
                    if self.previous_state is not None and self.previous_action is not None:
                        # 報酬計算
                        reward = self.environment.compute_reward(
                            self.previous_action, tcp_data, eeg_classification
                        )
                        
                        # 経験バッファに追加
                        experience = Experience(
                            state=self.previous_state,
                            action=self.previous_action,
                            reward=reward,
                            next_state=state,
                            done=False
                        )
                        self.agent.memory.push(experience)
                        
                        # エージェント更新
                        if len(self.agent.memory) > 64:
                            critic_loss, actor_loss = self.agent.update()
                            if critic_loss is not None:
                                self.stats['learning_updates'] += 1
                    
                    # 新しいアクション選択
                    action = self.agent.select_action(state, add_noise=True)
                    
                    # 状態更新
                    self.previous_state = state
                    self.previous_action = action
                    
                    # エピソード統計
                    self.episode_count += 1
                    self.stats['total_episodes'] = self.episode_count
                    
                    # 定期的な進捗表示
                    if self.episode_count % 50 == 0:
                        print(f"📈 進捗: エピソード{self.episode_count}/{self.target_episodes}, "
                              f"学習更新{self.stats['learning_updates']}回")
                
                else:
                    time.sleep(0.1)  # データ待機
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ データ処理ループエラー: {e}")
                time.sleep(0.1)
    
    def _learning_loop(self):
        """メイン学習ループ"""
        print(f"🎓 TypeB学習ループ開始")
        
        start_time = time.time()
        self.stats['start_time'] = start_time
        
        while self.is_running and self.episode_count < self.target_episodes:
            try:
                # 健全性チェック
                if self.episode_count % 100 == 0:
                    health_result = self.health_checker.run_all_checks()
                    if not health_result['all_passed']:
                        print(f"⚠️ システム健全性チェック失敗")
                        for name, result in health_result['results'].items():
                            if not result['success']:
                                print(f"   {name}: {result['details']}")
                
                # モデル保存（定期的）
                if self.episode_count > 0 and self.episode_count % 200 == 0:
                    self.save_models(f"checkpoint_episode_{self.episode_count}")
                
                time.sleep(1.0)  # CPUロード軽減
                
            except KeyboardInterrupt:
                print(f"\n⏹️ ユーザー中断")
                break
            except Exception as e:
                print(f"❌ 学習ループエラー: {e}")
                time.sleep(1.0)
        
        print(f"✅ 学習完了: {self.episode_count}エピソード, 実行時間: {time.time() - start_time:.1f}秒")
    
    def start_learning(self):
        """学習開始"""
        if self.is_running:
            print(f"⚠️ 既に実行中")
            return False
        
        try:
            # サブシステム開始
            if not self.data_collector.start_collection():
                print(f"❌ データ収集システム開始失敗")
                return False
            
            if not self.feedback_interface.start_server():
                print(f"❌ フィードバック通信開始失敗")
                return False
            
            self.is_running = True
            
            # スレッド開始
            self.data_processing_thread = threading.Thread(target=self._data_processing_loop)
            self.learning_thread = threading.Thread(target=self._learning_loop)
            
            self.data_processing_thread.start()
            self.learning_thread.start()
            
            print(f"🚀 TypeB DDPG学習システム開始")
            return True
            
        except Exception as e:
            print(f"❌ システム開始エラー: {e}")
            self.stop_learning()
            return False
    
    def stop_learning(self):
        """学習停止"""
        print(f"⏹️ システム停止中...")
        
        self.is_running = False
        
        # スレッド停止待機
        if self.learning_thread and self.learning_thread.is_alive():
            self.learning_thread.join(timeout=5.0)
        
        if self.data_processing_thread and self.data_processing_thread.is_alive():
            self.data_processing_thread.join(timeout=5.0)
        
        # サブシステム停止
        if hasattr(self.data_collector, 'stop_collection'):
            self.data_collector.stop_collection()
        
        if hasattr(self.feedback_interface, 'stop_server'):
            self.feedback_interface.stop_server()
        
        # 最終モデル保存
        self.save_models("final")
        
        print(f"✅ システム停止完了")
    
    def save_models(self, suffix=""):
        """モデル保存"""
        try:
            model_path = os.path.join(self.model_save_dir, f"actor_{suffix}.pth")
            torch.save(self.agent.actor.state_dict(), model_path)
            
            critic_path = os.path.join(self.model_save_dir, f"critic_{suffix}.pth")
            torch.save(self.agent.critic.state_dict(), critic_path)
            
            # 統計情報保存
            stats_path = os.path.join(self.model_save_dir, f"stats_{suffix}.json")
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2, default=str)
            
            print(f"💾 モデル保存完了: {suffix}")
            
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
    
    def _setup_health_checks(self):
        """健全性チェック項目の設定"""
        
        def check_lsl_connection():
            """LSL接続チェック"""
            return {
                'success': getattr(self.data_collector, 'is_running', True),
                'details': f"LSLデータ収集: {'稼働中' if getattr(self.data_collector, 'is_running', True) else '停止中'}"
            }
        
        def check_feedback_interface():
            """フィードバック通信チェック"""
            return {
                'success': getattr(self.feedback_interface, 'is_connected', True),
                'details': f"フィードバック通信: {'接続中' if getattr(self.feedback_interface, 'is_connected', True) else '未接続'}"
            }
        
        def check_learning_progress():
            """学習進捗チェック"""
            buffer_ratio = len(self.agent.memory) / self.agent.memory.capacity
            return {
                'success': True,
                'details': f"バッファ利用率: {buffer_ratio:.1%}, エピソード: {self.episode_count}/{self.target_episodes}"
            }
        
        # チェック項目登録
        self.health_checker.register_check("lsl_connection", check_lsl_connection, critical=True)
        self.health_checker.register_check("feedback_interface", check_feedback_interface, critical=True)
        self.health_checker.register_check("learning_progress", check_learning_progress, critical=False)
    
    def print_status(self):
        """ステータス表示"""
        print(f"\n🤖 TypeB DDPG LSLシステム ステータス")
        print(f"=" * 60)
        print(f"   セッションID: {self.session_id}")
        print(f"   実験タイプ: {self.experiment_type}")
        print(f"   実行状態: {'稼働中' if self.is_running else '停止中'}")
        print(f"   エピソード進捗: {self.episode_count}/{self.target_episodes}")
        print(f"   学習更新回数: {self.stats['learning_updates']}")
        print(f"   EEGデータ処理回数: {self.stats['eeg_data_count']}")
        
        if self.stats['grip_force_history']:
            avg_grip_force = np.mean(self.stats['grip_force_history'][-50:])
            print(f"   平均把持力（最新50）: {avg_grip_force:.2f}N")
        
        print(f"   経験バッファサイズ: {len(self.agent.memory)}")
        
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            print(f"   稼働時間: {uptime:.1f}秒")
        
        print(f"   モデル保存先: {self.model_save_dir}")
    
    def run_demo(self):
        """デモ実行"""
        print(f"🚀 TypeB DDPG LSLフィードバックシステム デモ実行")
        
        if self.start_learning():
            try:
                print(f"\n💡 システム稼働中...")
                print(f"   📡 LSLストリーム: {self.lsl_stream_name}")
                print(f"   📡 TCPポート: {self.tcp_port}")
                print(f"   📡 フィードバックポート: {self.feedback_port}")
                print(f"   🧠 EEG分類機: {'有効' if self.classifier else '無効'}")
                print(f"   🎓 DDPGリアルタイム学習実行中")
                print(f"   目標エピソード: {self.target_episodes}")
                print(f"   Ctrl+C で終了")
                
                # メインループ
                while self.is_running and self.episode_count < self.target_episodes:
                    time.sleep(5.0)
                    
                    # 定期的な進捗表示
                    progress_percent = (self.episode_count / self.target_episodes) * 100
                    print(f"📈 進捗: {progress_percent:.1f}% "
                          f"({self.episode_count}/{self.target_episodes} エピソード), "
                          f"学習更新{self.stats['learning_updates']}回")
                    
            except KeyboardInterrupt:
                print(f"\n⏹️ ユーザー停止")
            finally:
                self.stop_learning()
        else:
            print(f"❌ システム開始失敗")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="TypeB DDPG LSLフィードバック学習システム")
    parser.add_argument("--type", choices=["B_400", "B_long"], default="B_400",
                       help="実験タイプ")
    parser.add_argument("--lsl-stream", default="MockEEG",
                       help="LSLストリーム名")
    parser.add_argument("--tcp-port", type=int, default=12345,
                       help="TCPポート")
    parser.add_argument("--feedback-port", type=int, default=12346,
                       help="フィードバックポート")
    parser.add_argument("--model-path", default="models/improved_grip_force_classifier_*.pth",
                       help="EEG分類機モデルパス")
    
    args = parser.parse_args()
    
    print(f"🧠 TypeB DDPG LSLフィードバック学習システム")
    print(f"=" * 70)
    print(f"システム構成:")
    print(f"  📡 LSLリアルタイムEEGデータ受信")
    print(f"  🧠 EEG分類による把持力予測")
    print(f"  🤖 DDPGエージェントによる最適化学習")
    print(f"  📤 Unity側への把持力フィードバック送信")
    print(f"=" * 70)
    
    # システム初期化
    system = TypeBDDPGLSLSystem(
        model_path=args.model_path,
        lsl_stream_name=args.lsl_stream,
        tcp_host='127.0.0.1',
        tcp_port=args.tcp_port,
        feedback_port=args.feedback_port,
        experiment_type=args.type
    )
    
    # デモ実行
    system.run_demo()

if __name__ == "__main__":
    main()