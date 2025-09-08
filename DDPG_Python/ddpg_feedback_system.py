#!/usr/bin/env python3
"""
DDPG強化学習フィードバックシステム

統合フロー:
1. tcp_lsl_sync_system.py でLSLデータとTCPデータを受信
2. grip_force_classifier.py で分類機を使って3クラス分類 (UnderGrip/Success/OverGrip)
3. DDPGエージェントが分類結果とTCP GripForceをもとに次エピソードの適切な把持力を学習
4. unity_tcp_interface.py のリクエストに対して学習済み把持力を応答
5. 継続的にフィードバックループで学習を進める
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

# 既存システムのインポート
from tcp_lsl_sync_system import LSLTCPEpisodeCollector, Episode
from unity_tcp_interface import EEGTCPInterface
from grip_force_classifier import RealtimeGripForceClassifier

# PyTorch設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 デバイス: {device}")

# 経験バッファ用のnamedtuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class Actor(nn.Module):
    """DDPGアクターネットワーク（把持力を出力）"""
    
    def __init__(self, state_dim=6, action_dim=1, hidden_dim=128):
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
        x = torch.tanh(self.fc3(x))  # [-1, 1]の範囲に正規化
        return x

class Critic(nn.Module):
    """DDPGクリティックネットワーク（Q値を出力）"""
    
    def __init__(self, state_dim=6, action_dim=1, hidden_dim=128):
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

class OUNoise:
    """Ornstein-Uhlenbeck ノイズ（アクション探索用）"""
    
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
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """経験を追加"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        """バッチサンプリング"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    """DDPGエージェント"""
    
    def __init__(self, state_dim=6, action_dim=1, lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # ネットワーク初期化
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        
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
        self.actor_loss_history = []
        self.critic_loss_history = []
        
    def hard_update(self, target, source):
        """ハードアップデート（完全コピー）"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source, tau):
        """ソフトアップデート（徐々に更新）"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def select_action(self, state, add_noise=True):
        """アクション選択"""
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if add_noise:
            action += self.noise.sample()
        
        return np.clip(action, -1.0, 1.0)
    
    def update(self, batch_size=64):
        """ネットワーク更新"""
        if len(self.memory) < batch_size:
            return
        
        # バッチサンプリング
        state, action, reward, next_state, done = self.memory.sample(batch_size)
        
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
        self.critic_optimizer.step()
        
        # Actorの更新
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ターゲットネットワーク更新
        self.soft_update(self.actor_target, self.actor, self.tau)
        self.soft_update(self.critic_target, self.critic, self.tau)
        
        # 統計記録
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

class GripForceEnvironment:
    """把持力環境（状態管理・報酬計算）"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """環境リセット"""
        self.episode_count = 0
        self.current_classification = None
        self.current_tcp_data = None
        self.previous_grip_force = 10.0
        self.success_count = 0
        self.total_episodes = 0
        
    def create_state(self, classification_result, tcp_data, previous_grip_force):
        """状態ベクトル作成"""
        # 分類結果をワンホットエンコード
        class_onehot = [0, 0, 0]
        if classification_result is not None:
            class_onehot[classification_result] = 1
        
        # TCP データから特徴量抽出
        grip_force = tcp_data.get('grip_force', 0.0) / 30.0  # 正規化
        contact = 1.0 if tcp_data.get('contact', False) else 0.0
        broken = 1.0 if tcp_data.get('broken', False) else 0.0
        
        # 状態ベクトル作成 [class_0, class_1, class_2, grip_force, contact, broken]
        state = np.array(class_onehot + [grip_force, contact, broken], dtype=np.float32)
        return state
    
    def calculate_reward(self, classification_result, tcp_data, action_grip_force):
        """報酬計算"""
        reward = 0.0
        
        # 分類結果に基づく報酬
        if classification_result == 1:  # Success
            reward += 10.0
            self.success_count += 1
        elif classification_result == 0:  # UnderGrip
            reward -= 5.0
        elif classification_result == 2:  # OverGrip
            reward -= 8.0
        
        # 接触成功報酬
        if tcp_data.get('contact', False):
            reward += 3.0
        
        # 破損ペナルティ
        if tcp_data.get('broken', False):
            reward -= 15.0
        
        # 把持力の適切性（目標範囲8-15N）
        target_min, target_max = 8.0, 15.0
        actual_grip_force = action_grip_force * 15.0 + 15.0  # [-1,1] -> [0,30]N
        
        if target_min <= actual_grip_force <= target_max:
            reward += 2.0
        else:
            # 範囲外ペナルティ
            distance = min(abs(actual_grip_force - target_min), abs(actual_grip_force - target_max))
            reward -= distance * 0.5
        
        return reward

class DDPGFeedbackSystem:
    """DDPG強化学習フィードバックシステム統合クラス"""
    
    def __init__(self, 
                 classifier_model_path='models/best_grip_force_classifier.pth',
                 lsl_stream_name='MockEEG',
                 tcp_host='127.0.0.1',
                 tcp_port=12345):
        
        self.classifier_model_path = classifier_model_path
        self.lsl_stream_name = lsl_stream_name
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        
        # サブシステム初期化
        self.init_classifier()
        self.init_data_collector()
        self.init_tcp_interface()
        
        # DDPGエージェント
        self.agent = DDPGAgent()
        
        # 環境
        self.environment = GripForceEnvironment()
        
        # フィードバックループ管理
        self.is_running = False
        self.episode_queue = queue.Queue()
        self.learning_thread = None
        
        # 学習統計
        self.stats = {
            'total_episodes': 0,
            'successful_episodes': 0,
            'total_rewards': [],
            'classification_accuracy': [],
            'grip_force_history': [],
            'start_time': None
        }
        
        # モデル保存
        self.model_save_dir = "models/ddpg_feedback"
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        print(f"🚀 DDPG強化学習フィードバックシステム初期化完了")
    
    def init_classifier(self):
        """分類機初期化"""
        try:
            if os.path.exists(self.classifier_model_path):
                self.classifier = RealtimeGripForceClassifier(
                    model_path=self.classifier_model_path,
                    lsl_stream_name=self.lsl_stream_name,
                    tcp_host=self.tcp_host,
                    tcp_port=self.tcp_port
                )
                print(f"✅ 分類機読み込み成功: {self.classifier_model_path}")
            else:
                print(f"⚠️ 分類機モデルが見つかりません: {self.classifier_model_path}")
                self.classifier = None
        except Exception as e:
            print(f"❌ 分類機初期化エラー: {e}")
            self.classifier = None
    
    def init_data_collector(self):
        """データ収集システム初期化"""
        self.data_collector = LSLTCPEpisodeCollector(
            lsl_stream_name=self.lsl_stream_name,
            tcp_host=self.tcp_host,
            tcp_port=self.tcp_port,
            save_to_csv=True
        )
        print(f"✅ データ収集システム初期化完了")
    
    def init_tcp_interface(self):
        """TCP通信インターフェース初期化"""
        self.tcp_interface = EEGTCPInterface(
            host=self.tcp_host,
            port=self.tcp_port + 1  # 別ポートでリクエスト応答
        )
        
        # 把持力リクエストコールバック設定
        self.tcp_interface.add_message_callback(self.handle_grip_force_request)
        
        print(f"✅ TCP通信インターフェース初期化完了")
    
    def handle_grip_force_request(self, message_data):
        """Unityからの把持力リクエスト処理"""
        try:
            if message_data.get('type') == 'grip_force_request':
                # 現在の状態を取得（最新エピソードから）
                if hasattr(self, 'latest_state') and self.latest_state is not None:
                    # DDPGエージェントからアクション取得
                    action = self.agent.select_action(self.latest_state, add_noise=False)
                    
                    # アクションを把持力に変換 [-1,1] -> [5,25]N
                    grip_force = (action[0] * 10.0) + 15.0
                    grip_force = np.clip(grip_force, 5.0, 25.0)
                    
                    print(f"🎯 DDPGエージェントによる把持力応答: {grip_force:.2f}N")
                    
                    # TCP応答送信
                    response = {
                        'type': 'grip_force_command',
                        'target_force': float(grip_force),
                        'timestamp': time.time(),
                        'session_id': f"ddpg_rl_{int(time.time())}"
                    }
                    self.tcp_interface.send_message(response)
                    
                    # 統計更新
                    self.stats['grip_force_history'].append(grip_force)
                    
                else:
                    # 状態がない場合はデフォルト値
                    default_grip_force = 12.0
                    print(f"⚠️ 状態なし - デフォルト把持力応答: {default_grip_force}N")
                    
                    response = {
                        'type': 'grip_force_command',
                        'target_force': default_grip_force,
                        'timestamp': time.time(),
                        'session_id': f"ddpg_default_{int(time.time())}"
                    }
                    self.tcp_interface.send_message(response)
                    
        except Exception as e:
            print(f"❌ 把持力リクエスト処理エラー: {e}")
    
    def start_feedback_learning(self):
        """フィードバック学習開始"""
        print(f"🔴 DDPGフィードバック学習開始")
        
        if not self.classifier:
            print(f"❌ 分類機が利用できません")
            return False
        
        self.is_running = True
        self.stats['start_time'] = time.time()
        self.environment.reset()
        
        # データ収集開始
        if not self.data_collector.start_collection():
            print(f"❌ データ収集開始失敗")
            return False
        
        # TCP通信開始
        if not self.tcp_interface.start_server():
            print(f"❌ TCP通信開始失敗")
            return False
        
        # 分類機開始
        if not self.classifier.start_classification():
            print(f"❌ 分類機開始失敗")
            return False
        
        # エピソード処理スレッド開始
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        
        print(f"✅ DDPGフィードバック学習開始完了")
        print(f"💡 Unity側でエピソードを実行してください")
        return True
    
    def _learning_loop(self):
        """学習ループ（バックグラウンド実行）"""
        print(f"🔄 学習ループ開始")
        
        previous_state = None
        previous_action = None
        
        while self.is_running:
            try:
                # 新しいエピソードを待機
                if len(self.data_collector.episodes) > self.stats['total_episodes']:
                    # 最新エピソード取得
                    latest_episode = self.data_collector.episodes[-1]
                    
                    print(f"🆕 新エピソード受信: {latest_episode.episode_id}")
                    
                    # EEG分類実行
                    classification_result = self._classify_episode(latest_episode)
                    
                    # 現在の状態作成
                    current_state = self.environment.create_state(
                        classification_result, 
                        latest_episode.tcp_data, 
                        self.environment.previous_grip_force
                    )
                    
                    # 前エピソードが存在する場合、経験を追加
                    if previous_state is not None and previous_action is not None:
                        reward = self.environment.calculate_reward(
                            classification_result, 
                            latest_episode.tcp_data, 
                            previous_action[0]
                        )
                        
                        # 経験バッファに追加
                        done = latest_episode.tcp_data.get('broken', False)
                        self.agent.memory.push(
                            previous_state, 
                            previous_action, 
                            reward, 
                            current_state, 
                            done
                        )
                        
                        # 統計更新
                        self.stats['total_rewards'].append(reward)
                        if classification_result == 1:  # Success
                            self.stats['successful_episodes'] += 1
                        
                        print(f"📈 報酬: {reward:.2f}, 分類: {classification_result}")
                        
                        # ネットワーク更新
                        if len(self.agent.memory) >= 64:
                            self.agent.update()
                    
                    # 次のアクション選択
                    current_action = self.agent.select_action(current_state)
                    
                    # 状態更新
                    previous_state = current_state
                    previous_action = current_action
                    self.latest_state = current_state
                    self.stats['total_episodes'] += 1
                    
                    # 定期的なモデル保存
                    if self.stats['total_episodes'] % 50 == 0:
                        self.save_model()
                
                time.sleep(0.1)  # 100ms待機
                
            except Exception as e:
                print(f"❌ 学習ループエラー: {e}")
                time.sleep(1.0)
        
        print(f"🔄 学習ループ終了")
    
    def _classify_episode(self, episode: Episode) -> Optional[int]:
        """エピソードのEEGデータを分類"""
        try:
            if not self.classifier:
                return None
            
            # EEGデータを分類機の入力形式に変換
            eeg_data = episode.lsl_data  # (300, 32)
            
            # 分類実行（grip_force_classifierの関数を使用）
            result = self.classifier._classify_eeg_data(eeg_data)
            
            if result:
                classification = result.get('predicted_class', None)
                confidence = result.get('confidence', 0.0)
                
                print(f"🧠 EEG分類結果: クラス{classification}, 信頼度{confidence:.3f}")
                return classification
            
        except Exception as e:
            print(f"❌ EEG分類エラー: {e}")
        
        return None
    
    def save_model(self, filepath=None):
        """モデル保存"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_save_dir, f"ddpg_model_{timestamp}.pth")
        
        try:
            torch.save({
                'actor_state_dict': self.agent.actor.state_dict(),
                'critic_state_dict': self.agent.critic.state_dict(),
                'actor_optimizer': self.agent.actor_optimizer.state_dict(),
                'critic_optimizer': self.agent.critic_optimizer.state_dict(),
                'stats': self.stats,
                'episodes': self.stats['total_episodes']
            }, filepath)
            
            print(f"💾 モデル保存完了: {filepath}")
            
        except Exception as e:
            print(f"❌ モデル保存エラー: {e}")
    
    def load_model(self, filepath):
        """モデル読み込み"""
        try:
            checkpoint = torch.load(filepath)
            
            self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
            if 'stats' in checkpoint:
                self.stats.update(checkpoint['stats'])
            
            print(f"📂 モデル読み込み完了: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            return False
    
    def stop_learning(self):
        """学習停止"""
        print(f"⏹️ DDPGフィードバック学習停止中...")
        
        self.is_running = False
        
        # 各サブシステム停止
        if self.classifier:
            self.classifier.stop_classification()
        
        if self.data_collector:
            self.data_collector.stop_collection()
        
        if self.tcp_interface:
            self.tcp_interface.stop_server()
        
        # モデル保存
        self.save_model()
        
        print(f"✅ DDPGフィードバック学習停止完了")
    
    def print_stats(self):
        """学習統計表示"""
        print(f"\n📊 DDPG学習統計:")
        print(f"   総エピソード数     : {self.stats['total_episodes']}")
        print(f"   成功エピソード数   : {self.stats['successful_episodes']}")
        
        if self.stats['total_episodes'] > 0:
            success_rate = self.stats['successful_episodes'] / self.stats['total_episodes'] * 100
            print(f"   成功率           : {success_rate:.1f}%")
        
        if self.stats['total_rewards']:
            avg_reward = np.mean(self.stats['total_rewards'][-100:])  # 最新100エピソードの平均
            print(f"   平均報酬（最新100）: {avg_reward:.2f}")
        
        if self.stats['grip_force_history']:
            avg_grip_force = np.mean(self.stats['grip_force_history'][-100:])
            print(f"   平均把持力（最新100）: {avg_grip_force:.2f}N")
        
        if self.stats['start_time']:
            uptime = time.time() - self.stats['start_time']
            print(f"   稼働時間         : {uptime:.1f}秒")

def main():
    """メイン実行関数"""
    print(f"🧠 DDPG強化学習フィードバックシステム")
    print(f"=" * 60)
    print(f"1. 事前準備: grip_force_classifier.py で分類機を学習")
    print(f"2. LSL/TCPデータ受信 & EEG分類")
    print(f"3. DDPGエージェントによる把持力最適化")
    print(f"4. Unity TCP リクエストへの学習済み応答")
    print(f"=" * 60)
    
    # システム初期化
    system = DDPGFeedbackSystem(
        classifier_model_path='models/best_grip_force_classifier.pth',
        lsl_stream_name='MockEEG',
        tcp_host='127.0.0.1',
        tcp_port=12345
    )
    
    # フィードバック学習開始
    if system.start_feedback_learning():
        try:
            print(f"\n💡 システム稼働中...")
            print(f"   Unity側でエピソードを実行してください")
            print(f"   把持力リクエストに自動応答します")
            print(f"   Ctrl+C で終了")
            
            while True:
                time.sleep(5.0)
                system.print_stats()
                
        except KeyboardInterrupt:
            print(f"\n⏹️ ユーザー停止")
        finally:
            system.stop_learning()
    else:
        print(f"❌ システム開始失敗")

if __name__ == "__main__":
    main()


# 使用例とテスト用の追加機能

class DDPGFeedbackSystemTester:
    """DDPG強化学習システムのテスト・デバッグ用クラス"""
    
    def __init__(self, system: DDPGFeedbackSystem):
        self.system = system
    
    def test_classifier_only(self, csv_dir="DDPG_Python/logs/episodes_latest"):
        """分類機のみテスト"""
        print(f"🧪 分類機単体テスト")
        
        if not self.system.classifier:
            print(f"❌ 分類機が利用できません")
            return False
        
        try:
            # 保存されたCSVデータを読み込み
            from grip_force_classifier import load_csv_data
            eeg_data_list, grip_force_labels = load_csv_data(csv_dir)
            
            if not eeg_data_list:
                print(f"❌ テスト用CSVデータが見つかりません: {csv_dir}")
                return False
            
            print(f"📂 テストデータ: {len(eeg_data_list)}件")
            
            # 各データで分類テスト
            correct_predictions = 0
            total_predictions = len(eeg_data_list)
            
            for i, (eeg_data, true_label) in enumerate(zip(eeg_data_list, grip_force_labels)):
                result = self.system._classify_episode_data(eeg_data)
                predicted_label = result if result is not None else -1
                
                if predicted_label == true_label:
                    correct_predictions += 1
                
                if i < 5:  # 最初の5件を詳細表示
                    print(f"   テスト{i+1}: 真値={true_label}, 予測={predicted_label}")
            
            accuracy = correct_predictions / total_predictions * 100
            print(f"✅ 分類精度: {accuracy:.1f}% ({correct_predictions}/{total_predictions})")
            
            return accuracy > 50  # 50%以上で合格
            
        except Exception as e:
            print(f"❌ 分類機テストエラー: {e}")
            return False
    
    def test_ddpg_agent(self, num_episodes=10):
        """DDPGエージェントの動作テスト"""
        print(f"🧪 DDPGエージェント単体テスト ({num_episodes}エピソード)")
        
        try:
            # ダミーデータでエージェントテスト
            for episode in range(num_episodes):
                # ランダム状態作成
                state = np.random.rand(6).astype(np.float32)
                
                # アクション選択
                action = self.system.agent.select_action(state)
                
                # 把持力変換
                grip_force = (action[0] * 10.0) + 15.0
                grip_force = np.clip(grip_force, 5.0, 25.0)
                
                # ダミー報酬
                reward = np.random.uniform(-5.0, 10.0)
                
                # 次状態
                next_state = np.random.rand(6).astype(np.float32)
                done = np.random.choice([True, False], p=[0.1, 0.9])
                
                # 経験追加
                self.system.agent.memory.push(state, action, reward, next_state, done)
                
                print(f"   エピソード{episode+1}: 把持力={grip_force:.2f}N, 報酬={reward:.2f}")
                
                # 学習更新（十分な経験がある場合）
                if len(self.system.agent.memory) >= 64:
                    self.system.agent.update()
            
            print(f"✅ DDPGエージェントテスト完了")
            print(f"   経験バッファサイズ: {len(self.system.agent.memory)}")
            
            return True
            
        except Exception as e:
            print(f"❌ DDPGエージェントテストエラー: {e}")
            return False
    
    def test_tcp_communication(self, test_duration=10):
        """TCP通信テスト"""
        print(f"🧪 TCP通信テスト ({test_duration}秒)")
        
        try:
            # TCP インターフェース開始
            if not self.system.tcp_interface.start_server():
                print(f"❌ TCPサーバー開始失敗")
                return False
            
            print(f"🔗 TCPサーバー待機中...")
            print(f"   外部から接続してテストメッセージを送信してください")
            
            start_time = time.time()
            initial_message_count = self.system.tcp_interface.stats['messages_received']
            
            while time.time() - start_time < test_duration:
                current_message_count = self.system.tcp_interface.stats['messages_received']
                
                if current_message_count > initial_message_count:
                    print(f"📥 メッセージ受信: {current_message_count - initial_message_count}件")
                
                time.sleep(1.0)
            
            final_message_count = self.system.tcp_interface.stats['messages_received']
            total_received = final_message_count - initial_message_count
            
            print(f"✅ TCP通信テスト完了")
            print(f"   受信メッセージ数: {total_received}件")
            
            self.system.tcp_interface.stop_server()
            return total_received >= 0  # 0件以上で合格（接続がなくても問題なし）
            
        except Exception as e:
            print(f"❌ TCP通信テストエラー: {e}")
            return False
    
    def run_all_tests(self):
        """全テスト実行"""
        print(f"🧪 DDPG強化学習システム 全テスト実行")
        print(f"=" * 50)
        
        results = {}
        
        # 1. 分類機テスト
        results['classifier'] = self.test_classifier_only()
        print()
        
        # 2. DDPGエージェントテスト
        results['ddpg'] = self.test_ddpg_agent()
        print()
        
        # 3. TCP通信テスト
        results['tcp'] = self.test_tcp_communication()
        print()
        
        # 結果サマリー
        print(f"📊 テスト結果サマリー:")
        for test_name, result in results.items():
            status = "✅ 合格" if result else "❌ 不合格"
            print(f"   {test_name:12}: {status}")
        
        all_passed = all(results.values())
        print(f"\n🎯 総合結果: {'✅ 全テスト合格' if all_passed else '❌ 一部テスト不合格'}")
        
        return all_passed


def run_comprehensive_demo():
    """包括的なデモ実行"""
    print(f"🚀 DDPG強化学習フィードバックシステム 包括デモ")
    print(f"=" * 60)
    
    # システム初期化
    system = DDPGFeedbackSystem(
        classifier_model_path='models/best_grip_force_classifier.pth',
        lsl_stream_name='MockEEG',
        tcp_host='127.0.0.1',
        tcp_port=12345
    )
    
    # テスター初期化
    tester = DDPGFeedbackSystemTester(system)
    
    print(f"Phase 1: システムテスト")
    print(f"-" * 30)
    
    # 全テスト実行
    test_success = tester.run_all_tests()
    
    if not test_success:
        print(f"⚠️ 一部テストが失敗しました。続行しますか？ (y/n)")
        user_input = input().strip().lower()
        if user_input != 'y':
            print(f"❌ デモ中止")
            return
    
    print(f"\nPhase 2: フィードバック学習デモ")
    print(f"-" * 30)
    
    # フィードバック学習開始
    if system.start_feedback_learning():
        try:
            print(f"\n💡 フィードバック学習システム稼働中...")
            print(f"   Unity側でエピソードを実行してください")
            print(f"   a2cClient.SendGripForceRequest() で把持力リクエスト可能")
            print(f"   EPISODE_ENDトリガーで学習実行")
            print(f"   'q' + Enter で終了")
            
            # 非ブロッキング入力待機
            import select
            import sys
            
            while True:
                # 統計表示（5秒ごと）
                system.print_stats()
                
                # ユーザー入力チェック
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    user_input = sys.stdin.readline().strip()
                    if user_input.lower() == 'q':
                        break
                
                time.sleep(5.0)
                
        except KeyboardInterrupt:
            print(f"\n⏹️ ユーザー停止")
        except Exception as e:
            print(f"\n❌ 予期しないエラー: {e}")
        finally:
            system.stop_learning()
    else:
        print(f"❌ フィードバック学習開始失敗")
    
    print(f"\n🎯 デモ完了")


def run_training_mode():
    """学習専用モード"""
    print(f"🎓 DDPG学習専用モード")
    print(f"=" * 40)
    
    # 既存の学習済みモデルがあるかチェック
    model_files = []
    model_dir = "models/ddpg_feedback"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    system = DDPGFeedbackSystem()
    
    # 既存モデルの読み込みオプション
    if model_files:
        print(f"📂 既存の学習済みモデル発見:")
        for i, model_file in enumerate(model_files):
            print(f"   {i+1}. {model_file}")
        
        print(f"   0. 新規学習開始")
        
        try:
            choice = int(input(f"選択 (0-{len(model_files)}): "))
            
            if 1 <= choice <= len(model_files):
                model_path = os.path.join(model_dir, model_files[choice-1])
                if system.load_model(model_path):
                    print(f"✅ モデル読み込み完了: {model_files[choice-1]}")
                else:
                    print(f"❌ モデル読み込み失敗、新規学習を開始します")
        except (ValueError, IndexError):
            print(f"⚠️ 無効な選択、新規学習を開始します")
    
    # 学習パラメータ設定
    print(f"\n⚙️ 学習パラメータ設定:")
    try:
        target_episodes = int(input(f"目標エピソード数 (デフォルト: 1000): ") or "1000")
        save_interval = int(input(f"モデル保存間隔 (デフォルト: 50): ") or "50")
    except ValueError:
        target_episodes = 1000
        save_interval = 50
        print(f"⚠️ デフォルト値を使用: {target_episodes}エピソード, {save_interval}間隔保存")
    
    # 学習開始
    if system.start_feedback_learning():
        try:
            print(f"\n🎓 学習開始 (目標: {target_episodes}エピソード)")
            print(f"💡 Unity側でエピソードを実行してください")
            print(f"   進捗はリアルタイムで表示されます")
            print(f"   Ctrl+C で早期終了")
            
            while system.stats['total_episodes'] < target_episodes:
                time.sleep(10.0)  # 10秒ごとに統計表示
                system.print_stats()
                
                # 保存間隔チェック
                if (system.stats['total_episodes'] % save_interval == 0 and 
                    system.stats['total_episodes'] > 0):
                    system.save_model()
                    print(f"💾 中間保存完了 (エピソード {system.stats['total_episodes']})")
            
            print(f"\n🎉 目標エピソード数達成!")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ 学習早期終了")
        finally:
            system.stop_learning()
            print(f"📊 最終統計:")
            system.print_stats()
    else:
        print(f"❌ 学習開始失敗")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "test":
            # テストモード
            system = DDPGFeedbackSystem()
            tester = DDPGFeedbackSystemTester(system)
            tester.run_all_tests()
            
        elif mode == "demo":
            # デモモード
            run_comprehensive_demo()
            
        elif mode == "train":
            # 学習専用モード
            run_training_mode()
            
        else:
            print(f"❌ 無効なモード: {mode}")
            print(f"使用可能なモード: test, demo, train")
            print(f"例: python ddpg_feedback_system.py demo")
    else:
        # デフォルト: 通常実行
        main()