#!/usr/bin/env python3
"""
DDPG学習監視・可視化ツール（完全版）

機能:
1. リアルタイム学習進捗の監視
2. 報酬履歴とEEG分類精度の可視化
3. 把持力分布の分析
4. モデル性能の評価レポート生成
5. セッション比較機能
6. 収束・安定性分析
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional
import seaborn as sns
import json

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DDPGTrainingMonitor:
    """DDPG学習監視クラス"""
    
    def __init__(self, 
                 stats_dir="models/ddpg",
                 update_interval=5.0,
                 save_plots=True):
        
        self.stats_dir = stats_dir
        self.update_interval = update_interval
        self.save_plots = save_plots
        
        # データ格納
        self.training_data = {
            'episodes': [],
            'rewards': [],
            'classification_accuracy': [],
            'grip_forces': [],
            'timestamps': []
        }
        
        # プロット設定
        self.fig = None
        self.axes = None
        self.is_monitoring = False
        
        # 統計情報
        self.current_stats = None
        
        print(f"📊 DDPG学習監視システム初期化")
        print(f"   統計ディレクトリ: {stats_dir}")
        print(f"   更新間隔: {update_interval}秒")
    
    def start_monitoring(self):
        """監視開始"""
        print(f"🔍 DDPG学習監視開始")
        
        self.is_monitoring = True
        
        # プロット初期化
        self._setup_plots()
        
        # 監視スレッド開始
        monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        monitor_thread.start()
        
        return True
    
    def stop_monitoring(self):
        """監視停止"""
        print(f"⏹️ DDPG学習監視停止")
        self.is_monitoring = False
    
    def _setup_plots(self):
        """プロット設定"""
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('DDPG把持力強化学習 - リアルタイム監視', fontsize=16)
        
        # サブプロットタイトル
        self.axes[0, 0].set_title('エピソード報酬')
        self.axes[0, 1].set_title('EEG分類精度')
        self.axes[1, 0].set_title('把持力分布')
        self.axes[1, 1].set_title('学習統計')
        
        plt.tight_layout()
        plt.ion()  # インタラクティブモード
        plt.show()
    
    def _monitoring_loop(self):
        """監視ループ"""
        print(f"🔄 監視ループ開始")
        
        while self.is_monitoring:
            try:
                # 最新統計ファイルを読み込み
                latest_stats = self._load_latest_stats()
                
                if latest_stats:
                    self.current_stats = latest_stats
                    self._update_training_data(latest_stats)
                    self._update_plots()
                
                time.sleep(self.update_interval)
                
            except Exception as e:
                print(f"⚠️ 監視エラー: {e}")
                time.sleep(self.update_interval)
        
        print(f"🔄 監視ループ終了")
    
    def _load_latest_stats(self):
        """最新の統計ファイルを読み込み"""
        try:
            if not os.path.exists(self.stats_dir):
                return None
            
            # 統計ファイルを検索
            stats_files = [f for f in os.listdir(self.stats_dir) 
                          if f.startswith('training_stats_') and f.endswith('.pkl')]
            
            if not stats_files:
                return None
            
            # 最新ファイルを選択
            latest_file = max(stats_files, key=lambda x: os.path.getmtime(os.path.join(self.stats_dir, x)))
            latest_path = os.path.join(self.stats_dir, latest_file)
            
            # 統計データ読み込み
            with open(latest_path, 'rb') as f:
                stats_data = pickle.load(f)
            
            return stats_data
            
        except Exception as e:
            print(f"⚠️ 統計ファイル読み込みエラー: {e}")
            return None
    
    def _update_training_data(self, stats):
        """訓練データの更新"""
        if not stats:
            return
        
        # 新しいデータのみ追加
        current_episodes = len(self.training_data['episodes'])
        new_episodes = len(stats.get('total_rewards', []))
        
        if new_episodes > current_episodes:
            # 新しいエピソードデータを追加
            for i in range(current_episodes, new_episodes):
                self.training_data['episodes'].append(i + 1)
                
                if i < len(stats['total_rewards']):
                    self.training_data['rewards'].append(stats['total_rewards'][i])
                
                if i < len(stats.get('classification_accuracy', [])):
                    self.training_data['classification_accuracy'].append(
                        stats['classification_accuracy'][i]
                    )
                
                self.training_data['timestamps'].append(time.time())
            
            # 把持力履歴の更新
            if 'grip_force_history' in stats:
                self.training_data['grip_forces'] = stats['grip_force_history']
    
    def _update_plots(self):
        """プロットの更新"""
        if not self.training_data['episodes']:
            return
        
        # プロットクリア
        for ax in self.axes.flat:
            ax.clear()
        
        # 1. エピソード報酬
        self._plot_episode_rewards()
        
        # 2. EEG分類精度
        self._plot_classification_accuracy()
        
        # 3. 把持力分布
        self._plot_grip_force_distribution()
        
        # 4. 学習統計
        self._plot_learning_statistics()
        
        # 図の更新
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
        
        # プロット保存
        if self.save_plots:
            self._save_current_plots()
    
    def _plot_episode_rewards(self):
        """エピソード報酬のプロット"""
        ax = self.axes[0, 0]
        
        if len(self.training_data['rewards']) < 2:
            ax.text(0.5, 0.5, 'データ不足', ha='center', va='center', transform=ax.transAxes)
            return
        
        episodes = self.training_data['episodes']
        rewards = self.training_data['rewards']
        
        # 報酬プロット
        ax.plot(episodes, rewards, 'b-', alpha=0.3, label='エピソード報酬')
        
        # 移動平均
        if len(rewards) >= 10:
            window = min(20, len(rewards) // 2)
            moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, moving_avg, 'r-', linewidth=2, label=f'移動平均({window})')
        
        ax.set_xlabel('エピソード')
        ax.set_ylabel('報酬')
        ax.set_title('エピソード報酬推移')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_classification_accuracy(self):
        """EEG分類精度のプロット"""
        ax = self.axes[0, 1]
        
        if not self.training_data['classification_accuracy']:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        episodes = self.training_data['episodes'][:len(self.training_data['classification_accuracy'])]
        accuracy = self.training_data['classification_accuracy']
        
        # 精度プロット（成功率として）
        success_rate = [acc * 100 for acc in accuracy]
        ax.plot(episodes, success_rate, 'g-', alpha=0.5, label='分類精度')
        
        # 移動平均
        if len(success_rate) >= 10:
            window = min(20, len(success_rate) // 2)
            moving_avg = pd.Series(success_rate).rolling(window=window, min_periods=1).mean()
            ax.plot(episodes, moving_avg, 'darkgreen', linewidth=2, label=f'移動平均({window})')
        
        ax.set_xlabel('エピソード')
        ax.set_ylabel('成功率 (%)')
        ax.set_title('EEG分類精度 (Success)')
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_grip_force_distribution(self):
        """把持力分布のプロット"""
        ax = self.axes[1, 0]
        
        if not self.training_data['grip_forces']:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        grip_forces = self.training_data['grip_forces']
        
        # ヒストグラム
        ax.hist(grip_forces, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        
        # 統計線
        mean_force = np.mean(grip_forces)
        std_force = np.std(grip_forces)
        
        ax.axvline(mean_force, color='red', linestyle='--', 
                  label=f'平均: {mean_force:.2f}N')
        ax.axvline(mean_force + std_force, color='orange', linestyle=':', 
                  label=f'+1σ: {mean_force + std_force:.2f}N')
        ax.axvline(mean_force - std_force, color='orange', linestyle=':', 
                  label=f'-1σ: {mean_force - std_force:.2f}N')
        
        # 理想範囲
        ax.axvspan(8, 15, alpha=0.2, color='green', label='理想範囲 (8-15N)')
        
        ax.set_xlabel('把持力 (N)')
        ax.set_ylabel('頻度')
        ax.set_title('把持力分布')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_statistics(self):
        """学習統計のプロット"""
        ax = self.axes[1, 1]
        
        if not self.current_stats:
            ax.text(0.5, 0.5, 'データなし', ha='center', va='center', transform=ax.transAxes)
            return
        
        # 統計テキスト表示
        stats_text = []
        
        # 基本統計
        total_episodes = self.current_stats.get('total_episodes', 0)
        stats_text.append(f'総エピソード数: {total_episodes}')
        
        # 報酬統計
        if self.current_stats.get('total_rewards'):
            rewards = self.current_stats['total_rewards']
            avg_reward = np.mean(rewards)
            max_reward = np.max(rewards)
            recent_avg = np.mean(rewards[-20:]) if len(rewards) >= 20 else avg_reward
            
            stats_text.append(f'平均報酬: {avg_reward:.3f}')
            stats_text.append(f'最高報酬: {max_reward:.3f}')
            stats_text.append(f'最近20EP平均: {recent_avg:.3f}')
        
        # 分類精度
        if self.current_stats.get('classification_accuracy'):
            accuracy = self.current_stats['classification_accuracy']
            avg_accuracy = np.mean(accuracy) * 100
            recent_accuracy = np.mean(accuracy[-20:]) * 100 if len(accuracy) >= 20 else avg_accuracy
            
            stats_text.append(f'平均分類精度: {avg_accuracy:.1f}%')
            stats_text.append(f'最近20EP精度: {recent_accuracy:.1f}%')
        
        # 把持力統計
        if self.current_stats.get('grip_force_history'):
            forces = self.current_stats['grip_force_history']
            avg_force = np.mean(forces)
            std_force = np.std(forces)
            
            stats_text.append(f'平均把持力: {avg_force:.2f}±{std_force:.2f}N')
        
        # 学習時間
        if self.current_stats.get('start_time'):
            elapsed = time.time() - self.current_stats['start_time']
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            stats_text.append(f'学習時間: {hours:02d}:{minutes:02d}:{seconds:02d}')
        
        # テキスト表示
        y_pos = 0.9
        for line in stats_text:
            ax.text(0.05, y_pos, line, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', fontfamily='monospace')
            y_pos -= 0.12
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('学習統計')
        ax.axis('off')
    
    def _save_current_plots(self):
        """現在のプロットを保存"""
        try:
            # 保存ディレクトリの確保
            os.makedirs(self.stats_dir, exist_ok=True)
            
            # タイムスタンプ付きファイル名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(self.stats_dir, f"training_monitor_{timestamp}.png")
            
            # 高品質で保存
            self.fig.savefig(plot_path, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
            
            # 定期的なクリーンアップ（古いファイル削除）
            self._cleanup_old_plots()
            
        except Exception as e:
            print(f"⚠️ プロット保存エラー: {e}")
    
    def _cleanup_old_plots(self):
        """古いプロットファイルのクリーンアップ"""
        try:
            if not os.path.exists(self.stats_dir):
                return
            
            # monitoring_*.pngファイルを検索
            plot_files = [f for f in os.listdir(self.stats_dir) 
                         if f.startswith('training_monitor_') and f.endswith('.png')]
            
            # 最新10個以外を削除
            if len(plot_files) > 10:
                plot_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.stats_dir, x)))
                old_files = plot_files[:-10]  # 古い分
                
                for old_file in old_files:
                    old_path = os.path.join(self.stats_dir, old_file)
                    os.remove(old_path)
                
                print(f"🗑️ 古いプロットファイル {len(old_files)}個を削除")
                
        except Exception as e:
            print(f"⚠️ プロットクリーンアップエラー: {e}")
    
    def generate_training_report(self):
        """学習レポート生成"""
        if not self.current_stats:
            print(f"❌ 統計データがありません")
            return None
        
        print(f"📋 学習レポート生成開始")
        
        # レポートデータ
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_episodes': self.current_stats.get('total_episodes', 0),
            'learning_time': None,
            'reward_stats': {},
            'classification_stats': {},
            'grip_force_stats': {},
            'performance_metrics': {}
        }
        
        # 学習時間
        if self.current_stats.get('start_time'):
            elapsed = time.time() - self.current_stats['start_time']
            report['learning_time'] = f"{elapsed:.1f} seconds"
        
        # 報酬統計
        if self.current_stats.get('total_rewards'):
            rewards = self.current_stats['total_rewards']
            report['reward_stats'] = {
                'mean': float(np.mean(rewards)),
                'std': float(np.std(rewards)),
                'max': float(np.max(rewards)),
                'min': float(np.min(rewards)),
                'recent_20_mean': float(np.mean(rewards[-20:])) if len(rewards) >= 20 else None
            }
        
        # 分類統計
        if self.current_stats.get('classification_accuracy'):
            accuracy = self.current_stats['classification_accuracy']
            report['classification_stats'] = {
                'mean_accuracy': float(np.mean(accuracy)),
                'recent_20_accuracy': float(np.mean(accuracy[-20:])) if len(accuracy) >= 20 else None,
                'success_rate': float(np.mean(accuracy))
            }
        
        # 把持力統計
        if self.current_stats.get('grip_force_history'):
            forces = self.current_stats['grip_force_history']
            
            # 理想範囲内の把持力率
            ideal_range_count = sum(1 for f in forces if 8 <= f <= 15)
            ideal_range_rate = ideal_range_count / len(forces) if forces else 0
            
            report['grip_force_stats'] = {
                'mean': float(np.mean(forces)),
                'std': float(np.std(forces)),
                'min': float(np.min(forces)),
                'max': float(np.max(forces)),
                'ideal_range_rate': float(ideal_range_rate)
            }
        
        # 性能指標
        if report['reward_stats'] and report['classification_stats']:
            # 学習効率指標
            learning_efficiency = (
                report['reward_stats']['mean'] * 
                report['classification_stats']['success_rate']
            )
            
            report['performance_metrics'] = {
                'learning_efficiency': float(learning_efficiency),
                'convergence_indicator': self._calculate_convergence_indicator(),
                'stability_score': self._calculate_stability_score()
            }
        
        # レポート保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.stats_dir, f"training_report_{timestamp}.json")
        
        try:
            os.makedirs(self.stats_dir, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 学習レポート保存: {report_path}")
            
        except Exception as e:
            print(f"⚠️ レポート保存エラー: {e}")
        
        return report
    
    def _calculate_convergence_indicator(self):
        """収束指標の計算"""
        if not self.current_stats.get('total_rewards'):
            return 0.0
        
        rewards = self.current_stats['total_rewards']
        if len(rewards) < 50:
            return 0.0
        
        # 最新50エピソードの分散を収束指標とする
        recent_rewards = rewards[-50:]
        variance = np.var(recent_rewards)
        
        # 分散が小さいほど収束（0-1でスコア化）
        max_variance = np.var(rewards)
        convergence_score = max(0, 1 - (variance / max_variance)) if max_variance > 0 else 1.0
        
        return float(convergence_score)
    
    def _calculate_stability_score(self):
        """安定性スコアの計算"""
        if not self.current_stats.get('classification_accuracy'):
            return 0.0
        
        accuracy = self.current_stats['classification_accuracy']
        if len(accuracy) < 20:
            return 0.0
        
        # 最新20エピソードの成功率の標準偏差
        recent_accuracy = accuracy[-20:]
        stability = 1.0 - np.std(recent_accuracy)  # 標準偏差が小さいほど安定
        
        return max(0.0, float(stability))
    
    def print_current_status(self):
        """現在のステータス表示"""
        if not self.current_stats:
            print(f"📊 現在の統計データなし")
            return
        
        print(f"\n📊 DDPG学習システム現在のステータス")
        print(f"=" * 60)
        
        # 基本情報
        total_episodes = self.current_stats.get('total_episodes', 0)
        print(f"総エピソード数: {total_episodes}")
        
        if self.current_stats.get('start_time'):
            elapsed = time.time() - self.current_stats['start_time']
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            print(f"学習時間: {hours:02d}時間{minutes:02d}分")
        
        # 報酬統計
        if self.current_stats.get('total_rewards'):
            rewards = self.current_stats['total_rewards']
            print(f"\n🎯 報酬統計:")
            print(f"   平均報酬: {np.mean(rewards):.3f}")
            print(f"   最高報酬: {np.max(rewards):.3f}")
            
            if len(rewards) >= 20:
                recent_avg = np.mean(rewards[-20:])
                print(f"   最新20EP平均: {recent_avg:.3f}")
        
        # 分類精度
        if self.current_stats.get('classification_accuracy'):
            accuracy = self.current_stats['classification_accuracy']
            avg_accuracy = np.mean(accuracy) * 100
            print(f"\n🧠 EEG分類統計:")
            print(f"   平均精度: {avg_accuracy:.1f}%")
            
            if len(accuracy) >= 20:
                recent_accuracy = np.mean(accuracy[-20:]) * 100
                print(f"   最新20EP精度: {recent_accuracy:.1f}%")
        
        # 把持力統計
        if self.current_stats.get('grip_force_history'):
            forces = self.current_stats['grip_force_history']
            ideal_count = sum(1 for f in forces if 8 <= f <= 15)
            ideal_rate = (ideal_count / len(forces)) * 100 if forces else 0
            
            print(f"\n🤏 把持力統計:")
            print(f"   平均把持力: {np.mean(forces):.2f}N")
            print(f"   理想範囲率: {ideal_rate:.1f}% (8-15N)")
        
        print(f"=" * 60)


def create_training_comparison_report(stats_files: List[str], output_path: str = None):
    """複数の学習セッションの比較レポート作成"""
    print(f"📊 学習セッション比較レポート作成")
    
    if len(stats_files) < 2:
        print(f"❌ 比較には最低2つの統計ファイルが必要です")
        return None
    
    # データ読み込み
    sessions_data = []
    for i, stats_file in enumerate(stats_files):
        try:
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)
            
            session_name = f"Session_{i+1}"
            sessions_data.append({
                'name': session_name,
                'file': stats_file,
                'stats': stats
            })
            
        except Exception as e:
            print(f"⚠️ ファイル読み込みエラー {stats_file}: {e}")
            continue
    
    if len(sessions_data) < 2:
        print(f"❌ 有効なデータが不足しています")
        return None
    
    # 比較プロット作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('DDPG学習セッション比較', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # 1. 報酬比較
    ax = axes[0, 0]
    for i, session in enumerate(sessions_data):
        rewards = session['stats'].get('total_rewards', [])
        if rewards:
            episodes = list(range(1, len(rewards) + 1))
            
            # 移動平均
            window = min(20, len(rewards) // 4) if len(rewards) > 20 else 1
            if window > 1:
                moving_avg = pd.Series(rewards).rolling(window=window, min_periods=1).mean()
                ax.plot(episodes, moving_avg, color=colors[i % len(colors)], 
                       label=f"{session['name']} (平均)", linewidth=2)
            else:
                ax.plot(episodes, rewards, color=colors[i % len(colors)], 
                       label=session['name'], alpha=0.7)
    
    ax.set_xlabel('エピソード')
    ax.set_ylabel('報酬')
    ax.set_title('報酬推移比較')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 分類精度比較
    ax = axes[0, 1]
    for i, session in enumerate(sessions_data):
        accuracy = session['stats'].get('classification_accuracy', [])
        if accuracy:
            episodes = list(range(1, len(accuracy) + 1))
            accuracy_percent = [acc * 100 for acc in accuracy]
            
            # 移動平均
            window = min(20, len(accuracy) // 4) if len(accuracy) > 20 else 1
            if window > 1:
                moving_avg = pd.Series(accuracy_percent).rolling(window=window, min_periods=1).mean()
                ax.plot(episodes, moving_avg, color=colors[i % len(colors)], 
                       label=f"{session['name']}", linewidth=2)
            else:
                ax.plot(episodes, accuracy_percent, color=colors[i % len(colors)], 
                       label=session['name'], alpha=0.7)
    
    ax.set_xlabel('エピソード')
    ax.set_ylabel('成功率 (%)')
    ax.set_title('EEG分類精度比較')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 把持力分布比較
    ax = axes[1, 0]
    for i, session in enumerate(sessions_data):
        forces = session['stats'].get('grip_force_history', [])
        if forces:
            ax.hist(forces, bins=20, alpha=0.5, color=colors[i % len(colors)], 
                   label=f"{session['name']}", density=True)
    
    # 理想範囲表示
    ax.axvspan(8, 15, alpha=0.2, color='green', label='理想範囲')
    ax.set_xlabel('把持力 (N)')
    ax.set_ylabel('密度')
    ax.set_title('把持力分布比較')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 統計サマリー
    ax = axes[1, 1]
    summary_data = []
    
    for session in sessions_data:
        stats = session['stats']
        
        # 基本統計計算
        rewards = stats.get('total_rewards', [])
        accuracy = stats.get('classification_accuracy', [])
        forces = stats.get('grip_force_history', [])
        
        summary = {
            'セッション': session['name'],
            'エピソード数': len(rewards),
            '平均報酬': f"{np.mean(rewards):.3f}" if rewards else "N/A",
            '平均精度': f"{np.mean(accuracy)*100:.1f}%" if accuracy else "N/A",
            '平均把持力': f"{np.mean(forces):.2f}N" if forces else "N/A"
        }
        summary_data.append(summary)
    
    # テーブル表示
    df = pd.DataFrame(summary_data)
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title('統計サマリー')
    
    plt.tight_layout()
    
    # 保存
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 比較レポート保存: {output_path}")
    
    plt.show()
    
    return sessions_data


def main():
    """メイン実行関数"""
    print(f"📊 DDPG学習監視・可視化ツール")
    print(f"=" * 60)
    
    print(f"実行モードを選択してください:")
    print(f"1. リアルタイム監視")
    print(f"2. 学習レポート生成")
    print(f"3. セッション比較")
    print(f"4. 統計データ表示")
    
    choice = input(f"選択 (1-4): ").strip()
    
    try:
        if choice == "1":
            # リアルタイム監視
            print(f"\n🔍 リアルタイム監視モード")
            
            stats_dir = input(f"統計ディレクトリパス (空でデフォルト): ").strip()
            if not stats_dir:
                stats_dir = "models/ddpg"
            
            monitor = DDPGTrainingMonitor(stats_dir=stats_dir)
            
            if monitor.start_monitoring():
                try:
                    print(f"💡 監視開始 - リアルタイムプロット表示中")
                    print(f"   統計ファイルの更新を監視中...")
                    print(f"   Ctrl+C で終了")
                    
                    while monitor.is_monitoring:
                        time.sleep(1.0)
                        
                        # 定期的なステータス表示
                        if int(time.time()) % 30 == 0:  # 30秒ごと
                            monitor.print_current_status()
                        
                except KeyboardInterrupt:
                    print(f"\n⏹️ 監視停止")
                finally:
                    monitor.stop_monitoring()
        
        elif choice == "2":
            # 学習レポート生成
            print(f"\n📋 学習レポート生成")
            
            stats_dir = input(f"統計ディレクトリパス (空でデフォルト): ").strip()
            if not stats_dir:
                stats_dir = "models/ddpg"
            
            monitor = DDPGTrainingMonitor(stats_dir=stats_dir)
            
            # 最新統計読み込み
            latest_stats = monitor._load_latest_stats()
            if latest_stats:
                monitor.current_stats = latest_stats
                report = monitor.generate_training_report()
                
                if report:
                    print(f"✅ レポート生成完了")
                    
                    # 詳細表示
                    print(f"\n📋 レポート詳細:")
                    print(json.dumps(report, indent=2, ensure_ascii=False))
                else:
                    print(f"❌ レポート生成失敗")
            else:
                print(f"❌ 統計データが見つかりません")
        
        elif choice == "3":
            # セッション比較
            print(f"\n📊 セッション比較")
            
            stats_dir = input(f"統計ディレクトリパス (空でデフォルト): ").strip()
            if not stats_dir:
                stats_dir = "models/ddpg"
            
            if not os.path.exists(stats_dir):
                print(f"❌ ディレクトリが存在しません: {stats_dir}")
                return
            
            # 統計ファイル検索
            stats_files = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir)
                          if f.startswith('training_stats_') and f.endswith('.pkl')]
            
            if len(stats_files) < 2:
                print(f"❌ 比較用の統計ファイルが不足しています (最低2つ必要)")
                print(f"   見つかったファイル数: {len(stats_files)}")
                return
            
            # 最新の複数ファイルを選択
            stats_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            selected_files = stats_files[:4]  # 最新4セッション
            
            print(f"比較対象セッション:")
            for i, file in enumerate(selected_files):
                print(f"   {i+1}. {os.path.basename(file)}")
            
            output_path = os.path.join(stats_dir, f"session_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            
            comparison_data = create_training_comparison_report(selected_files, output_path)
            
            if comparison_data:
                print(f"✅ セッション比較完了")
            else:
                print(f"❌ セッション比較失敗")
        
        elif choice == "4":
            # 統計データ表示
            print(f"\n📊 統計データ表示")
            
            stats_dir = input(f"統計ディレクトリパス (空でデフォルト): ").strip()
            if not stats_dir:
                stats_dir = "models/ddpg"
            
            monitor = DDPGTrainingMonitor(stats_dir=stats_dir)
            
            # 最新統計読み込み
            latest_stats = monitor._load_latest_stats()
            if latest_stats:
                monitor.current_stats = latest_stats
                monitor.print_current_status()
            else:
                print(f"❌ 統計データが見つかりません")
        
        else:
            print(f"❌ 無効な選択です")
    
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()


# ユーティリティ関数
def analyze_training_convergence(stats_file: str):
    """学習収束の詳細分析"""
    try:
        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)
        
        rewards = stats.get('total_rewards', [])
        if len(rewards) < 50:
            print(f"❌ 収束分析には最低50エピソード必要")
            return None
        
        # 収束分析
        window_size = 20
        convergence_data = {
            'episode_windows': [],
            'mean_rewards': [],
            'variance_rewards': [],
            'trend_slopes': []
        }
        
        for i in range(window_size, len(rewards)):
            window_rewards = rewards[i-window_size:i]
            
            convergence_data['episode_windows'].append(i)
            convergence_data['mean_rewards'].append(np.mean(window_rewards))
            convergence_data['variance_rewards'].append(np.var(window_rewards))
            
            # トレンド計算（線形回帰の傾き）
            x = np.arange(len(window_rewards))
            slope = np.polyfit(x, window_rewards, 1)[0]
            convergence_data['trend_slopes'].append(slope)
        
        # 収束点の特定
        variance_threshold = np.mean(convergence_data['variance_rewards']) * 0.5
        slope_threshold = np.std(convergence_data['trend_slopes']) * 0.1
        
        convergence_point = None
        for i, (var, slope) in enumerate(zip(convergence_data['variance_rewards'], 
                                           convergence_data['trend_slopes'])):
            if var < variance_threshold and abs(slope) < slope_threshold:
                convergence_point = convergence_data['episode_windows'][i]
                break
        
        # 可視化
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('学習収束分析', fontsize=16)
        
        # 報酬推移
        axes[0, 0].plot(range(1, len(rewards)+1), rewards, alpha=0.3)
        axes[0, 0].plot(convergence_data['episode_windows'], convergence_data['mean_rewards'], 
                       linewidth=2, label=f'{window_size}EP移動平均')
        if convergence_point:
            axes[0, 0].axvline(convergence_point, color='red', linestyle='--', 
                             label=f'収束点 (EP{convergence_point})')
        axes[0, 0].set_title('報酬推移と収束点')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # 分散推移
        axes[0, 1].plot(convergence_data['episode_windows'], convergence_data['variance_rewards'])
        axes[0, 1].axhline(variance_threshold, color='red', linestyle='--', label='収束閾値')
        axes[0, 1].set_title('報酬分散推移')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # トレンド推移
        axes[1, 0].plot(convergence_data['episode_windows'], convergence_data['trend_slopes'])
        axes[1, 0].axhline(slope_threshold, color='red', linestyle='--', label='トレンド閾値')
        axes[1, 0].axhline(-slope_threshold, color='red', linestyle='--')
        axes[1, 0].set_title('学習トレンド（傾き）')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 収束サマリー
        axes[1, 1].axis('off')
        summary_text = [
            f"総エピソード数: {len(rewards)}",
            f"収束点: {f'エピソード {convergence_point}' if convergence_point else '未収束'}",
            f"最終{window_size}EP平均報酬: {np.mean(rewards[-window_size:]):.3f}",
            f"最終{window_size}EP分散: {np.var(rewards[-window_size:]):.3f}",
            f"収束後安定性: {'良好' if convergence_point and len(rewards) - convergence_point > 50 else '要改善'}"
        ]
        
        y_pos = 0.8
        for text in summary_text:
            axes[1, 1].text(0.1, y_pos, text, fontsize=12, transform=axes[1, 1].transAxes)
            y_pos -= 0.15
        
        axes[1, 1].set_title('収束分析サマリー')
        
        plt.tight_layout()
        
        # 保存
        output_path = stats_file.replace('.pkl', '_convergence_analysis.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ 収束分析保存: {output_path}")
        
        plt.show()
        
        return {
            'convergence_point': convergence_point,
            'final_mean_reward': np.mean(rewards[-window_size:]),
            'final_variance': np.var(rewards[-window_size:]),
            'converged': convergence_point is not None
        }
        
    except Exception as e:
        print(f"❌ 収束分析エラー: {e}")
        return None


def generate_performance_summary_report(stats_dir: str):
    """全セッションの性能サマリーレポート生成"""
    try:
        if not os.path.exists(stats_dir):
            print(f"❌ ディレクトリが存在しません: {stats_dir}")
            return None
        
        # 統計ファイル検索
        stats_files = [os.path.join(stats_dir, f) for f in os.listdir(stats_dir)
                      if f.startswith('training_stats_') and f.endswith('.pkl')]
        
        if not stats_files:
            print(f"❌ 統計ファイルが見つかりません")
            return None
        
        print(f"📊 {len(stats_files)}セッションの性能サマリー生成中...")
        
        summary_data = []
        
        for stats_file in stats_files:
            try:
                with open(stats_file, 'rb') as f:
                    stats = pickle.load(f)
                
                # 基本統計
                rewards = stats.get('total_rewards', [])
                accuracy = stats.get('classification_accuracy', [])
                forces = stats.get('grip_force_history', [])
                
                if not rewards:
                    continue
                
                # 性能指標計算
                session_data = {
                    'session_file': os.path.basename(stats_file),
                    'total_episodes': len(rewards),
                    'mean_reward': np.mean(rewards),
                    'max_reward': np.max(rewards),
                    'final_20_mean': np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards),
                    'reward_std': np.std(rewards),
                    'mean_accuracy': np.mean(accuracy) if accuracy else 0,
                    'final_20_accuracy': np.mean(accuracy[-20:]) if len(accuracy) >= 20 else (np.mean(accuracy) if accuracy else 0),
                    'mean_grip_force': np.mean(forces) if forces else 0,
                    'ideal_grip_rate': sum(1 for f in forces if 8 <= f <= 15) / len(forces) if forces else 0,
                    'learning_efficiency': np.mean(rewards) * (np.mean(accuracy) if accuracy else 0.5),
                    'start_time': stats.get('start_time'),
                    'learning_duration': time.time() - stats.get('start_time', time.time()) if stats.get('start_time') else 0
                }
                
                summary_data.append(session_data)
                
            except Exception as e:
                print(f"⚠️ ファイル読み込みエラー: {os.path.basename(stats_file)} - {e}")
                continue
        
        if not summary_data:
            print(f"❌ 有効なデータがありません")
            return None
        
        # データフレーム作成
        df = pd.DataFrame(summary_data)
        
        # 性能ランキング
        df_sorted = df.sort_values('learning_efficiency', ascending=False)
        
        # CSVレポート保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(stats_dir, f"performance_summary_{timestamp}.csv")
        df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 可視化レポート
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DDPG学習セッション性能サマリー', fontsize=16)
        
        # 1. 学習効率ランキング
        top_sessions = df_sorted.head(10)
        axes[0, 0].barh(range(len(top_sessions)), top_sessions['learning_efficiency'])
        axes[0, 0].set_yticks(range(len(top_sessions)))
        axes[0, 0].set_yticklabels([f"S{i+1}" for i in range(len(top_sessions))])
        axes[0, 0].set_title('学習効率ランキング (Top10)')
        axes[0, 0].set_xlabel('学習効率')
        
        # 2. 報酬分布
        axes[0, 1].hist(df['mean_reward'], bins=15, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('平均報酬分布')
        axes[0, 1].set_xlabel('平均報酬')
        axes[0, 1].set_ylabel('セッション数')
        
        # 3. 分類精度分布
        axes[0, 2].hist(df['mean_accuracy'] * 100, bins=15, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('平均分類精度分布')
        axes[0, 2].set_xlabel('平均精度 (%)')
        axes[0, 2].set_ylabel('セッション数')
        
        # 4. 報酬 vs 精度散布図
        axes[1, 0].scatter(df['mean_accuracy'] * 100, df['mean_reward'], alpha=0.6)
        axes[1, 0].set_xlabel('平均分類精度 (%)')
        axes[1, 0].set_ylabel('平均報酬')
        axes[1, 0].set_title('分類精度 vs 報酬の関係')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 把持力理想範囲率
        axes[1, 1].hist(df['ideal_grip_rate'] * 100, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('理想把持力範囲率分布')
        axes[1, 1].set_xlabel('理想範囲率 (%)')
        axes[1, 1].set_ylabel('セッション数')
        
        # 6. 学習時間分布
        learning_hours = df['learning_duration'] / 3600  # 時間変換
        axes[1, 2].hist(learning_hours, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('学習時間分布')
        axes[1, 2].set_xlabel('学習時間 (時間)')
        axes[1, 2].set_ylabel('セッション数')
        
        plt.tight_layout()
        
        # 可視化レポート保存
        plot_path = os.path.join(stats_dir, f"performance_summary_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # サマリー統計表示
        print(f"\n📊 性能サマリー統計:")
        print(f"=" * 60)
        print(f"総セッション数: {len(df)}")
        print(f"平均報酬: {df['mean_reward'].mean():.3f} ± {df['mean_reward'].std():.3f}")
        print(f"最高報酬: {df['mean_reward'].max():.3f}")
        print(f"平均分類精度: {df['mean_accuracy'].mean()*100:.1f}% ± {df['mean_accuracy'].std()*100:.1f}%")
        print(f"平均理想把持力率: {df['ideal_grip_rate'].mean()*100:.1f}%")
        print(f"平均学習時間: {df['learning_duration'].mean()/3600:.1f}時間")
        print(f"\n✅ レポート保存:")
        print(f"   CSV: {csv_path}")
        print(f"   図表: {plot_path}")
        
        return df_sorted
        
    except Exception as e:
        print(f"❌ 性能サマリー生成エラー: {e}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "convergence" and len(sys.argv) > 2:
            # 収束分析実行
            stats_file = sys.argv[2]
            analyze_training_convergence(stats_file)
        
        elif command == "summary" and len(sys.argv) > 2:
            # 性能サマリー実行
            stats_dir = sys.argv[2]
            generate_performance_summary_report(stats_dir)
        
        else:
            print(f"使用方法:")
            print(f"  python {sys.argv[0]}                    # 通常実行")
            print(f"  python {sys.argv[0]} convergence <file> # 収束分析")
            print(f"  python {sys.argv[0]} summary <dir>      # 性能サマリー")
    else:
        # 通常のメイン実行
        main()