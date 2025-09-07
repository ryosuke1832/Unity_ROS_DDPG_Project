#!/usr/bin/env python3
"""
DDPG強化学習システム統合実行スクリプト

使用方法:
1. 分類器の学習: python ddpg_main_runner.py train_classifier
2. リアルタイム学習: python ddpg_main_runner.py realtime_learning
3. 長期学習: python ddpg_main_runner.py self_training
4. 学習監視: python ddpg_main_runner.py monitor
5. 全システム統合実行: python ddpg_main_runner.py full_system

機能:
- 各システムの自動セットアップ
- 依存関係の確認
- 設定ファイルの管理
- ログ出力の統合
"""

import os
import sys
import time
import subprocess
import threading
import signal
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import glob

# プロジェクトモジュールのインポート
try:
    from grip_force_classifier import train_grip_force_classifier, RealtimeGripForceClassifier
    from tcp_lsl_sync_system import LSLTCPEpisodeCollector
    from unity_tcp_interface import EEGTCPInterface
    
    # DDPGシステム（上記で作成したコード）
    from ddpg_grip_force_rl_system import DDPGGripForceSystem
    from ddpg_training_monitor import DDPGTrainingMonitor
    
    IMPORTS_OK = True
except ImportError as e:
    print(f"⚠️ モジュールインポートエラー: {e}")
    print(f"   必要なファイルが同一ディレクトリにあることを確認してください")
    IMPORTS_OK = False


class DDPGSystemConfig:
    """DDPG システム設定管理"""
    
    def __init__(self, config_path="config/ddpg_config.json"):
        self.config_path = config_path
        self.config = self._load_default_config()
        
        # 設定ファイルから読み込み
        if os.path.exists(config_path):
            self._load_config()
        else:
            self._save_config()  # デフォルト設定を保存
    
    def _load_default_config(self):
        """デフォルト設定"""
        return {
            "system": {
                "lsl_stream_name": "MockEEG",
                "tcp_host": "127.0.0.1",
                "tcp_port": 12345,
                "sampling_rate": 250,
                "enable_gpu": True
            },
            "classifier": {
                "model_path": "models/best_grip_force_classifier.pth",
                "csv_data_dir": "DDPG_Python/logs",
                "min_episodes_for_training": 10
            },
            "ddpg": {
                "state_dim": 5,
                "action_dim": 1,
                "lr_actor": 1e-4,
                "lr_critic": 1e-3,
                "gamma": 0.99,
                "tau": 0.001,
                "noise_std": 0.2,
                "buffer_capacity": 100000,
                "batch_size": 64
            },
            "training": {
                "max_episodes": 1000,
                "save_interval": 50,
                "evaluation_interval": 100,
                "early_stopping_patience": 200
            },
            "monitoring": {
                "update_interval": 5.0,
                "save_plots": True,
                "stats_dir": "models/ddpg"
            }
        }
    
    def _load_config(self):
        """設定ファイル読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
            
            # デフォルト設定に上書き
            self._update_config(self.config, loaded_config)
            print(f"✅ 設定ファイル読み込み: {self.config_path}")
            
        except Exception as e:
            print(f"⚠️ 設定ファイル読み込みエラー: {e}")
            print(f"   デフォルト設定を使用します")
    
    def _update_config(self, base_config, new_config):
        """設定の再帰的更新"""
        for key, value in new_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _save_config(self):
        """設定ファイル保存"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            
            print(f"💾 設定ファイル保存: {self.config_path}")
            
        except Exception as e:
            print(f"⚠️ 設定ファイル保存エラー: {e}")
    
    def get(self, section, key=None):
        """設定値取得"""
        if key is None:
            return self.config.get(section, {})
        return self.config.get(section, {}).get(key)
    
    def set(self, section, key, value):
        """設定値設定"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
        self._save_config()


class DDPGSystemRunner:
    """DDPG システム統合実行クラス"""
    
    def __init__(self, config_path="config/ddpg_config.json"):
        self.config = DDPGSystemConfig(config_path)
        self.running_processes = []
        self.running_threads = []
        self.is_running = False
        
        # シグナルハンドラー設定
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"🚀 DDPG システム統合ランナー初期化完了")
    
    def _signal_handler(self, signum, frame):
        """シグナルハンドラー（Ctrl+C対応）"""
        print(f"\n🛑 終了シグナル受信: {signum}")
        self.stop_all_systems()
        sys.exit(0)
    
    def check_dependencies(self):
        """依存関係チェック"""
        print(f"🔍 依存関係チェック開始")
        
        # 必須ディレクトリ
        required_dirs = ["models", "logs", "config"]
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                print(f"📁 ディレクトリ作成: {dir_name}")
        
        # Pythonモジュール
        required_modules = [
            "torch", "numpy", "pandas", "matplotlib", 
            "seaborn", "scikit-learn", "pylsl"
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print(f"❌ 不足モジュール: {missing_modules}")
            print(f"   以下のコマンドでインストールしてください:")
            print(f"   pip install {' '.join(missing_modules)}")
            return False
        
        # プロジェクトファイル
        if not IMPORTS_OK:
            print(f"❌ プロジェクトモジュールが不足しています")
            return False
        
        print(f"✅ 依存関係チェック完了")
        return True
    
    def train_classifier(self, csv_data_dir=None):
        """分類器学習"""
        print(f"🎓 EEG把持力分類器学習開始")
        
        if csv_data_dir is None:
            csv_data_dir = self.config.get("classifier", "csv_data_dir")
        
        # 最新のエピソードディレクトリを検索
        if os.path.exists(csv_data_dir):
            episode_dirs = glob.glob(os.path.join(csv_data_dir, "episodes_*"))
            if episode_dirs:
                latest_dir = max(episode_dirs, key=os.path.getmtime)
                csv_data_dir = latest_dir
                print(f"📂 使用データディレクトリ: {csv_data_dir}")
            else:
                print(f"❌ エピソードデータが見つかりません: {csv_data_dir}")
                return False
        else:
            print(f"❌ データディレクトリが存在しません: {csv_data_dir}")
            return False
        
        # 分類器学習実行
        try:
            model_path = self.config.get("classifier", "model_path")
            result = train_grip_force_classifier(csv_data_dir, model_path)
            
            if result and result.get('test_accuracy', 0) > 30:
                print(f"✅ 分類器学習成功")
                print(f"   モデル保存: {result['model_path']}")
                print(f"   テスト精度: {result['test_accuracy']:.1f}%")
                return True
            else:
                print(f"❌ 分類器学習失敗または精度不足")
                return False
                
        except Exception as e:
            print(f"❌ 分類器学習エラー: {e}")
            return False
    
    def run_realtime_learning(self):
        """リアルタイム学習実行"""
        print(f"🔴 リアルタイム学習モード開始")
        
        if not self.check_dependencies():
            return False
        
        try:
            # DDPG システム初期化
            ddpg_system = DDPGGripForceSystem(
                classifier_model_path=self.config.get("classifier", "model_path"),
                lsl_stream_name=self.config.get("system", "lsl_stream_name"),
                tcp_host=self.config.get("system", "tcp_host"),
                tcp_port=self.config.get("system", "tcp_port")
            )
            
            # リアルタイム学習開始
            if ddpg_system.start_realtime_learning_mode():
                self.is_running = True
                
                print(f"✅ リアルタイム学習開始完了")
                print(f"💡 システム稼働中...")
                print(f"   LSL/TCPデータ受信 → EEG分類 → DDPG学習")
                print(f"   Unity把持力リクエストに自動応答")
                print(f"   Ctrl+C で終了")
                
                # メインループ
                try:
                    while self.is_running and ddpg_system.is_running:
                        time.sleep(1.0)
                        
                        # 定期的な統計表示
                        if ddpg_system.stats['total_episodes'] > 0:
                            if ddpg_system.stats['total_episodes'] % 20 == 0:
                                self._print_learning_progress(ddpg_system)
                
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    ddpg_system.stop_learning()
                    self.is_running = False
                
                return True
            else:
                print(f"❌ リアルタイム学習開始失敗")
                return False
                
        except Exception as e:
            print(f"❌ リアルタイム学習エラー: {e}")
            return False
    
    def run_self_training(self, pretrained_model=None):
        """長期学習モード実行"""
        print(f"🔵 長期学習モード（自己学習）開始")
        
        if not self.check_dependencies():
            return False
        
        try:
            # DDPG システム初期化
            ddpg_system = DDPGGripForceSystem(
                classifier_model_path=self.config.get("classifier", "model_path"),
                lsl_stream_name=self.config.get("system", "lsl_stream_name"),
                tcp_host=self.config.get("system", "tcp_host"),
                tcp_port=self.config.get("system", "tcp_port")
            )
            
            # 事前学習モデル読み込み
            if pretrained_model and os.path.exists(pretrained_model):
                ddpg_system.agent.load_model(pretrained_model)
                print(f"📂 事前学習モデル読み込み: {pretrained_model}")
            
            # 長期学習開始
            if ddpg_system.start_self_training_mode(pretrained_model):
                self.is_running = True
                
                print(f"✅ 長期学習開始完了")
                print(f"💡 自己学習実行中...")
                print(f"   シミュレーション環境でDDPGエージェント学習")
                print(f"   Ctrl+C で終了")
                
                # メインループ
                try:
                    while self.is_running and ddpg_system.is_running:
                        time.sleep(5.0)
                        
                        # 定期的な進捗表示
                        if ddpg_system.stats['total_episodes'] > 0:
                            if ddpg_system.stats['total_episodes'] % 50 == 0:
                                self._print_learning_progress(ddpg_system)
                
                except KeyboardInterrupt:
                    print(f"\n⏹️ ユーザー停止")
                finally:
                    ddpg_system.stop_learning()
                    self.is_running = False
                
                return True
            else:
                print(f"❌ 長期学習開始失敗")
                return False
                
        except Exception as e:
            print(f"❌ 長期学習エラー: {e}")
            return False
    
    def run_monitoring(self):
        """学習監視実行"""
        print(f"📊 DDPG学習監視開始")
        
        try:
            stats_dir = self.config.get("monitoring", "stats_dir")
            update_interval = self.config.get("monitoring", "update_interval")
            
            monitor = DDPGTrainingMonitor(
                stats_dir=stats_dir,
                update_interval=update_interval,
                save_plots=self.config.get("monitoring", "save_plots")
            )
            
            if monitor.start_monitoring():
                self.is_running = True
                
                print(f"✅ 監視開始完了")
                print(f"💡 リアルタイムプロット表示中...")
                print(f"   統計ファイルの更新を監視")
                print(f"   Ctrl+C で終了")
                
                try:
                    while self.is_running and monitor.is_monitoring:
                        time.sleep(1.0)
                        
                        # 定期的なステータス表示
                        if int(time.time()) % 30 == 0:  # 30秒ごと
                            monitor.print_current_status()
                
                except KeyboardInterrupt:
                    print(f"\n⏹️ 監視停止")
                finally:
                    monitor.stop_monitoring()
                    self.is_running = False
                
                return True
            else:
                print(f"❌ 監視開始失敗")
                return False
                
        except Exception as e:
            print(f"❌ 監視エラー: {e}")
            return False
    
    def run_full_system(self):
        """全システム統合実行"""
        print(f"🔄 DDPG強化学習システム全体統合実行")
        print(f"=" * 60)
        
        if not self.check_dependencies():
            return False
        
        # Step 1: 分類器の確認・学習
        classifier_path = self.config.get("classifier", "model_path")
        if not os.path.exists(classifier_path):
            print(f"🎓 Step 1: EEG分類器学習")
            if not self.train_classifier():
                print(f"❌ 分類器学習失敗 - システム停止")
                return False
        else:
            print(f"✅ Step 1: 既存分類器を使用: {classifier_path}")
        
        # Step 2: データ収集システム開始
        print(f"📡 Step 2: データ収集システム開始")
        data_collector = self._start_data_collection()
        if not data_collector:
            print(f"❌ データ収集システム開始失敗")
            return False
        
        # Step 3: 学習監視システム開始
        print(f"📊 Step 3: 学習監視システム開始")
        monitor_thread = threading.Thread(target=self._run_monitoring_thread, daemon=True)
        monitor_thread.start()
        self.running_threads.append(monitor_thread)
        
        # Step 4: DDPG学習システム開始
        print(f"🤖 Step 4: DDPG学習システム開始")
        learning_success = self._start_ddpg_learning()
        
        if learning_success:
            print(f"✅ 全システム統合実行完了")
            print(f"💡 統合システム稼働中...")
            print(f"   データ収集 → EEG分類 → DDPG学習 → 把持力出力")
            print(f"   リアルタイム監視プロット表示")
            print(f"   Ctrl+C で全システム停止")
            
            try:
                while self.is_running:
                    time.sleep(2.0)
            except KeyboardInterrupt:
                print(f"\n⏹️ ユーザー停止")
            finally:
                self.stop_all_systems()
            
            return True
        else:
            print(f"❌ DDPG学習システム開始失敗")
            self.stop_all_systems()
            return False
    
    def _start_data_collection(self):
        """データ収集システム開始"""
        try:
            collector = LSLTCPEpisodeCollector(
                lsl_stream_name=self.config.get("system", "lsl_stream_name"),
                tcp_host=self.config.get("system", "tcp_host"),
                tcp_port=self.config.get("system", "tcp_port"),
                sampling_rate=self.config.get("system", "sampling_rate"),
                save_to_csv=True
            )
            
            if collector.start_collection():
                print(f"✅ データ収集システム開始完了")
                return collector
            else:
                return None
                
        except Exception as e:
            print(f"❌ データ収集システムエラー: {e}")
            return None
    
    def _run_monitoring_thread(self):
        """監視スレッド実行"""
        try:
            self.run_monitoring()
        except Exception as e:
            print(f"⚠️ 監視スレッドエラー: {e}")
    
    def _start_ddpg_learning(self):
        """DDPG学習システム開始"""
        try:
            return self.run_realtime_learning()
        except Exception as e:
            print(f"❌ DDPG学習システムエラー: {e}")
            return False
    
    def _print_learning_progress(self, ddpg_system):
        """学習進捗表示"""
        stats = ddpg_system.stats
        
        if stats['total_rewards']:
            recent_rewards = stats['total_rewards'][-10:]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            print(f"📈 学習進捗: EP={stats['total_episodes']}, "
                  f"平均報酬={avg_reward:.3f}, "
                  f"学習ステップ={ddpg_system.agent.training_step}")
    
    def stop_all_systems(self):
        """全システム停止"""
        print(f"🛑 全システム停止処理開始...")
        
        self.is_running = False
        
        # 実行中スレッドの停止
        for thread in self.running_threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
        
        # 実行中プロセスの停止
        for process in self.running_processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5.0)
        
        print(f"✅ 全システム停止完了")
    
    def show_status(self):
        """システム状態表示"""
        print(f"\n📊 DDPG強化学習システム状態")
        print(f"=" * 50)
        
        # 設定確認
        print(f"🔧 設定:")
        print(f"   LSLストリーム: {self.config.get('system', 'lsl_stream_name')}")
        print(f"   TCP接続: {self.config.get('system', 'tcp_host')}:{self.config.get('system', 'tcp_port')}")
        print(f"   分類器モデル: {self.config.get('classifier', 'model_path')}")
        
        # ファイル存在確認
        print(f"\n📁 ファイル確認:")
        classifier_path = self.config.get("classifier", "model_path")
        print(f"   分類器モデル: {'✅' if os.path.exists(classifier_path) else '❌'} {classifier_path}")
        
        stats_dir = self.config.get("monitoring", "stats_dir")
        print(f"   統計ディレクトリ: {'✅' if os.path.exists(stats_dir) else '❌'} {stats_dir}")
        
        # 最新統計
        if os.path.exists(stats_dir):
            stats_files = glob.glob(os.path.join(stats_dir, "training_stats_*.pkl"))
            print(f"   統計ファイル数: {len(stats_files)}")
            
            if stats_files:
                latest_file = max(stats_files, key=os.path.getmtime)
                mod_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
                print(f"   最新統計: {os.path.basename(latest_file)} ({mod_time.strftime('%Y-%m-%d %H:%M:%S')})")


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="DDPG強化学習システム統合実行")
    parser.add_argument("command", choices=[
        "train_classifier", "realtime_learning", "self_training", 
        "monitor", "full_system", "status", "check_deps"
    ], help="実行コマンド")
    
    parser.add_argument("--config", default="config/ddpg_config.json", 
                       help="設定ファイルパス")
    parser.add_argument("--csv-dir", help="分類器学習用CSVディレクトリ")
    parser.add_argument("--pretrained-model", help="事前学習モデルパス")
    
    args = parser.parse_args()
    
    print(f"🚀 DDPG強化学習システム統合ランナー")
    print(f"=" * 60)
    print(f"コマンド: {args.command}")
    print(f"設定ファイル: {args.config}")
    print(f"")
    
    # システムランナー初期化
    runner = DDPGSystemRunner(args.config)
    
    try:
        if args.command == "check_deps":
            # 依存関係チェック
            success = runner.check_dependencies()
            print(f"依存関係チェック: {'✅ 完了' if success else '❌ 失敗'}")
        
        elif args.command == "train_classifier":
            # 分類器学習
            success = runner.train_classifier(args.csv_dir)
            print(f"分類器学習: {'✅ 成功' if success else '❌ 失敗'}")
        
        elif args.command == "realtime_learning":
            # リアルタイム学習
            success = runner.run_realtime_learning()
            print(f"リアルタイム学習: {'✅ 完了' if success else '❌ 失敗'}")
        
        elif args.command == "self_training":
            # 長期学習
            success = runner.run_self_training(args.pretrained_model)
            print(f"長期学習: {'✅ 完了' if success else '❌ 失敗'}")
        
        elif args.command == "monitor":
            # 学習監視
            success = runner.run_monitoring()
            print(f"学習監視: {'✅ 完了' if success else '❌ 失敗'}")
        
        elif args.command == "full_system":
            # 全システム統合実行
            success = runner.run_full_system()
            print(f"全システム統合実行: {'✅ 完了' if success else '❌ 失敗'}")
        
        elif args.command == "status":
            # システム状態表示
            runner.show_status()
        
        else:
            print(f"❌ 不明なコマンド: {args.command}")
    
    except Exception as e:
        print(f"❌ 実行エラー: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        runner.stop_all_systems()
    
    print(f"\n👋 DDPG強化学習システム終了")


def interactive_mode():
    """インタラクティブモード"""
    print(f"🎮 DDPG強化学習システム - インタラクティブモード")
    print(f"=" * 60)
    
    runner = DDPGSystemRunner()
    
    while True:
        print(f"\n実行したい操作を選択してください:")
        print(f"1. 依存関係チェック")
        print(f"2. EEG分類器学習")
        print(f"3. リアルタイム学習モード")
        print(f"4. 長期学習モード（自己学習）")
        print(f"5. 学習監視")
        print(f"6. 全システム統合実行")
        print(f"7. システム状態確認")
        print(f"8. 終了")
        
        choice = input(f"\n選択 (1-8): ").strip()
        
        try:
            if choice == "1":
                runner.check_dependencies()
            
            elif choice == "2":
                csv_dir = input(f"CSVディレクトリパス (空でデフォルト): ").strip()
                runner.train_classifier(csv_dir if csv_dir else None)
            
            elif choice == "3":
                runner.run_realtime_learning()
            
            elif choice == "4":
                model_path = input(f"事前学習モデルパス (空でスキップ): ").strip()
                runner.run_self_training(model_path if model_path else None)
            
            elif choice == "5":
                runner.run_monitoring()
            
            elif choice == "6":
                runner.run_full_system()
            
            elif choice == "7":
                runner.show_status()
            
            elif choice == "8":
                print(f"👋 終了します")
                break
            
            else:
                print(f"❌ 無効な選択です")
        
        except KeyboardInterrupt:
            print(f"\n⏹️ 操作中断")
        except Exception as e:
            print(f"❌ エラー: {e}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 引数なしの場合はインタラクティブモード
        interactive_mode()
    else:
        # 引数ありの場合はコマンドラインモード
        main()