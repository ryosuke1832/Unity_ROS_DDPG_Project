#!/usr/bin/env python3
"""
LSL リアルタイム分類器（関数化版）
学習済みEEG判別機を使用してリアルタイムでOverGrip/UnderGrip/Correctを判定

使用フロー:
1. TCPトリガー受信
2. トリガーから3.2秒さかのぼって1.2秒間のEEGデータを取得
3. 関数化された分類器で分類
4. 結果をデバッグ出力
"""

import numpy as np
import socket
import time
import threading
from collections import deque
from datetime import datetime
import json
import sys

# 既存システムから拡張
from systems.episode_contact_sync_system import EpisodeContactSynchronizer
from lsl_data_send.eeg_neuroadaptation_preprocessor import NeuroadaptationEEGPreprocessor

# 関数化されたEEG判別機をインポート
from eeg_classifier_function import classify_eeg_epoch

# LSL関連
try:
    from pylsl import StreamInlet, resolve_streams
    print("✅ pylsl インポート成功")
except ImportError as e:
    print(f"❌ pylsl インポートエラー: {e}")
    print("pip install pylsl を実行してください")
    sys.exit(1)
except Exception as e:
    print(f"❌ pylsl 読み込み時の予期しないエラー: {e}")
    print("詳細なエラー情報:")
    import traceback
    traceback.print_exc()
    sys.exit(1)


class LSLRealtimeClassifier:
    """
    LSL リアルタイム分類器（関数化版）
    関数化されたEEG判別機を使用してリアルタイム判定
    """
    
    def __init__(self, 
                 model_path='./models/best_eeg_classifier.pth',
                 tcp_host='127.0.0.1', 
                 tcp_port=12345,
                 lsl_stream_name='MockEEG',
                 epoch_duration=1.2,  # エポック長 (秒)
                 lookback_duration=3.2,  # さかのぼり時間 (秒)
                 sampling_rate=250,
                 enable_preprocessing=True):
        
        self.model_path = model_path
        self.tcp_host = tcp_host
        self.tcp_port = tcp_port
        self.lsl_stream_name = lsl_stream_name
        self.epoch_duration = epoch_duration
        self.lookback_duration = lookback_duration
        self.sampling_rate = sampling_rate
        self.enable_preprocessing = enable_preprocessing
        
        # 設定
        self.epoch_samples = int(epoch_duration * sampling_rate)  # 300サンプル
        self.lookback_samples = int(lookback_duration * sampling_rate)  # 800サンプル
        
        # 分類ラベル
        self.class_names = ['Correct', 'UnderGrip', 'OverGrip']
        
        # データバッファ
        self.eeg_buffer = deque(maxlen=self.lookback_samples)
        self.buffer_lock = threading.Lock()
        
        # 前処理器（オプション）
        if self.enable_preprocessing:
            self.preprocessor = NeuroadaptationEEGPreprocessor(
                sampling_rate=sampling_rate,
                filter_lowcut=1,
                filter_highcut=40,
                enable_asr=True,
                enable_ica=False
            )
        else:
            self.preprocessor = None
        
        # 状態管理
        self.running = False
        
        # 統計
        self.classification_count = 0
        self.start_time = None
        
        print(f"🤖 LSL リアルタイム分類器（関数化版） 初期化完了")
        print(f"   モデル: {model_path}")
        print(f"   TCP: {tcp_host}:{tcp_port}")
        print(f"   LSL: {lsl_stream_name}")
        print(f"   エポック: {epoch_duration}秒 ({self.epoch_samples}サンプル)")
        print(f"   前処理: {'有効' if enable_preprocessing else '無効'}")
    
    def setup_lsl_connection(self):
        """LSL接続の設定"""
        try:
            print(f"🔍 LSLストリーム検索中: {self.lsl_stream_name}")
            
            # ストリーム検索
            streams = resolve_streams()
            
            if not streams:
                raise RuntimeError(f"LSLストリームが見つかりません")
            
            # 指定した名前のストリームを検索
            target_stream = None
            print("利用可能なストリーム:")
            for stream in streams:
                stream_name = stream.name()
                print(f"  - {stream_name} ({stream.type()})")
                if stream_name == self.lsl_stream_name:
                    target_stream = stream
            
            if target_stream is None:
                raise RuntimeError(f"指定されたストリーム '{self.lsl_stream_name}' が見つかりません")
            
            # ストリームに接続
            self.lsl_inlet = StreamInlet(target_stream)
            
            # ストリーム情報取得
            info = self.lsl_inlet.info()
            self.n_channels = info.channel_count()
            self.stream_sampling_rate = info.nominal_srate()
            
            print(f"✅ LSL接続完了")
            print(f"   チャンネル数: {self.n_channels}")
            print(f"   サンプリングレート: {self.stream_sampling_rate} Hz")
            
            # サンプリングレート検証
            if abs(self.stream_sampling_rate - self.sampling_rate) > 1:
                print(f"⚠️ サンプリングレート不一致: 期待値{self.sampling_rate}, 実際{self.stream_sampling_rate}")
            
            return True
            
        except Exception as e:
            print(f"❌ LSL接続エラー: {e}")
            print("📝 トラブルシューティング:")
            print("   1. mock_eeg_sender.py が起動しているか確認")
            print("   2. sender側で 'start' コマンドを実行")
            print("   3. ストリーム名が正しいか確認")
            return False
    
    def setup_tcp_connection(self):
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
                # LSLからデータ取得
                sample, timestamp = self.lsl_inlet.pull_sample(timeout=1.0)
                
                if sample is not None:
                    with self.buffer_lock:
                        # バッファに追加（32チャンネル対応）
                        if len(sample) >= 32:
                            self.eeg_buffer.append(sample[:32])
                        else:
                            # チャンネル数が足りない場合はゼロパディング
                            padded_sample = sample + [0.0] * (32 - len(sample))
                            self.eeg_buffer.append(padded_sample)
                
            except Exception as e:
                if self.running:  # 停止時のエラーは無視
                    print(f"⚠️ LSLデータ受信エラー: {e}")
                time.sleep(0.001)
        
        print(f"🔄 LSLデータ受信終了")
    
    def tcp_trigger_thread(self):
        """TCPトリガー受信スレッド"""
        print(f"📡 TCP待機開始")
        
        while self.running:
            try:
                # クライアント接続待機
                client_socket, client_address = self.tcp_socket.accept()
                print(f"📡 TCP接続受付: {client_address}")
                
                while self.running:
                    try:
                        # データ受信
                        data = client_socket.recv(1024)
                        if not data:
                            break
                        
                        # トリガー処理
                        trigger_data = data.decode('utf-8').strip()
                        self.process_trigger(trigger_data)
                        
                    except Exception as e:
                        print(f"⚠️ TCPデータ受信エラー: {e}")
                        break
                
                client_socket.close()
                print(f"📡 TCP接続終了: {client_address}")
                
            except Exception as e:
                if self.running:
                    print(f"⚠️ TCP接続エラー: {e}")
                time.sleep(1.0)
        
        print(f"📡 TCP待機終了")
    
    def process_trigger(self, trigger_data):
        """トリガー処理とEEG分類"""
        try:
            trigger_time = time.time()
            
            print(f"\n🎯 トリガー受信: {trigger_data}")
            print(f"   受信時刻: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            # EEGエポック取得
            eeg_epoch = self.extract_eeg_epoch(trigger_time)
            
            if eeg_epoch is None:
                print(f"❌ EEGエポック取得失敗")
                return
            
            # 関数化された分類器で分類実行
            prediction, class_id, confidence = self.classify_eeg_with_function(eeg_epoch)
            
            # 結果出力
            self.output_classification_result(prediction, class_id, confidence, trigger_data)
            
        except Exception as e:
            print(f"❌ トリガー処理エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def extract_eeg_epoch(self, trigger_time):
        """EEGエポックの抽出"""
        try:
            with self.buffer_lock:
                if len(self.eeg_buffer) < self.epoch_samples:
                    print(f"⚠️ バッファ不足: {len(self.eeg_buffer)}/{self.epoch_samples}")
                    return None
                
                # 3.2秒さかのぼって1.2秒間のデータを取得
                start_idx = len(self.eeg_buffer) - self.lookback_samples
                end_idx = start_idx + self.epoch_samples
                
                if start_idx < 0:
                    print(f"⚠️ さかのぼり範囲不足")
                    # 最初から1.2秒分取得
                    epoch_data = list(self.eeg_buffer)[-self.epoch_samples:]
                else:
                    # 指定範囲から取得
                    epoch_data = list(self.eeg_buffer)[start_idx:end_idx]
                
                # numpy配列に変換 (300, 32)
                eeg_epoch = np.array(epoch_data, dtype=np.float32)
                
                print(f"🔍 EEGエポック抽出: shape={eeg_epoch.shape}")
                
                return eeg_epoch
                
        except Exception as e:
            print(f"❌ EEGエポック抽出エラー: {e}")
            return None
    
    def classify_eeg_with_function(self, eeg_epoch):
        """
        関数化されたEEG分類器を使用してエポックを分類
        
        Args:
            eeg_epoch (np.ndarray): EEGエポックデータ (300, 32)
            
        Returns:
            tuple: (予測クラス名, クラスID, 信頼度)
        """
        try:
            print(f"🔍 関数化分類器呼び出し: shape={eeg_epoch.shape}")
            
            # オプション：前処理実行
            if self.enable_preprocessing and self.preprocessor is not None:
                print(f"🔧 前処理実行中...")
                preprocess_result = self.preprocessor.preprocess_epoch(eeg_epoch)
                
                # 前処理結果から実際のデータを取得
                if isinstance(preprocess_result, dict):
                    processed_epoch = preprocess_result.get('processed_epoch', eeg_epoch)
                else:
                    processed_epoch = preprocess_result
                
                print(f"🔧 前処理完了: shape={processed_epoch.shape}")
            else:
                processed_epoch = eeg_epoch
                print(f"🔧 前処理スキップ")
            
            # 関数化された分類器で分類実行
            class_name, class_id, confidence = classify_eeg_epoch(
                eeg_data=processed_epoch,
                model_path=self.model_path,
                class_names=self.class_names
            )
            
            print(f"🎯 関数化分類完了: {class_name} (ID: {class_id}, 信頼度: {confidence:.3f})")
            
            return class_name, class_id, confidence
            
        except Exception as e:
            print(f"❌ 関数化EEG分類エラー: {e}")
            import traceback
            traceback.print_exc()
            return "ERROR", -1, 0.0
    
    def output_classification_result(self, prediction, class_id, confidence, trigger_data):
        """分類結果の出力"""
        self.classification_count += 1
        
        # 分類結果
        if prediction != "ERROR":
            print(f"🎯 分類結果: {prediction}")
            print(f"   信頼度: {confidence:.3f}")
            print(f"   予測クラスID: {class_id}")
        else:
            print(f"❌ 分類失敗")
        
        # 統計情報
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        classification_rate = self.classification_count / elapsed_time if elapsed_time > 0 else 0
        
        print(f"📊 統計: {self.classification_count}回分類, {classification_rate:.2f}回/秒")
        print(f"   時刻: {datetime.now().strftime('%H:%M:%S')}")
        
        # デバッグ出力（要求仕様）
        debug_output = {
            'timestamp': datetime.now().isoformat(),
            'trigger': trigger_data,
            'classification': prediction,
            'prediction_class': class_id,
            'confidence': confidence,
            'count': self.classification_count,
            'method': 'function_based'  # 関数化版であることを明示
        }
        
        # print(f"🐛 DEBUG: {json.dumps(debug_output, ensure_ascii=False)}")
    
    def run(self):
        """メイン実行"""
        print(f"\n🚀 LSL リアルタイム分類器（関数化版） 開始")
        
        # 初期化
        if not self.setup_lsl_connection():
            return False
        
        if not self.setup_tcp_connection():
            return False
        
        self.running = True
        self.start_time = time.time()
        
        try:
            # データ受信スレッド開始
            lsl_thread = threading.Thread(target=self.lsl_data_thread, daemon=True)
            tcp_thread = threading.Thread(target=self.tcp_trigger_thread, daemon=True)
            
            lsl_thread.start()
            tcp_thread.start()
            
            print(f"✅ 全システム稼働中（関数化版）")
            print(f"   LSLデータ受信: 開始")
            print(f"   TCP待機: {self.tcp_host}:{self.tcp_port}")
            print(f"   分類器: 関数化版待機中")
            print(f"\n💡 使用方法:")
            print(f"   1. Unity等からTCP {self.tcp_port}にトリガー送信")
            print(f"   2. 自動でEEG分類実行（関数化版）")
            print(f"   3. 結果をデバッグ出力")
            print(f"   4. Ctrl+Cで終了")
            
            # メインループ
            while True:
                time.sleep(1.0)
                
                # バッファ状況表示（1分おき）
                if int(time.time()) % 60 == 0:
                    with self.buffer_lock:
                        buffer_size = len(self.eeg_buffer)
                        buffer_percentage = (buffer_size / self.lookback_samples) * 100
                        
                    print(f"📊 バッファ状況: {buffer_size}/{self.lookback_samples} ({buffer_percentage:.1f}%)")
                    
        except KeyboardInterrupt:
            print(f"\n⏹️ 停止要求受信")
        except Exception as e:
            print(f"\n❌ 実行エラー: {e}")
        finally:
            self.stop()
            
        return True
    
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
        
        # 統計出力
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        print(f"📊 最終統計（関数化版）:")
        print(f"   実行時間: {elapsed_time:.1f}秒")
        print(f"   分類回数: {self.classification_count}回")
        print(f"   平均レート: {self.classification_count/elapsed_time:.2f}回/秒" if elapsed_time > 0 else "   平均レート: N/A")
        
        print(f"✅ システム停止完了")


def main():
    """メイン関数 - 関数化版分類器"""
    print("🧠 LSL リアルタイム分類器（関数化版）")
    print("=" * 50)
    
    # モデルパス指定
    model_path = './models/best_eeg_classifier.pth'
    
    print(f"📋 設定:")
    print(f"   使用モデル: {model_path}")
    print(f"   TCP接続: 127.0.0.1:12345")
    print(f"   LSLストリーム: MockEEG")
    print(f"   分類対象: OverGrip/UnderGrip/Correct")
    print(f"   エポック: 3.2秒さかのぼり + 1.2秒間")
    print(f"   実装: 関数化されたEEG判別機使用")
    
    # 分類器作成・実行
    classifier = LSLRealtimeClassifier(
        model_path=model_path,
        tcp_host='127.0.0.1',
        tcp_port=12345,
        lsl_stream_name='MockEEG',
        epoch_duration=1.2,
        lookback_duration=3.2,
        sampling_rate=250,
        enable_preprocessing=True  # 前処理の有効/無効を選択可能
    )
    
    # 実行
    success = classifier.run()
    
    if success:
        print(f"✅ 正常終了")
    else:
        print(f"❌ エラー終了")
        sys.exit(1)


if __name__ == "__main__":
    main()