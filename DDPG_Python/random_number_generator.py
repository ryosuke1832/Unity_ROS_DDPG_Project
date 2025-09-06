#!/usr/bin/env python3
"""
ランダム数値生成関数
2から30の間でランダムな数字を生成する関数を提供
"""

import random
import time

def generate_random_grip_force(min_value=2.0, max_value=30.0):
    """
    指定された範囲でランダムな把持力値を生成
    
    Args:
        min_value (float): 最小値（デフォルト: 2.0）
        max_value (float): 最大値（デフォルト: 30.0）
    
    Returns:
        float: 生成されたランダムな数値
    
    Example:
        >>> force = generate_random_grip_force()
        >>> print(f"ランダムな把持力: {force:.2f}N")
        ランダムな把持力: 15.47N
    """
    return random.uniform(min_value, max_value)


def generate_random_integer(min_value=2, max_value=30):
    """
    指定された範囲でランダムな整数を生成
    
    Args:
        min_value (int): 最小値（デフォルト: 2）
        max_value (int): 最大値（デフォルト: 30）
    
    Returns:
        int: 生成されたランダムな整数
    
    Example:
        >>> number = generate_random_integer()
        >>> print(f"ランダムな整数: {number}")
        ランダムな整数: 18
    """
    return random.randint(min_value, max_value)


def generate_seeded_random(min_value=2.0, max_value=30.0, seed=None):
    """
    シード値を指定してランダムな数値を生成（再現可能）
    
    Args:
        min_value (float): 最小値（デフォルト: 2.0）
        max_value (float): 最大値（デフォルト: 30.0）
        seed (int, optional): シード値（Noneの場合は現在時刻を使用）
    
    Returns:
        float: 生成されたランダムな数値
    
    Example:
        >>> # 同じシードで同じ結果を得る
        >>> force1 = generate_seeded_random(seed=42)
        >>> force2 = generate_seeded_random(seed=42)
        >>> print(force1 == force2)  # True
    """
    if seed is None:
        seed = int(time.time() * 1000000) % 2147483647  # マイクロ秒ベースのシード
    
    # 一時的にシードを設定
    random.seed(seed)
    result = random.uniform(min_value, max_value)
    
    # シードをリセット（他のランダム関数に影響しないように）
    random.seed()
    
    return result


def generate_weighted_random(min_value=2.0, max_value=30.0, target_value=15.0, weight=0.3):
    """
    特定の値に偏重したランダムな数値を生成
    
    Args:
        min_value (float): 最小値
        max_value (float): 最大値
        target_value (float): 偏重させたい値
        weight (float): 偏重の強さ（0.0-1.0）
    
    Returns:
        float: 生成されたランダムな数値
    
    Example:
        >>> # 15.0付近の値が出やすくなる
        >>> force = generate_weighted_random(target_value=15.0, weight=0.5)
    """
    # 通常のランダム値
    random_value = random.uniform(min_value, max_value)
    
    # 目標値との重み付き平均
    weighted_value = (1 - weight) * random_value + weight * target_value
    
    # 範囲内にクランプ
    return max(min_value, min(max_value, weighted_value))


def generate_normal_distribution_random(mean=16.0, std_dev=5.0, min_value=2.0, max_value=30.0):
    """
    正規分布に基づいてランダムな数値を生成
    
    Args:
        mean (float): 平均値
        std_dev (float): 標準偏差
        min_value (float): 最小値（クランプ用）
        max_value (float): 最大値（クランプ用）
    
    Returns:
        float: 生成されたランダムな数値
    
    Example:
        >>> # 平均16.0、標準偏差5.0の正規分布
        >>> force = generate_normal_distribution_random()
    """
    value = random.gauss(mean, std_dev)
    
    # 指定範囲内にクランプ
    return max(min_value, min(max_value, value))


class RandomGripForceGenerator:
    """
    把持力のランダム生成クラス
    設定を保持して複数回呼び出す場合に便利
    """
    
    def __init__(self, min_value=2.0, max_value=30.0, generation_mode='uniform'):
        """
        初期化
        
        Args:
            min_value (float): 最小値
            max_value (float): 最大値
            generation_mode (str): 生成モード ('uniform', 'normal', 'weighted')
        """
        self.min_value = min_value
        self.max_value = max_value
        self.generation_mode = generation_mode
        
        # 正規分布用パラメータ
        self.mean = (min_value + max_value) / 2
        self.std_dev = (max_value - min_value) / 6
        
        # 重み付き用パラメータ
        self.target_value = (min_value + max_value) / 2
        self.weight = 0.3
        
        print(f"🎲 ランダム生成器初期化: 範囲[{min_value}, {max_value}], モード: {generation_mode}")
    
    def generate(self):
        """
        設定に基づいてランダムな値を生成
        
        Returns:
            float: 生成されたランダムな数値
        """
        if self.generation_mode == 'uniform':
            return generate_random_grip_force(self.min_value, self.max_value)
        elif self.generation_mode == 'normal':
            return generate_normal_distribution_random(
                self.mean, self.std_dev, self.min_value, self.max_value
            )
        elif self.generation_mode == 'weighted':
            return generate_weighted_random(
                self.min_value, self.max_value, self.target_value, self.weight
            )
        else:
            # デフォルトは一様分布
            return generate_random_grip_force(self.min_value, self.max_value)
    
    def generate_multiple(self, count):
        """
        複数の値を一度に生成
        
        Args:
            count (int): 生成する数値の個数
        
        Returns:
            list: 生成された数値のリスト
        """
        return [self.generate() for _ in range(count)]
    
    def update_range(self, min_value, max_value):
        """
        生成範囲を更新
        
        Args:
            min_value (float): 新しい最小値
            max_value (float): 新しい最大値
        """
        self.min_value = min_value
        self.max_value = max_value
        self.mean = (min_value + max_value) / 2
        self.std_dev = (max_value - min_value) / 6
        self.target_value = (min_value + max_value) / 2


# 🎯 プロジェクト統合用の関数（既存コードとの互換性）
def generate_random_feedback(episode_info=None, min_value=2.0, max_value=30.0):
    """
    既存のepisode_contact_sync_system.pyとの互換性を保つ関数
    
    Args:
        episode_info (dict, optional): エピソード情報（今回は未使用）
        min_value (float): 最小値
        max_value (float): 最大値
    
    Returns:
        float: 生成されたランダムな把持力値
    """
    return generate_random_grip_force(min_value, max_value)


# 使用例とテスト
def demo_random_generation():
    """ランダム生成のデモとテスト"""
    print("🎲 ランダム数値生成デモ")
    print("=" * 50)
    
    # 1. 基本的な一様分布
    print("\n1. 基本的な一様分布（浮動小数点）:")
    for i in range(5):
        value = generate_random_grip_force()
        print(f"   生成値 {i+1}: {value:.2f}N")
    
    # 2. 整数版
    print("\n2. 整数版:")
    for i in range(5):
        value = generate_random_integer()
        print(f"   生成値 {i+1}: {value}")
    
    # 3. シード付き（再現可能）
    print("\n3. シード付き（再現可能）:")
    seed_value = 12345
    for i in range(3):
        value = generate_seeded_random(seed=seed_value)
        print(f"   シード {seed_value} → {value:.2f}N")
    
    # 4. 正規分布
    print("\n4. 正規分布（平均16.0、標準偏差5.0）:")
    for i in range(5):
        value = generate_normal_distribution_random()
        print(f"   生成値 {i+1}: {value:.2f}N")
    
    # 5. 重み付き分布
    print("\n5. 重み付き分布（15.0付近に偏重）:")
    for i in range(5):
        value = generate_weighted_random(target_value=15.0, weight=0.5)
        print(f"   生成値 {i+1}: {value:.2f}N")
    
    # 6. クラス版
    print("\n6. クラス版（複数生成）:")
    generator = RandomGripForceGenerator(
        min_value=5.0, 
        max_value=25.0, 
        generation_mode='normal'
    )
    
    values = generator.generate_multiple(5)
    for i, value in enumerate(values):
        print(f"   生成値 {i+1}: {value:.2f}N")
    
    # 7. 統計確認
    print("\n7. 統計確認（1000回生成）:")
    test_values = [generate_random_grip_force() for _ in range(1000)]
    
    import statistics
    mean_val = statistics.mean(test_values)
    min_val = min(test_values)
    max_val = max(test_values)
    
    print(f"   平均値: {mean_val:.2f}")
    print(f"   最小値: {min_val:.2f}")
    print(f"   最大値: {max_val:.2f}")
    print(f"   範囲内確認: 全て2.0-30.0の範囲内: {all(2.0 <= v <= 30.0 for v in test_values)}")
    
    print("\n✅ デモ完了")


if __name__ == "__main__":
    demo_random_generation()



# 基本的な使い方
# 最もシンプルな関数：
# pythonfrom random_number_generator import generate_random_grip_force

# # 2.0から30.0の間でランダムな浮動小数点数を生成
# random_value = generate_random_grip_force()
# print(f"ランダムな値: {random_value:.2f}")
# 整数が欲しい場合：
# pythonfrom random_number_generator import generate_random_integer

# # 2から30の間でランダムな整数を生成
# random_int = generate_random_integer()
# print(f"ランダムな整数: {random_int}")
# 他のコードから呼び出す方法
# 既存のプロジェクトに統合する場合は、以下のように簡単に呼び出せます：
# python# 既存のコードに追加
# from random_number_generator import generate_random_grip_force, RandomGripForceGenerator

# # 1. 簡単な一回呼び出し
# grip_force = generate_random_grip_force(min_value=5.0, max_value=25.0)

# # 2. 複数回使用する場合はクラス版が便利
# generator = RandomGripForceGenerator(min_value=2.0, max_value=30.0)
# for i in range(10):
#     value = generator.generate()
#     print(f"把持力 {i+1}: {value:.2f}N")