using UnityEngine;

/// <summary>
/// グリッパー制御に関する共通の型定義
/// </summary>

/// <summary>
/// 把持状態情報構造体
/// </summary>
[System.Serializable]
public struct GraspingState
{
    public bool isGrasping;
    public float currentForce;
    public float targetForce;
    public float gripperPosition;
    public bool isSuccessful;
    public float softness;
}

/// <summary>
/// 詳細な力情報構造体
/// </summary>
[System.Serializable]
public struct ForceInfo
{
    public float leftForce;
    public float rightForce;
    public float leftVelocity;
    public float rightVelocity;
    public float leftPosition;
    public float rightPosition;
    public float averageForce;
    public float forceBalance;
}

/// <summary>
/// 制御モード
/// </summary>
public enum GripControlMode
{
    Standard,    // 標準制御
    Adaptive,    // 適応制御
    Experimental // 実験的制御
}

/// <summary>
/// プリセットタイプ
/// </summary>
public enum GripForcePresetType
{
    Custom,        // カスタム（プリセット未使用）
    SoftObject,    // 柔らかい物体用
    MediumObject,  // 中程度の硬さの物体用
    HardObject,    // 硬い物体用
    Experimental   // 実験用
}

/// <summary>
/// プリセットデータ
/// </summary>
[System.Serializable]
public struct GripForcePreset
{
    public float baseForce;
    public float variability;
    public float changeRate;
    public bool adaptiveEnabled;
    public float adaptiveGain;
    public float damping;
    public bool noiseEnabled;
    public float noiseStrength;
}

/// <summary>
/// 実験データポイント
/// </summary>
[System.Serializable]
public struct ExperimentDataPoint
{
    public float timestamp;
    public float targetForce;
    public float actualForce;
    public float baseForce;
    public string controlMode;
    public bool isGrasping;
    public bool isSuccessful;
}