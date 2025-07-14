using UnityEngine;

/// <summary>
/// 把持評価に関する型定義
/// 他のスクリプトで共通して使用される列挙型とクラス
/// </summary>

/// <summary>
/// 把持結果の種類
/// </summary>
public enum GraspResult
{
    Success,        // 成功
    UnderGrip,      // 把持力不足
    OverGrip,       // 把持力過多
    Failure,        // 失敗
    NoContact,      // 接触なし
    Broken          // 物体破損
}

/// <summary>
/// 把持評価の詳細情報
/// </summary>
[System.Serializable]
public class GraspEvaluation
{
    [Header("評価結果")]
    public GraspResult result = GraspResult.Failure;
    
    [Header("力の情報")]
    public float appliedForce = 0f;         // 適用された力
    public float targetForce = 0f;          // 目標力
    public float forceDeviation = 0f;       // 力の偏差
    
    [Header("変形情報")]
    public float deformation = 0f;          // 変形量
    public float maxDeformation = 0f;       // 最大変形量
    
    [Header("状態情報")]
    public bool isBroken = false;           // 破損状態
    public bool hasContact = false;         // 接触状態
    public bool isGripping = false;         // 把持状態
    
    [Header("評価指標")]
    [Range(0f, 1f)]
    public float confidence = 0f;           // 信頼度 (0-1)
    [Range(0f, 1f)]
    public float stability = 0f;            // 安定性 (0-1)
    [Range(0f, 1f)]
    public float efficiency = 0f;           // 効率性 (0-1)
    
    [Header("タイミング情報")]
    public float evaluationTime = 0f;       // 評価時刻
    public float graspDuration = 0f;        // 把持継続時間
    
    /// <summary>
    /// 総合スコアを計算
    /// </summary>
    public float GetOverallScore()
    {
        if (result == GraspResult.Broken)
            return 0f;
        
        if (result == GraspResult.Success)
            return (confidence + stability + efficiency) / 3f;
        
        // 部分的な成功の場合
        float baseScore = 0.3f;
        if (result == GraspResult.UnderGrip)
            baseScore += confidence * 0.4f;
        else if (result == GraspResult.OverGrip)
            baseScore += stability * 0.4f;
        
        return Mathf.Clamp01(baseScore);
    }
    
    /// <summary>
    /// 評価結果の文字列表現
    /// </summary>
    public override string ToString()
    {
        return $"GraspEvaluation: {result}, Score: {GetOverallScore():F2}, " +
               $"Force: {appliedForce:F1}N, Deformation: {deformation:F3}, " +
               $"Confidence: {confidence:F2}";
    }
    
    /// <summary>
    /// 簡単な評価結果の作成
    /// </summary>
    public static GraspEvaluation CreateSimple(GraspResult result, float force = 0f, float deformation = 0f)
    {
        return new GraspEvaluation
        {
            result = result,
            appliedForce = force,
            deformation = deformation,
            evaluationTime = Time.time,
            confidence = result == GraspResult.Success ? 0.8f : 0.3f
        };
    }
    
    /// <summary>
    /// 失敗評価の作成
    /// </summary>
    public static GraspEvaluation CreateFailure(string reason = "Unknown")
    {
        return new GraspEvaluation
        {
            result = GraspResult.Failure,
            confidence = 0f,
            evaluationTime = Time.time
        };
    }
}

/// <summary>
/// 把持状態の詳細情報
/// SimpleGripForceController や GripperForceController から取得
/// </summary>
[System.Serializable]
public class GraspingState
{
    [Header("基本状態")]
    public bool isGrasping = false;         // 把持中かどうか
    public bool isSuccessful = false;       // 成功状態
    public bool hasContact = false;         // 接触状態
    
    [Header("力の情報")]
    public float currentForce = 0f;         // 現在の力
    public float targetForce = 0f;          // 目標力
    public float leftGripperForce = 0f;     // 左グリッパーの力
    public float rightGripperForce = 0f;    // 右グリッパーの力
    
    [Header("位置情報")]
    public float leftGripperPosition = 0f;  // 左グリッパーの位置
    public float rightGripperPosition = 0f; // 右グリッパーの位置
    public float gripperOpening = 0f;       // グリッパー開度
    
    [Header("タイミング")]
    public float graspStartTime = 0f;       // 把持開始時刻
    public float lastUpdateTime = 0f;       // 最終更新時刻
    
    /// <summary>
    /// 把持継続時間を取得
    /// </summary>
    public float GetGraspDuration()
    {
        if (!isGrasping) return 0f;
        return Time.time - graspStartTime;
    }
    
    /// <summary>
    /// 力のバランスを取得（0に近いほど良い）
    /// </summary>
    public float GetForceBalance()
    {
        return Mathf.Abs(leftGripperForce - rightGripperForce);
    }
    
    /// <summary>
    /// 簡単な把持状態の作成
    /// </summary>
    public static GraspingState CreateDefault()
    {
        return new GraspingState
        {
            lastUpdateTime = Time.time
        };
    }
}

/// <summary>
/// 把持評価のユーティリティクラス
/// </summary>
public static class GraspEvaluationUtils
{
    /// <summary>
    /// 力に基づく結果判定
    /// </summary>
    public static GraspResult EvaluateByForce(float appliedForce, float targetForce, float tolerance = 0.2f)
    {
        if (appliedForce < 0.1f)
            return GraspResult.NoContact;
        
        float deviation = Mathf.Abs(appliedForce - targetForce) / targetForce;
        
        if (deviation <= tolerance)
            return GraspResult.Success;
        else if (appliedForce < targetForce)
            return GraspResult.UnderGrip;
        else
            return GraspResult.OverGrip;
    }
    
    /// <summary>
    /// 変形に基づく結果判定
    /// </summary>
    public static GraspResult EvaluateByDeformation(float deformation, float maxAllowed, float breakingPoint)
    {
        if (deformation >= breakingPoint)
            return GraspResult.Broken;
        
        if (deformation > maxAllowed)
            return GraspResult.OverGrip;
        
        if (deformation < 0.01f)
            return GraspResult.UnderGrip;
        
        return GraspResult.Success;
    }
    
    /// <summary>
    /// 信頼度の計算
    /// </summary>
    public static float CalculateConfidence(bool hasContact, bool isStable, float forceAccuracy)
    {
        float confidence = 0f;
        
        if (hasContact) confidence += 0.4f;
        if (isStable) confidence += 0.3f;
        confidence += forceAccuracy * 0.3f;
        
        return Mathf.Clamp01(confidence);
    }
}