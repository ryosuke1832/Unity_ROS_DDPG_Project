using UnityEngine;

/// <summary>
/// 把持評価に関する型定義（GraspingStateは削除し、既存のものを使用）
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