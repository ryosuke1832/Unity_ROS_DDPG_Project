using UnityEngine;

/// <summary>
/// 基本的な型定義
/// 他のスクリプトの依存関係を解決するため
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

public enum GraspResult
{
    Success,
    UnderGrip,
    OverGrip,
    Failure,
    NoContact,
    Broken
}

[System.Serializable]
public class GraspEvaluation
{
    public GraspResult result = GraspResult.Failure;
    public float appliedForce = 0f;
    public float deformation = 0f;
    public bool isBroken = false;
    public bool hasContact = false;
    public bool isGripping = false;
    public float confidence = 0f;
    public float evaluationTime = 0f;
    
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
}

[System.Serializable]
public struct ObjectState
{
    public float deformation;
    public float appliedForce;
    public bool isBroken;
    public bool isBeingGrasped;
    public int materialType; // enum の代わりに int を使用
    public float softness;
}