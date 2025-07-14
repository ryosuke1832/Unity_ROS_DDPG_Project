using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 把持力制御システム
/// </summary>
public class GripperForceController : MonoBehaviour
{
    [Header("グリッパー設定")]
    public ArticulationBody leftGripper;
    public ArticulationBody rightGripper;
    
    [Header("力制御設定")]
    [Range(0.1f, 100f)]
    public float targetGripForce = 10f;
    
    [Range(0f, 1f)]
    public float softness = 0.5f;
    
    [Header("制御パラメータ")]
    public float forceControlSpeed = 2f;
    public float maxForce = 50f;
    public float minForce = 0.1f;
    
    [Header("デバッグ")]
    public bool showDebugInfo = false;
    public bool enableForceLogging = false;
    
    // 内部状態
    private bool isGrasping = false;
    private bool isForceControlActive = false;
    private float currentLeftForce = 0f;
    private float currentRightForce = 0f;
    private float averageGripForce = 0f;
    
    void Start()
    {
        // グリッパーの自動検索
        if (leftGripper == null || rightGripper == null)
        {
            FindGrippers();
        }
        
        // 初期化
        isGrasping = false;
        isForceControlActive = false;
        
        Debug.Log("GripperForceController initialized");
    }
    
    void Update()
    {
        UpdateForceControl();
        UpdateDebugInfo();
        
        if (enableForceLogging && Time.time % 1f < Time.deltaTime)
        {
            Debug.Log($"Grip Force - Target: {targetGripForce:F2}N, Average: {averageGripForce:F2}N");
        }
    }
    
    /// <summary>
    /// グリッパーの自動検索
    /// </summary>
    private void FindGrippers()
    {
        ArticulationBody[] allBodies = FindObjectsOfType<ArticulationBody>();
        
        foreach (var body in allBodies)
        {
            if (body.name.Contains("left_gripper") || body.name.Contains("LeftGripper"))
            {
                leftGripper = body;
            }
            else if (body.name.Contains("right_gripper") || body.name.Contains("RightGripper"))
            {
                rightGripper = body;
            }
        }
        
        if (leftGripper != null && rightGripper != null)
        {
            Debug.Log($"Grippers found: {leftGripper.name}, {rightGripper.name}");
        }
        else
        {
            Debug.LogWarning("Could not find both grippers automatically");
        }
    }
    
    /// <summary>
    /// 力制御の更新
    /// </summary>
    private void UpdateForceControl()
    {
        if (!isForceControlActive || leftGripper == null || rightGripper == null)
            return;
        
        // 現在の力を取得
        GetCurrentGripForces();
        
        // 力制御を適用
        ApplyForceControl();
    }
    
    /// <summary>
    /// 現在のグリッパー力を取得
    /// </summary>
    private void GetCurrentGripForces()
    {
        if (leftGripper != null)
        {
            var jointForce = leftGripper.jointForce;
            currentLeftForce = jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
        }
        
        if (rightGripper != null)
        {
            var jointForce = rightGripper.jointForce;
            currentRightForce = jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
        }
        
        averageGripForce = (currentLeftForce + currentRightForce) / 2f;
    }
    
    /// <summary>
    /// 力制御を適用
    /// </summary>
    private void ApplyForceControl()
    {
        float forceError = targetGripForce - averageGripForce;
        float adjustment = forceError * forceControlSpeed * Time.deltaTime;
        
        // グリッパーの目標位置を調整
        if (leftGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            leftDrive.target = Mathf.Clamp(leftDrive.target - adjustment * 0.001f, -0.02f, 0f);
            leftGripper.xDrive = leftDrive;
        }
        
        if (rightGripper != null)
        {
            var rightDrive = rightGripper.xDrive;
            rightDrive.target = Mathf.Clamp(rightDrive.target + adjustment * 0.001f, 0f, 0.02f);
            rightGripper.xDrive = rightDrive;
        }
    }
    
    /// <summary>
    /// 目標把持力を設定
    /// </summary>
    public void SetTargetGripForce(float force)
    {
        targetGripForce = Mathf.Clamp(force, minForce, maxForce);
        
        if (showDebugInfo)
        {
            Debug.Log($"Target grip force set to: {targetGripForce:F2}N");
        }
    }
    
    /// <summary>
    /// 力制御を有効/無効にする
    /// </summary>
    public void SetForceControlActive(bool active)
    {
        isForceControlActive = active;
        
        if (showDebugInfo)
        {
            Debug.Log($"Force control {(active ? "activated" : "deactivated")}");
        }
    }
    
    /// <summary>
    /// 把持状態を開始
    /// </summary>
    public void StartGrasping()
    {
        isGrasping = true;
        isForceControlActive = true;
        
        if (showDebugInfo)
        {
            Debug.Log("Grasping started");
        }
    }
    
    /// <summary>
    /// 把持状態を停止
    /// </summary>
    public void StopGrasping()
    {
        isGrasping = false;
        isForceControlActive = false;
        
        if (showDebugInfo)
        {
            Debug.Log("Grasping stopped");
        }
    }
    
    /// <summary>
    /// 現在の把持力を取得
    /// </summary>
    public float GetCurrentGripperForce()
    {
        return averageGripForce;
    }
    
    /// <summary>
    /// グリッパーの位置を取得
    /// </summary>
    public float GetGripperPosition()
    {
        if (leftGripper != null && leftGripper.jointPosition.dofCount > 0)
        {
            return leftGripper.jointPosition[0];
        }
        return 0f;
    }
    
    /// <summary>
    /// 把持が成功しているかチェック
    /// </summary>
    public bool IsGraspSuccessful()
    {
        float currentPos = GetGripperPosition();
        float currentForce = GetCurrentGripperForce();
        
        // 物体を掴んでいる判定：位置が中間値で、力が適切範囲内
        bool hasObject = Mathf.Abs(currentPos) < 0.008f && Mathf.Abs(currentPos) > 0.002f;
        bool appropriateForce = currentForce > 0.1f && currentForce < targetGripForce * 1.2f;
        
        return hasObject && appropriateForce;
    }
    
    /// <summary>
    /// 把持状態の取得
    /// </summary>
    public GraspingState GetGraspingState()
    {
        return new GraspingState
        {
            isGrasping = this.isGrasping,
            currentForce = GetCurrentGripperForce(),
            targetForce = this.targetGripForce,
            gripperPosition = GetGripperPosition(),
            isSuccessful = IsGraspSuccessful(),
            softness = this.softness
        };
    }
    
    /// <summary>
    /// 詳細な力情報を取得
    /// </summary>
    public ForceInfo GetDetailedForceInfo()
    {
        ForceInfo info = new ForceInfo();
        
        if (leftGripper != null)
        {
            var jointForce = leftGripper.jointForce;
            var jointVelocity = leftGripper.jointVelocity;
            var jointPosition = leftGripper.jointPosition;
            
            info.leftForce = jointForce.dofCount > 0 ? jointForce[0] : 0f;
            info.leftVelocity = jointVelocity.dofCount > 0 ? jointVelocity[0] : 0f;
            info.leftPosition = jointPosition.dofCount > 0 ? jointPosition[0] : 0f;
        }
        
        if (rightGripper != null)
        {
            var jointForce = rightGripper.jointForce;
            var jointVelocity = rightGripper.jointVelocity;
            var jointPosition = rightGripper.jointPosition;
            
            info.rightForce = jointForce.dofCount > 0 ? jointForce[0] : 0f;
            info.rightVelocity = jointVelocity.dofCount > 0 ? jointVelocity[0] : 0f;
            info.rightPosition = jointPosition.dofCount > 0 ? jointPosition[0] : 0f;
        }
        
        info.averageForce = (Mathf.Abs(info.leftForce) + Mathf.Abs(info.rightForce)) / 2f;
        info.forceBalance = info.leftForce + info.rightForce; // バランス（0に近いほど良い）
        
        return info;
    }
    
    /// <summary>
    /// デバッグ情報の更新
    /// </summary>
    private void UpdateDebugInfo()
    {
        if (leftGripper != null) 
        {
            var jointForce = leftGripper.jointForce;
            currentLeftForce = jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
        }
        if (rightGripper != null) 
        {
            var jointForce = rightGripper.jointForce;
            currentRightForce = jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
        }
        averageGripForce = (currentLeftForce + currentRightForce) / 2f;
    }
    
    /// <summary>
    /// 外部からの目標力設定
    /// </summary>
    public void SetSoftness(float softnessValue)
    {
        softness = Mathf.Clamp01(softnessValue);
    }
    
    // OnGUI デバッグ表示
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        var forceInfo = GetDetailedForceInfo();
        
        GUILayout.BeginArea(new Rect(10, 10, 300, 250));
        GUILayout.Label("=== 把持力制御デバッグ ===");
        GUILayout.Label($"把持中: {isGrasping}");
        GUILayout.Label($"力制御: {isForceControlActive}");
        GUILayout.Label($"目標力: {targetGripForce:F2} N");
        GUILayout.Label($"現在力: {averageGripForce:F2} N");
        GUILayout.Label($"左力: {forceInfo.leftForce:F2} N");
        GUILayout.Label($"右力: {forceInfo.rightForce:F2} N");
        GUILayout.Label($"力バランス: {forceInfo.forceBalance:F2}");
        GUILayout.Label($"柔軟性: {softness:F2}");
        GUILayout.Label($"成功判定: {IsGraspSuccessful()}");
        GUILayout.Label($"位置: {GetGripperPosition():F4}");
        GUILayout.EndArea();
    }
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