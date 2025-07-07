using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Robotics.ROSTCPConnector;

/// <summary>
/// 把持力制御拡張スクリプト
/// 研究用途：人間フィードバックによる強化学習での把持力制御
/// </summary>
public class GripperForceController : MonoBehaviour
{
    [Header("把持力制御パラメータ")]
    [Range(0.1f, 100f)]
    [SerializeField] protected float targetGripForce = 10f;
    
    [Range(0f, 1f)]
    [SerializeField] protected float softness = 0.5f; // 0=硬い把持, 1=柔らかい把持
    
    [Range(1f, 50f)]
    [SerializeField] protected float forceControlSpeed = 10f; // 力制御の応答速度
    
    [Header("PID制御パラメータ")]
    [SerializeField] protected float kp = 1.0f; // 比例ゲイン
    [SerializeField] protected float ki = 0.1f; // 積分ゲイン
    [SerializeField] protected float kd = 0.05f; // 微分ゲイン
    
    [Header("グリッパー参照")]
    [SerializeField] public  ArticulationBody leftGripper;
    [SerializeField] public  ArticulationBody rightGripper;
    
    [Header("デバッグ情報")]
    [SerializeField] protected bool showDebugInfo = true;
    [SerializeField] protected float currentLeftForce = 0f;
    [SerializeField] protected float currentRightForce = 0f;
    [SerializeField] protected float averageGripForce = 0f;
    
    // PID制御用変数
    protected float integral = 0f;
    protected float lastError = 0f;
    protected float currentForce = 0f;
    
    // 把持状態管理
    protected bool isGrasping = false;
    protected bool isForceControlActive = false;
    
    // パフォーマンス最適化
    protected float lastUpdateTime;
    protected const float UPDATE_INTERVAL = 0.02f; // 50Hz更新
    
    void Start()
    {
        // グリッパー参照の自動取得（未設定の場合）
        if (leftGripper == null || rightGripper == null)
        {
            FindGrippers();
        }
        
        lastUpdateTime = Time.time;
        
        if (showDebugInfo)
        {
            Debug.Log("GripperForceController初期化完了");
        }
    }
    
    protected virtual void Update()
    {
        // 高頻度更新を避けるための時間チェック
        if (Time.time - lastUpdateTime < UPDATE_INTERVAL) return;
        lastUpdateTime = Time.time;
        
        if (isForceControlActive && isGrasping)
        {
            UpdateForceControl();
        }
        
        UpdateDebugInfo();
    }
    
    /// <summary>
    /// デバッグ情報の更新
    /// </summary>
    protected void UpdateDebugInfo()
    {
        // 左右グリッパーの力を更新
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
    public virtual void SetTargetGripForce(float force)
    {
        targetGripForce = Mathf.Clamp(force, 0.1f, 100f);
        if (showDebugInfo)
        {
            Debug.Log($"目標把持力を{targetGripForce}Nに設定");
        }
    }
    
    /// <summary>
    /// 柔軟性パラメータ設定
    /// </summary>
    public virtual void SetSoftness(float softnessValue)
    {
        softness = Mathf.Clamp01(softnessValue);
    }
    
    /// <summary>
    /// 把持状態の取得
    /// </summary>
    public virtual GraspingState GetGraspingState()
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
    /// グリッパーの自動検出
    /// </summary>
    protected void FindGrippers()
    {
        // 階層からグリッパーを検索
        ArticulationBody[] bodies = FindObjectsOfType<ArticulationBody>();
        
        foreach (var body in bodies)
        {
            if (body.name.Contains("left_gripper") || body.name.Contains("LeftGripper"))
                leftGripper = body;
            else if (body.name.Contains("right_gripper") || body.name.Contains("RightGripper"))
                rightGripper = body;
        }
        
        if (leftGripper == null || rightGripper == null)
        {
            Debug.LogWarning("グリッパーが見つかりません。手動で設定してください。");
        }
    }
    
    /// <summary>
    /// 把持開始（力制御付き）
    /// </summary>
    /// <param name="targetForce">目標把持力（N）</param>
    /// <param name="enableForceControl">力制御を有効にするか</param>
    public virtual void StartGrasping(float targetForce = -1f, bool enableForceControl = true)
    {
        if (targetForce > 0) this.targetGripForce = targetForce;
        
        isGrasping = true;
        isForceControlActive = enableForceControl;
        
        // PID制御をリセット
        integral = 0f;
        lastError = 0f;
        
        // 基本的な把持動作を開始
        CloseGrippers();
        
        if (showDebugInfo)
        {
            Debug.Log($"把持開始 - 目標力: {this.targetGripForce}N, 力制御: {enableForceControl}");
        }
    }
    
    /// <summary>
    /// 把持停止
    /// </summary>
    public virtual void StopGrasping()
    {
        isGrasping = false;
        isForceControlActive = false;
        
        OpenGrippers();
        
        if (showDebugInfo)
        {
            Debug.Log("把持停止");
        }
    }
    
    /// <summary>
    /// リアルタイム力制御更新
    /// </summary>
    protected void UpdateForceControl()
    {
        // 現在の把持力を取得
        currentForce = GetCurrentGripperForce();
        
        // PID制御による力調整
        float forceAdjustment = ControlGripperForce(targetGripForce, currentForce, UPDATE_INTERVAL);
        
        // 力調整を適用
        ApplyForceAdjustment(forceAdjustment);
    }
    
    /// <summary>
    /// PID制御による力調整計算
    /// </summary>
    protected float ControlGripperForce(float targetForce, float currentForce, float deltaTime)
    {
        float error = targetForce - currentForce;
        
        // 積分項の計算（ワインドアップ対策付き）
        integral += error * deltaTime;
        integral = Mathf.Clamp(integral, -10f, 10f);
        
        // 微分項の計算
        float derivative = (error - lastError) / deltaTime;
        
        // PID出力
        float output = kp * error + ki * integral + kd * derivative;
        
        lastError = error;
        
        return output;
    }
    
    /// <summary>
    /// 力調整をグリッパーに適用
    /// </summary>
    protected void ApplyForceAdjustment(float adjustment)
    {
        // 調整値をクランプ
        float adjustmentClamped = Mathf.Clamp(adjustment, -10f, 10f);
        
        if (leftGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            leftDrive.forceLimit = Mathf.Max(1f, leftDrive.forceLimit + adjustmentClamped);
            leftGripper.xDrive = leftDrive;
        }
        
        if (rightGripper != null)
        {
            var rightDrive = rightGripper.xDrive;
            rightDrive.forceLimit = Mathf.Max(1f, rightDrive.forceLimit + adjustmentClamped);
            rightGripper.xDrive = rightDrive;
        }
    }
    
    /// <summary>
    /// グリッパーを閉じる
    /// </summary>
    protected void CloseGrippers()
    {
        if (leftGripper != null && rightGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            var rightDrive = rightGripper.xDrive;
            
            leftDrive.target = 0.01f;   // 左グリッパーを閉じる
            rightDrive.target = -0.01f; // 右グリッパーを閉じる
            
            leftGripper.xDrive = leftDrive;
            rightGripper.xDrive = rightDrive;
        }
    }
    
    /// <summary>
    /// グリッパーを開く
    /// </summary>
    protected void OpenGrippers()
    {
        if (leftGripper != null && rightGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            var rightDrive = rightGripper.xDrive;
            
            leftDrive.target = -0.01f;  // 左グリッパーを開く
            rightDrive.target = 0.01f;  // 右グリッパーを開く
            
            leftGripper.xDrive = leftDrive;
            rightGripper.xDrive = rightDrive;
        }
    }
    
    /// <summary>
    /// 現在の把持力を取得
    /// </summary>
    public virtual float GetCurrentGripperForce()
    {
        float leftForce = 0f, rightForce = 0f;
        
        if (leftGripper != null)
        {
            // 関節力を取得（ArticulationReducedSpaceから）
            var jointForce = leftGripper.jointForce;
            if (jointForce.dofCount > 0)
            {
                leftForce = Mathf.Abs(jointForce[0]); // 最初のDOFの力
            }
            
            // 代替方法：関節加速度から力を推定
            if (leftForce == 0f)
            {
                var jointAccel = leftGripper.jointAcceleration;
                if (jointAccel.dofCount > 0)
                {
                    leftForce = Mathf.Abs(jointAccel[0]) * leftGripper.mass;
                }
            }
        }
        
        if (rightGripper != null)
        {
            var jointForce = rightGripper.jointForce;
            if (jointForce.dofCount > 0)
            {
                rightForce = Mathf.Abs(jointForce[0]); // 最初のDOFの力
            }
            
            // 代替方法：関節加速度から力を推定
            if (rightForce == 0f)
            {
                var jointAccel = rightGripper.jointAcceleration;
                if (jointAccel.dofCount > 0)
                {
                    rightForce = Mathf.Abs(jointAccel[0]) * rightGripper.mass;
                }
            }
        }
        
        return (leftForce + rightForce) / 2f; // 平均把持力
    }
    
    /// <summary>
    /// より詳細な力情報を取得
    /// </summary>
    public virtual ForceInfo GetDetailedForceInfo()
    {
        var info = new ForceInfo();
        
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
        
        info.averageForce = (info.leftForce + info.rightForce) / 2f;
        info.forceBalance = Mathf.Abs(info.leftForce - info.rightForce);
        
        return info;
    }
    
    /// <summary>
    /// グリッパー位置の取得
    /// </summary>
    protected float GetGripperPosition()
    {
        if (leftGripper != null && rightGripper != null)
        {
            return Vector3.Distance(leftGripper.transform.position, rightGripper.transform.position);
        }
        return 0f;
    }
    
    /// <summary>
    /// 把持成功判定
    /// </summary>
    protected bool IsGraspSuccessful()
    {
        return isGrasping && currentForce > 1f && GetGripperPosition() < 0.1f;
    }
    
    // OnGUI デバッグ表示
    protected virtual void OnGUI()
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
    
    protected virtual void OnDrawGizmos()
    {
        // デフォルトでは何も描画しない
        // 継承クラスでオーバーライド可能
    }
}