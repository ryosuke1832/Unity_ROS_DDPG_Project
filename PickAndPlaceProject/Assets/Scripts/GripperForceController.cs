using UnityEngine;

/// <summary>
/// 変形可能ターゲット対応の拡張把持力コントローラー
/// </summary>
public class dGripperForceController : MonoBehaviour
{
    [Header("=== 把持力制御 ===")]
    [SerializeField] private ArticulationBody leftGripper;
    [SerializeField] private ArticulationBody rightGripper;
    [SerializeField, Range(0.1f, 100f)] private float targetGripForce = 10f;
    [SerializeField, Range(0f, 1f)] private float softness = 0.5f;
    
    [Header("=== 変形ターゲット検出 ===")]
    [SerializeField] private LayerMask targetLayers = -1;
    [SerializeField] private float detectionRadius = 0.05f;
    [SerializeField] private bool enableForceTransmission = true;
    
    [Header("=== PID制御パラメータ ===")]
    [SerializeField, Range(0f, 2f)] private float proportionalGain = 1f;
    [SerializeField, Range(0f, 1f)] private float integralGain = 0.1f;
    [SerializeField, Range(0f, 1f)] private float derivativeGain = 0.05f;
    
    [Header("=== デバッグ ===")]
    [SerializeField] private bool showDebugInfo = true;
    
    // 制御変数
    private bool isGrasping = false;
    private bool isForceControlActive = true;
    private float currentForce = 0f;
    private float integral = 0f;
    private float lastError = 0f;
    private const float UPDATE_INTERVAL = 0.02f; // 50Hz更新
    
    // 変形ターゲット追跡
    private DeformableTarget currentTarget = null;
    private Vector3 lastContactPoint = Vector3.zero;
    private Vector3 lastContactNormal = Vector3.zero;
    
    // 内部計算用
    private float leftForce = 0f;
    private float rightForce = 0f;
    private float averageGripForce = 0f;
    
    void Start()
    {
        InitializeController();
        
        // 50Hz間隔で力制御更新
        InvokeRepeating(nameof(UpdateForceControl), 0f, UPDATE_INTERVAL);
    }
    
    void Update()
    {
        UpdateForceReadings();
        DetectAndInteractWithTargets();
    }
    
    /// <summary>
    /// コントローラー初期化
    /// </summary>
    private void InitializeController()
    {
        if (leftGripper == null || rightGripper == null)
        {
            FindGrippers();
        }
        
        if (showDebugInfo)
        {
            Debug.Log("EnhancedGripperForceController初期化完了");
        }
    }
    
    /// <summary>
    /// グリッパーの自動検出
    /// </summary>
    private void FindGrippers()
    {
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
    /// 力の読み取り更新
    /// </summary>
    private void UpdateForceReadings()
    {
        // 左グリッパーの力
        if (leftGripper != null)
        {
            var jointForce = leftGripper.jointForce;
            leftForce = jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
        }
        
        // 右グリッパーの力
        if (rightGripper != null) 
        {
            var jointForce = rightGripper.jointForce;
            rightForce = jointForce.dofCount > 0 ? Mathf.Abs(jointForce[0]) : 0f;
        }
        
        // 平均把持力
        averageGripForce = (leftForce + rightForce) / 2f;
        currentForce = averageGripForce;
    }
    
    /// <summary>
    /// ターゲット検出と相互作用
    /// </summary>
    private void DetectAndInteractWithTargets()
    {
        if (!enableForceTransmission || !isGrasping) return;
        
        DeformableTarget detectedTarget = null;
        Vector3 contactPoint = Vector3.zero;
        Vector3 contactNormal = Vector3.zero;
        
        // 左グリッパー周辺での検出
        if (leftGripper != null)
        {
            var result = DetectTargetNearGripper(leftGripper.transform.position);
            if (result.target != null)
            {
                detectedTarget = result.target;
                contactPoint = result.contactPoint;
                contactNormal = result.contactNormal;
            }
        }
        
        // 右グリッパー周辺での検出（左で見つからなかった場合）
        if (detectedTarget == null && rightGripper != null)
        {
            var result = DetectTargetNearGripper(rightGripper.transform.position);
            if (result.target != null)
            {
                detectedTarget = result.target;
                contactPoint = result.contactPoint;
                contactNormal = result.contactNormal;
            }
        }
        
        // ターゲットが変わった場合の処理
        if (currentTarget != detectedTarget)
        {
            if (currentTarget != null)
            {
                currentTarget.StopGrasping();
            }
            currentTarget = detectedTarget;
        }
        
        // 現在のターゲットに力を適用
        if (currentTarget != null && currentForce > 0.1f)
        {
            Vector3 forceDirection = contactNormal;
            if (forceDirection == Vector3.zero)
            {
                // デフォルトの力の方向（グリッパー間の方向）
                if (leftGripper != null && rightGripper != null)
                {
                    forceDirection = (rightGripper.transform.position - leftGripper.transform.position).normalized;
                }
            }
            
            currentTarget.ApplyGripForce(currentForce, contactPoint, forceDirection);
            lastContactPoint = contactPoint;
            lastContactNormal = contactNormal;
            
            if (showDebugInfo)
            {
                Debug.DrawRay(contactPoint, forceDirection * 0.1f, Color.red);
            }
        }
    }
    
    /// <summary>
    /// グリッパー近辺でのターゲット検出
    /// </summary>
    private (DeformableTarget target, Vector3 contactPoint, Vector3 contactNormal) DetectTargetNearGripper(Vector3 gripperPosition)
    {
        Collider[] colliders = Physics.OverlapSphere(gripperPosition, detectionRadius, targetLayers);
        
        foreach (var collider in colliders)
        {
            DeformableTarget target = collider.GetComponent<DeformableTarget>();
            if (target != null)
            {
                // 接触点の計算
                Vector3 contactPoint = collider.ClosestPoint(gripperPosition);
                Vector3 contactNormal = (gripperPosition - contactPoint).normalized;
                
                return (target, contactPoint, contactNormal);
            }
        }
        
        return (null, Vector3.zero, Vector3.zero);
    }
    
    /// <summary>
    /// 力制御更新（50Hz）
    /// </summary>
    private void UpdateForceControl()
    {
        if (!isForceControlActive || !isGrasping) return;
        
        // PID制御による力調整
        float error = targetGripForce - currentForce;
        integral += error * UPDATE_INTERVAL;
        float derivative = (error - lastError) / UPDATE_INTERVAL;
        
        float adjustment = proportionalGain * error + 
                          integralGain * integral + 
                          derivativeGain * derivative;
        
        // ターゲットの硬さに応じた力調整
        if (currentTarget != null)
        {
            float targetSoftness = currentTarget.Softness;
            adjustment *= (1f - targetSoftness * 0.5f); // 柔らかい物体には弱い力
        }
        
        // グリッパーに力調整を適用
        ApplyForceAdjustment(adjustment);
        
        lastError = error;
    }
    
    /// <summary>
    /// 力調整をグリッパーに適用
    /// </summary>
    private void ApplyForceAdjustment(float adjustment)
    {
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
    /// 把持開始
    /// </summary>
    public void StartGrasping(float force = -1f)
    {
        if (force > 0) targetGripForce = force;
        
        isGrasping = true;
        isForceControlActive = true;
        
        // PID制御リセット
        integral = 0f;
        lastError = 0f;
        
        CloseGrippers();
        
        if (showDebugInfo)
        {
            Debug.Log($"把持開始 - 目標力: {targetGripForce}N");
        }
    }
    
    /// <summary>
    /// 把持停止
    /// </summary>
    public void StopGrasping()
    {
        isGrasping = false;
        isForceControlActive = false;
        
        if (currentTarget != null)
        {
            currentTarget.StopGrasping();
            currentTarget = null;
        }
        
        OpenGrippers();
        
        if (showDebugInfo)
        {
            Debug.Log("把持停止");
        }
    }
    
    /// <summary>
    /// グリッパーを閉じる
    /// </summary>
    private void CloseGrippers()
    {
        if (leftGripper != null && rightGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            var rightDrive = rightGripper.xDrive;
            
            leftDrive.target = 0.01f;
            rightDrive.target = -0.01f;
            
            leftGripper.xDrive = leftDrive;
            rightGripper.xDrive = rightDrive;
        }
    }
    
    /// <summary>
    /// グリッパーを開く
    /// </summary>
    private void OpenGrippers()
    {
        if (leftGripper != null && rightGripper != null)
        {
            var leftDrive = leftGripper.xDrive;
            var rightDrive = rightGripper.xDrive;
            
            leftDrive.target = -0.01f;
            rightDrive.target = 0.01f;
            
            leftGripper.xDrive = leftDrive;
            rightGripper.xDrive = rightDrive;
        }
    }
    
    /// <summary>
    /// 目標把持力の設定
    /// </summary>
    public void SetTargetGripForce(float force)
    {
        targetGripForce = Mathf.Clamp(force, 0.1f, 100f);
    }
    
    /// <summary>
    /// 現在の把持状態取得
    /// </summary>
    public GraspingState GetGraspingState()
    {
        return new GraspingState
        {
            isGrasping = this.isGrasping,
            currentForce = this.currentForce,
            targetForce = this.targetGripForce,
            gripperPosition = GetGripperPosition(),
            isSuccessful = IsGraspSuccessful(),
            softness = this.softness
        };
    }
    
    private float GetGripperPosition()
    {
        if (leftGripper != null && rightGripper != null)
        {
            return Vector3.Distance(leftGripper.transform.position, rightGripper.transform.position);
        }
        return 0f;
    }
    
    private bool IsGraspSuccessful()
    {
        return isGrasping && currentForce > 1f && currentTarget != null;
    }
    
    void OnDrawGizmos()
    {
        if (showDebugInfo)
        {
            // 検出範囲の表示
            if (leftGripper != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawWireSphere(leftGripper.transform.position, detectionRadius);
            }
            
            if (rightGripper != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawWireSphere(rightGripper.transform.position, detectionRadius);
            }
            
            // 現在のターゲットとの接触点
            if (currentTarget != null && lastContactPoint != Vector3.zero)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawWireSphere(lastContactPoint, 0.01f);
                
                Gizmos.color = Color.red;
                Gizmos.DrawRay(lastContactPoint, lastContactNormal * 0.05f);
            }
        }
    }
    
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 300, 200));
        GUILayout.Label("=== 拡張把持力制御デバッグ ===");
        GUILayout.Label($"把持中: {isGrasping}");
        GUILayout.Label($"力制御: {isForceControlActive}");
        GUILayout.Label($"目標力: {targetGripForce:F2} N");
        GUILayout.Label($"現在力: {currentForce:F2} N");
        GUILayout.Label($"左力: {leftForce:F2} N");
        GUILayout.Label($"右力: {rightForce:F2} N");
        GUILayout.Label($"現在のターゲット: {(currentTarget != null ? currentTarget.name : "なし")}");
        if (currentTarget != null)
        {
            GUILayout.Label($"ターゲット変形度: {currentTarget.CurrentDeformation:F3}");
            GUILayout.Label($"ターゲット柔軟性: {currentTarget.Softness:F2}");
        }
        GUILayout.EndArea();
    }
}