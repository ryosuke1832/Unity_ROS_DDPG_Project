using UnityEngine;

/// <summary>
/// 変形ターゲット対応の拡張把持力コントローラー
/// GripperForceControllerを継承して変形機能を追加
/// </summary>
public class EnhancedGripperForceController : GripperForceController
{
    [Header("=== 変形ターゲット検出 ===")]
    [SerializeField] private LayerMask targetLayers = -1;
    [SerializeField] private float detectionRadius = 0.05f;
    [SerializeField] private bool enableForceTransmission = true;
    
    // 変形ターゲット追跡
    private DeformableTarget currentTarget = null;
    private Vector3 lastContactPoint = Vector3.zero;
    private Vector3 lastContactNormal = Vector3.zero;
    
    protected override void Update()
    {
        base.Update(); // 基本的な力制御処理
        
        // 変形ターゲットとの相互作用
        DetectAndInteractWithTargets();
    }
    
    /// <summary>
    /// ターゲット検出と相互作用
    /// </summary>
    private void DetectAndInteractWithTargets()
    {
        if (!enableForceTransmission || !GetGraspingState().isGrasping) return;
        
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
        if (currentTarget != null && GetCurrentGripperForce() > 0.1f)
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
            
            float currentForce = GetCurrentGripperForce();
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
    /// 把持停止時のオーバーライド
    /// </summary>
    public override void StopGrasping()
    {
        base.StopGrasping();
        
        if (currentTarget != null)
        {
            currentTarget.StopGrasping();
            currentTarget = null;
        }
    }
    
    /// <summary>
    /// 現在のターゲット取得
    /// </summary>
    public DeformableTarget GetCurrentTarget()
    {
        return currentTarget;
    }
    
    /// <summary>
    /// 検出範囲の設定
    /// </summary>
    public void SetDetectionRadius(float radius)
    {
        detectionRadius = Mathf.Clamp(radius, 0.01f, 0.5f);
    }
    
    /// <summary>
    /// 力伝達の有効/無効切り替え
    /// </summary>
    public void SetForceTransmissionEnabled(bool enabled)
    {
        enableForceTransmission = enabled;
    }
    
    protected override void OnDrawGizmos()
    {
        base.OnDrawGizmos();
        
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
    
    protected override void OnGUI()
    {
        base.OnGUI();
        
        if (!showDebugInfo) return;
        
        // 追加のデバッグ情報
        GUILayout.BeginArea(new Rect(320, 10, 250, 150));
        GUILayout.Label("=== 変形ターゲット情報 ===");
        GUILayout.Label($"検出範囲: {detectionRadius:F3}m");
        GUILayout.Label($"現在のターゲット: {(currentTarget != null ? currentTarget.name : "なし")}");
        if (currentTarget != null)
        {
            GUILayout.Label($"ターゲット変形度: {currentTarget.CurrentDeformation:F3}");
            GUILayout.Label($"ターゲット柔軟性: {currentTarget.Softness:F2}");
            GUILayout.Label($"変形中: {(currentTarget.IsDeformed ? "はい" : "いいえ")}");
        }
        GUILayout.Label($"力伝達: {(enableForceTransmission ? "有効" : "無効")}");
        GUILayout.EndArea();
    }

    
// EnhancedGripperForceController.cs に以下のメソッドを追加

/// <summary>
/// 現在のグリッパー力を取得
/// </summary>
/// <returns>平均把持力</returns>
public float GetCurrentGripperForce()
{
    return averageGripForce;
}

/// <summary>
/// 把持開始（オーバーライド）
/// </summary>
/// <param name="targetForce">目標把持力</param>
public override void StartGrasping(float targetForce)
{
    base.StartGrasping(targetForce);
    
    if (showDebugInfo)
    {
        Debug.Log($"変形対応把持開始 - 目標力: {targetForce}N");
    }
}

/// <summary>
/// DeformableTargetクラス内で使用されるメソッド
/// （もしDeformableTarget.cs内でcurrentTarget.ApplyGripForce()を呼び出している場合）
/// </summary>
/// <param name="force">把持力</param>
/// <param name="contactPoint">接触点</param>
/// <param name="forceDirection">力の方向</param>
public void ApplyGripForce(float force, Vector3 contactPoint, Vector3 forceDirection)
{
    // この機能がDeformableTargetで必要な場合に実装
    if (currentTarget != null)
    {
        currentTarget.StartGrasping(force, contactPoint, forceDirection);
    }
}
}
