using UnityEngine;

/// <summary>
/// 接触検出のデバッグと分析ツール（改良版）
/// </summary>
public class ContactDetectionDebug : MonoBehaviour
{
    [Header("デバッグ対象")]
    public GripperTargetInterface gripperInterface;
    public DeformableTarget target;
    
    [Header("デバッグ設定")]
    public bool enableDetailedLogging = true;
    public bool enableGizmoDisplay = true;
    public float logInterval = 1f;
    
    [Header("検出対象の手動指定")]
    [Tooltip("実際のグリッパーエンドエフェクター")]
    public Transform actualGripperEndEffector;
    [Tooltip("左グリッパーの指先")]
    public Transform leftGripperTip;
    [Tooltip("右グリッパーの指先")]
    public Transform rightGripperTip;
    
    private float lastLogTime = 0f;
    
    void Start()
    {
        // 自動検出
        if (gripperInterface == null)
            gripperInterface = FindObjectOfType<GripperTargetInterface>();
        
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
        
        // エンドエフェクターの自動検出
        AutoDetectGripperComponents();
    }
    
    void Update()
    {
        if (enableDetailedLogging && Time.time - lastLogTime >= logInterval)
        {
            LogDistanceInfo();
            lastLogTime = Time.time;
        }
    }
    
    /// <summary>
    /// グリッパーコンポーネントの自動検出
    /// </summary>
    private void AutoDetectGripperComponents()
    {
        if (actualGripperEndEffector == null || leftGripperTip == null || rightGripperTip == null)
        {
            Debug.Log("=== グリッパーコンポーネント自動検出 ===");
            
            // シーン内のすべてのTransformを検索
            Transform[] allTransforms = FindObjectsOfType<Transform>();
            
            foreach (Transform t in allTransforms)
            {
                string name = t.name.ToLower();
                
                // 左グリッパー検出
                if (leftGripperTip == null && (name.Contains("left") && (name.Contains("gripper") || name.Contains("finger") || name.Contains("tip"))))
                {
                    leftGripperTip = t;
                    Debug.Log($"左グリッパー検出: {t.name} at {t.position}");
                }
                
                // 右グリッパー検出
                if (rightGripperTip == null && (name.Contains("right") && (name.Contains("gripper") || name.Contains("finger") || name.Contains("tip"))))
                {
                    rightGripperTip = t;
                    Debug.Log($"右グリッパー検出: {t.name} at {t.position}");
                }
                
                // エンドエフェクター検出
                if (actualGripperEndEffector == null && (name.Contains("endeffector") || name.Contains("end_effector") || name.Contains("eef")))
                {
                    actualGripperEndEffector = t;
                    Debug.Log($"エンドエフェクター検出: {t.name} at {t.position}");
                }
            }
        }
    }
    
    private void LogDistanceInfo()
    {
        if (target == null) return;
        
        Debug.Log("=== 接触検出詳細情報 ===");
        
        // 現在のGripperTargetInterfaceの参照位置
        if (gripperInterface != null)
        {
            Vector3 interfacePos = gripperInterface.transform.position;
            float interfaceDistance = Vector3.Distance(interfacePos, target.transform.position);
            
            Debug.Log($"GripperInterface位置: {interfacePos}");
            Debug.Log($"Interface距離: {interfaceDistance:F3}m");
            Debug.Log($"検出半径: {gripperInterface.contactDetectionRadius:F3}m");
            Debug.Log($"Interface接触判定: {(interfaceDistance <= gripperInterface.contactDetectionRadius ? "接触中" : "範囲外")}");
        }
        
        // 実際のエンドエフェクター位置
        if (actualGripperEndEffector != null)
        {
            Vector3 endEffectorPos = actualGripperEndEffector.position;
            float endEffectorDistance = Vector3.Distance(endEffectorPos, target.transform.position);
            
            Debug.Log($"実エンドエフェクター位置: {endEffectorPos}");
            Debug.Log($"実エンドエフェクター距離: {endEffectorDistance:F3}m");
        }
        
        // 左右グリッパー位置
        if (leftGripperTip != null && rightGripperTip != null)
        {
            Vector3 leftPos = leftGripperTip.position;
            Vector3 rightPos = rightGripperTip.position;
            Vector3 centerPos = (leftPos + rightPos) / 2f;
            
            float leftDistance = Vector3.Distance(leftPos, target.transform.position);
            float rightDistance = Vector3.Distance(rightPos, target.transform.position);
            float centerDistance = Vector3.Distance(centerPos, target.transform.position);
            
            Debug.Log($"左グリッパー位置: {leftPos} (距離: {leftDistance:F3}m)");
            Debug.Log($"右グリッパー位置: {rightPos} (距離: {rightDistance:F3}m)");
            Debug.Log($"グリッパー中心位置: {centerPos} (距離: {centerDistance:F3}m)");
            
            // 推奨設定
            float recommendedRadius = Mathf.Max(leftDistance, rightDistance, centerDistance) + 0.05f;
            Debug.Log($"推奨検出半径: {recommendedRadius:F3}m");
        }
        
        Debug.Log($"ターゲット位置: {target.transform.position}");
        
        // 問題の診断
        DiagnoseProblem();
    }
    
    private void DiagnoseProblem()
    {
        Debug.Log("=== 問題診断 ===");
        
        if (gripperInterface == null)
        {
            Debug.LogError("GripperTargetInterfaceが見つかりません");
            return;
        }
        
        if (target == null)
        {
            Debug.LogError("DeformableTargetが見つかりません");
            return;
        }
        
        Vector3 gripperPos = gripperInterface.transform.position;
        Vector3 targetPos = target.transform.position;
        float distance = Vector3.Distance(gripperPos, targetPos);
        float radius = gripperInterface.contactDetectionRadius;
        
        if (distance > 10f)
        {
            Debug.LogWarning("グリッパーとターゲットが非常に離れています。GripperTargetInterfaceが間違ったオブジェクトにアタッチされている可能性があります。");
            Debug.LogWarning("解決策: GripperTargetInterfaceを実際のエンドエフェクターまたはグリッパーにアタッチしてください。");
        }
        else if (distance > radius * 2)
        {
            Debug.LogWarning($"検出半径が小さすぎます。現在の距離{distance:F3}mに対して、半径を{distance + 0.1f:F3}m以上に設定してください。");
        }
        else if (distance <= radius)
        {
            Debug.Log("接触検出は正常に動作するはずです。力伝達システムを確認してください。");
        }
    }
    
    void OnDrawGizmos()
    {
        if (!enableGizmoDisplay) return;
        
        // ターゲット表示
        if (target != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireCube(target.transform.position, Vector3.one * 0.1f);
        }
        
        // GripperInterface位置
        if (gripperInterface != null)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(gripperInterface.transform.position, 0.05f);
            Gizmos.DrawWireSphere(gripperInterface.transform.position, gripperInterface.contactDetectionRadius);
            
            // ターゲットとの接続線
            if (target != null)
            {
                Gizmos.color = Color.yellow;
                Gizmos.DrawLine(gripperInterface.transform.position, target.transform.position);
            }
        }
        
        // 実際のエンドエフェクター位置
        if (actualGripperEndEffector != null)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(actualGripperEndEffector.position, 0.03f);
            
            if (target != null)
            {
                Gizmos.color = Color.cyan;
                Gizmos.DrawLine(actualGripperEndEffector.position, target.transform.position);
            }
        }
        
        // 左右グリッパー
        if (leftGripperTip != null)
        {
            Gizmos.color = Color.magenta;
            Gizmos.DrawWireSphere(leftGripperTip.position, 0.02f);
        }
        
        if (rightGripperTip != null)
        {
            Gizmos.color = Color.magenta;
            Gizmos.DrawWireSphere(rightGripperTip.position, 0.02f);
        }
        
        // グリッパー中心
        if (leftGripperTip != null && rightGripperTip != null)
        {
            Vector3 center = (leftGripperTip.position + rightGripperTip.position) / 2f;
            Gizmos.color = Color.white;
            Gizmos.DrawWireSphere(center, 0.04f);
        }
    }
}