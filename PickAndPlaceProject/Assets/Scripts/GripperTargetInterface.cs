using UnityEngine;

/// <summary>
/// グリッパーとターゲット間の力伝達システム（修正版）
/// 適切なエンドエフェクター参照による接触検出
/// </summary>
public class GripperTargetInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public SimpleGripForceController simpleGripperController;
    public DeformableTarget target;
    
    [Header("エンドエフェクター参照")]
    [Tooltip("実際のグリッパーの指先またはエンドエフェクターのTransform")]
    public Transform gripperEndEffector;  // 新規追加
    [Tooltip("左グリッパーの指先（より正確な位置検出用）")]
    public Transform leftGripperTip;      // 新規追加
    [Tooltip("右グリッパーの指先（より正確な位置検出用）")]
    public Transform rightGripperTip;     // 新規追加
    
    [Header("力伝達設定")]
    public float forceTransferRate = 1f;
    public float contactDetectionRadius = 0.15f;  // 適切な値に調整
    public LayerMask targetLayer = -1;
    
    [Header("デバッグ")]
    public bool showForceGizmos = true;
    public bool enableForceLogging = false;
    
    // 内部状態
    private bool isInContact = false;
    private Vector3 lastContactPoint;
    private float lastForceTransferred = 0f;
    private Vector3 effectiveGripperPosition;
    
    void Start()
    {
        if (simpleGripperController == null)
            simpleGripperController = GetComponent<SimpleGripForceController>();
        
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
            
        InitializeInterface();
    }
    
    void FixedUpdate()
    {
        UpdateEffectiveGripperPosition();
        UpdateContactDetection();
        TransferForceToTarget();
    }
    
    private void InitializeInterface()
    {
        if (simpleGripperController == null || target == null)
        {
            Debug.LogError("GripperTargetInterface: 必要なコンポーネントが見つかりません");
            enabled = false;
            return;
        }
        
        // エンドエフェクター参照の自動検出
        if (gripperEndEffector == null)
        {
            AutoDetectEndEffector();
        }
        
        Debug.Log("GripperTargetInterface initialized successfully");
    }
    
    /// <summary>
    /// エンドエフェクターの自動検出
    /// </summary>
    private void AutoDetectEndEffector()
    {
        // 1. 指定されたエンドエフェクターを使用
        if (gripperEndEffector != null)
        {
            effectiveGripperPosition = gripperEndEffector.position;
            return;
        }
        
        // 2. 左右のグリッパー先端の中点を使用
        if (leftGripperTip != null && rightGripperTip != null)
        {
            effectiveGripperPosition = (leftGripperTip.position + rightGripperTip.position) / 2f;
            return;
        }
        
        // 3. 子オブジェクトから"EndEffector"または"Gripper"を含む名前を検索
        Transform[] children = GetComponentsInChildren<Transform>();
        foreach (Transform child in children)
        {
            if (child.name.ToLower().Contains("endeffector") || 
                child.name.ToLower().Contains("gripper") ||
                child.name.ToLower().Contains("tip"))
            {
                gripperEndEffector = child;
                Debug.Log($"自動検出: エンドエフェクター = {child.name}");
                break;
            }
        }
        
        // 4. フォールバック: 現在のTransformを使用（警告付き）
        if (gripperEndEffector == null)
        {
            Debug.LogWarning("エンドエフェクターが見つかりません。現在のTransformを使用しますが、距離が不正確になる可能性があります。");
            gripperEndEffector = transform;
        }
    }
    
    /// <summary>
    /// 有効なグリッパー位置の更新
    /// </summary>
    private void UpdateEffectiveGripperPosition()
    {
        if (leftGripperTip != null && rightGripperTip != null)
        {
            // 左右のグリッパー先端の中点を使用（最も正確）
            effectiveGripperPosition = (leftGripperTip.position + rightGripperTip.position) / 2f;
        }
        else if (gripperEndEffector != null)
        {
            // 指定されたエンドエフェクターを使用
            effectiveGripperPosition = gripperEndEffector.position;
        }
        else
        {
            // フォールバック
            effectiveGripperPosition = transform.position;
        }
    }
    
    private void UpdateContactDetection()
    {
        if (target == null) return;
        
        // 修正された距離計算
        float distance = Vector3.Distance(effectiveGripperPosition, target.transform.position);
        isInContact = distance <= contactDetectionRadius;
        
        if (isInContact)
        {
            lastContactPoint = target.transform.position;
        }
        
        // デバッグログ（定期的に出力）
        if (Time.fixedTime % 1f < Time.fixedDeltaTime)
        {
            Debug.Log($"接触検出: 距離={distance:F3}m, 閾値={contactDetectionRadius:F3}m, 接触={isInContact}");
        }
    }
    
    private void TransferForceToTarget()
    {
        // 接触状態をログ出力（詳細デバッグ用）
        if (Time.fixedTime % 0.5f < Time.fixedDeltaTime)
        {
            Debug.Log($"力伝達スキップ - 接触:{isInContact}, Target:{target != null}, Controller:{simpleGripperController != null}");
        }
        
        if (!isInContact || target == null || simpleGripperController == null) return;
        
        // SimpleGripForceControllerから現在の力を取得
        float currentForce = GetCurrentForceFromSimpleController() * forceTransferRate;
        
        // ターゲットに力を伝達
        if (target != null && currentForce > 0.1f)
        {
            target.ApplyGripperForce(currentForce, lastContactPoint);
            Debug.Log($"力伝達実行: {currentForce:F2}N");
        }
        
        lastForceTransferred = currentForce;
        
        // ログ出力
        if (enableForceLogging && Time.fixedTime % 0.1f < Time.fixedDeltaTime)
        {
            Debug.Log($"Force Transfer: {currentForce:F2}N to target object");
        }
    }
    
    /// <summary>
    /// SimpleGripForceControllerから力を取得（改良版）
    /// </summary>
    private float GetCurrentForceFromSimpleController()
    {
        if (simpleGripperController == null) return 0f;
        
        // より正確な力取得のため、SimpleGripForceControllerに公開プロパティを追加する必要があります
        // 暫定的にbaseGripForceを使用
        return simpleGripperController.baseGripForce;
    }
    
    /// <summary>
    /// 把持状態の評価
    /// </summary>
    public GraspEvaluation EvaluateGrasp()
    {
        if (target == null || simpleGripperController == null)
            return new GraspEvaluation { result = GraspResult.Failure };
        
        var objectState = target.GetCurrentState();
        
        // 評価ロジック
        GraspResult result = DetermineGraspResult(objectState);
        
        return new GraspEvaluation
        {
            result = result,
            appliedForce = objectState.appliedForce,
            deformation = objectState.deformation,
            isBroken = objectState.isBroken,
            confidence = 0.8f // 仮の値
        };
    }
    
    private GraspResult DetermineGraspResult(DeformableTarget.ObjectState objectState)
    {
        // 破損チェック
        if (objectState.isBroken)
            return GraspResult.OverGrip;
        
        // 把持力による判定
        float force = objectState.appliedForce;
        
        if (force < 1f)
            return GraspResult.UnderGrip;
        else if (force > 50f)
            return GraspResult.OverGrip;
        else
            return GraspResult.Success;
    }
    
    void OnDrawGizmos()
    {
        // 有効なグリッパー位置の表示
        Gizmos.color = Color.blue;
        Gizmos.DrawWireSphere(effectiveGripperPosition, 0.02f);
        
        // 接触検出範囲の表示
        Gizmos.color = isInContact ? Color.green : Color.yellow;
        Gizmos.DrawWireSphere(effectiveGripperPosition, contactDetectionRadius);
        
        // 力の可視化
        if (showForceGizmos && isInContact)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(lastContactPoint, 0.05f);
            Gizmos.DrawLine(effectiveGripperPosition, lastContactPoint);
        }
        
        // ターゲットとの距離線
        if (target != null)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawLine(effectiveGripperPosition, target.transform.position);
        }
    }
}

// 必要な列挙型とクラス
public enum GraspResult
{
    Success,
    UnderGrip,
    OverGrip,
    Failure
}

public class GraspEvaluation
{
    public GraspResult result;
    public float appliedForce;
    public float deformation;
    public bool isBroken;
    public float confidence;
}