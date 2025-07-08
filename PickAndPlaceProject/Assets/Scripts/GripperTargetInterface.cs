using UnityEngine;

/// <summary>
/// グリッパーとターゲット間の力伝達システム（SimpleGripForceController対応版）
/// </summary>
public class GripperTargetInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public SimpleGripForceController simpleGripperController;  // 変更点
    public DeformableTarget target;
    
    [Header("力伝達設定")]
    public float forceTransferRate = 1f;        // 力伝達係数
    public float contactDetectionRadius = 0.1f;  // 接触検出半径
    public LayerMask targetLayer = -1;           // ターゲットレイヤー
    
    [Header("デバッグ")]
    public bool showForceGizmos = true;
    public bool enableForceLogging = false;
    
    // 内部状態
    private bool isInContact = false;
    private Vector3 lastContactPoint;
    private float lastForceTransferred = 0f;
    
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
        
        Debug.Log("GripperTargetInterface initialized successfully");
    }
    
    private void UpdateContactDetection()
    {
        // グリッパーとターゲット間の距離チェック
        if (target == null) return;
        
        float distance = Vector3.Distance(transform.position, target.transform.position);
        isInContact = distance <= contactDetectionRadius;
        
        if (isInContact)
        {
            lastContactPoint = target.transform.position;
        }
    }
    
    private void TransferForceToTarget()
    {
        if (!isInContact || target == null || simpleGripperController == null) return;
        
        // SimpleGripForceControllerから現在の力を取得
        float currentForce = GetCurrentForceFromSimpleController() * forceTransferRate;
        
        // ターゲットに力を伝達
        if (target != null)
        {
            target.ApplyGripperForce(currentForce, lastContactPoint);
        }
        
        lastForceTransferred = currentForce;
        
        // ログ出力
        if (enableForceLogging && Time.fixedTime % 0.1f < Time.fixedDeltaTime)
        {
            Debug.Log($"Force Transfer: {currentForce:F2}N to target object");
        }
    }
    
    /// <summary>
    /// SimpleGripForceControllerから力を取得
    /// </summary>
    private float GetCurrentForceFromSimpleController()
    {
        if (simpleGripperController == null) return 0f;
        
        // SimpleGripForceControllerの内部変数にアクセスする必要があるため
        // 公開プロパティを追加するか、リフレクションを使用する
        // ここでは基本値を返す（後で改善）
        return 10f; // 仮の値
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
        if (showForceGizmos && isInContact)
        {
            Gizmos.color = Color.red;
            Gizmos.DrawWireSphere(lastContactPoint, 0.05f);
            Gizmos.DrawLine(transform.position, lastContactPoint);
        }
        
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(transform.position, contactDetectionRadius);
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