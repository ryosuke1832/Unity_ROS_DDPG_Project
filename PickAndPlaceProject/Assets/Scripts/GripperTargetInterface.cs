using UnityEngine;

/// <summary>
/// グリッパーとターゲット間の力伝達システム（SimpleGripForceController対応版）
/// 完全修正版
/// </summary>
public class GripperTargetInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public SimpleGripForceController simpleGripperController;
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
        bool wasInContact = isInContact;
        isInContact = distance <= contactDetectionRadius;
        
        if (isInContact)
        {
            lastContactPoint = target.transform.position;
        }
        
        // 接触状態の変化をログ
        if (enableForceLogging && isInContact != wasInContact)
        {
            Debug.Log($"接触状態変化: {wasInContact} → {isInContact}, 距離: {distance:F3}m");
        }
    }
    
    private void TransferForceToTarget()
    {
        // 前提条件チェック
        if (!isInContact || target == null || simpleGripperController == null) 
        {
            if (enableForceLogging && Time.fixedTime % 1f < Time.fixedDeltaTime)
            {
                Debug.Log($"力伝達スキップ - 接触:{isInContact}, Target:{target != null}, Controller:{simpleGripperController != null}");
            }
            return;
        }
        
        // SimpleGripForceControllerから現在の力を取得
        float rawForce = GetCurrentForceFromSimpleController();
        float currentForce = rawForce * forceTransferRate;
        
        // ターゲットに力を伝達
        if (target != null && currentForce > 0f)
        {
            target.ApplyGripperForce(currentForce, lastContactPoint);
            lastForceTransferred = currentForce;
            
            if (enableForceLogging)
            {
                Debug.Log($"力伝達成功 - Raw: {rawForce:F2}N, Applied: {currentForce:F2}N, Contact: {lastContactPoint}");
            }
        }
        else
        {
            if (enableForceLogging && Time.fixedTime % 0.5f < Time.fixedDeltaTime)
            {
                Debug.Log($"力伝達失敗 - Force: {currentForce:F2}N, Target: {target != null}");
            }
        }
    }
    
    /// <summary>
    /// SimpleGripForceControllerから力を取得（修正版）
    /// </summary>
    private float GetCurrentForceFromSimpleController()
    {
        if (simpleGripperController == null) return 0f;
        
        // SimpleGripForceControllerが有効でない場合は0を返す
        if (!simpleGripperController.enabled) 
        {
            if (enableForceLogging && Time.fixedTime % 2f < Time.fixedDeltaTime)
                Debug.Log("SimpleGripForceController が無効です");
            return 0f;
        }
        
        // 力制御システムから実際の力を取得
        try
        {
            // 新しく追加予定のメソッドを使用
            if (HasMethod(simpleGripperController, "GetCurrentTargetForce"))
            {
                return simpleGripperController.GetCurrentTargetForce();
            }
            else
            {
                // フォールバック: 基本力を返す
                return simpleGripperController.GetBaseGripForce();
            }
        }
        catch (System.Exception e)
        {
            if (enableForceLogging)
                Debug.LogWarning($"力取得エラー: {e.Message}");
            
            // フォールバック: 基本力を返す
            return simpleGripperController.baseGripForce;
        }
    }
    
    /// <summary>
    /// メソッドの存在確認用ヘルパー
    /// </summary>
    private bool HasMethod(object obj, string methodName)
    {
        return obj.GetType().GetMethod(methodName) != null;
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
            confidence = CalculateConfidence(objectState)
        };
    }
    
    private GraspResult DetermineGraspResult(DeformableTarget.ObjectState objectState)
    {
        // 破損チェック
        if (objectState.isBroken)
            return GraspResult.OverGrip;
        
        // 把持力による判定
        float force = objectState.appliedForce;
        
        // 材質に応じた閾値
        float successMin = GetSuccessForceMin(objectState.materialType);
        float successMax = GetSuccessForceMax(objectState.materialType);
        float overGripThreshold = GetOverGripThreshold(objectState.materialType);
        
        if (force < successMin)
            return GraspResult.UnderGrip;
        else if (force > overGripThreshold)
            return GraspResult.OverGrip;
        else if (force >= successMin && force <= successMax)
            return GraspResult.Success;
        else
            return GraspResult.Failure;
    }
    
    // 材質別の力閾値設定
    private float GetSuccessForceMin(DeformableTarget.MaterialType material)
    {
        return material switch
        {
            DeformableTarget.MaterialType.Soft => 2f,
            DeformableTarget.MaterialType.Medium => 5f,
            DeformableTarget.MaterialType.Hard => 8f,
            DeformableTarget.MaterialType.Fragile => 1f,
            _ => 5f
        };
    }
    
    private float GetSuccessForceMax(DeformableTarget.MaterialType material)
    {
        return material switch
        {
            DeformableTarget.MaterialType.Soft => 15f,
            DeformableTarget.MaterialType.Medium => 25f,
            DeformableTarget.MaterialType.Hard => 40f,
            DeformableTarget.MaterialType.Fragile => 8f,
            _ => 25f
        };
    }
    
    private float GetOverGripThreshold(DeformableTarget.MaterialType material)
    {
        return material switch
        {
            DeformableTarget.MaterialType.Soft => 20f,
            DeformableTarget.MaterialType.Medium => 35f,
            DeformableTarget.MaterialType.Hard => 60f,
            DeformableTarget.MaterialType.Fragile => 12f,
            _ => 35f
        };
    }
    
    private float CalculateConfidence(DeformableTarget.ObjectState objectState)
    {
        // 力の安定性、変形の一貫性などから信頼度を計算
        float forceStability = objectState.appliedForce > 0 ? 0.8f : 0.4f;
        float deformationConsistency = 1f - Mathf.Abs(objectState.deformation - 0.5f);
        
        return (forceStability + deformationConsistency) / 2f;
    }
    
    // デバッグ表示
    void OnDrawGizmos()
    {
        if (!showForceGizmos) return;
        
        // 接触範囲の表示
        Gizmos.color = isInContact ? Color.green : Color.yellow;
        Gizmos.DrawWireSphere(transform.position, contactDetectionRadius);
        
        // 力の可視化
        if (isInContact && lastForceTransferred > 0)
        {
            Gizmos.color = Color.red;
            Vector3 forceDirection = (lastContactPoint - transform.position).normalized;
            Gizmos.DrawRay(transform.position, forceDirection * lastForceTransferred * 0.1f);
        }
    }
}

// 把持評価結果
public enum GraspResult
{
    Success,   // 成功
    OverGrip,  // 過把持
    UnderGrip, // 不足把持
    Failure    // 失敗
}

[System.Serializable]
public class GraspEvaluation
{
    public GraspResult result;
    public float appliedForce;
    public float deformation;
    public bool isBroken;
    public float confidence;
}