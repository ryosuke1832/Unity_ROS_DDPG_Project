using UnityEngine;

/// <summary>
/// アルミ缶専用のグリッパーインターフェース
/// DeformableTargetは使用せず、IntegratedAluminumCanのみに対応
/// </summary>
public class GripperTargetInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public SimpleGripForceController simpleGripperController;
    public IntegratedAluminumCan target;
    
    [Header("グリッパー設定")]
    public Transform leftGripperTip;
    public Transform rightGripperTip;
    public ArticulationBody leftGripperBody;
    public ArticulationBody rightGripperBody;
    
    [Header("接触判定設定")]
    public float gripperCloseThreshold = 0.01f;
    public float contactForceThreshold = 0.1f;
    public bool requireBothGrippersContact = true;
    
    [Header("デバッグ")]
    public bool showContactGizmos = true;
    public bool enableDetailedLogging = false;
    
    // 内部状態
    private bool leftGripperInContact = false;
    private bool rightGripperInContact = false;
    private Vector3 leftContactPoint;
    private Vector3 rightContactPoint;
    private Vector3 leftContactNormal = Vector3.up;
    private Vector3 rightContactNormal = Vector3.up;
    private Vector3 lastContactNormal = Vector3.up;
    private bool isGripperClosed = false;
    private float currentLeftPosition = 0f;
    private float currentRightPosition = 0f;
    private bool hasLoggedForceTransfer = false;
    
    void Start()
    {
        // コンポーネント検索
        if (simpleGripperController == null)
            simpleGripperController = GetComponent<SimpleGripForceController>();
        
        if (target == null)
            target = FindObjectOfType<IntegratedAluminumCan>();
        
        // グリッパーの自動検索
        AutoFindGrippers();
        
        // トリガーの設定
        SetupGripperColliders();
        
        Debug.Log($"GripperTargetInterface initialized with AluminumCan: {(target != null ? "✅" : "❌")}");
    }
    
    void FixedUpdate()
    {
        UpdateGripperState();
        TransferForceToTarget();
    }
    
    private void TransferForceToTarget()
    {
        if (target == null || simpleGripperController == null) return;
        
        bool canTransferForce = isGripperClosed && HasValidContact();
        
        if (!canTransferForce)
        {
            hasLoggedForceTransfer = false;
            return;
        }
        
        float currentForce = simpleGripperController.GetCurrentTargetForce();
        
        if (currentForce >= contactForceThreshold)
        {
            Vector3 contactPoint = CalculateContactPoint();
            lastContactNormal = CalculateAggregateContactNormal();
            
            if (!hasLoggedForceTransfer)
            {
                Debug.Log($"=== アルミ缶への力伝達開始 ===");
                Debug.Log($"接触点: {contactPoint}");
                Debug.Log($"接触法線: {lastContactNormal}");
                Debug.Log($"伝達力: {currentForce:F2}N");
                hasLoggedForceTransfer = true;
            }
            
            // IntegratedAluminumCan のメソッド呼び出し
            target.ApplyGripperForceWithDirection(currentForce, contactPoint, lastContactNormal);
            
            Debug.Log($"アルミ缶に力を適用: {currentForce:F2}N");
        }
        else
        {
            hasLoggedForceTransfer = false;
            Debug.Log($"力が閾値以下: {currentForce:F2}N < {contactForceThreshold:F2}N");
        }
    }
    
    private void AutoFindGrippers()
    {
        ArticulationBody[] allBodies = FindObjectsOfType<ArticulationBody>();
        
        foreach (var body in allBodies)
        {
            if (body.name.Contains("left_gripper"))
            {
                leftGripperBody = body;
                leftGripperTip = body.transform;
                if (enableDetailedLogging)
                    Debug.Log($"Found left gripper: {body.name}");
            }
            else if (body.name.Contains("right_gripper"))
            {
                rightGripperBody = body;
                rightGripperTip = body.transform;
                if (enableDetailedLogging)
                    Debug.Log($"Found right gripper: {body.name}");
            }
        }
    }
    
    private void SetupGripperColliders()
    {
        if (leftGripperTip != null)
        {
            SetupSingleGripperCollider(leftGripperTip.gameObject, true);
        }
        
        if (rightGripperTip != null)
        {
            SetupSingleGripperCollider(rightGripperTip.gameObject, false);
        }
    }
    
    private void SetupSingleGripperCollider(GameObject gripperObj, bool isLeft)
    {
        // 既存のコライダーをチェック
        Collider existingCollider = gripperObj.GetComponent<Collider>();
        if (existingCollider == null)
        {
            // 小さなトリガーコライダーを追加
            SphereCollider triggerCollider = gripperObj.AddComponent<SphereCollider>();
            triggerCollider.isTrigger = true;
            triggerCollider.radius = 0.02f;
        }
        else
        {
            existingCollider.isTrigger = true;
        }
        
        // 距離ベースの接触検出を使用（SimpleContactDetectorは使わない）
        Debug.Log($"Setup gripper collider for {(isLeft ? "left" : "right")} gripper: {gripperObj.name}");
    }
    
    private void UpdateGripperState()
    {
        if (leftGripperBody != null && rightGripperBody != null)
        {
            currentLeftPosition = leftGripperBody.xDrive.target;
            currentRightPosition = rightGripperBody.xDrive.target;
            
            bool leftClosed = currentLeftPosition <= -gripperCloseThreshold;
            bool rightClosed = currentRightPosition >= gripperCloseThreshold;
            
            isGripperClosed = leftClosed && rightClosed;
        }
        
        // 距離ベースの接触検出
        UpdateDistanceBasedContact();
        
        if (leftGripperInContact || rightGripperInContact)
        {
            CalculateGripperForceDirection();
        }
    }
    
    /// <summary>
    /// 距離ベースの接触検出（SimpleContactDetectorの代替）
    /// </summary>
    private void UpdateDistanceBasedContact()
    {
        if (target == null) return;
        
        float contactDistance = 0.05f; // 5cm以内で接触とみなす
        
        // 左グリッパーの接触チェック
        if (leftGripperTip != null)
        {
            float leftDistance = Vector3.Distance(leftGripperTip.position, target.transform.position);
            bool wasInContact = leftGripperInContact;
            leftGripperInContact = leftDistance <= contactDistance;
            
            if (leftGripperInContact)
            {
                leftContactPoint = target.transform.position;
            }
            
            // 接触状態の変化をログ出力
            if (leftGripperInContact != wasInContact && enableDetailedLogging)
            {
                Debug.Log($"Left gripper contact: {leftGripperInContact} (distance: {leftDistance:F3}m)");
            }
        }
        
        // 右グリッパーの接触チェック
        if (rightGripperTip != null)
        {
            float rightDistance = Vector3.Distance(rightGripperTip.position, target.transform.position);
            bool wasInContact = rightGripperInContact;
            rightGripperInContact = rightDistance <= contactDistance;
            
            if (rightGripperInContact)
            {
                rightContactPoint = target.transform.position;
            }
            
            // 接触状態の変化をログ出力
            if (rightGripperInContact != wasInContact && enableDetailedLogging)
            {
                Debug.Log($"Right gripper contact: {rightGripperInContact} (distance: {rightDistance:F3}m)");
            }
        }
    }
    
    private void CalculateGripperForceDirection()
    {
        if (leftGripperTip == null || rightGripperTip == null) return;
        
        Vector3 gripperVector = rightGripperTip.position - leftGripperTip.position;
        Vector3 gripperDirection = gripperVector.normalized;
        
        leftContactNormal = gripperDirection;
        rightContactNormal = -gripperDirection;
        
        // 接触点の更新
        if (leftGripperInContact && target != null)
        {
            leftContactPoint = target.transform.position;
        }
        
        if (rightGripperInContact && target != null)
        {
            rightContactPoint = target.transform.position;
        }
    }
    
    private bool HasValidContact()
    {
        return requireBothGrippersContact ? 
            (leftGripperInContact && rightGripperInContact) : 
            (leftGripperInContact || rightGripperInContact);
    }
    
    private Vector3 CalculateContactPoint()
    {
        if (leftGripperInContact && rightGripperInContact)
        {
            return (leftContactPoint + rightContactPoint) * 0.5f;
        }
        else if (leftGripperInContact)
        {
            return leftContactPoint;
        }
        else if (rightGripperInContact)
        {
            return rightContactPoint;
        }
        
        return target != null ? target.transform.position : Vector3.zero;
    }
    
    private Vector3 CalculateAggregateContactNormal()
    {
        Vector3 aggregateNormal = Vector3.zero;
        int contactCount = 0;
        
        if (leftGripperInContact)
        {
            aggregateNormal += leftContactNormal;
            contactCount++;
        }
        
        if (rightGripperInContact)
        {
            aggregateNormal += rightContactNormal;
            contactCount++;
        }
        
        if (contactCount > 0)
        {
            aggregateNormal /= contactCount;
            return aggregateNormal.normalized;
        }
        
        return Vector3.up;
    }
    
    /// <summary>
    /// アルミ缶専用の把持評価
    /// </summary>
    public GraspEvaluation EvaluateGrasp()
    {
        if (target == null || simpleGripperController == null)
        {
            return GraspEvaluation.CreateSimple(GraspResult.Failure);
        }
        
        // アルミ缶の現在状態を取得
        ObjectState objectState = target.GetCurrentState();
        
        // グリッパーの状態を取得
        GraspingState graspingState = simpleGripperController.GetGraspingStateForInterface();
        
        // 接触の有効性をチェック
        bool hasValidContact = HasValidContact();
        bool isGripping = isGripperClosed && hasValidContact;
        
        // アルミ缶専用の評価ロジック
        GraspResult result = DetermineAluminumCanGraspResult(objectState, graspingState, isGripping);
        
        // 評価結果を作成
        GraspEvaluation evaluation = new GraspEvaluation
        {
            result = result,
            appliedForce = objectState.appliedForce,
            deformation = objectState.deformation,
            isBroken = objectState.isBroken,
            hasContact = hasValidContact,
            isGripping = isGripping,
            confidence = CalculateConfidence(objectState, graspingState, hasValidContact, isGripping),
            evaluationTime = Time.time
        };
        
        if (enableDetailedLogging)
        {
            Debug.Log($"アルミ缶把持評価: {result}, 力: {objectState.appliedForce:F2}N, " +
                    $"変形: {objectState.deformation:F3}, 信頼度: {evaluation.confidence:F2}");
        }
        
        return evaluation;
    }
    
    private GraspResult DetermineAluminumCanGraspResult(ObjectState objectState, GraspingState graspingState, bool isGripping)
    {
        // アルミ缶がつぶれてしまった場合
        if (objectState.isBroken)
            return GraspResult.OverGrip;
        
        // 接触していない場合
        if (!isGripping)
            return GraspResult.NoContact;
        
        // 力による判定（アルミ缶専用の閾値）
        float force = objectState.appliedForce;
        
        if (force < 2f)
            return GraspResult.UnderGrip;
        else if (force > 20f && !objectState.isBroken) // まだつぶれていないが危険な力
            return GraspResult.OverGrip;
        else if (force >= 2f && force <= 15f) // アルミ缶に適切な力範囲
            return GraspResult.Success;
        else
            return GraspResult.Failure;
    }
    
    private float CalculateConfidence(ObjectState objectState, GraspingState graspingState, bool hasValidContact, bool isGripping)
    {
        float confidence = 0f;
        
        // 接触状態
        if (hasValidContact) confidence += 0.3f;
        
        // 把持状態
        if (isGripping) confidence += 0.3f;
        
        // グリッパー閉状態
        if (isGripperClosed) confidence += 0.2f;
        
        // アルミ缶がまだ破損していない
        if (!objectState.isBroken) confidence += 0.2f;
        
        return Mathf.Clamp01(confidence);
    }
    
    /// <summary>
    /// デバッグ用の状態確認
    /// </summary>
    [ContextMenu("Check Current State")]
    public void CheckCurrentState()
    {
        if (target == null)
        {
            Debug.LogError("❌ Target (IntegratedAluminumCan) が設定されていません");
            return;
        }
        
        Debug.Log("=== アルミ缶把持状態 ===");
        Debug.Log($"グリッパー閉じ状態: {isGripperClosed}");
        Debug.Log($"左グリッパー接触: {leftGripperInContact}");
        Debug.Log($"右グリッパー接触: {rightGripperInContact}");
        Debug.Log($"有効な接触: {HasValidContact()}");
        
        var state = target.GetCurrentState();
        Debug.Log($"缶の状態: {(state.isBroken ? "つぶれた" : "正常")}");
        Debug.Log($"適用力: {state.appliedForce:F2}N");
        Debug.Log($"蓄積力: {target.GetAccumulatedForce():F2}N");
        Debug.Log($"変形進行: {(state.deformation * 100):F1}%");
        
        if (simpleGripperController != null)
        {
            float currentForce = simpleGripperController.GetCurrentTargetForce();
            Debug.Log($"グリッパー制御力: {currentForce:F2}N");
        }
    }
    
    void OnDrawGizmos()
    {
        if (!showContactGizmos) return;
        
        // グリッパーの状態を色で表示
        if (leftGripperTip != null)
        {
            Gizmos.color = leftGripperInContact ? Color.green : Color.red;
            Gizmos.DrawWireSphere(leftGripperTip.position, 0.01f);
        }
        
        if (rightGripperTip != null)
        {
            Gizmos.color = rightGripperInContact ? Color.green : Color.red;
            Gizmos.DrawWireSphere(rightGripperTip.position, 0.01f);
        }
        
        // 接触点の表示
        if (leftGripperInContact)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(leftContactPoint, 0.02f);
        }
        
        if (rightGripperInContact)
        {
            Gizmos.color = Color.cyan;
            Gizmos.DrawWireSphere(rightContactPoint, 0.02f);
        }
        
        // 力の方向表示
        if (target != null && (leftGripperInContact || rightGripperInContact))
        {
            Gizmos.color = Color.yellow;
            Vector3 forceDirection = lastContactNormal * 0.05f;
            Gizmos.DrawRay(target.transform.position, forceDirection);
        }
    }
}