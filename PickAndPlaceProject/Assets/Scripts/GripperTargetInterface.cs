using UnityEngine;

public class GripperTargetInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public SimpleGripForceController simpleGripperController;
    public DeformableTarget target;
    
    [Header("グリッパー設定")]
    public Transform leftGripperTip;
    public Transform rightGripperTip;
    public ArticulationBody leftGripperBody;
    public ArticulationBody rightGripperBody;
    
    [Header("接触判定設定")]
    public float gripperCloseThreshold = 0.005f;
    public float contactForceThreshold = 0.5f;
    public bool requireBothGrippersContact = true;
    
    [Header("デバッグ")]
    public bool showContactGizmos = true;
    public bool enableDetailedLogging = false;
    
    // 内部状態
    private bool leftGripperInContact = false;
    private bool rightGripperInContact = false;
    private Vector3 leftContactPoint;
    private Vector3 rightContactPoint;
    private Vector3 lastContactNormal = Vector3.up;
    private bool isGripperClosed = false;
    private float currentLeftPosition = 0f;
    private float currentRightPosition = 0f;
    
    void Start()
    {
        // コンポーネント検索
        if (simpleGripperController == null)
            simpleGripperController = GetComponent<SimpleGripForceController>();
        
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
        
        // グリッパーの自動検索
        AutoFindGrippers();
        
        // トリガーの設定
        SetupGripperColliders();
        
        Debug.Log("GripperTargetInterface initialized");
    }
    
    void FixedUpdate()
    {
        UpdateGripperState();
        TransferForceToTarget();
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
            AddTriggerToGripper(leftGripperTip.gameObject, true);
        }
        
        if (rightGripperTip != null)
        {
            AddTriggerToGripper(rightGripperTip.gameObject, false);
        }
    }
    
    private void AddTriggerToGripper(GameObject gripperObject, bool isLeft)
    {
        // トリガーコライダーの追加
        Collider existingCollider = gripperObject.GetComponent<Collider>();
        if (existingCollider == null)
        {
            BoxCollider triggerCollider = gripperObject.AddComponent<BoxCollider>();
            triggerCollider.isTrigger = true;
            triggerCollider.size = Vector3.one * 0.05f;
            
            if (enableDetailedLogging)
                Debug.Log($"Added trigger to {(isLeft ? "left" : "right")} gripper");
        }
        
        // 接触検出コンポーネントの追加
        SimpleContactDetector detector = gripperObject.GetComponent<SimpleContactDetector>();
        if (detector == null)
        {
            detector = gripperObject.AddComponent<SimpleContactDetector>();
            detector.Initialize(this, isLeft);
        }
    }
    
    private void UpdateGripperState()
    {
        // グリッパーの位置を取得
        if (leftGripperBody != null && leftGripperBody.jointPosition.dofCount > 0)
        {
            currentLeftPosition = leftGripperBody.jointPosition[0];
        }
        
        if (rightGripperBody != null && rightGripperBody.jointPosition.dofCount > 0)
        {
            currentRightPosition = rightGripperBody.jointPosition[0];
        }
        
        // グリッパーが閉じているかチェック
        bool wasGripperClosed = isGripperClosed;
        bool leftClosed = currentLeftPosition < -gripperCloseThreshold;
        bool rightClosed = currentRightPosition > gripperCloseThreshold;
        isGripperClosed = leftClosed && rightClosed;
        
        // 状態変化をログ
        if (enableDetailedLogging && wasGripperClosed != isGripperClosed)
        {
            Debug.Log($"Gripper state changed: {(isGripperClosed ? "CLOSED" : "OPEN")}");
            Debug.Log($"Positions - Left: {currentLeftPosition:F4}, Right: {currentRightPosition:F4}");
        }
    }
    
    private void TransferForceToTarget()
    {
        if (target == null || simpleGripperController == null) return;
        
        // 力を伝達できる条件をチェック
        bool canTransferForce = isGripperClosed && HasValidContact();
        
        if (!canTransferForce) return;
        
        // 現在の力を取得
        float currentForce = simpleGripperController.GetCurrentTargetForce();
        
        // 閾値以上の場合のみ伝達
        if (currentForce >= contactForceThreshold)
        {
            Vector3 contactPoint = CalculateContactPoint();
            
            // 方向を考慮した力伝達
            target.ApplyGripperForceWithDirection(currentForce, contactPoint, lastContactNormal);
            
            if (enableDetailedLogging && Time.fixedTime % 0.2f < Time.fixedDeltaTime)
            {
                Debug.Log($"Force Transfer: {currentForce:F2}N at {contactPoint}");
            }
        }
    }
    
    private bool HasValidContact()
    {
        if (requireBothGrippersContact)
        {
            return leftGripperInContact && rightGripperInContact;
        }
        else
        {
            return leftGripperInContact || rightGripperInContact;
        }
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
        
        return target.transform.position;
    }
    
    // 外部から呼び出される接触イベント
    public void OnGripperContactEnter(Collider collider, bool isLeftGripper, Vector3 contactPoint, Vector3 contactNormal)
    {
        if (collider.gameObject != target.gameObject) return;
        
        if (isLeftGripper)
        {
            leftGripperInContact = true;
            leftContactPoint = contactPoint;
        }
        else
        {
            rightGripperInContact = true;
            rightContactPoint = contactPoint;
        }
        
        lastContactNormal = contactNormal;
        
        if (enableDetailedLogging)
        {
            Debug.Log($"{(isLeftGripper ? "Left" : "Right")} gripper contact ENTER");
        }
    }
    
    public void OnGripperContactExit(Collider collider, bool isLeftGripper)
    {
        if (collider.gameObject != target.gameObject) return;
        
        if (isLeftGripper)
        {
            leftGripperInContact = false;
        }
        else
        {
            rightGripperInContact = false;
        }
        
        if (enableDetailedLogging)
        {
            Debug.Log($"{(isLeftGripper ? "Left" : "Right")} gripper contact EXIT");
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
            Gizmos.color = Color.cyan;
            Gizmos.DrawSphere(leftContactPoint, 0.005f);
        }
        
        if (rightGripperInContact)
        {
            Gizmos.color = Color.magenta;
            Gizmos.DrawSphere(rightContactPoint, 0.005f);
        }
        
        // 接触法線の表示
        if (HasValidContact())
        {
            Gizmos.color = Color.yellow;
            Vector3 contactPoint = CalculateContactPoint();
            Gizmos.DrawRay(contactPoint, lastContactNormal * 0.05f);
        }
    }

/// <summary>
    /// 把持状態の評価
    /// TrajectoryPlannerDeform から呼び出される
    /// </summary>
    public GraspEvaluation EvaluateGrasp()
    {
        if (target == null || simpleGripperController == null)
            return GraspEvaluation.CreateSimple(GraspResult.Failure);
        
        // ターゲットの現在状態を取得
        ObjectState objectState = target.GetCurrentState();
        
        // グリッパーの状態を取得
        GraspingState graspingState = simpleGripperController.GetGraspingStateForInterface();
        
        // 接触の有効性をチェック
        bool hasValidContact = HasValidContact();
        bool isGripping = isGripperClosed && hasValidContact;
        
        // 把持結果を判定
        GraspResult result = DetermineGraspResult(objectState, graspingState, isGripping);
        
        // 評価結果を作成
        GraspEvaluation evaluation = new GraspEvaluation
        {
            result = result,
            appliedForce = objectState.appliedForce,
            deformation = objectState.deformation,
            isBroken = objectState.isBroken,
            hasContact = hasValidContact,
            isGripping = isGripping,
            confidence = CalculateConfidence(hasValidContact, isGripping, objectState),
            evaluationTime = Time.time
        };
        
        if (enableDetailedLogging)
        {
            Debug.Log($"Grasp Evaluation: {result}, Force: {objectState.appliedForce:F2}N, " +
                     $"Deformation: {objectState.deformation:F3}, Confidence: {evaluation.confidence:F2}");
        }
        
        return evaluation;
    }
    
    /// <summary>
    /// 把持結果を判定
    /// </summary>
    private GraspResult DetermineGraspResult(ObjectState objectState, GraspingState graspingState, bool isGripping)
    {
        // 破損チェック
        if (objectState.isBroken)
            return GraspResult.Broken;
        
        // 接触チェック
        if (!isGripping)
            return GraspResult.NoContact;
        
        // 力による判定
        float force = objectState.appliedForce;
        
        if (force < 1f)
            return GraspResult.UnderGrip;
        else if (force > 50f)
            return GraspResult.OverGrip;
        else
            return GraspResult.Success;
    }
    
    /// <summary>
    /// 信頼度を計算
    /// </summary>
    private float CalculateConfidence(bool hasValidContact, bool isGripping, ObjectState objectState)
    {
        float confidence = 0f;
        
        // 接触状態
        if (hasValidContact) confidence += 0.3f;
        
        // 把持状態
        if (isGripping) confidence += 0.3f;
        
        // グリッパー閉状態
        if (isGripperClosed) confidence += 0.2f;
        
        // 破損していない
        if (!objectState.isBroken) confidence += 0.2f;
        
        return Mathf.Clamp01(confidence);
    }

    // GripperTargetInterface.cs の IsGripperInClosedState メソッドを以下に変更

/// <summary>
/// グリッパーが閉じた状態かどうかを判定（URDF準拠）
/// </summary>
private bool IsGripperInClosedState()
{
    // URDFファイルによると：
    // CloseGripper: leftDrive.target = -0.01f, rightDrive.target = 0.01f
    // つまり：左グリッパーが負の方向、右グリッパーが正の方向で閉じる
    
    bool leftClosed = currentLeftPosition < -gripperCloseThreshold;
    bool rightClosed = currentRightPosition > gripperCloseThreshold;
    
    if (enableDetailedLogging)
    {
        Debug.Log($"Gripper positions - Left: {currentLeftPosition:F4} (closed: {leftClosed}), " +
                 $"Right: {currentRightPosition:F4} (closed: {rightClosed})");
    }
    
    return leftClosed && rightClosed;
}

}

// シンプルな接触検出コンポーネント
public class SimpleContactDetector : MonoBehaviour
{
    private GripperTargetInterface parentInterface;
    private bool isLeftGripper;
    
    public void Initialize(GripperTargetInterface targetInterface, bool isLeft)
    {
        parentInterface = targetInterface;
        isLeftGripper = isLeft;
    }
    
    void OnTriggerEnter(Collider other)
    {
        if (parentInterface == null) return;
        
        Vector3 contactPoint = other.ClosestPoint(transform.position);
        Vector3 contactNormal = (transform.position - other.transform.position).normalized;
        
        parentInterface.OnGripperContactEnter(other, isLeftGripper, contactPoint, contactNormal);
    }
    
    void OnTriggerExit(Collider other)
    {
        if (parentInterface == null) return;
        
        parentInterface.OnGripperContactExit(other, isLeftGripper);
    }
}