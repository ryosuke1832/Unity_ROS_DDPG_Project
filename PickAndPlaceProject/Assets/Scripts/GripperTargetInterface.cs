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
    public float gripperCloseThreshold = 0.001f;
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
            target = FindObjectOfType<DeformableTarget>();
        
        // グリッパーの自動検索
        AutoFindGrippers();
        
        // トリガーの設定
        SetupGripperColliders();
        
        Debug.Log("GripperTargetInterface initialized");
    }
    
    void FixedUpdate()
    {
        // UpdateGripperState();
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
        
        // 接触検出コンポーネントを追加
        SimpleContactDetector detector = gripperObj.GetComponent<SimpleContactDetector>();
        if (detector == null)
        {
            detector = gripperObj.AddComponent<SimpleContactDetector>();
        }
        detector.Initialize(this, isLeft);
    }
    
    private void UpdateGripperState()
    {
        // グリッパーの位置を取得
        if (leftGripperBody != null)
        {
            currentLeftPosition = leftGripperBody.jointPosition[0];
        }
        
        if (rightGripperBody != null)
        {
            currentRightPosition = rightGripperBody.jointPosition[0];
        }
        
        // グリッパーの閉状態を判定
        isGripperClosed = IsGripperInClosedState();
    }
    
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
        else
        {
            return target != null ? target.transform.position : Vector3.zero;
        }
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
        
        return Vector3.up; // デフォルト
    }
    
    private void TransferForceToTarget()
    {
        if (target == null || simpleGripperController == null) return;
        
        bool canTransferForce = isGripperClosed && HasValidContact();
        
        if (!canTransferForce)
        {
            hasLoggedForceTransfer = false; // リセット
            return;
        }
        
        float currentForce = simpleGripperController.GetCurrentTargetForce();
        
        if (currentForce >= contactForceThreshold)
        {
            Vector3 contactPoint = CalculateContactPoint();
            lastContactNormal = CalculateAggregateContactNormal();
            
            // 力伝達開始時に一度だけ詳細ログを出力
            if (!hasLoggedForceTransfer)
            {
                Debug.Log($"=== FORCE TRANSFER STARTED ===");
                Debug.Log($"Using contact normal: {lastContactNormal}");
                Debug.Log($"Left normal: {leftContactNormal}");
                Debug.Log($"Right normal: {rightContactNormal}");
                Debug.Log($"Aggregate X component: {Mathf.Abs(Vector3.Dot(lastContactNormal, Vector3.right)):F3}");
                Debug.Log($"Aggregate Y component: {Mathf.Abs(Vector3.Dot(lastContactNormal, Vector3.up)):F3}");
                Debug.Log($"Aggregate Z component: {Mathf.Abs(Vector3.Dot(lastContactNormal, Vector3.forward)):F3}");
                Debug.Log($"Contact point: {contactPoint}");
                Debug.Log($"Force: {currentForce:F2}N");
                hasLoggedForceTransfer = true;
            }
            
            // 方向を考慮した力伝達
            target.ApplyGripperForceWithDirection(currentForce, contactPoint, lastContactNormal);
        }
        else
        {
            hasLoggedForceTransfer = false; // 力が閾値以下になったらリセット
        }
    }
    
    public void OnGripperContactEnter(Collider collider, bool isLeftGripper, Vector3 contactPoint, Vector3 contactNormal)
    {
        if (collider.gameObject != target.gameObject) return;
        
        if (isLeftGripper)
        {
            leftGripperInContact = true;
            leftContactPoint = contactPoint;
            leftContactNormal = contactNormal;
        }
        else
        {
            rightGripperInContact = true;
            rightContactPoint = contactPoint;
            rightContactNormal = contactNormal;
        }
        
        if (enableDetailedLogging)
        {
            Debug.Log($"{(isLeftGripper ? "Left" : "Right")} gripper contact ENTER");
            Debug.Log($"Contact point: {contactPoint}");
            Debug.Log($"Contact normal: {contactNormal}");
            Debug.Log($"Contact state - Left: {leftGripperInContact}, Right: {rightGripperInContact}");
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
            Debug.Log($"Contact state - Left: {leftGripperInContact}, Right: {rightGripperInContact}");
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
        
        // デバッグ情報を詳細に出力
        if (enableDetailedLogging)
        {
            Debug.Log($"=== 詳細評価ログ ===");
            Debug.Log($"Left gripper in contact: {leftGripperInContact}");
            Debug.Log($"Right gripper in contact: {rightGripperInContact}");
            Debug.Log($"Require both grippers: {requireBothGrippersContact}");
            Debug.Log($"Is gripper closed: {isGripperClosed}");
            Debug.Log($"Left position: {currentLeftPosition:F4}, threshold: {-gripperCloseThreshold}");
            Debug.Log($"Right position: {currentRightPosition:F4}, threshold: {gripperCloseThreshold}");
        }
        
        // ターゲットの現在状態を取得
        ObjectState objectState = target.GetCurrentState();
        
        // グリッパーの状態を取得
        GraspingState graspingState = simpleGripperController.GetGraspingStateForInterface();
        
        // 接触の有効性をチェック
        bool hasValidContact = HasValidContact();
        bool isGripping = isGripperClosed && hasValidContact;
        
        // 評価ロジック
        GraspResult result = DetermineGraspResult(objectState, graspingState, isGripping);
        
        // 評価結果を作成（BasicTypes.csのGraspEvaluationクラスを使用）
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
            Debug.Log($"Grasp Evaluation: {result}, Force: {objectState.appliedForce:F2}N, " +
                    $"Deformation: {objectState.deformation:F3}, Confidence: {evaluation.confidence:F2}");
        }
        
        return evaluation;
    }
    
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
    private float CalculateConfidence(ObjectState objectState, GraspingState graspingState, bool hasValidContact, bool isGripping)
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
        
        // 接触点を表示
        if (leftGripperInContact)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawSphere(leftContactPoint, 0.005f);
        }
        
        if (rightGripperInContact)
        {
            Gizmos.color = Color.yellow;
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
}

// SimpleContactDetector クラス
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
        Vector3 contactNormal = (other.transform.position - transform.position).normalized;
        
        // デバッグログを削除し、直接コールバック
        parentInterface.OnGripperContactEnter(other, isLeftGripper, contactPoint, contactNormal);
    }
    
    void OnTriggerExit(Collider other)
    {
        if (parentInterface == null) return;
        parentInterface.OnGripperContactExit(other, isLeftGripper);
    }
}