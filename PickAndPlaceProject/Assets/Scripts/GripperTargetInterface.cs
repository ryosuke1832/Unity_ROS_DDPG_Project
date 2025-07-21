using UnityEngine;

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
        if (leftGripperBody != null && rightGripperBody != null)
        {
            // グリッパーの実際の位置を取得
            currentLeftPosition = leftGripperBody.xDrive.target;
            currentRightPosition = rightGripperBody.xDrive.target;
            
            // 実際のjoint位置も取得
            float leftJointPosition = leftGripperBody.jointPosition[0];
            float rightJointPosition = rightGripperBody.jointPosition[0];
            
            // デバッグ: グリッパーの詳細状態
            // Debug.Log($"=== グリッパー状態詳細 ===");
            // Debug.Log($"Left - Target: {currentLeftPosition:F4}, Actual: {leftJointPosition:F4}");
            // Debug.Log($"Right - Target: {currentRightPosition:F4}, Actual: {rightJointPosition:F4}");
            // Debug.Log($"Left Threshold: {gripperCloseThreshold:F4}");
            // Debug.Log($"Right Threshold: {-gripperCloseThreshold:F4}");
            
            bool leftClosed = currentLeftPosition <= -gripperCloseThreshold;  // -0.01 <= -0.001 = True
            bool rightClosed = currentRightPosition >= gripperCloseThreshold; // +0.01 >= +0.001 = True
            
            // Debug.Log($"Left Closed Check: {leftClosed} ({currentLeftPosition:F4} >= {gripperCloseThreshold:F4})");
            // Debug.Log($"Right Closed Check: {rightClosed} ({currentRightPosition:F4} <= {-gripperCloseThreshold:F4})");
            
            isGripperClosed = leftClosed && rightClosed;
            // Debug.Log($"Overall Gripper Closed: {isGripperClosed}");
        }
        
        // 改良された接触検出
        // CheckColliderContactImproved();
    }


    private void CheckColliderContactImproved()
    {
        if (target == null) return;
        
        float contactDistanceThreshold = 0.5f; // 12cm以内
        
        Debug.Log($"=== 接触検出デバッグ ===");
        Debug.Log($"Contact Distance Threshold: {contactDistanceThreshold:F4}");
        
        // 左グリッパーの接触検出
        if (leftGripperTip != null)
        {
            float leftDistance = Vector3.Distance(leftGripperTip.position, target.transform.position);
            Debug.Log($"Left Gripper Distance to Target: {leftDistance:F4}");
            
            bool leftDistanceContact = leftDistance <= contactDistanceThreshold;
            Debug.Log($"Left Distance Contact: {leftDistanceContact} ({leftDistance:F4} <= {contactDistanceThreshold:F4})");
            
            leftGripperInContact = leftDistanceContact;
            
            if (leftGripperInContact)
            {
                leftContactPoint = target.transform.position;
            }
        }
        
        // 右グリッパーの接触検出
        if (rightGripperTip != null)
        {
            float rightDistance = Vector3.Distance(rightGripperTip.position, target.transform.position);
            Debug.Log($"Right Gripper Distance to Target: {rightDistance:F4}");
            
            bool rightDistanceContact = rightDistance <= contactDistanceThreshold;
            Debug.Log($"Right Distance Contact: {rightDistanceContact} ({rightDistance:F4} <= {contactDistanceThreshold:F4})");
            
            rightGripperInContact = rightDistanceContact;
            
            if (rightGripperInContact)
            {
                rightContactPoint = target.transform.position;
            }
        }
        
        // *** 正しい方法: グリッパー間のベクトルから力方向を計算 ***
        if (leftGripperInContact || rightGripperInContact)
        {
            CalculateGripperForceDirection();
        }
        
        // Debug.Log($"Contact Status - Left: {leftGripperInContact}, Right: {rightGripperInContact}");
        // Debug.Log($"Valid Contact: {HasValidContact()}");
        // Debug.Log($"Can Transfer Force: {isGripperClosed && HasValidContact()}");
    }

    // *** 新しいメソッド: グリッパー間の幾何学的関係から力方向を計算 ***
    private void CalculateGripperForceDirection()
    {
        if (leftGripperTip == null || rightGripperTip == null) return;
        
        // グリッパー間のベクトルを計算
        Vector3 gripperVector = rightGripperTip.position - leftGripperTip.position;
        Vector3 gripperDirection = gripperVector.normalized;
        
        // Debug.Log($"=== グリッパー幾何学計算 ===");
        // Debug.Log($"Left Gripper Position: {leftGripperTip.position}");
        // Debug.Log($"Right Gripper Position: {rightGripperTip.position}");
        // Debug.Log($"Gripper Vector: {gripperVector}");
        // Debug.Log($"Gripper Direction: {gripperDirection}");
        // Debug.Log($"Gripper Distance: {gripperVector.magnitude:F4}");
        
        // 左右のグリッパーから中心への力方向を設定
        // 左グリッパー → 右方向への力
        leftContactNormal = gripperDirection; // 左から右方向
        
        // 右グリッパー → 左方向への力  
        rightContactNormal = -gripperDirection; // 右から左方向
        
        // Debug.Log($"Left Contact Normal (L→R): {leftContactNormal}");
        // Debug.Log($"Left Normal Components - X: {leftContactNormal.x:F3}, Y: {leftContactNormal.y:F3}, Z: {leftContactNormal.z:F3}");
        
        // Debug.Log($"Right Contact Normal (R→L): {rightContactNormal}");
        // Debug.Log($"Right Normal Components - X: {rightContactNormal.x:F3}, Y: {rightContactNormal.y:F3}, Z: {rightContactNormal.z:F3}");
        
        // グリッパー間ベクトルの成分分析
        float xComponent = Mathf.Abs(gripperDirection.x);
        float yComponent = Mathf.Abs(gripperDirection.y);
        float zComponent = Mathf.Abs(gripperDirection.z);
        
        // Debug.Log($"グリッパー方向成分 - X: {xComponent:F3}, Y: {yComponent:F3}, Z: {zComponent:F3}");
        
        // string primaryDirection = "";
        // if (xComponent > yComponent && xComponent > zComponent)
        //     primaryDirection = "X軸（左右）";
        // else if (yComponent > zComponent)
        //     primaryDirection = "Y軸（上下）";
        // else
        //     primaryDirection = "Z軸（前後）";
        
        // Debug.Log($"主要把持方向: {primaryDirection}");
    }

    // *** 修正された統合法線計算 ***
    private Vector3 CalculateAggregateContactNormal()
    {
        Vector3 aggregateNormal = Vector3.zero;
        int contactCount = 0;
        
        if (leftGripperInContact)
        {
            aggregateNormal += leftContactNormal;
            contactCount++;
            Debug.Log($"Adding left normal: {leftContactNormal}");
        }
        
        if (rightGripperInContact)
        {
            aggregateNormal += rightContactNormal;
            contactCount++;
            Debug.Log($"Adding right normal: {rightContactNormal}");
        }
        
        if (contactCount > 0)
        {
            aggregateNormal /= contactCount;
            
            // *** 重要: 左右からの力なので統合すると相殺される（ゼロベクトルに近くなる） ***
            // これは物理的に正しい - 両方向から等しい力が加わる
            Debug.Log($"Raw aggregate normal: {aggregateNormal}");
            Debug.Log($"Aggregate magnitude: {aggregateNormal.magnitude:F3}");
            
            // 統合法線が小さすぎる場合は、グリッパー方向を使用
            if (aggregateNormal.magnitude < 0.1f && leftGripperTip != null && rightGripperTip != null)
            {
                Vector3 gripperDirection = (rightGripperTip.position - leftGripperTip.position).normalized;
                aggregateNormal = gripperDirection;
                Debug.Log($"Using gripper direction as aggregate normal: {aggregateNormal}");
            }
            else
            {
                aggregateNormal = aggregateNormal.normalized;
            }
            
            Debug.Log($"Final aggregate normal: {aggregateNormal}");
            Debug.Log($"Final components - X: {aggregateNormal.x:F3}, Y: {aggregateNormal.y:F3}, Z: {aggregateNormal.z:F3}");
            
            return aggregateNormal;
        }
        
        // フォールバック: グリッパーが存在する場合はグリッパー方向を使用
        if (leftGripperTip != null && rightGripperTip != null)
        {
            Vector3 gripperDirection = (rightGripperTip.position - leftGripperTip.position).normalized;
            Debug.Log($"Using fallback gripper direction: {gripperDirection}");
            return gripperDirection;
        }
        
        // 最終フォールバック
        Vector3 defaultNormal = Vector3.right;
        Debug.Log($"Using final fallback normal: {defaultNormal}");
        return defaultNormal;
    }

    // *** デバッグ用: グリッパー幾何学の可視化 ***
    [ContextMenu("Debug Gripper Geometry")]
    public void DebugGripperGeometry()
    {
        if (leftGripperTip == null || rightGripperTip == null)
        {
            Debug.LogError("グリッパーのTransformが設定されていません");
            return;
        }
        
        Debug.Log("=== グリッパー幾何学分析 ===");
        
        Vector3 leftPos = leftGripperTip.position;
        Vector3 rightPos = rightGripperTip.position;
        Vector3 centerPos = (leftPos + rightPos) / 2f;
        Vector3 gripperVector = rightPos - leftPos;
        
        Debug.Log($"左グリッパー位置: {leftPos}");
        Debug.Log($"右グリッパー位置: {rightPos}");
        Debug.Log($"グリッパー中心: {centerPos}");
        Debug.Log($"グリッパー間距離: {gripperVector.magnitude:F4}");
        Debug.Log($"グリッパー方向: {gripperVector.normalized}");
        
        if (target != null)
        {
            Vector3 targetPos = target.transform.position;
            Debug.Log($"ターゲット位置: {targetPos}");
            Debug.Log($"中心からターゲットまでの距離: {Vector3.Distance(centerPos, targetPos):F4}");
            
            // ターゲットがグリッパー間の軸上にあるかチェック
            Vector3 toTarget = targetPos - centerPos;
            float dotProduct = Vector3.Dot(toTarget.normalized, gripperVector.normalized);
            Debug.Log($"ターゲットの軸上配置度: {Mathf.Abs(dotProduct):F3} (1.0に近いほど軸上)");
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
        else
        {
            return target != null ? target.transform.position : Vector3.zero;
        }
    }
    
    // private Vector3 CalculateAggregateContactNormal()
    // {
    //     Vector3 aggregateNormal = Vector3.zero;
    //     int contactCount = 0;
        
    //     if (leftGripperInContact)
    //     {
    //         aggregateNormal += leftContactNormal;
    //         contactCount++;
    //     }
        
    //     if (rightGripperInContact)
    //     {
    //         aggregateNormal += rightContactNormal;
    //         contactCount++;
    //     }
        
    //     if (contactCount > 0)
    //     {
    //         aggregateNormal /= contactCount;
    //         return aggregateNormal.normalized;
    //     }
        
    //     return Vector3.up; // デフォルト
    // }
    
    private void TransferForceToTarget()
    {
        if (target == null || simpleGripperController == null) return;
        
        bool canTransferForce = isGripperClosed && HasValidContact();
        
        // 詳細な状態デバッグ
        // Debug.Log($"=== 力伝達判定 ===");
        // Debug.Log($"グリッパー閉じ状態: {isGripperClosed}");
        // Debug.Log($"有効な接触: {HasValidContact()}");
        // Debug.Log($"力伝達可能: {canTransferForce}");
        
        if (!canTransferForce)
        {
            hasLoggedForceTransfer = false;
            return;
        }
        
        float currentForce = simpleGripperController.GetCurrentTargetForce();
        // Debug.Log($"現在の目標力: {currentForce:F2}N, 閾値: {contactForceThreshold:F2}N");
        
        if (currentForce >= contactForceThreshold)
        {
            Vector3 contactPoint = CalculateContactPoint();
            lastContactNormal = CalculateAggregateContactNormal();
            
            // 力伝達開始時に一度だけ詳細ログを出力
            if (!hasLoggedForceTransfer)
            {
                Debug.Log($"=== 力伝達開始 ===");
                Debug.Log($"接触点: {contactPoint}");
                Debug.Log($"接触法線: {lastContactNormal}");
                Debug.Log($"伝達力: {currentForce:F2}N");
                hasLoggedForceTransfer = true;
            }
            
            // *** 修正 - voidメソッドを呼び出し ***
            target.ApplyGripperForceWithDirection(currentForce, contactPoint, lastContactNormal);
            Debug.Log($"力を正常に適用: {currentForce:F2}N");
        }
        else
        {
            hasLoggedForceTransfer = false;
            Debug.Log($"力が閾値以下: {currentForce:F2}N < {contactForceThreshold:F2}N");
        }
    }

    // *** 修正 - SimpleGripForceControllerの実際のメソッドを使用 ***
    [ContextMenu("Check Force Controller State")]
    public void CheckForceControllerState()
    {
        if (simpleGripperController == null)
        {
            Debug.LogError("SimpleGripForceController が見つかりません");
            return;
        }
        
        Debug.Log($"=== 力制御状態 ===");
        Debug.Log($"有効状態: {simpleGripperController.enabled}");
        Debug.Log($"現在の目標力: {simpleGripperController.GetCurrentTargetForce():F2}N");
        
        // 力制御の有効化 - SetForceControlEnabledメソッドを使用
        if (!simpleGripperController.enabled)
        {
            Debug.Log("力制御を手動で有効化します");
            simpleGripperController.SetForceControlEnabled(true);
        }
    }

    // デバッグ用のターゲット状態確認メソッド
    [ContextMenu("Check Target State")]
    public void CheckTargetState()
    {
        if (target == null)
        {
            Debug.LogError("DeformableTarget が見つかりません");
            return;
        }
        
        var state = target.GetCurrentState();
        Debug.Log($"=== ターゲット状態 ===");
        Debug.Log($"変形量: {state.deformation:F3}");
        Debug.Log($"適用力: {state.appliedForce:F2}N");
        Debug.Log($"破損状態: {state.isBroken}");
        Debug.Log($"材質タイプ: {state.materialType}");
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
            Debug.Log($"Left position: {currentLeftPosition:F4}, threshold: {gripperCloseThreshold}");
            Debug.Log($"Right position: {currentRightPosition:F4}, threshold: {-gripperCloseThreshold}");
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