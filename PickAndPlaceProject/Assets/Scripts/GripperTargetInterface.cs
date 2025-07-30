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
    public float gripperCloseThreshold = 0.015f;
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
        // SetupGripperColliders();
        
        Debug.Log($"GripperTargetInterface initialized with AluminumCan: {(target != null ? "✅" : "❌")}");
    }
    
    void FixedUpdate()
    {
        UpdateGripperState();
        LogContactState();
        TransferForceToTarget();
    }
    
        // GripperTargetInterface.cs の TransferForceToTarget() メソッドに追加
        private void TransferForceToTarget()
        {
            if (target == null || simpleGripperController == null)
            {
                Debug.LogWarning("⚠️ Target または GripperController が null です");
                return;
            }
            
            // 基本状態のログ
            bool canTransferForce = isGripperClosed && HasValidContact();
            float currentForce = simpleGripperController.GetCurrentTargetForce();
            
            if (enableDetailedLogging)
            {
                // Debug.Log($"[状態チェック] グリッパー閉じ: {isGripperClosed}, " +
                //         $"有効接触: {HasValidContact()}, " +
                //         $"力伝達可能: {canTransferForce}, " +
                //         $"現在力: {currentForce:F2}N");
            }
            
            if (!canTransferForce)
            {
                if (!hasLoggedForceTransfer)
                {
                    // Debug.LogWarning($"❌ 力伝達不可 - グリッパー閉じ: {isGripperClosed}, 接触: {HasValidContact()}");
                }
                return;
            }
            
            // 力伝達の実行とログ
            Vector3 contactPoint = CalculateContactPoint();
            target.ApplyGripperForceWithDirection(currentForce, contactPoint, lastContactNormal);
            
            if (enableDetailedLogging)
            {
                // Debug.Log($"✅ 力伝達実行: {currentForce:F2}N → 接触点: {contactPoint}");
            }
        }
    
        
    // GripperTargetInterface.cs の AutoFindGrippers() メソッドを診断機能付きで拡張

    private void AutoFindGrippers()
    {
        Debug.Log("🔍 グリッパー自動検索開始...");
        
        // すべてのArticulationBodyを検索
        ArticulationBody[] allBodies = FindObjectsOfType<ArticulationBody>();
        Debug.Log($"見つかったArticulationBody数: {allBodies.Length}");
        
        foreach (var body in allBodies)
        {
            Debug.Log($"  - {body.name} (親: {(body.transform.parent ? body.transform.parent.name : "なし")})");
            
            if (body.name.ToLower().Contains("left") && body.name.ToLower().Contains("gripper"))
            {
                leftGripperBody = body;
                Debug.Log($"✅ 左グリッパー発見: {body.name}");
            }
            else if (body.name.ToLower().Contains("right") && body.name.ToLower().Contains("gripper"))
            {
                rightGripperBody = body;
                Debug.Log($"✅ 右グリッパー発見: {body.name}");
            }
        }
        
        // Transform検索（Tipを探す）
        Transform[] allTransforms = FindObjectsOfType<Transform>();
        Debug.Log($"見つかったTransform数: {allTransforms.Length}");
        
        foreach (var trans in allTransforms)
        {
            if ((trans.name.ToLower().Contains("left") && trans.name.ToLower().Contains("gripper")) ||
                trans.name.ToLower().Contains("left_gripper"))
            {
                if (trans.name.ToLower().Contains("tip") || trans.name.ToLower().Contains("finger"))
                {
                    leftGripperTip = trans;
                    Debug.Log($"✅ 左グリッパーTip発見: {trans.name}");
                }
            }
            else if ((trans.name.ToLower().Contains("right") && trans.name.ToLower().Contains("gripper")) ||
                    trans.name.ToLower().Contains("right_gripper"))
            {
                if (trans.name.ToLower().Contains("tip") || trans.name.ToLower().Contains("finger"))
                {
                    rightGripperTip = trans;
                    Debug.Log($"✅ 右グリッパーTip発見: {trans.name}");
                }
            }
        }
        
        // 結果の確認
        Debug.Log("=== グリッパー検索結果 ===");
        Debug.Log($"左グリッパーBody: {(leftGripperBody != null ? leftGripperBody.name : "❌ 未発見")}");
        Debug.Log($"右グリッパーBody: {(rightGripperBody != null ? rightGripperBody.name : "❌ 未発見")}");
        Debug.Log($"左グリッパーTip: {(leftGripperTip != null ? leftGripperTip.name : "❌ 未発見")}");
        Debug.Log($"右グリッパーTip: {(rightGripperTip != null ? rightGripperTip.name : "❌ 未発見")}");
        
        // 手動で設定が必要かもしれない場合の案内
        if (leftGripperBody == null || rightGripperBody == null || leftGripperTip == null || rightGripperTip == null)
        {
            Debug.LogWarning("⚠️ 一部のグリッパーコンポーネントが見つかりませんでした。");
            Debug.LogWarning("Inspectorで手動設定してください。");
            
            // 参考のため、gripper関連の全オブジェクト名をリストアップ
            Debug.Log("=== Gripper関連オブジェクト一覧 ===");
            foreach (var trans in allTransforms)
            {
                if (trans.name.ToLower().Contains("gripper") || 
                    trans.name.ToLower().Contains("finger") ||
                    trans.name.ToLower().Contains("tip"))
                {
                    Debug.Log($"  候補: {trans.name} (親: {(trans.parent ? trans.parent.name : "なし")})");
                }
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
            existingCollider.isTrigger = false;
        }
        
        // 距離ベースの接触検出を使用（SimpleContactDetectorは使わない）
        // Debug.Log($"Setup gripper collider for {(isLeft ? "left" : "right")} gripper: {gripperObj.name}");
    }
    
    // // UpdateGripperState() メソッドに追加
    // private void UpdateGripperState()
    // {
    //     // 現在の位置を取得
    //     if (leftGripperBody != null)
    //         currentLeftPosition = leftGripperBody.jointPosition[0];
    //     if (rightGripperBody != null)
    //         currentRightPosition = rightGripperBody.jointPosition[0];

    //     float gripperDistance = Mathf.Abs(currentLeftPosition - currentRightPosition);
    //     bool wasGripperClosed = isGripperClosed;
    //     isGripperClosed = gripperDistance < gripperCloseThreshold;

    //     // 詳細な診断ログを追加
    //     if (enableDetailedLogging)
    //     {
    //         Debug.Log($"🔍 グリッパー診断:");
    //         Debug.Log($"  左位置: {currentLeftPosition:F4}");
    //         Debug.Log($"  右位置: {currentRightPosition:F4}");
    //         Debug.Log($"  距離: {gripperDistance:F4}");
    //         Debug.Log($"  閉じ閾値: {gripperCloseThreshold:F4}");
    //         Debug.Log($"  閉じ状態: {isGripperClosed}");
    //         Debug.Log($"  左グリッパーBody: {(leftGripperBody != null ? "✅" : "❌")}");
    //         Debug.Log($"  右グリッパーBody: {(rightGripperBody != null ? "✅" : "❌")}");
            
    //         // ArticulationBodyの詳細状態
    //         if (leftGripperBody != null)
    //         {
    //             var leftDrive = leftGripperBody.xDrive;
    //             Debug.Log($"  左ドライブ - Target: {leftDrive.target:F4}, Force: {leftDrive.forceLimit:F2}");
    //         }
            
    //         if (rightGripperBody != null)
    //         {
    //             var rightDrive = rightGripperBody.xDrive;
    //             Debug.Log($"  右ドライブ - Target: {rightDrive.target:F4}, Force: {rightDrive.forceLimit:F2}");
    //         }
    //     }

    //     // 状態変化時のログ
    //     if (wasGripperClosed != isGripperClosed)
    //     {
    //         Debug.Log($"🔄 グリッパー状態変化: {(isGripperClosed ? "閉じた" : "開いた")} " +
    //                 $"(距離: {gripperDistance:F4}, 閾値: {gripperCloseThreshold:F4})");
    //     }
    // }

    // 接触検出の詳細ログも追加
    private void LogContactState()
    {
        if (enableDetailedLogging && Time.time % 0.5f < Time.deltaTime)
        {
            Debug.Log($"📍 接触状態診断:");
            Debug.Log($"  左グリッパー接触: {leftGripperInContact}");
            Debug.Log($"  右グリッパー接触: {rightGripperInContact}");
            Debug.Log($"  両方接触必要: {requireBothGrippersContact}");
            Debug.Log($"  有効接触: {HasValidContact()}");
            Debug.Log($"  左接触点: {leftContactPoint}");
            Debug.Log($"  右接触点: {rightContactPoint}");
            
            // アルミ缶の位置も確認
            if (target != null)
            {
                Debug.Log($"  アルミ缶位置: {target.transform.position}");
                Debug.Log($"  左グリッパーとの距離: {Vector3.Distance(leftGripperTip.position, target.transform.position):F3}");
                Debug.Log($"  右グリッパーとの距離: {Vector3.Distance(rightGripperTip.position, target.transform.position):F3}");
            }
        }
    }

        // OnTriggerEnter と OnTriggerExit にログを追加
    void OnTriggerEnter(Collider other)
    {
        if (other.CompareTag("AluminumCan"))
        {
            bool wasLeftContact = leftGripperInContact;
            bool wasRightContact = rightGripperInContact;
            
            if (transform.name.Contains("left"))
            {
                leftGripperInContact = true;
                leftContactPoint = other.ClosestPoint(transform.position);
            }
            else if (transform.name.Contains("right"))
            {
                rightGripperInContact = true;
                rightContactPoint = other.ClosestPoint(transform.position);
            }
            
            if (enableDetailedLogging)
            {
                // Debug.Log($"📍 接触検出: {transform.name} → {other.name} " +
                //         $"at {(transform.name.Contains("left") ? leftContactPoint : rightContactPoint)}");
            }
        }
    }
    // GripperTargetInterface.cs の UpdateDistanceBasedContact() を修正

private void UpdateDistanceBasedContact()
{
    if (target == null || leftGripperTip == null || rightGripperTip == null) return;
    
    // アルミ缶のサイズを考慮した接触判定
    float canRadius = 0.03f; // アルミ缶の半径（推定3cm）
    float contactThreshold = 0.04f; // 接触とみなす距離（2cm）
    
    // 左グリッパーの接触チェック
    Vector3 leftToCanCenter = target.transform.position - leftGripperTip.position;
    float leftDistanceToCenter = leftToCanCenter.magnitude;
    
    // アルミ缶表面までの距離を計算
    float leftDistanceToSurface = leftDistanceToCenter - canRadius;
    bool wasLeftInContact = leftGripperInContact;
    leftGripperInContact = leftDistanceToSurface <= contactThreshold;
    
    // 右グリッパーの接触チェック  
    Vector3 rightToCanCenter = target.transform.position - rightGripperTip.position;
    float rightDistanceToCenter = rightToCanCenter.magnitude;
    
    // アルミ缶表面までの距離を計算
    float rightDistanceToSurface = rightDistanceToCenter - canRadius;
    bool wasRightInContact = rightGripperInContact;
    rightGripperInContact = rightDistanceToSurface <= contactThreshold;
    
    // 詳細ログ（デバッグ用）
    if (enableDetailedLogging)
    {
        // Debug.Log($"🔍 正しい接触検出:");
        // Debug.Log($"  缶半径: {canRadius:F3}m, 接触閾値: {contactThreshold:F3}m");
        // Debug.Log($"  左グリッパー → 缶中心: {leftDistanceToCenter:F3}m");
        // Debug.Log($"  左グリッパー → 缶表面: {leftDistanceToSurface:F3}m → 接触: {leftGripperInContact}");
        // Debug.Log($"  右グリッパー → 缶中心: {rightDistanceToCenter:F3}m");
        // Debug.Log($"  右グリッパー → 缶表面: {rightDistanceToSurface:F3}m → 接触: {rightGripperInContact}");
        // Debug.Log($"  有効接触: {HasValidContact()}");
    }
    
    // 接触点と法線の更新
    if (leftGripperInContact)
    {
        // 缶表面上の接触点を計算
        leftContactPoint = target.transform.position - leftToCanCenter.normalized * canRadius;
        leftContactNormal = leftToCanCenter.normalized;
    }
    
    if (rightGripperInContact)
    {
        // 缶表面上の接触点を計算
        rightContactPoint = target.transform.position - rightToCanCenter.normalized * canRadius;
        rightContactNormal = rightToCanCenter.normalized;
    }
    
    // 接触状態の変化をログ
    if (leftGripperInContact != wasLeftInContact)
    {
        // Debug.Log($"🔄 左グリッパー接触変化: {leftGripperInContact} (表面距離: {leftDistanceToSurface:F3}m)");
    }
    
    if (rightGripperInContact != wasRightInContact)
    {
        // Debug.Log($"🔄 右グリッパー接触変化: {rightGripperInContact} (表面距離: {rightDistanceToSurface:F3}m)");
    }
}

// より高度な接触検出（オプション）
private void UpdateAdvancedContactDetection()
{
    if (target == null || leftGripperTip == null || rightGripperTip == null) return;
    
    // アルミ缶のColliderを使用した正確な接触判定
    Collider canCollider = target.GetComponent<Collider>();
    if (canCollider == null) return;
    
    float contactThreshold = 0.04f; // 2cm以内で接触
    
    // 左グリッパーの最近点を取得
    Vector3 leftClosestPoint = canCollider.ClosestPoint(leftGripperTip.position);
    float leftDistanceToSurface = Vector3.Distance(leftGripperTip.position, leftClosestPoint);
    bool wasLeftInContact = leftGripperInContact;
    leftGripperInContact = leftDistanceToSurface <= contactThreshold;
    
    // 右グリッパーの最近点を取得
    Vector3 rightClosestPoint = canCollider.ClosestPoint(rightGripperTip.position);
    float rightDistanceToSurface = Vector3.Distance(rightGripperTip.position, rightClosestPoint);
    bool wasRightInContact = rightGripperInContact;
    rightGripperInContact = rightDistanceToSurface <= contactThreshold;
    
    // 詳細ログ
    if (enableDetailedLogging)
    {
        // Debug.Log($"🎯 高精度接触検出:");
        // Debug.Log($"  左最近点: {leftClosestPoint}");
        // Debug.Log($"  左表面距離: {leftDistanceToSurface:F3}m → 接触: {leftGripperInContact}");
        // Debug.Log($"  右最近点: {rightClosestPoint}");
        // Debug.Log($"  右表面距離: {rightDistanceToSurface:F3}m → 接触: {rightGripperInContact}");
    }
    
    // 接触点と法線の更新
    if (leftGripperInContact)
    {
        leftContactPoint = leftClosestPoint;
        leftContactNormal = (leftGripperTip.position - leftClosestPoint).normalized;
    }
    
    if (rightGripperInContact)
    {
        rightContactPoint = rightClosestPoint;
        rightContactNormal = (rightGripperTip.position - rightClosestPoint).normalized;
    }
    
    // 変化ログ
    if (leftGripperInContact != wasLeftInContact)
    {
        Debug.Log($"🔄 左グリッパー接触変化(高精度): {leftGripperInContact}");
    }
    
    if (rightGripperInContact != wasRightInContact)
    {
        Debug.Log($"🔄 右グリッパー接触変化(高精度): {rightGripperInContact}");
    }
}

// UpdateGripperState()で呼び出す部分を変更
private void UpdateGripperState()
{
    // グリッパー閉じ判定（既存コード）
    if (leftGripperBody != null && rightGripperBody != null)
    {
        currentLeftPosition = leftGripperBody.jointPosition[0];
        currentRightPosition = rightGripperBody.jointPosition[0];
        
        float gripperDistance = Mathf.Abs(currentLeftPosition - currentRightPosition);
        bool wasGripperClosed = isGripperClosed;
        isGripperClosed = gripperDistance < gripperCloseThreshold;
        
        if (enableDetailedLogging && wasGripperClosed != isGripperClosed)
        {
            Debug.Log($"🔄 グリッパー状態変化: {(isGripperClosed ? "閉じた" : "開いた")}");
        }
    }
    
    // 新しい接触検出を使用
    UpdateAdvancedContactDetection(); // または UpdateDistanceBasedContact();
    
    if (leftGripperInContact || rightGripperInContact)
    {
        CalculateGripperForceDirection();
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

    // GripperTargetInterface.cs に完全な診断メソッドを追加

/// <summary>
/// 完全な状態診断（コンテキストメニューから実行）
/// </summary>
[ContextMenu("完全診断実行")]
public void CompleteSystemDiagnosis()
{
    Debug.Log("=== 🔍 完全システム診断開始 ===");
    
    // 1. コンポーネント存在確認
    Debug.Log("--- 1. コンポーネント確認 ---");
    Debug.Log($"SimpleGripperController: {(simpleGripperController != null ? "✅" : "❌")}");
    Debug.Log($"Target (AluminumCan): {(target != null ? "✅" : "❌")}");
    Debug.Log($"LeftGripperTip: {(leftGripperTip != null ? "✅" : "❌")}");
    Debug.Log($"RightGripperTip: {(rightGripperTip != null ? "✅" : "❌")}");
    Debug.Log($"LeftGripperBody: {(leftGripperBody != null ? "✅" : "❌")}");
    Debug.Log($"RightGripperBody: {(rightGripperBody != null ? "✅" : "❌")}");
    
    // 2. 位置情報の詳細確認
    Debug.Log("--- 2. 位置情報 ---");
    if (leftGripperTip != null)
        Debug.Log($"左グリッパー位置: {leftGripperTip.position}");
    if (rightGripperTip != null)
        Debug.Log($"右グリッパー位置: {rightGripperTip.position}");
    if (target != null)
        Debug.Log($"アルミ缶位置: {target.transform.position}");
    
    // 3. グリッパー間の距離
    if (leftGripperTip != null && rightGripperTip != null)
    {
        float gripperDistance = Vector3.Distance(leftGripperTip.position, rightGripperTip.position);
        Debug.Log($"グリッパー間距離: {gripperDistance:F3}m");
        
        Vector3 centerPoint = (leftGripperTip.position + rightGripperTip.position) * 0.5f;
        Debug.Log($"グリッパー中心点: {centerPoint}");
    }
    
    // 4. コライダー情報の詳細確認
    Debug.Log("--- 3. コライダー情報 ---");
    DiagnoseColliders();
    
    // 5. ArticulationBody の状態
    Debug.Log("--- 4. ArticulationBody 状態 ---");
    DiagnoseArticulationBodies();
    
    // 6. 接触計算の詳細
    Debug.Log("--- 5. 接触計算詳細 ---");
    DiagnoseContactCalculation();
}

private void DiagnoseColliders()
{
    // アルミ缶のコライダー
    if (target != null)
    {
        Collider canCollider = target.GetComponent<Collider>();
        if (canCollider != null)
        {
            Debug.Log($"アルミ缶Collider: {canCollider.GetType().Name}");
            Debug.Log($"  IsTrigger: {canCollider.isTrigger}");
            Debug.Log($"  Enabled: {canCollider.enabled}");
            
            if (canCollider is BoxCollider boxCol)
            {
                Debug.Log($"  BoxCollider Size: {boxCol.size}");
                Debug.Log($"  BoxCollider Center: {boxCol.center}");
                Debug.Log($"  World Bounds: {boxCol.bounds}");
            }
        }
        else
        {
            Debug.LogError("❌ アルミ缶にColliderがありません！");
        }
    }
    
    // グリッパーのコライダー
    if (leftGripperTip != null)
    {
        Collider[] leftColliders = leftGripperTip.GetComponents<Collider>();
        Debug.Log($"左グリッパーCollider数: {leftColliders.Length}");
        
        for (int i = 0; i < leftColliders.Length; i++)
        {
            var col = leftColliders[i];
            Debug.Log($"  [{i}] {col.GetType().Name}: IsTrigger={col.isTrigger}, Enabled={col.enabled}");
        }
    }
    
    if (rightGripperTip != null)
    {
        Collider[] rightColliders = rightGripperTip.GetComponents<Collider>();
        Debug.Log($"右グリッパーCollider数: {rightColliders.Length}");
        
        for (int i = 0; i < rightColliders.Length; i++)
        {
            var col = rightColliders[i];
            Debug.Log($"  [{i}] {col.GetType().Name}: IsTrigger={col.isTrigger}, Enabled={col.enabled}");
        }
    }
}

private void DiagnoseArticulationBodies()
{
    if (leftGripperBody != null)
    {
        Debug.Log($"左ArticulationBody: {leftGripperBody.name}");
        Debug.Log($"  Position: {leftGripperBody.transform.position}");
        Debug.Log($"  JointPosition: {leftGripperBody.jointPosition[0]:F4}");
        Debug.Log($"  XDrive Target: {leftGripperBody.xDrive.target:F4}");
    }
    
    if (rightGripperBody != null)
    {
        Debug.Log($"右ArticulationBody: {rightGripperBody.name}");
        Debug.Log($"  Position: {rightGripperBody.transform.position}");
        Debug.Log($"  JointPosition: {rightGripperBody.jointPosition[0]:F4}");
        Debug.Log($"  XDrive Target: {rightGripperBody.xDrive.target:F4}");
    }
}

private void DiagnoseContactCalculation()
{
    if (target == null || leftGripperTip == null || rightGripperTip == null) return;
    
    Collider canCollider = target.GetComponent<Collider>();
    if (canCollider == null) return;
    
    // 手動で最近点を計算
    Vector3 leftClosest = canCollider.ClosestPoint(leftGripperTip.position);
    Vector3 rightClosest = canCollider.ClosestPoint(rightGripperTip.position);
    
    float leftDist = Vector3.Distance(leftGripperTip.position, leftClosest);
    float rightDist = Vector3.Distance(rightGripperTip.position, rightClosest);
    
    Debug.Log($"手動計算結果:");
    Debug.Log($"  左グリッパー → 最近点: {leftClosest} (距離: {leftDist:F3}m)");
    Debug.Log($"  右グリッパー → 最近点: {rightClosest} (距離: {rightDist:F3}m)");
    
    // 最近点が同じかチェック
    if (Vector3.Distance(leftClosest, rightClosest) < 0.001f)
    {
        Debug.LogWarning("⚠️ 左右の最近点が同じです！これは異常です。");
        Debug.LogWarning("   原因: アルミ缶のColliderが小さすぎるか、グリッパーが同じ方向にある");
    }
    
    // アルミ缶を中心とした距離も計算
    float leftToCenter = Vector3.Distance(leftGripperTip.position, target.transform.position);
    float rightToCenter = Vector3.Distance(rightGripperTip.position, target.transform.position);
    
    Debug.Log($"中心からの距離:");
    Debug.Log($"  左グリッパー → 中心: {leftToCenter:F3}m");
    Debug.Log($"  右グリッパー → 中心: {rightToCenter:F3}m");
}


}