using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 修正版：物理ベースの接触検出とグリッパー方向を考慮した力伝達システム
/// 既存コードとの競合を回避
/// </summary>
public class GripperTargetInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public SimpleGripForceController simpleGripperController;
    public DeformableTarget target;
    
    [Header("グリッパー設定")]
    public Transform leftGripperTip;    // 左グリッパーの先端
    public Transform rightGripperTip;   // 右グリッパーの先端
    public ArticulationBody leftGripperBody;   // 左グリッパーのArticulationBody
    public ArticulationBody rightGripperBody;  // 右グリッパーのArticulationBody
    
    [Header("接触判定設定")]
    public float gripperOpenThreshold = 0.01f;     // グリッパーが開いている閾値
    public float gripperCloseThreshold = 0.005f;   // グリッパーが閉じている閾値
    public float contactForceThreshold = 0.5f;     // 接触を判定する最小力
    public LayerMask targetLayerMask = -1;
    
    [Header("力伝達設定")]
    public float forceTransferRate = 1f;
    public bool requireBothGrippersContact = true;  // 両方のグリッパーでの接触を要求
    
    [Header("デバッグ")]
    public bool showContactGizmos = true;
    public bool enableDetailedLogging = false;
    
    // 内部状態
    private bool leftGripperInContact = false;
    private bool rightGripperInContact = false;
    private Vector3 leftContactPoint;
    private Vector3 rightContactPoint;
    private Vector3 lastContactNormal;
    
    // グリッパー状態
    private float currentLeftPosition = 0f;
    private float currentRightPosition = 0f;
    private bool isGripperClosed = false;
    private bool wasGripperClosed = false;
    
    // 接触情報を格納するコンポーネント
    private List<ContactPointInfo> activeContacts = new List<ContactPointInfo>();
    
    [System.Serializable]
    public struct ContactPointInfo
    {
        public Vector3 point;
        public Vector3 normal;
        public float force;
        public bool isLeftGripper;
        public float timestamp;
    }
    
    void Start()
    {
        InitializeGripperInterface();
        SetupGripperColliders();
    }
    
    void FixedUpdate()
    {
        UpdateGripperState();
        UpdateContactDetection();
        TransferForceToTarget();
        CleanupOldContacts();
    }
    
    /// <summary>
    /// 初期化処理
    /// </summary>
    private void InitializeGripperInterface()
    {
        if (simpleGripperController == null)
            simpleGripperController = GetComponent<SimpleGripForceController>();
        
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
        
        // グリッパーの自動検索
        if (leftGripperBody == null || rightGripperBody == null)
        {
            FindGripperComponents();
        }
        
        if (simpleGripperController == null || target == null)
        {
            Debug.LogError("GripperTargetInterface: 必要なコンポーネントが見つかりません");
            enabled = false;
            return;
        }
        
        Debug.Log("GripperTargetInterface initialized with physics-based contact detection");
    }
    
    /// <summary>
    /// グリッパーコンポーネントの自動検索
    /// </summary>
    private void FindGripperComponents()
    {
        ArticulationBody[] allBodies = FindObjectsOfType<ArticulationBody>();
        
        foreach (var body in allBodies)
        {
            if (body.name.Contains("left_gripper") || body.name.Contains("LeftGripper"))
            {
                leftGripperBody = body;
                leftGripperTip = body.transform;
            }
            else if (body.name.Contains("right_gripper") || body.name.Contains("RightGripper"))
            {
                rightGripperBody = body;
                rightGripperTip = body.transform;
            }
        }
        
        if (enableDetailedLogging)
        {
            Debug.Log($"Found grippers - Left: {leftGripperBody?.name}, Right: {rightGripperBody?.name}");
        }
    }
    
    /// <summary>
    /// グリッパーにコライダーとトリガーを設定
    /// </summary>
    private void SetupGripperColliders()
    {
        if (leftGripperTip != null)
        {
            SetupGripperCollider(leftGripperTip.gameObject, true);
        }
        
        if (rightGripperTip != null)
        {
            SetupGripperCollider(rightGripperTip.gameObject, false);
        }
    }
    
    private void SetupGripperCollider(GameObject gripperObject, bool isLeft)
    {
        // 既存のコライダーを確認
        Collider existingCollider = gripperObject.GetComponent<Collider>();
        if (existingCollider == null)
        {
            // 新しいトリガーコライダーを追加
            BoxCollider triggerCollider = gripperObject.AddComponent<BoxCollider>();
            triggerCollider.isTrigger = true;
            triggerCollider.size = Vector3.one * 0.02f; // 小さなトリガー領域
        }
        
        // 接触検出コンポーネントを追加
        GripperContactDetector detector = gripperObject.GetComponent<GripperContactDetector>();
        if (detector == null)
        {
            detector = gripperObject.AddComponent<GripperContactDetector>();
        }
        detector.Initialize(this, isLeft);
    }
    
    /// <summary>
    /// グリッパーの現在状態を更新
    /// </summary>
    private void UpdateGripperState()
    {
        // ArticulationBodyから実際の位置を取得
        if (leftGripperBody != null && leftGripperBody.jointPosition.dofCount > 0)
        {
            currentLeftPosition = leftGripperBody.jointPosition[0];
        }
        
        if (rightGripperBody != null && rightGripperBody.jointPosition.dofCount > 0)
        {
            currentRightPosition = rightGripperBody.jointPosition[0];
        }
        
        // グリッパーが閉じているかどうかの判定
        wasGripperClosed = isGripperClosed;
        isGripperClosed = IsGripperInClosedState();
        
        // 状態変化のログ
        if (enableDetailedLogging && wasGripperClosed != isGripperClosed)
        {
            Debug.Log($"Gripper state changed: {(isGripperClosed ? "CLOSED" : "OPEN")}");
            Debug.Log($"Left position: {currentLeftPosition:F4}, Right position: {currentRightPosition:F4}");
        }
    }
    
    /// <summary>
    /// グリッパーが閉じた状態かどうかを判定
    /// </summary>
    private bool IsGripperInClosedState()
    {
        // 左グリッパーが負の方向、右グリッパーが正の方向に移動している
        bool leftClosed = currentLeftPosition < -gripperCloseThreshold;
        bool rightClosed = currentRightPosition > gripperCloseThreshold;
        
        return leftClosed && rightClosed;
    }
    
    /// <summary>
    /// 物理ベースの接触検出更新
    /// </summary>
    private void UpdateContactDetection()
    {
        // グリッパーが閉じている状態でのみ接触を有効とする
        if (!isGripperClosed)
        {
            leftGripperInContact = false;
            rightGripperInContact = false;
            return;
        }
        
        // 両方のグリッパーでの接触が必要な場合
        if (requireBothGrippersContact)
        {
            // 両方が接触している場合のみ有効
            bool validContact = leftGripperInContact && rightGripperInContact;
            
            if (enableDetailedLogging && validContact != (leftGripperInContact || rightGripperInContact))
            {
                Debug.Log($"Contact state: Left={leftGripperInContact}, Right={rightGripperInContact}, Valid={validContact}");
            }
        }
    }
    
    /// <summary>
    /// 力をターゲットに伝達（修正版）
    /// </summary>
    private void TransferForceToTarget()
    {
        if (target == null || simpleGripperController == null) return;
        
        // グリッパーが閉じていて、適切な接触がある場合のみ力を伝達
        bool canTransferForce = isGripperClosed && HasValidContact();
        
        if (!canTransferForce)
        {
            return;
        }
        
        // SimpleGripForceControllerから現在の力を取得
        float currentForce = GetCurrentForceFromController() * forceTransferRate;
        
        // 力が閾値以上の場合のみ伝達
        if (currentForce >= contactForceThreshold)
        {
            // 接触点の計算（両方のグリッパーの中点）
            Vector3 contactPoint = CalculateContactPoint();
            
            // ターゲットに力を伝達（方向情報付き）
            target.ApplyGripperForceWithDirection(currentForce, contactPoint, lastContactNormal);
            
            if (enableDetailedLogging && Time.fixedTime % 0.2f < Time.fixedDeltaTime)
            {
                Debug.Log($"Force Transfer: {currentForce:F2}N at {contactPoint} (Normal: {lastContactNormal})");
            }
        }
    }
    
    /// <summary>
    /// 有効な接触があるかチェック
    /// </summary>
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
    
    /// <summary>
    /// 接触点を計算
    /// </summary>
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
        
        // フォールバック：ターゲットの中心
        return target.transform.position;
    }
    
    /// <summary>
    /// SimpleGripForceControllerから力を取得
    /// </summary>
    private float GetCurrentForceFromController()
    {
        if (simpleGripperController == null) return 0f;
        
        // GetCurrentTargetForce()メソッドを使用
        return simpleGripperController.GetCurrentTargetForce();
    }
    
    /// <summary>
    /// 外部から呼び出される接触イベント（GripperContactDetectorから）
    /// </summary>
    public void OnGripperContactEnter(Collision collision, bool isLeftGripper)
    {
        if (!IsTargetObject(collision.gameObject)) return;
        
        ContactPoint contact = collision.contacts[0];
        
        if (isLeftGripper)
        {
            leftGripperInContact = true;
            leftContactPoint = contact.point;
        }
        else
        {
            rightGripperInContact = true;
            rightContactPoint = contact.point;
        }
        
        lastContactNormal = contact.normal;
        
        // 接触情報を記録
        var contactInfo = new ContactPointInfo
        {
            point = contact.point,
            normal = contact.normal,
            force = 0f, // 後で更新
            isLeftGripper = isLeftGripper,
            timestamp = Time.time
        };
        activeContacts.Add(contactInfo);
        
        if (enableDetailedLogging)
        {
            Debug.Log($"{(isLeftGripper ? "Left" : "Right")} gripper contact ENTER at {contact.point}");
        }
    }
    
    public void OnGripperContactExit(Collision collision, bool isLeftGripper)
    {
        if (!IsTargetObject(collision.gameObject)) return;
        
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
    
    /// <summary>
    /// Colliderベースの接触開始（GripperContactDetectorから呼び出し）
    /// </summary>
    public void OnGripperContactWithCollider(Collider collider, bool isLeftGripper, Vector3 contactPoint, Vector3 contactNormal)
    {
        if (!IsTargetObject(collider.gameObject)) return;
        
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
        
        // 接触情報を記録
        var contactInfo = new ContactPointInfo
        {
            point = contactPoint,
            normal = contactNormal,
            force = 0f, // 後で更新
            isLeftGripper = isLeftGripper,
            timestamp = Time.time
        };
        activeContacts.Add(contactInfo);
        
        if (enableDetailedLogging)
        {
            Debug.Log($"{(isLeftGripper ? "Left" : "Right")} gripper contact ENTER (Collider) at {contactPoint}");
        }
    }
    
    /// <summary>
    /// Colliderベースの接触終了
    /// </summary>
    public void OnGripperContactExitWithCollider(Collider collider, bool isLeftGripper)
    {
        if (!IsTargetObject(collider.gameObject)) return;
        
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
            Debug.Log($"{(isLeftGripper ? "Left" : "Right")} gripper contact EXIT (Collider)");
        }
    }
    
    /// <summary>
    /// ターゲットオブジェクトかどうかチェック
    /// </summary>
    private bool IsTargetObject(GameObject obj)
    {
        return obj == target.gameObject || obj.transform.IsChildOf(target.transform);
    }
    
    /// <summary>
    /// 古い接触情報をクリーンアップ
    /// </summary>
    private void CleanupOldContacts()
    {
        float currentTime = Time.time;
        activeContacts.RemoveAll(contact => currentTime - contact.timestamp > 1f);
    }
    
    /// <summary>
    /// 把持状態の評価（既存のGraspingStateを使用）
    /// </summary>
    public GraspEvaluation EvaluateGraspCompatible()
    {
        if (target == null || simpleGripperController == null)
            return GraspEvaluation.CreateSimple(GraspResult.Failure);
        
        var objectState = target.GetCurrentState();
        bool hasValidContact = HasValidContact();
        bool isGripping = isGripperClosed && hasValidContact;
        
        // 既存のGraspingStateを取得
        GraspingState graspingState = simpleGripperController.GetGraspingStateForInterface();
        
        GraspResult result = DetermineGraspResultCompatible(objectState, graspingState, isGripping);
        
        return new GraspEvaluation
        {
            result = result,
            appliedForce = objectState.appliedForce,
            deformation = objectState.deformation,
            isBroken = objectState.isBroken,
            confidence = CalculateConfidenceCompatible(hasValidContact, isGripping),
            hasContact = hasValidContact,
            isGripping = isGripping,
            evaluationTime = Time.time
        };
    }
    
    private GraspResult DetermineGraspResultCompatible(ObjectState objectState, GraspingState graspingState, bool isGripping)
    {
        if (objectState.isBroken)
            return GraspResult.Broken;
        
        if (!isGripping)
            return GraspResult.NoContact;
        
        float force = objectState.appliedForce;
        
        if (force < 1f)
            return GraspResult.UnderGrip;
        else if (force > 50f)
            return GraspResult.OverGrip;
        else
            return GraspResult.Success;
    }
    
    private float CalculateConfidenceCompatible(bool hasValidContact, bool isGripping)
    {
        float confidence = 0f;
        
        if (hasValidContact) confidence += 0.4f;
        if (isGripping) confidence += 0.4f;
        if (isGripperClosed) confidence += 0.2f;
        
        return confidence;
    }
    
    void OnDrawGizmos()
    {
        if (!showContactGizmos) return;
        
        // グリッパーの位置
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
        
        // 接触点
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
        
        // 接触法線
        if (HasValidContact())
        {
            Gizmos.color = Color.yellow;
            Vector3 contactPoint = CalculateContactPoint();
            Gizmos.DrawRay(contactPoint, lastContactNormal * 0.05f);
        }
    }
}