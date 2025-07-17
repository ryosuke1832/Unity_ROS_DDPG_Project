using UnityEngine;

/// <summary>
/// ペットボトルターゲットとグリッパーシステムの連携インターフェース
/// 既存のGripperForceControllerと新しいPetBottleTargetを接続
/// </summary>
public class BottleGripperInterface : MonoBehaviour
{
    [Header("連携コンポーネント")]
    public PetBottleTarget bottleTarget;
    public Transform leftGripper;
    public Transform rightGripper;
    
    [Header("接触検出設定")]
    public float contactDetectionRadius = 0.05f;
    public LayerMask bottleLayer = -1;
    public bool enableContinuousContact = true;
    
    [Header("力計算設定")]
    public float baseGripForce = 10f;
    public float forceMultiplier = 2f;
    public float maxGripForce = 50f;
    public AnimationCurve forceDistanceCurve = AnimationCurve.EaseInOut(0f, 0f, 1f, 1f);
    
    [Header("デバッグ")]
    public bool showContactGizmos = true;
    public bool enableForceLogging = false;
    
    // 内部状態
    private bool leftInContact = false;
    private bool rightInContact = false;
    private Vector3 leftContactPoint;
    private Vector3 rightContactPoint;
    private Vector3 leftContactNormal;
    private Vector3 rightContactNormal;
    private float currentGripDistance = 0f;
    private float targetGripDistance = 0f;
    
    // キャッシュされたコンポーネント
    private Collider bottleCollider;
    private ArticulationBody leftGripperBody;
    private ArticulationBody rightGripperBody;
    
    void Start()
    {
        InitializeInterface();
    }
    
    void Update()
    {
        UpdateContactDetection();
        CalculateGripForces();
        ApplyForcesToBottle();
    }
    
    private void InitializeInterface()
    {
        // ボトルターゲットを自動検索
        if (bottleTarget == null)
        {
            bottleTarget = FindObjectOfType<PetBottleTarget>();
        }
        
        // グリッパーを自動検索
        if (leftGripper == null || rightGripper == null)
        {
            FindGrippers();
        }
        
        // コンポーネントをキャッシュ
        if (bottleTarget != null)
        {
            bottleCollider = bottleTarget.GetComponent<Collider>();
        }
        
        if (leftGripper != null)
        {
            leftGripperBody = leftGripper.GetComponent<ArticulationBody>();
        }
        
        if (rightGripper != null)
        {
            rightGripperBody = rightGripper.GetComponent<ArticulationBody>();
        }
        
        Debug.Log("BottleGripperInterface initialized");
    }
    
    private void FindGrippers()
    {
        // TrajectoryPlannerからグリッパー情報を取得
        var trajectoryPlanner = FindObjectOfType<TrajectoryPlanner>();
        if (trajectoryPlanner != null)
        {
            // TrajectoryPlannerのm_LeftGripperとm_RightGripperにアクセス
            // プライベートフィールドのため、Reflectionを使用
            var leftField = typeof(TrajectoryPlanner).GetField("m_LeftGripper", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var rightField = typeof(TrajectoryPlanner).GetField("m_RightGripper", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            if (leftField != null)
                leftGripper = ((ArticulationBody)leftField.GetValue(trajectoryPlanner))?.transform;
            if (rightField != null)
                rightGripper = ((ArticulationBody)rightField.GetValue(trajectoryPlanner))?.transform;
        }
        
        // 手動検索の代替方法
        if (leftGripper == null || rightGripper == null)
        {
            var grippers = FindObjectsOfType<ArticulationBody>();
            foreach (var gripper in grippers)
            {
                if (gripper.name.ToLower().Contains("left") && gripper.name.ToLower().Contains("gripper"))
                {
                    leftGripper = gripper.transform;
                }
                else if (gripper.name.ToLower().Contains("right") && gripper.name.ToLower().Contains("gripper"))
                {
                    rightGripper = gripper.transform;
                }
            }
        }
    }
    
    private void UpdateContactDetection()
    {
        if (bottleTarget == null || bottleCollider == null) return;
        
        // 左グリッパーの接触検出
        leftInContact = false;
        if (leftGripper != null)
        {
            float leftDistance = Vector3.Distance(leftGripper.position, bottleTarget.transform.position);
            if (leftDistance <= contactDetectionRadius + bottleTarget.bottleRadius)
            {
                RaycastHit hit;
                Vector3 direction = (bottleTarget.transform.position - leftGripper.position).normalized;
                
                if (Physics.Raycast(leftGripper.position, direction, out hit, contactDetectionRadius * 2f, bottleLayer))
                {
                    if (hit.collider == bottleCollider)
                    {
                        leftInContact = true;
                        leftContactPoint = hit.point;
                        leftContactNormal = hit.normal;
                    }
                }
            }
        }
        
        // 右グリッパーの接触検出
        rightInContact = false;
        if (rightGripper != null)
        {
            float rightDistance = Vector3.Distance(rightGripper.position, bottleTarget.transform.position);
            if (rightDistance <= contactDetectionRadius + bottleTarget.bottleRadius)
            {
                RaycastHit hit;
                Vector3 direction = (bottleTarget.transform.position - rightGripper.position).normalized;
                
                if (Physics.Raycast(rightGripper.position, direction, out hit, contactDetectionRadius * 2f, bottleLayer))
                {
                    if (hit.collider == bottleCollider)
                    {
                        rightInContact = true;
                        rightContactPoint = hit.point;
                        rightContactNormal = hit.normal;
                    }
                }
            }
        }
        
        // グリップ距離を計算
        if (leftGripper != null && rightGripper != null)
        {
            currentGripDistance = Vector3.Distance(leftGripper.position, rightGripper.position);
        }
    }
    
    private void CalculateGripForces()
    {
        if (bottleTarget == null) return;
        
        // グリッパー間の距離に基づく力の計算
        float normalizedDistance = Mathf.Clamp01(currentGripDistance / (bottleTarget.bottleRadius * 2f));
        float forceFromDistance = forceDistanceCurve.Evaluate(1f - normalizedDistance) * baseGripForce;
        
        // ArticulationBodyの駆動設定から力を推定
        float additionalForce = 0f;
        if (leftGripperBody != null && rightGripperBody != null)
        {
            // グリッパーの目標位置から閉じる力を推定
            var leftDrive = leftGripperBody.xDrive;
            var rightDrive = rightGripperBody.xDrive;
            
            // 現在の位置を取得
            var leftPosition = leftGripperBody.jointPosition;
            var rightPosition = rightGripperBody.jointPosition;
            float leftCurrentPos = leftPosition.dofCount > 0 ? leftPosition[0] : 0f;
            float rightCurrentPos = rightPosition.dofCount > 0 ? rightPosition[0] : 0f;
            
            // 現在の位置と目標位置の差から力を推定
            float leftPositionError = Mathf.Abs(leftDrive.target - leftCurrentPos);
            float rightPositionError = Mathf.Abs(rightDrive.target - rightCurrentPos);
            
            float leftEstimatedForce = leftPositionError * leftDrive.stiffness * 0.1f;
            float rightEstimatedForce = rightPositionError * rightDrive.stiffness * 0.1f;
            
            additionalForce = (leftEstimatedForce + rightEstimatedForce) * forceMultiplier;
        }
        
        float totalForce = Mathf.Clamp(forceFromDistance + additionalForce, 0f, maxGripForce);
        
        if (enableForceLogging && (leftInContact || rightInContact))
        {
            Debug.Log($"Grip force calculated: {totalForce:F2}N (Distance: {currentGripDistance:F3}m)");
        }
    }
    
    private void ApplyForcesToBottle()
    {
        if (bottleTarget == null) return;
        
        // 左グリッパーからの力を適用
        if (leftInContact)
        {
            float force = CalculateGripForceForContact(leftGripper, leftContactPoint);
            bottleTarget.ApplyGripperForceWithDirection(force, leftContactPoint, -leftContactNormal);
        }
        
        // 右グリッパーからの力を適用
        if (rightInContact)
        {
            float force = CalculateGripForceForContact(rightGripper, rightContactPoint);
            bottleTarget.ApplyGripperForceWithDirection(force, rightContactPoint, -rightContactNormal);
        }
        
        // 両方のグリッパーが接触していない場合
        if (!leftInContact && !rightInContact)
        {
            // 力をゼロにして元の形に戻す
            bottleTarget.ApplyGripperForce(0f, Vector3.zero);
        }
    }
    
    private float CalculateGripForceForContact(Transform gripper, Vector3 contactPoint)
    {
        if (gripper == null || bottleTarget == null) return 0f;
        
        // 距離ベースの力計算
        float distance = Vector3.Distance(gripper.position, contactPoint);
        float normalizedDistance = Mathf.Clamp01(distance / contactDetectionRadius);
        float proximityForce = (1f - normalizedDistance) * baseGripForce;
        
        // グリッパーの速度を考慮
        ArticulationBody gripperBody = gripper.GetComponent<ArticulationBody>();
        float velocityForce = 0f;
        if (gripperBody != null)
        {
            // 関節の速度を取得 - dofCountプロパティを使用
            var jointVelocity = gripperBody.jointVelocity;
            float velocity = jointVelocity.dofCount > 0 ? Mathf.Abs(jointVelocity[0]) : 0f;
            velocityForce = velocity * forceMultiplier * 10f; // 速度を力に変換
            
            // 目標位置と現在位置の差からも力を推定
            var drive = gripperBody.xDrive;
            var jointPosition = gripperBody.jointPosition;
            float currentPosition = jointPosition.dofCount > 0 ? jointPosition[0] : 0f;
            float positionError = Mathf.Abs(drive.target - currentPosition);
            float stiffnessForce = positionError * drive.stiffness * 0.01f;
            velocityForce += stiffnessForce;
        }
        
        return Mathf.Clamp(proximityForce + velocityForce, 0f, maxGripForce);
    }
    
    public void SetGripTarget(float targetDistance)
    {
        targetGripDistance = targetDistance;
    }
    
    public bool IsBothGrippersInContact()
    {
        return leftInContact && rightInContact;
    }
    
    public bool IsEitherGripperInContact()
    {
        return leftInContact || rightInContact;
    }
    
    public float GetCurrentGripDistance()
    {
        return currentGripDistance;
    }
    
    public Vector3 GetAverageContactPoint()
    {
        if (leftInContact && rightInContact)
        {
            return (leftContactPoint + rightContactPoint) * 0.5f;
        }
        else if (leftInContact)
        {
            return leftContactPoint;
        }
        else if (rightInContact)
        {
            return rightContactPoint;
        }
        
        return Vector3.zero;
    }
    
    void OnDrawGizmos()
    {
        if (!showContactGizmos) return;
        
        // 接触検出範囲を表示
        if (leftGripper != null)
        {
            Gizmos.color = leftInContact ? Color.green : Color.gray;
            Gizmos.DrawWireSphere(leftGripper.position, contactDetectionRadius);
            
            if (leftInContact)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(leftContactPoint, 0.01f);
                Gizmos.DrawRay(leftContactPoint, leftContactNormal * 0.05f);
            }
        }
        
        if (rightGripper != null)
        {
            Gizmos.color = rightInContact ? Color.green : Color.gray;
            Gizmos.DrawWireSphere(rightGripper.position, contactDetectionRadius);
            
            if (rightInContact)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(rightContactPoint, 0.01f);
                Gizmos.DrawRay(rightContactPoint, rightContactNormal * 0.05f);
            }
        }
        
        // グリッパー間の距離を表示
        if (leftGripper != null && rightGripper != null)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawLine(leftGripper.position, rightGripper.position);
        }
        
        // ボトルの概要
        if (bottleTarget != null)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireSphere(bottleTarget.transform.position, bottleTarget.bottleRadius);
        }
    }
}