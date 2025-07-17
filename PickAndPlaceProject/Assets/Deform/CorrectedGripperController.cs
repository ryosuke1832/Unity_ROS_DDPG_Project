using UnityEngine;
using Deform;

/// <summary>
/// SquashAndStretchDeformerを使用した修正版グリッパーコントローラー
/// </summary>
public class CorrectedGripperController : MonoBehaviour
{
    [Header("連携設定")]
    public SquashAndStretchDeformer squashDeformer;
    public Transform leftGripper;
    public Transform rightGripper;
    
    [Header("接触検出設定")]
    public float detectionDistance = 0.15f;
    public bool enableContinuousContact = true;
    
    [Header("変形設定")]
    [Range(0f, 2f)]
    public float maxSquash = 1f;
    [Range(0.1f, 5f)]
    public float deformationSpeed = 2f;
    [Range(0f, 1f)]
    public float recoverySpeed = 0.5f;
    
    [Header("デバッグ")]
    public bool showContactGizmos = true;
    public bool enableDebugLogs = false;
    
    // 内部状態
    private bool isGrasping = false;
    private float currentSquash = 0f;
    private float targetSquash = 0f;
    private bool leftInContact = false;
    private bool rightInContact = false;
    
    void Start()
    {
        InitializeController();
    }
    
    void Update()
    {
        DetectGraspingState();
        UpdateDeformation();
    }
    
    private void InitializeController()
    {
        // SquashAndStretchDeformerを自動検索
        if (squashDeformer == null)
        {
            squashDeformer = GetComponent<SquashAndStretchDeformer>();
        }
        
        // グリッパーを自動検索
        if (leftGripper == null || rightGripper == null)
        {
            FindGrippers();
        }
        
        Debug.Log("CorrectedGripperController initialized");
    }
    
    private void FindGrippers()
    {
        // TrajectoryPlannerからグリッパー情報を取得
        var trajectoryPlanner = FindObjectOfType<TrajectoryPlanner>();
        if (trajectoryPlanner != null)
        {
            // Reflectionを使ってプライベートフィールドにアクセス
            var leftField = typeof(TrajectoryPlanner).GetField("m_LeftGripper", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            var rightField = typeof(TrajectoryPlanner).GetField("m_RightGripper", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            if (leftField != null)
            {
                var leftGripperBody = (ArticulationBody)leftField.GetValue(trajectoryPlanner);
                leftGripper = leftGripperBody?.transform;
            }
            
            if (rightField != null)
            {
                var rightGripperBody = (ArticulationBody)rightField.GetValue(trajectoryPlanner);
                rightGripper = rightGripperBody?.transform;
            }
        }
        
        // 手動検索の代替方法
        if (leftGripper == null || rightGripper == null)
        {
            var grippers = FindObjectsOfType<ArticulationBody>();
            foreach (var gripper in grippers)
            {
                string name = gripper.name.ToLower();
                if (name.Contains("left") && name.Contains("gripper"))
                {
                    leftGripper = gripper.transform;
                }
                else if (name.Contains("right") && name.Contains("gripper"))
                {
                    rightGripper = gripper.transform;
                }
            }
        }
    }
    
    private void DetectGraspingState()
    {
        bool leftNear = false;
        bool rightNear = false;
        
        if (leftGripper != null)
        {
            float leftDistance = Vector3.Distance(leftGripper.position, transform.position);
            leftNear = leftDistance < detectionDistance;
        }
        
        if (rightGripper != null)
        {
            float rightDistance = Vector3.Distance(rightGripper.position, transform.position);
            rightNear = rightDistance < detectionDistance;
        }
        
        leftInContact = leftNear;
        rightInContact = rightNear;
        
        bool newGraspState = leftNear && rightNear;
        
        if (newGraspState != isGrasping)
        {
            isGrasping = newGraspState;
            OnGraspStateChanged(isGrasping);
        }
        
        // 把持力の計算
        if (isGrasping)
        {
            float gripperDistance = Vector3.Distance(leftGripper.position, rightGripper.position);
            float normalizedGrip = Mathf.Clamp01(1f - (gripperDistance / (detectionDistance * 2f)));
            targetSquash = normalizedGrip * maxSquash;
        }
        else
        {
            targetSquash = 0f;
        }
    }
    
    private void UpdateDeformation()
    {
        if (squashDeformer == null) return;
        
        // スムーズな変形更新
        float speed = isGrasping ? deformationSpeed : recoverySpeed;
        currentSquash = Mathf.Lerp(currentSquash, targetSquash, speed * Time.deltaTime);
        
        // SquashAndStretchDeformerに適用
        squashDeformer.Factor = currentSquash;
        
        if (enableDebugLogs && Time.frameCount % 30 == 0) // 30フレームごとにログ
        {
            Debug.Log($"Deformation: {currentSquash:F3}, Target: {targetSquash:F3}, Grasping: {isGrasping}");
        }
    }
    
    private void OnGraspStateChanged(bool grasping)
    {
        if (enableDebugLogs)
        {
            Debug.Log($"Grasp state changed: {grasping}");
        }
    }
    
    /// <summary>
    /// 外部からの力適用（既存システムとの互換性用）
    /// </summary>
    public void ApplyGripperForce(float force, Vector3 contactPosition)
    {
        if (squashDeformer != null)
        {
            float normalizedForce = Mathf.Clamp01(force / 50f); // 50Nを最大とする
            targetSquash = normalizedForce * maxSquash;
        }
    }
    
    /// <summary>
    /// 変形をリセット
    /// </summary>
    public void ResetDeformation()
    {
        currentSquash = 0f;
        targetSquash = 0f;
        if (squashDeformer != null)
        {
            squashDeformer.Factor = 0f;
        }
    }
    
    /// <summary>
    /// 現在の把持状態を取得
    /// </summary>
    public bool IsGrasping()
    {
        return isGrasping;
    }
    
    /// <summary>
    /// 変形量を取得
    /// </summary>
    public float GetDeformationAmount()
    {
        return currentSquash;
    }
    
    void OnDrawGizmos()
    {
        if (!showContactGizmos) return;
        
        // 検出範囲を表示
        Gizmos.color = isGrasping ? Color.red : Color.yellow;
        Gizmos.DrawWireSphere(transform.position, detectionDistance);
        
        // グリッパーとの線
        if (leftGripper != null)
        {
            Gizmos.color = leftInContact ? Color.green : Color.blue;
            Gizmos.DrawLine(transform.position, leftGripper.position);
        }
        
        if (rightGripper != null)
        {
            Gizmos.color = rightInContact ? Color.green : Color.blue;
            Gizmos.DrawLine(transform.position, rightGripper.position);
        }
        
        // 変形の強さを表示
        if (currentSquash > 0.01f)
        {
            Gizmos.color = Color.red;
            Vector3 scale = transform.localScale;
            scale.y *= (1f - currentSquash * 0.5f);
            Gizmos.matrix = Matrix4x4.TRS(transform.position, transform.rotation, scale);
            Gizmos.DrawWireCube(Vector3.zero, Vector3.one);
            Gizmos.matrix = Matrix4x4.identity;
        }
    }
}