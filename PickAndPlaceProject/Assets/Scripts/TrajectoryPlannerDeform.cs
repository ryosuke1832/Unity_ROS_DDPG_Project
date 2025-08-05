using System.Collections;
using UnityEngine;

/// <summary>
/// アルミ缶専用のTrajectoryPlanner拡張
/// シンプルな重複送信防止版
/// </summary>
public class TrajectoryPlannerDeform : MonoBehaviour
{
    [Header("アルミ缶システム連携")]
    public GripperTargetInterface gripperInterface;
    public IntegratedAluminumCan target;
    public SimpleGripForceController forceController;

    [Header("初期位置設定")]
    public Transform robotInitialPosition;
    public Transform aluminumCanInitialPosition;
    public bool resetOnStart = true;
    
    [Header("把持設定")]
    public float graspEvaluationDelay = 1f;
    public bool enableDeformationLogging = true;
    
    // 元のTrajectoryPlannerへの参照
    private TrajectoryPlanner originalTrajectoryPlanner;

    // 初期位置の記録用
    private Vector3 robotOriginalPosition;
    private Quaternion robotOriginalRotation;
    private Vector3 aluminumCanOriginalPosition;
    private Quaternion aluminumCanOriginalRotation;
    
    // 把持状態管理
    private bool isCurrentlyGrasping = false;
    private Coroutine graspEvaluationCoroutine;
    
    // 🔥 シンプルな重複防止フラグ（追加）
    private bool hasEvaluatedThisAttempt = false;
    
    // イベント定義
    public System.Action<GraspEvaluation> OnGraspEvaluated;
    
    void Start()
    {
        originalTrajectoryPlanner = GetComponent<TrajectoryPlanner>();
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerDeform: 同じGameObjectにTrajectoryPlannerが見つかりません");
            enabled = false;
            return;
        }
        
        RecordInitialPositions();
        InitializeAluminumCanSystem();

        if (resetOnStart)
        {
            ResetToInitialPositions();
        }
        
        StartCoroutine(MonitorGripperMovement());
    }

    /// <summary>
    /// スタートボタン用: 初期位置リセット + 評価フラグリセット
    /// </summary>
    public void PublishJointAlminumCanWithReset()
    {
        if (enableDeformationLogging)
            Debug.Log("=== スタートボタン押下: 初期位置リセット + アルミ缶システム開始 ===");
        
        // 🔥 評価フラグをリセット（追加）
        hasEvaluatedThisAttempt = false;
        
        ResetToInitialPositions();
        StartCoroutine(ExecuteAfterReset());
    }

    private IEnumerator ExecuteAfterReset()
    {
        yield return new WaitForSeconds(0.5f);
        PublishJointAlminumCan();
    }

    public void PublishJointAlminumCan()
    {
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerが見つかりません");
            return;
        }
        
        if (enableDeformationLogging)
            Debug.Log("アルミ缶システム対応のPublishJoints実行");
        
        PrepareAluminumCanSystem();
        
        if (forceController != null)
        {
            forceController.enabled = true;
            if (enableDeformationLogging)
                Debug.Log("力制御システムを有効化しました");
        }
        
        originalTrajectoryPlanner.PublishJoints();
    }

    private void RecordInitialPositions()
    {
        if (robotInitialPosition != null)
        {
            robotOriginalPosition = robotInitialPosition.position;
            robotOriginalRotation = robotInitialPosition.rotation;
        }
        else if (originalTrajectoryPlanner?.NiryoOne != null)
        {
            robotOriginalPosition = originalTrajectoryPlanner.NiryoOne.transform.position;
            robotOriginalRotation = originalTrajectoryPlanner.NiryoOne.transform.rotation;
        }
        
        if (aluminumCanInitialPosition != null)
        {
            aluminumCanOriginalPosition = aluminumCanInitialPosition.position;
            aluminumCanOriginalRotation = aluminumCanInitialPosition.rotation;
        }
        else if (target != null)
        {
            aluminumCanOriginalPosition = target.transform.position;
            aluminumCanOriginalRotation = target.transform.rotation;
        }
        
        if (enableDeformationLogging)
        {
            Debug.Log($"初期位置記録完了:");
            Debug.Log($"- ロボット: {robotOriginalPosition}, {robotOriginalRotation.eulerAngles}");
            Debug.Log($"- アルミ缶: {aluminumCanOriginalPosition}, {aluminumCanOriginalRotation.eulerAngles}");
        }
    }

    public void ResetToInitialPositions()
    {
        if (enableDeformationLogging)
            Debug.Log("初期位置へのリセット開始");
        
        ResetRobotPosition();
        ResetAluminumCanPosition();
        ResetGraspingState();
        
        if (enableDeformationLogging)
            Debug.Log("初期位置へのリセット完了");
    }

    private void ResetRobotPosition()
    {
        GameObject robotObject = null;
        
        if (robotInitialPosition != null)
        {
            robotObject = robotInitialPosition.gameObject;
        }
        else if (originalTrajectoryPlanner?.NiryoOne != null)
        {
            robotObject = originalTrajectoryPlanner.NiryoOne;
        }
        
        if (robotObject != null)
        {
            robotObject.transform.position = robotOriginalPosition;
            robotObject.transform.rotation = robotOriginalRotation;
            
            ResetRobotJoints(robotObject);
            
            if (enableDeformationLogging)
                Debug.Log($"ロボット位置リセット: {robotOriginalPosition}");
        }
    }
    
    private void ResetRobotJoints(GameObject robotObject)
    {
        ArticulationBody[] joints = robotObject.GetComponentsInChildren<ArticulationBody>();
        
        foreach (var joint in joints)
        {
            if (joint.isRoot) continue;
            
            var drive = joint.xDrive;
            drive.target = 0f;
            joint.xDrive = drive;
            
            joint.velocity = Vector3.zero;
            joint.angularVelocity = Vector3.zero;
        }
        
        if (enableDeformationLogging)
            Debug.Log($"ロボット関節リセット完了: {joints.Length}個の関節");
    }
    
    private void ResetAluminumCanPosition()
    {
        if (target != null)
        {
            target.transform.position = aluminumCanOriginalPosition;
            target.transform.rotation = aluminumCanOriginalRotation;
            
            Rigidbody canRigidBody = target.GetComponent<Rigidbody>();
            if (canRigidBody != null)
            {
                canRigidBody.velocity = Vector3.zero;
                canRigidBody.angularVelocity = Vector3.zero;
            }
            
            target.ResetCan();
            
            if (enableDeformationLogging)
                Debug.Log($"アルミ缶位置リセット: {aluminumCanOriginalPosition}");
        }
    }

    private void ResetGraspingState()
    {
        isCurrentlyGrasping = false;
        
        if (graspEvaluationCoroutine != null)
        {
            StopCoroutine(graspEvaluationCoroutine);
            graspEvaluationCoroutine = null;
        }
        
        if (forceController != null)
        {
            forceController.enabled = false;
        }
        
        if (originalTrajectoryPlanner != null)
        {
            var openGripperMethod = originalTrajectoryPlanner.GetType().GetMethod("OpenGripper", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            openGripperMethod?.Invoke(originalTrajectoryPlanner, null);
        }
        
        if (enableDeformationLogging)
            Debug.Log("把持状態リセット完了");
    }

    private void InitializeAluminumCanSystem()
    {
        if (gripperInterface == null)
            gripperInterface = GetComponent<GripperTargetInterface>();
            
        if (target == null)
            target = FindObjectOfType<IntegratedAluminumCan>();
            
        if (forceController == null)
            forceController = GetComponent<SimpleGripForceController>();
        
        Debug.Log($"アルミ缶システム初期化:");
        Debug.Log($"- GripperInterface: {(gripperInterface != null ? "OK" : "NG")}");
        Debug.Log($"- AluminumCan: {(target != null ? "OK" : "NG")}");
        Debug.Log($"- ForceController: {(forceController != null ? "OK" : "NG")}");
    }
    
    private void PrepareAluminumCanSystem()
    {
        if (forceController != null)
        {
            forceController.enabled = false;
        }
        
        if (target != null)
        {
            target.ResetCan();
        }
    }

    public void PublishJointsWithAluminumCan()
    {
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerが見つかりません");
            return;
        }
        
        if (enableDeformationLogging)
            Debug.Log("アルミ缶システム対応のPublishJoints実行");
        
        PrepareAluminumCanSystem();
        
        if (forceController != null)
        {
            forceController.enabled = true;
            if (enableDeformationLogging)
                Debug.Log("力制御システムを有効化しました");
        }
        
        originalTrajectoryPlanner.PublishJoints();
    }

    private void StartGraspWithAluminumCan()
    {
        if (graspEvaluationCoroutine != null)
        {
            StopCoroutine(graspEvaluationCoroutine);
        }
        
        isCurrentlyGrasping = true;
        graspEvaluationCoroutine = StartCoroutine(EvaluateGraspAfterDelay());
    }
    
    private void StopGraspWithAluminumCan()
    {
        if (graspEvaluationCoroutine != null)
        {
            StopCoroutine(graspEvaluationCoroutine);
            graspEvaluationCoroutine = null;
        }
        
        isCurrentlyGrasping = false;
        
        if (forceController != null)
        {
            forceController.enabled = false;
        }
    }
    
    private IEnumerator EvaluateGraspAfterDelay()
    {
        yield return new WaitForSeconds(graspEvaluationDelay);
        
        // 🔥 重複チェック（修正）
        if (gripperInterface != null && !hasEvaluatedThisAttempt)
        {
            hasEvaluatedThisAttempt = true; // フラグを立てる
            
            var evaluation = gripperInterface.EvaluateGrasp();
            
            if (enableDeformationLogging)
            {
                Debug.Log($"📊 アルミ缶把持評価結果（一度だけ送信）:");
                Debug.Log($"- 結果: {evaluation.result}");
                Debug.Log($"- 適用力: {evaluation.appliedForce:F2}N");
                Debug.Log($"- 変形度: {evaluation.deformation:F3}");
                Debug.Log($"- つぶれ状態: {evaluation.isBroken}");
                Debug.Log($"- 信頼度: {evaluation.confidence:F2}");
            }
            
            // 一度だけ送信
            OnGraspEvaluated?.Invoke(evaluation);
        }
    }

    private IEnumerator MonitorGripperMovement()
    {
        ArticulationBody leftGripper = null;
        ArticulationBody rightGripper = null;
        
        ArticulationBody[] allBodies = FindObjectsOfType<ArticulationBody>();
        foreach (var body in allBodies)
        {
            if (body.name.Contains("left_gripper"))
                leftGripper = body;
            if (body.name.Contains("right_gripper"))
                rightGripper = body;
        }
        
        if (leftGripper == null || rightGripper == null)
        {
            if (enableDeformationLogging)
                Debug.LogWarning("グリッパーのArticulationBodyが見つかりません");
            yield break;
        }
        
        bool wasGrasping = false;
        
        while (true)
        {
            yield return new WaitForSeconds(0.1f);
            
            float currentLeftTarget = leftGripper.xDrive.target;
            float currentRightTarget = rightGripper.xDrive.target;
            
            bool isCurrentlyGrasping = (currentLeftTarget < -0.005f && currentRightTarget > 0.005f);
            
            if (isCurrentlyGrasping && !wasGrasping)
            {
                if (enableDeformationLogging)
                    Debug.Log("グリッパー閉じ動作を検出 - アルミ缶力制御開始");
                StartGraspWithAluminumCan();
            }
            else if (!isCurrentlyGrasping && wasGrasping)
            {
                if (enableDeformationLogging)
                    Debug.Log("グリッパー開き動作を検出 - アルミ缶力制御終了");
                StopGraspWithAluminumCan();
            }
            
            wasGrasping = isCurrentlyGrasping;
        }
    }
}