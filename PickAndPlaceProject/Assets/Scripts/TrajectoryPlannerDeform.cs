using System.Collections;
using UnityEngine;

/// <summary>
/// アルミ缶専用のTrajectoryPlanner拡張
/// DeformableTargetは使用せず、IntegratedAluminumCanのみに対応
/// </summary>
public class TrajectoryPlannerDeform : MonoBehaviour
{
    [Header("アルミ缶システム連携")]
    public GripperTargetInterface gripperInterface;
    public IntegratedAluminumCan target;
    public SimpleGripForceController forceController;

        [Header("初期位置設定")]
    public Transform robotInitialPosition;      // ロボットの初期位置参照
    public Transform aluminumCanInitialPosition; // アルミ缶の初期位置参照
    public bool resetOnStart = true;             // スタート時に初期位置にリセット
    
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
    
    void Start()
    {
        // 同じGameObjectのTrajectoryPlannerを取得
        originalTrajectoryPlanner = GetComponent<TrajectoryPlanner>();
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerDeform: 同じGameObjectにTrajectoryPlannerが見つかりません");
            enabled = false;
            return;
        }
                // 初期位置を記録
        RecordInitialPositions();

        InitializeAluminumCanSystem();

        // 初期位置にリセット（オプション）
        if (resetOnStart)
        {
            ResetToInitialPositions();
        }
        
        // グリッパーの動作監視を開始
        StartCoroutine(MonitorGripperMovement());
    }


    /// <summary>
    /// 初期位置を記録する
    /// </summary>
    private void RecordInitialPositions()
    {
        // ロボットの初期位置を記録
        if (robotInitialPosition != null)
        {
            robotOriginalPosition = robotInitialPosition.position;
            robotOriginalRotation = robotInitialPosition.rotation;
        }
        else if (originalTrajectoryPlanner?.NiryoOne != null)
        {
            // TrajectoryPlannerのNiryoOneから取得
            robotOriginalPosition = originalTrajectoryPlanner.NiryoOne.transform.position;
            robotOriginalRotation = originalTrajectoryPlanner.NiryoOne.transform.rotation;
        }
        
        // アルミ缶の初期位置を記録
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

    /// <summary>
    /// ロボットとアルミ缶を初期位置に戻す
    /// </summary>
    public void ResetToInitialPositions()
    {
        if (enableDeformationLogging)
            Debug.Log("初期位置へのリセット開始");
        
        // ロボットを初期位置に戻す
        ResetRobotPosition();
        
        // アルミ缶を初期位置に戻す
        ResetAluminumCanPosition();
        
        // 把持状態をリセット
        ResetGraspingState();
        
        if (enableDeformationLogging)
            Debug.Log("初期位置へのリセット完了");
    }


    /// <summary>
    /// ロボットを初期位置に戻す
    /// </summary>
    private void ResetRobotPosition()
    {
        GameObject robotObject = null;
        
        // ロボットオブジェクトを特定
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
            // 位置と回転をリセット
            robotObject.transform.position = robotOriginalPosition;
            robotObject.transform.rotation = robotOriginalRotation;
            
            // 関節角度もリセット（ArticulationBodyがある場合）
            ResetRobotJoints(robotObject);
            
            if (enableDeformationLogging)
                Debug.Log($"ロボット位置リセット: {robotOriginalPosition}");
        }
    }
    
    /// <summary>
    /// ロボットの関節角度をリセット
    /// </summary>
    private void ResetRobotJoints(GameObject robotObject)
    {
        ArticulationBody[] joints = robotObject.GetComponentsInChildren<ArticulationBody>();
        
        foreach (var joint in joints)
        {
            if (joint.isRoot) continue;
            
            // 関節を初期角度（0度）にリセット
            var drive = joint.xDrive;
            drive.target = 0f;
            joint.xDrive = drive;
            
            // 速度と角速度もリセット
            joint.velocity = Vector3.zero;
            joint.angularVelocity = Vector3.zero;
        }
        
        if (enableDeformationLogging)
            Debug.Log($"ロボット関節リセット完了: {joints.Length}個の関節");
    }


    
    /// <summary>
    /// アルミ缶を初期位置に戻す
    /// </summary>
    private void ResetAluminumCanPosition()
    {
        if (target != null)
        {
            // 位置と回転をリセット
            target.transform.position = aluminumCanOriginalPosition;
            target.transform.rotation = aluminumCanOriginalRotation;
            
            // Rigidbodyの速度もリセット
            Rigidbody canRigidBody = target.GetComponent<Rigidbody>();
            if (canRigidBody != null)
            {
                canRigidBody.velocity = Vector3.zero;
                canRigidBody.angularVelocity = Vector3.zero;
            }
            
            // アルミ缶の状態もリセット
            target.ResetCan();
            
            if (enableDeformationLogging)
                Debug.Log($"アルミ缶位置リセット: {aluminumCanOriginalPosition}");
        }
    }


    /// <summary>
    /// 把持状態をリセット
    /// </summary>
    private void ResetGraspingState()
    {
        // 把持状態をクリア
        isCurrentlyGrasping = false;
        
        // 実行中のコルーチンを停止
        if (graspEvaluationCoroutine != null)
        {
            StopCoroutine(graspEvaluationCoroutine);
            graspEvaluationCoroutine = null;
        }
        
        // 力制御システムを停止
        if (forceController != null)
        {
            forceController.enabled = false;
        }
        
        // グリッパーを開く
        if (originalTrajectoryPlanner != null)
        {
            // TrajectoryPlannerのOpenGripper()メソッドを呼び出し
            var openGripperMethod = originalTrajectoryPlanner.GetType().GetMethod("OpenGripper", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            openGripperMethod?.Invoke(originalTrajectoryPlanner, null);
        }
        
        if (enableDeformationLogging)
            Debug.Log("把持状態リセット完了");
    }

    /// <summary>
    /// スタートボタン用: 初期位置リセット + PublishJointAlminumCan実行
    /// </summary>
    public void PublishJointAlminumCanWithReset()
    {
        if (enableDeformationLogging)
            Debug.Log("=== スタートボタン押下: 初期位置リセット + アルミ缶システム開始 ===");
        
        // まず初期位置にリセット
        ResetToInitialPositions();
        
        // 少し待ってからPublishJointAlminumCanを実行
        StartCoroutine(ExecuteAfterReset());
    }


    /// <summary>
    /// リセット後にPublishJointAlminumCanを実行
    /// </summary>
    private IEnumerator ExecuteAfterReset()
    {
        // リセット処理の安定化を待つ
        yield return new WaitForSeconds(0.5f);
        
        // 元のPublishJointAlminumCan処理を実行
        PublishJointAlminumCan();
    }


    /// <summary>
    /// 元のPublishJointAlminumCan処理（既存コードを保持）
    /// </summary>
    public void PublishJointAlminumCan()
    {
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerが見つかりません");
            return;
        }
        
        if (enableDeformationLogging)
            Debug.Log("アルミ缶システム対応のPublishJoints実行");
        
        // まずアルミ缶システムを準備
        PrepareAluminumCanSystem();
        
        // 力制御を有効化（把持動作前に準備）
        if (forceController != null)
        {
            forceController.enabled = true;
            if (enableDeformationLogging)
                Debug.Log("力制御システムを有効化しました");
        }
        
        // 元のPublishJointsを実行
        originalTrajectoryPlanner.PublishJoints();
    }

    private void InitializeAluminumCanSystem()
    {
        // 自動的にコンポーネントを検索
        if (gripperInterface == null)
            gripperInterface = GetComponent<GripperTargetInterface>();
            
        if (target == null)
            target = FindObjectOfType<IntegratedAluminumCan>();
            
        if (forceController == null)
            forceController = GetComponent<SimpleGripForceController>();
        
        // 接続状況をログ出力
        Debug.Log($"アルミ缶システム初期化:");
        Debug.Log($"- GripperInterface: {(gripperInterface != null ? "OK" : "NG")}");
        Debug.Log($"- AluminumCan: {(target != null ? "OK" : "NG")}");
        Debug.Log($"- ForceController: {(forceController != null ? "OK" : "NG")}");
        
        // GripperInterfaceの設定
        if (gripperInterface != null)
        {
            gripperInterface.simpleGripperController = forceController;
            gripperInterface.target = target;
        }
    }
    
    /// <summary>
    /// アルミ缶システム対応の把持開始
    /// 外部から呼び出し可能（例：UI、ボタンイベント）
    /// </summary>
    public void StartGraspWithAluminumCan()
    {
        if (enableDeformationLogging)
            Debug.Log("アルミ缶把持開始 - 力制御システム有効");
        
        // 把持システムの開始
        isCurrentlyGrasping = true;
        
        // 力制御開始
        if (forceController != null)
        {
            forceController.enabled = true;
            if (enableDeformationLogging)
                Debug.Log("力制御システム開始");
        }
        
        // 把持評価の開始
        if (graspEvaluationCoroutine != null)
            StopCoroutine(graspEvaluationCoroutine);
            
        graspEvaluationCoroutine = StartCoroutine(EvaluateGraspAfterDelay());
    }
    
    /// <summary>
    /// アルミ缶システム対応の把持終了
    /// </summary>
    public void StopGraspWithAluminumCan()
    {
        if (enableDeformationLogging)
            Debug.Log("アルミ缶把持終了 - 力制御システム停止");
        
        // 把持システムの停止
        isCurrentlyGrasping = false;
        
        // 力制御停止
        if (forceController != null)
        {
            forceController.enabled = false;
        }
        
        // アルミ缶のリセット
        if (target != null)
        {
            target.ResetCan(); // IntegratedAluminumCanのResetCanメソッドを使用
        }
    }
    
    private IEnumerator EvaluateGraspAfterDelay()
    {
        yield return new WaitForSeconds(graspEvaluationDelay);
        
        if (gripperInterface != null && isCurrentlyGrasping)
        {
            var evaluation = gripperInterface.EvaluateGrasp();
            
            if (enableDeformationLogging)
            {
                Debug.Log($"アルミ缶把持評価結果:");
                Debug.Log($"- 結果: {evaluation.result}");
                Debug.Log($"- 適用力: {evaluation.appliedForce:F2}N");
                Debug.Log($"- 変形度: {evaluation.deformation:F3}");
                Debug.Log($"- つぶれ状態: {evaluation.isBroken}");
                Debug.Log($"- 信頼度: {evaluation.confidence:F2}");
            }
            
            // 将来のEEG/DDPG統合用
            OnGraspEvaluated?.Invoke(evaluation);
        }
    }
    
    // イベント定義
    public System.Action<GraspEvaluation> OnGraspEvaluated;
    
    /// <summary>
    /// 元のPublishJointsにアルミ缶システムを統合したバージョン
    /// </summary>
    public void PublishJointsWithAluminumCan()
    {
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerが見つかりません");
            return;
        }
        
        if (enableDeformationLogging)
            Debug.Log("アルミ缶システム対応のPublishJoints実行");
        
        // まずアルミ缶システムを準備
        PrepareAluminumCanSystem();
        
        // 力制御を有効化（把持動作前に準備）
        if (forceController != null)
        {
            forceController.enabled = true;
            if (enableDeformationLogging)
                Debug.Log("力制御システムを有効化しました");
        }
        
        // 元のPublishJointsを実行
        originalTrajectoryPlanner.PublishJoints();
    }
    
    private void PrepareAluminumCanSystem()
    {
        // アルミ缶システムの準備処理
        if (forceController != null)
        {
            forceController.enabled = false; // 初期状態では無効
        }
        
        if (target != null)
        {
            target.ResetCan(); // アルミ缶をリセット
        }
    }
    
    /// <summary>
    /// 手動での把持テスト用
    /// </summary>
    [ContextMenu("Test Grip With Aluminum Can")]
    public void TestGripWithAluminumCan()
    {
        if (isCurrentlyGrasping)
        {
            StopGraspWithAluminumCan();
        }
        else
        {
            StartGraspWithAluminumCan();
        }
    }
    
    /// <summary>
    /// デバッグ用：現在の状態表示
    /// </summary>
    public void ShowCurrentStatus()
    {
        Debug.Log("=== アルミ缶システム状態 ===");
        Debug.Log($"把持中: {isCurrentlyGrasping}");
        
        if (target != null)
        {
            var state = target.GetCurrentState();
            Debug.Log($"アルミ缶変形: {state.deformation:F3}");
            Debug.Log($"適用力: {state.appliedForce:F2}N");
            Debug.Log($"つぶれ状態: {state.isBroken}");
        }
        
        if (gripperInterface != null)
        {
            Debug.Log($"GripperInterface: 接続済み");
        }
        
        if (forceController != null)
        {
            Debug.Log($"力制御有効: {forceController.enabled}");
        }
    }
    
    /// <summary>
    /// 外部から把持開始/終了を制御するためのパブリックメソッド
    /// </summary>
    public void OnGraspPhaseStarted()
    {
        StartGraspWithAluminumCan();
    }
    
    public void OnGraspPhaseEnded()
    {
        StopGraspWithAluminumCan();
    }
    
    /// <summary>
    /// グリッパーの動作を監視して自動的に力制御を管理
    /// </summary>
    private IEnumerator MonitorGripperMovement()
    {
        // グリッパーのArticulationBodyを探す
        ArticulationBody leftGripper = null;
        ArticulationBody rightGripper = null;
        
        yield return new WaitForSeconds(0.1f); // 初期化待ち
        
        // グリッパーを検索
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
        
        float previousLeftTarget = leftGripper.xDrive.target;
        float previousRightTarget = rightGripper.xDrive.target;
        bool wasGrasping = false;
        
        while (true)
        {
            yield return new WaitForSeconds(0.1f);
            
            float currentLeftTarget = leftGripper.xDrive.target;
            float currentRightTarget = rightGripper.xDrive.target;
            
            // グリッパーが閉じる動作を検出（目標値の変化で判定）
            bool isCurrentlyGrasping = (currentLeftTarget < -0.005f && currentRightTarget > 0.005f);
            
            if (isCurrentlyGrasping && !wasGrasping)
            {
                // 把持開始を検出
                if (enableDeformationLogging)
                    Debug.Log("グリッパー閉じ動作を検出 - アルミ缶力制御開始");
                StartGraspWithAluminumCan();
            }
            else if (!isCurrentlyGrasping && wasGrasping)
            {
                // 把持終了を検出
                if (enableDeformationLogging)
                    Debug.Log("グリッパー開き動作を検出 - アルミ缶力制御終了");
                StopGraspWithAluminumCan();
            }
            
            wasGrasping = isCurrentlyGrasping;
            previousLeftTarget = currentLeftTarget;
            previousRightTarget = currentRightTarget;
        }
    }
}