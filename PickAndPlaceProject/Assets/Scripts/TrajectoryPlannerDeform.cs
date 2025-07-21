using System.Collections;
using UnityEngine;

/// <summary>
/// TrajectoryPlannerに変形システムを統合するコンポーネント
/// 既存のTrajectoryPlannerと同じGameObjectにアタッチして使用
/// </summary>
public class TrajectoryPlannerDeform : MonoBehaviour
{
    [Header("変形システム連携")]
    public GripperTargetInterface gripperInterface;
    public IntegratedAluminumCan target;

    public SimpleGripForceController forceController;
    
    [Header("把持設定")]
    public float graspEvaluationDelay = 1f;
    public bool enableDeformationLogging = true;
    
    // 元のTrajectoryPlannerへの参照
    private TrajectoryPlanner originalTrajectoryPlanner;
    
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
        
        InitializeDeformationSystem();
        
        // グリッパーの動作監視を開始
        StartCoroutine(MonitorGripperMovement());
    }
    
    private void InitializeDeformationSystem()
    {
        // 自動的にコンポーネントを検索
        if (gripperInterface == null)
            gripperInterface = GetComponent<GripperTargetInterface>();
            
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
            
        if (forceController == null)
            forceController = GetComponent<SimpleGripForceController>();
        
        // 接続状況をログ出力
        Debug.Log($"変形システム初期化:");
        Debug.Log($"- GripperInterface: {(gripperInterface != null ? "OK" : "NG")}");
        Debug.Log($"- DeformableTarget: {(target != null ? "OK" : "NG")}");
        Debug.Log($"- ForceController: {(forceController != null ? "OK" : "NG")}");
        
        // GripperInterfaceの設定
        if (gripperInterface != null)
        {
            gripperInterface.simpleGripperController = forceController;
            gripperInterface.target = target;
        }
    }
    
    /// <summary>
    /// 変形システム対応の把持開始
    /// 外部から呼び出し可能（例：UI、ボタンイベント）
    /// </summary>
    public void StartGraspWithDeformation()
    {
        if (enableDeformationLogging)
            Debug.Log("把持開始 - 変形システム有効");
        
        // 変形システムの把持開始
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
    /// 変形システム対応の把持終了
    /// </summary>
    public void StopGraspWithDeformation()
    {
        if (enableDeformationLogging)
            Debug.Log("把持終了 - 変形システム停止");
        
        // 変形システムの停止
        isCurrentlyGrasping = false;
        
        // 力制御停止
        if (forceController != null)
        {
            forceController.enabled = false;
        }
        
        // ターゲットの変形リセット
        if (target != null)
        {
            target.ResetObject();
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
                Debug.Log($"把持評価結果:");
                Debug.Log($"- 結果: {evaluation.result}");
                Debug.Log($"- 適用力: {evaluation.appliedForce:F2}N");
                Debug.Log($"- 変形度: {evaluation.deformation:F3}");
                Debug.Log($"- 破損: {evaluation.isBroken}");
                Debug.Log($"- 信頼度: {evaluation.confidence:F2}");
            }
            
            // 将来のEEG/DDPG統合用
            OnGraspEvaluated?.Invoke(evaluation);
        }
    }
    
    // イベント定義（GraspEvaluationクラスはGripperTargetInterface.csで定義されている）
    public System.Action<GraspEvaluation> OnGraspEvaluated;
    
    /// <summary>
    /// 元のPublishJointsに変形システムを統合したバージョン
    /// </summary>
    public void PublishJointsWithDeformation()
    {
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerが見つかりません");
            return;
        }
        
        if (enableDeformationLogging)
            Debug.Log("変形システム対応のPublishJoints実行");
        
        // まず変形システムを準備
        PrepareDeformationSystem();
        
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
    
    private void PrepareDeformationSystem()
    {
        // 変形システムの準備処理
        if (forceController != null)
        {
            forceController.enabled = false; // 初期状態では無効
        }
        
        if (target != null)
        {
            target.ResetObject();
        }
    }
    
    /// <summary>
    /// 手動での把持テスト用
    /// </summary>
    [ContextMenu("Test Grip With Deformation")]
    public void TestGripWithDeformation()
    {
        if (isCurrentlyGrasping)
        {
            StopGraspWithDeformation();
        }
        else
        {
            StartGraspWithDeformation();
        }
    }
    
    /// <summary>
    /// デバッグ用：現在の状態表示
    /// </summary>
    public void ShowCurrentStatus()
    {
        Debug.Log("=== 変形システム状態 ===");
        Debug.Log($"把持中: {isCurrentlyGrasping}");
        
        if (target != null)
        {
            var state = target.GetCurrentState();
            Debug.Log($"ターゲット変形: {state.deformation:F3}");
            Debug.Log($"適用力: {state.appliedForce:F2}N");
            Debug.Log($"材質: {state.materialType}");
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
        StartGraspWithDeformation();
    }
    
    public void OnGraspPhaseEnded()
    {
        StopGraspWithDeformation();
    }
    
    /// <summary>
    /// グリッパーの動作を監視して自動的に力制御を管理
    /// </summary>
    private IEnumerator MonitorGripperMovement()
    {
        // グリッパーのArticulationBodyを探す
        ArticulationBody leftGripper = null;
        ArticulationBody rightGripper = null;
        
        yield return new WaitForSeconds(1f); // 初期化待ち
        
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
                    Debug.Log("グリッパー閉じ動作を検出 - 力制御開始");
                StartGraspWithDeformation();
            }
            else if (!isCurrentlyGrasping && wasGrasping)
            {
                // 把持終了を検出
                if (enableDeformationLogging)
                    Debug.Log("グリッパー開き動作を検出 - 力制御終了");
                StopGraspWithDeformation();
            }
            
            wasGrasping = isCurrentlyGrasping;
            previousLeftTarget = currentLeftTarget;
            previousRightTarget = currentRightTarget;
        }
    }
}