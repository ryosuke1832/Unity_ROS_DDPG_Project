// AutoEpisodeManager.cs
using System.Collections;
using UnityEngine;

/// <summary>
/// A2C学習用の自動エピソード管理システム
/// ロボットの動作完了を検出して自動的に次のエピソードを開始
/// </summary>
public class AutoEpisodeManager : MonoBehaviour
{
    [Header("コンポーネント参照")]
    public TrajectoryPlannerDeform trajectoryPlanner;
    public AluminumCanA2CClient a2cClient;
    public IntegratedAluminumCan aluminumCan;
    public GameObject niryoOneRobot; // 直接ロボットオブジェクトを参照
    public SimpleGripForceController gripForceController; // 把持力制御
    
    [Header("エピソード設定")]
    [Range(1f, 10f)]
    public float episodeDuration = 30f; // エピソードの最大時間（秒）
    
    [Range(0.5f, 5f)]
    public float resetDelay = 2f; // リセット後の待機時間
    
    [Range(0.1f, 2f)]
    public float completionCheckInterval = 0.5f; // 完了チェックの間隔
    
    [Header("把持力ランダム化")]
    public bool enableRandomGripForce = true;
    [Range(8f, 30f)]
    public float minGripForce = 8f;
    [Range(8f, 30f)]
    public float maxGripForce = 20f;
    public bool logGripForceChanges = true;
    
    [Header("自動実行設定")]
    public bool enableAutoEpisodes = true;
    public bool startOnAwake = true;
    public int maxEpisodesPerSession = 1000; // セッション当たりの最大エピソード数
    
    [Header("ロボット動作判定")]
    [Range(0.001f, 0.1f)]
    public float movementThreshold = 0.01f; // 停止判定の閾値
    
    [Range(1f, 10f)]
    public float stoppedTimeThreshold = 3f; // 停止と判定する時間
    
    [Header("デバッグ")]
    public bool enableDebugLogs = true;
    public bool showEpisodeStats = true;
    
    // 内部状態
    private EpisodeState currentState = EpisodeState.Idle;
    private int currentEpisodeNumber = 0;
    private float episodeStartTime = 0f;
    private float lastMovementTime = 0f;
    private Vector3 lastRobotPosition = Vector3.zero;
    private bool isRobotMoving = false;
    
    // 統計
    private int successfulEpisodes = 0;
    private int failedEpisodes = 0;
    private float totalEpisodeTime = 0f;
    
    // 把持力統計
    private float currentEpisodeGripForce = 0f;
    private System.Collections.Generic.List<float> usedGripForces = new System.Collections.Generic.List<float>();
    
    // 状態管理
    public enum EpisodeState
    {
        Idle,           // 待機中
        Starting,       // エピソード開始中
        Running,        // エピソード実行中
        Completing,     // エピソード完了処理中
        Resetting,      // リセット中
        Finished        // セッション終了
    }
    
    // イベント
    public System.Action<int> OnEpisodeStarted;
    public System.Action<int, bool> OnEpisodeCompleted; // episodeNumber, wasSuccessful
    public System.Action OnSessionCompleted;
    
    void Start()
    {
        InitializeComponents();
        
        if (startOnAwake && enableAutoEpisodes)
        {
            StartAutoEpisodes();
        }
    }
    
    void Update()
    {
        if (!enableAutoEpisodes) return;
        
        UpdateEpisodeState();
        UpdateMovementDetection();
        
        // デバッグUI表示
        if (enableDebugLogs && showEpisodeStats)
        {
            UpdateDebugDisplay();
        }
    }
    
    #region 初期化
    
    void InitializeComponents()
    {
        // コンポーネントの自動検索
        if (trajectoryPlanner == null)
            trajectoryPlanner = FindObjectOfType<TrajectoryPlannerDeform>();
            
        if (a2cClient == null)
            a2cClient = FindObjectOfType<AluminumCanA2CClient>();
            
        if (aluminumCan == null)
            aluminumCan = FindObjectOfType<IntegratedAluminumCan>();
        
        // 把持力制御の自動検索
        if (gripForceController == null)
            gripForceController = FindObjectOfType<SimpleGripForceController>();
        
        // ロボットオブジェクトの自動検索
        if (niryoOneRobot == null)
        {
            // NiryoOneという名前のGameObjectを検索
            niryoOneRobot = GameObject.Find("NiryoOne");
            
            // 見つからない場合はTrajectoryPlannerコンポーネントから取得を試行
            if (niryoOneRobot == null)
            {
                TrajectoryPlanner originalPlanner = FindObjectOfType<TrajectoryPlanner>();
                if (originalPlanner != null)
                {
                    // リフレクションでNiryoOneプロパティにアクセス
                    var niryoOneProperty = originalPlanner.GetType().GetProperty("NiryoOne");
                    if (niryoOneProperty != null)
                    {
                        niryoOneRobot = niryoOneProperty.GetValue(originalPlanner) as GameObject;
                    }
                }
            }
        }
        
        // 初期位置記録
        if (niryoOneRobot != null)
        {
            lastRobotPosition = niryoOneRobot.transform.position;
        }
        
        // 検索結果確認
        bool allComponentsFound = trajectoryPlanner != null && a2cClient != null && aluminumCan != null && niryoOneRobot != null;
        bool gripForceAvailable = gripForceController != null;
        
        if (enableDebugLogs)
        {
            Debug.Log("=== AutoEpisodeManager 初期化 ===");
            Debug.Log($"TrajectoryPlanner: {(trajectoryPlanner != null ? "✅" : "❌")}");
            Debug.Log($"A2CClient: {(a2cClient != null ? "✅" : "❌")}");
            Debug.Log($"AluminumCan: {(aluminumCan != null ? "✅" : "❌")}");
            Debug.Log($"NiryoOne Robot: {(niryoOneRobot != null ? "✅" : "❌")} {(niryoOneRobot != null ? niryoOneRobot.name : "Not Found")}");
            Debug.Log($"GripForceController: {(gripForceController != null ? "✅" : "❌")}");
            Debug.Log($"ランダム把持力: {(enableRandomGripForce && gripForceAvailable ? "有効" : "無効")}");
            if (enableRandomGripForce && gripForceAvailable)
            {
                Debug.Log($"把持力範囲: {minGripForce:F1}N - {maxGripForce:F1}N");
            }
            Debug.Log($"自動エピソード: {(allComponentsFound ? "準備完了" : "コンポーネント不足")}");
        }
        
        if (!allComponentsFound)
        {
            enableAutoEpisodes = false;
            Debug.LogError("必要なコンポーネントが見つかりません。自動エピソードを無効化します。");
        }
        
        // ランダム把持力が有効だが制御器がない場合の警告
        if (enableRandomGripForce && !gripForceAvailable)
        {
            Debug.LogWarning("ランダム把持力が有効ですが、SimpleGripForceControllerが見つかりません。");
            enableRandomGripForce = false;
        }
    }
    #endregion

    
    
    #region エピソード制御
    
    public void StartAutoEpisodes()
    {
        if (!enableAutoEpisodes) return;
        
        currentState = EpisodeState.Starting;
        currentEpisodeNumber = 0;
        successfulEpisodes = 0;
        failedEpisodes = 0;
        totalEpisodeTime = 0f;
        
        if (enableDebugLogs)
        {
            Debug.Log("🚀 自動エピソード開始！");
            Debug.Log($"最大エピソード数: {maxEpisodesPerSession}");
            Debug.Log($"エピソード時間: {episodeDuration}秒");
        }
        
        StartCoroutine(ExecuteEpisodeLoop());
    }
    
    public void StopAutoEpisodes()
    {
        enableAutoEpisodes = false;
        currentState = EpisodeState.Finished;
        
        if (enableDebugLogs)
        {
            Debug.Log("⏹️ 自動エピソード停止");
            ShowFinalStatistics();
        }
        
        OnSessionCompleted?.Invoke();
    }
    
    IEnumerator ExecuteEpisodeLoop()
    {
        while (enableAutoEpisodes && currentEpisodeNumber < maxEpisodesPerSession)
        {
            // 新しいエピソード開始
            yield return StartCoroutine(StartNewEpisode());
            
            // エピソード実行中
            yield return StartCoroutine(RunEpisode());
            
            // エピソード完了処理
            yield return StartCoroutine(CompleteEpisode());
            
            // リセット処理
            yield return StartCoroutine(ResetForNextEpisode());
        }
        
        // セッション完了
        currentState = EpisodeState.Finished;
        if (enableDebugLogs)
        {
            Debug.Log("🏁 すべてのエピソードが完了しました");
            ShowFinalStatistics();
        }
        
        OnSessionCompleted?.Invoke();
    }
    
    IEnumerator StartNewEpisode()
    {
        currentState = EpisodeState.Starting;
        currentEpisodeNumber++;
        episodeStartTime = Time.time;
        lastMovementTime = Time.time;
        
        // ランダム把持力の設定
        if (enableRandomGripForce && gripForceController != null)
        {
            SetRandomGripForce();
        }
        
        if (enableDebugLogs)
        {
            Debug.Log($"📋 エピソード {currentEpisodeNumber} 開始");
            if (enableRandomGripForce && gripForceController != null)
            {
                Debug.Log($"🎲 把持力: {currentEpisodeGripForce:F2}N");
            }
        }
        
        // A2Cクライアントにリセット通知
        if (a2cClient != null)
        {
            a2cClient.SendReset();
        }
        
        // 少し待機してからロボット動作開始
        yield return new WaitForSeconds(0.5f);
        
        // ロボット動作開始
        if (trajectoryPlanner != null)
        {
            trajectoryPlanner.PublishJointAlminumCan(); // 正しいメソッド名
        }
        
        OnEpisodeStarted?.Invoke(currentEpisodeNumber);
        
        yield return new WaitForSeconds(0.5f); // 動作開始の安定化待機
    }
    
    IEnumerator RunEpisode()
    {
        currentState = EpisodeState.Running;
        
        float episodeTime = 0f;
        bool episodeEnded = false;
        
        while (episodeTime < episodeDuration && !episodeEnded)
        {
            episodeTime = Time.time - episodeStartTime;
            
            // エピソード終了条件をチェック
            episodeEnded = CheckEpisodeEndConditions();
            
            yield return new WaitForSeconds(completionCheckInterval);
        }
        
        if (enableDebugLogs)
        {
            string endReason = episodeTime >= episodeDuration ? "時間切れ" : "完了条件達成";
            Debug.Log($"⏱️ エピソード実行終了: {endReason} ({episodeTime:F1}秒)");
        }
    }
    
    IEnumerator CompleteEpisode()
    {
        currentState = EpisodeState.Completing;
        
        float episodeTime = Time.time - episodeStartTime;
        totalEpisodeTime += episodeTime;
        
        // 成功/失敗の判定
        bool wasSuccessful = DetermineEpisodeSuccess();
        
        if (wasSuccessful)
            successfulEpisodes++;
        else
            failedEpisodes++;
        
        // A2Cクライアントにエピソード終了通知
        if (a2cClient != null)
        {
            a2cClient.SendEpisodeEnd();
        }
        
        // 統計表示
        if (enableDebugLogs)
        {
            float successRate = (float)successfulEpisodes / currentEpisodeNumber * 100f;
            Debug.Log($"🏁 エピソード {currentEpisodeNumber} 完了");
            Debug.Log($"   結果: {(wasSuccessful ? "✅成功" : "❌失敗")}");
            Debug.Log($"   時間: {episodeTime:F2}秒");
            if (enableRandomGripForce && currentEpisodeGripForce > 0)
            {
                Debug.Log($"   把持力: {currentEpisodeGripForce:F2}N");
            }
            Debug.Log($"   成功率: {successRate:F1}% ({successfulEpisodes}/{currentEpisodeNumber})");
        }
        
        OnEpisodeCompleted?.Invoke(currentEpisodeNumber, wasSuccessful);
        
        yield return new WaitForSeconds(0.5f);
    }
    
    IEnumerator ResetForNextEpisode()
    {
        currentState = EpisodeState.Resetting;
        
        if (enableDebugLogs)
        {
            Debug.Log("🔄 次のエピソードに向けてリセット中...");
        }
        
        // システムリセット
        if (trajectoryPlanner != null)
        {
            trajectoryPlanner.ResetToInitialPositions();
        }
        
        // アルミ缶リセット
        if (aluminumCan != null)
        {
            aluminumCan.ResetCan();
        }
        
        // リセット完了まで待機
        yield return new WaitForSeconds(resetDelay);
        
        // ロボット位置をリセット
        if (niryoOneRobot != null)
        {
            lastRobotPosition = niryoOneRobot.transform.position;
        }
        
        if (enableDebugLogs)
        {
            Debug.Log("✅ リセット完了");
        }
    }
    
    #endregion
    
    #region 状態判定
    
    void UpdateEpisodeState()
    {
        // 現在のステートに応じた処理は ExecuteEpisodeLoop で管理
    }
    
    void UpdateMovementDetection()
    {
        if (niryoOneRobot == null) return;
        
        Vector3 currentPosition = niryoOneRobot.transform.position;
        float distance = Vector3.Distance(currentPosition, lastRobotPosition);
        
        if (distance > movementThreshold)
        {
            isRobotMoving = true;
            lastMovementTime = Time.time;
            lastRobotPosition = currentPosition;
        }
        else
        {
            float timeSinceLastMovement = Time.time - lastMovementTime;
            isRobotMoving = timeSinceLastMovement < stoppedTimeThreshold;
        }
    }
    
    bool CheckEpisodeEndConditions()
    {
        // 1. アルミ缶がつぶれた場合
        if (aluminumCan != null && aluminumCan.IsBroken)
        {
            return true;
        }
        
        // 2. ロボットが長時間停止している場合
        if (!isRobotMoving && (Time.time - lastMovementTime) > stoppedTimeThreshold)
        {
            return true;
        }
        
        // 3. その他のタスク完了条件
        // TODO: 必要に応じて追加の完了条件を実装
        
        return false;
    }
    
    bool DetermineEpisodeSuccess()
    {
        // 成功条件：アルミ缶がつぶれていない
        if (aluminumCan != null)
        {
            return !aluminumCan.IsBroken;
        }
        
        return false;
    }
    
    #endregion
    
    #region UI・デバッグ
    
    void UpdateDebugDisplay()
    {
        // この関数は Update で呼ばれるため、フレームレート考慮
        if (Time.frameCount % 60 == 0) // 1秒ごとに更新
        {
            // 統計情報のログ出力は必要に応じて
        }
    }
    
    void ShowFinalStatistics()
    {
        float avgEpisodeTime = totalEpisodeTime / Mathf.Max(1, currentEpisodeNumber);
        float finalSuccessRate = (float)successfulEpisodes / Mathf.Max(1, currentEpisodeNumber) * 100f;
        
        Debug.Log(new string('=', 50));
        Debug.Log("📊 最終統計");
        Debug.Log(new string('-', 50));
        Debug.Log($"総エピソード数: {currentEpisodeNumber}");
        Debug.Log($"成功エピソード: {successfulEpisodes}");
        Debug.Log($"失敗エピソード: {failedEpisodes}");
        Debug.Log($"最終成功率: {finalSuccessRate:F2}%");
        Debug.Log($"平均エピソード時間: {avgEpisodeTime:F2}秒");
        Debug.Log($"総実行時間: {totalEpisodeTime:F2}秒");
        
        // 把持力統計
        if (enableRandomGripForce && usedGripForces.Count > 0)
        {
            var gripStats = GetGripForceStatistics();
            Debug.Log($"把持力統計:");
            Debug.Log($"- 平均把持力: {gripStats.averageForce:F2}N");
            Debug.Log($"- 使用範囲: {gripStats.minUsedForce:F2}N - {gripStats.maxUsedForce:F2}N");
            Debug.Log($"- 設定回数: {gripStats.totalForceSettings}回");
        }
        
        Debug.Log(new string('=', 50));
    }
    
    // GUI表示（ゲーム実行中の情報表示）
    void OnGUI()
    {
        if (!enableAutoEpisodes || !showEpisodeStats) return;
        
        GUIStyle style = new GUIStyle();
        style.fontSize = 14;
        style.normal.textColor = Color.white;
        
        float y = 10f;
        float lineHeight = 20f;
        
        GUI.Label(new Rect(10, y, 400, lineHeight), $"自動エピソード実行中", style);
        y += lineHeight;
        
        GUI.Label(new Rect(10, y, 400, lineHeight), $"状態: {currentState}", style);
        y += lineHeight;
        
        GUI.Label(new Rect(10, y, 400, lineHeight), $"エピソード: {currentEpisodeNumber}/{maxEpisodesPerSession}", style);
        y += lineHeight;
        
        if (currentState == EpisodeState.Running)
        {
            float episodeTime = Time.time - episodeStartTime;
            GUI.Label(new Rect(10, y, 400, lineHeight), $"経過時間: {episodeTime:F1}秒", style);
            y += lineHeight;
        }
        
        if (currentEpisodeNumber > 0)
        {
            float successRate = (float)successfulEpisodes / currentEpisodeNumber * 100f;
            GUI.Label(new Rect(10, y, 400, lineHeight), $"成功率: {successRate:F1}% ({successfulEpisodes}/{currentEpisodeNumber})", style);
            y += lineHeight;
        }
        
        // 把持力情報
        if (enableRandomGripForce && currentEpisodeGripForce > 0)
        {
            GUI.Label(new Rect(10, y, 400, lineHeight), $"現在の把持力: {currentEpisodeGripForce:F1}N", style);
            y += lineHeight;
        }
        
        GUI.Label(new Rect(10, y, 400, lineHeight), $"ロボット移動中: {(isRobotMoving ? "Yes" : "No")}", style);
        y += lineHeight;
        
        if (aluminumCan != null)
        {
            GUI.Label(new Rect(10, y, 400, lineHeight), $"缶の状態: {(aluminumCan.IsBroken ? "つぶれた" : "正常")}", style);
        }
    }
    
    #endregion
    
    #region 把持力ランダム化
    
    /// <summary>
    /// ランダムな把持力を設定
    /// </summary>
    void SetRandomGripForce()
    {
        if (gripForceController == null) return;
        
        // 範囲内でランダムに生成
        currentEpisodeGripForce = Random.Range(minGripForce, maxGripForce);
        gripForceController.baseGripForce = currentEpisodeGripForce;
        
        // 統計に追加
        usedGripForces.Add(currentEpisodeGripForce);
        
        if (logGripForceChanges)
        {
            Debug.Log($"🎲 把持力設定: {currentEpisodeGripForce:F2}N (範囲: {minGripForce:F1}-{maxGripForce:F1}N)");
            
            // アルミ缶の変形閾値と比較
            if (aluminumCan != null)
            {
                float threshold = aluminumCan.deformationThreshold;
                bool willCrush = currentEpisodeGripForce > threshold;
                Debug.Log($"   変形閾値: {threshold:F2}N -> {(willCrush ? "⚠️つぶれる可能性" : "✅安全範囲")}");
            }
        }
    }
    
    /// <summary>
    /// 把持力統計の取得
    /// </summary>
    public GripForceStatistics GetGripForceStatistics()
    {
        if (usedGripForces.Count == 0)
        {
            return new GripForceStatistics();
        }
        
        float avgForce = 0f;
        float minUsed = float.MaxValue;
        float maxUsed = float.MinValue;
        
        foreach (float force in usedGripForces)
        {
            avgForce += force;
            if (force < minUsed) minUsed = force;
            if (force > maxUsed) maxUsed = force;
        }
        
        avgForce /= usedGripForces.Count;
        
        return new GripForceStatistics
        {
            averageForce = avgForce,
            minUsedForce = minUsed,
            maxUsedForce = maxUsed,
            totalForceSettings = usedGripForces.Count,
            currentForce = currentEpisodeGripForce
        };
    }
    
    /// <summary>
    /// 把持力範囲の動的調整
    /// </summary>
    public void AdjustGripForceRange(float newMin, float newMax)
    {
        if (newMin >= newMax || newMin < 0 || newMax > 50)
        {
            Debug.LogWarning("無効な把持力範囲です。調整をスキップします。");
            return;
        }
        
        minGripForce = newMin;
        maxGripForce = newMax;
        
        if (enableDebugLogs)
        {
            Debug.Log($"把持力範囲を調整: {minGripForce:F1}N - {maxGripForce:F1}N");
        }
    }
    
    #endregion

    #region 公開メソッド
    
    /// <summary>
    /// 手動でエピソードを開始
    /// </summary>
    public void StartSingleEpisode()
    {
        if (currentState == EpisodeState.Idle || currentState == EpisodeState.Finished)
        {
            maxEpisodesPerSession = currentEpisodeNumber + 1;
            StartAutoEpisodes();
        }
    }
    
    /// <summary>
    /// 現在のエピソードを強制終了
    /// </summary>
    public void ForceEndCurrentEpisode()
    {
        if (currentState == EpisodeState.Running)
        {
            StopCoroutine(ExecuteEpisodeLoop());
            StartCoroutine(CompleteEpisode());
        }
    }
    
    /// <summary>
    /// 統計情報の取得
    /// </summary>
    public EpisodeStatistics GetStatistics()
    {
        return new EpisodeStatistics
        {
            totalEpisodes = currentEpisodeNumber,
            successfulEpisodes = successfulEpisodes,
            failedEpisodes = failedEpisodes,
            successRate = currentEpisodeNumber > 0 ? (float)successfulEpisodes / currentEpisodeNumber * 100f : 0f,
            averageEpisodeTime = currentEpisodeNumber > 0 ? totalEpisodeTime / currentEpisodeNumber : 0f,
            totalTime = totalEpisodeTime
        };
    }
    
    #endregion
}


/// <summary>
/// エピソード統計情報
/// </summary>
[System.Serializable]
public class EpisodeStatistics
{
    public int totalEpisodes;
    public int successfulEpisodes;
    public int failedEpisodes;
    public float successRate;
    public float averageEpisodeTime;
    public float totalTime;
}

/// <summary>
/// 把持力統計情報
/// </summary>
[System.Serializable]
public class GripForceStatistics
{
    public float averageForce;
    public float minUsedForce;
    public float maxUsedForce;
    public int totalForceSettings;
    public float currentForce;
}