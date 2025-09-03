// AutoEpisodeManager.cs
using System.Collections;
using UnityEngine;

/// <summary>
/// A2C学習用の自動エピソード管理システム
/// ロボットの動作完了を検出して自動的に次のエピソードを開始
/// TCP経由で受信した把持力指令を次のエピソードに適用する機能を追加
/// </summary>
public class AutoEpisodeManager : MonoBehaviour
{
    [Header("コンポーネント参照")]
    public TrajectoryPlannerDeform trajectoryPlanner;
    public AluminumCanA2CClient a2cClient;
    public IntegratedAluminumCan aluminumCan;
    public GameObject niryoOneRobot; // 直接ロボットオブジェクトを参照
    public SimpleGripForceController gripForceController; // 把持力制御
    public GripperTargetInterface gripperInterface;
    
    [Header("エピソード設定")]
    [Range(1f, 10f)]
    public float episodeDuration = 2f; // エピソードの最大時間（秒）
    
    [Range(0.5f, 5f)]
    public float resetDelay = 2f; // リセット後の待機時間
    
    [Range(0.1f, 2f)]
    public float completionCheckInterval = 0.5f; // 完了チェックの間隔
    
    [Header("🔥 TCP把持力制御")]
    [SerializeField] private bool enableTcpGripForceControl = false;
    [Range(1f, 30f)]
    public float tcpCommandWaitTimeout = 2f; // TCP指令待機のタイムアウト
    public bool waitForTcpCommandBeforeStart = true; // エピソード開始前にTCP指令を待機
    public bool useTcpForceWhenAvailable = true; // TCP指令が利用可能な場合に優先使用
    
    [Header("把持力ランダム化")]
    public bool enableRandomGripForce = true;
    [Range(2f, 30f)]
    public float minGripForce = 2f;
    [Range(2f, 30f)]
    public float maxGripForce = 30f;
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
    private Vector3 initialCanPosition = Vector3.zero;
    private Coroutine episodeLoopCoroutine;

    // 🔥 TCP把持力制御用の追加フィールド
    private float? pendingTcpGripForce = null; // TCPで受信した把持力指令
    private bool isWaitingForTcpCommand = false;
    private float tcpCommandWaitStartTime = 0f;
    private GripForceSource currentGripForceSource = GripForceSource.Random;
    // エピソード結果の一時保管（リセット直前に送信）
    private bool? pendingEpisodeResult = null;
    
    // 統計
    private int successfulEpisodes = 0;
    private int failedEpisodes = 0;
    private float totalEpisodeTime = 0f;
    
    // 把持力統計
    private float currentEpisodeGripForce = 0f;
    private System.Collections.Generic.List<float> usedGripForces = new System.Collections.Generic.List<float>();
    private System.Collections.Generic.List<GripForceSource> gripForceSources = new System.Collections.Generic.List<GripForceSource>();
    
    // 🔥 TCP統計
    private int tcpCommandsReceived = 0;
    private int tcpCommandsUsed = 0;
    private int tcpTimeouts = 0;
    
    // 状態管理
    public enum EpisodeState
    {
        Idle,           // 待機中
        WaitingForTcp,  // 🔥 新規追加：TCP指令待機中
        Starting,       // エピソード開始中
        Running,        // エピソード実行中
        Completing,     // エピソード完了処理中
        Resetting,      // リセット中
        Finished        // セッション終了
    }
    
    // 🔥 把持力のソース種別
    public enum GripForceSource
    {
        Random,    // ランダム生成
        Tcp,       // TCP指令
        Default    // デフォルト値
    }
    
    // イベント
    public System.Action<int> OnEpisodeStarted;
    public System.Action<int, bool> OnEpisodeCompleted; // episodeNumber, wasSuccessful
    public System.Action OnSessionCompleted;
    public System.Action<float, GripForceSource> OnGripForceApplied; // 🔥 新規追加
    
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
        
        // 🔥 TCP指令の処理
        HandleTcpGripForceCommands();
        
        // デバッグUI表示
        if (enableDebugLogs && showEpisodeStats)
        {
            UpdateDebugDisplay();
        }
    }
    
    #region 初期化
    
    void InitializeComponents()
    {
        // 既存の初期化処理...
        
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
        
        if (gripperInterface == null)
            gripperInterface = FindObjectOfType<GripperTargetInterface>();
        
        // ロボットオブジェクトの自動検索
        if (niryoOneRobot == null)
        {
            niryoOneRobot = GameObject.Find("NiryoOne");
            
            if (niryoOneRobot == null)
            {
                TrajectoryPlanner originalPlanner = FindObjectOfType<TrajectoryPlanner>();
                if (originalPlanner != null)
                {
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
        
        // 🔥 A2CClientとの連携設定
        SetupA2CClientIntegration();
        
        // 検証とログ出力
        bool allComponentsFound = trajectoryPlanner != null && a2cClient != null && aluminumCan != null && niryoOneRobot != null;
        bool gripForceAvailable = gripForceController != null;
        bool tcpControlReady = enableTcpGripForceControl && a2cClient != null;
        
        if (enableDebugLogs)
        {
            Debug.Log("=== AutoEpisodeManager 初期化 ===");
            Debug.Log($"TrajectoryPlanner: {(trajectoryPlanner != null ? "✅" : "❌")}");
            Debug.Log($"A2CClient: {(a2cClient != null ? "✅" : "❌")}");
            Debug.Log($"AluminumCan: {(aluminumCan != null ? "✅" : "❌")}");
            Debug.Log($"NiryoOne Robot: {(niryoOneRobot != null ? "✅" : "❌")} {(niryoOneRobot != null ? niryoOneRobot.name : "Not Found")}");
            Debug.Log($"GripForceController: {(gripForceController != null ? "✅" : "❌")}");
            Debug.Log($"🔥 TCP把持力制御: {(tcpControlReady ? "✅有効" : "❌無効")}");
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
        
        // TCP制御の妥当性確認
        if (enableTcpGripForceControl && !tcpControlReady)
        {
            Debug.LogWarning("TCP把持力制御が有効ですが、A2CClientが見つかりません。TCP制御を無効化します。");
            enableTcpGripForceControl = false;
        }
        
        // ランダム把持力が有効だが制御器がない場合の警告
        if (enableRandomGripForce && !gripForceAvailable)
        {
            Debug.LogWarning("ランダム把持力が有効ですが、SimpleGripForceControllerが見つかりません。");
            enableRandomGripForce = false;
        }
    }
    
    // 🔥 A2CClientとの連携設定
    void SetupA2CClientIntegration()
    {
        if (a2cClient == null) return;
        
        // TCP把持力指令受信のための直接メソッド呼び出し設定
        // A2CClientから直接OnTcpGripForceCommandReceivedを呼び出してもらう
        
        if (enableDebugLogs)
        {
            Debug.Log("🔥 A2CClientとの連携を設定しました");
            Debug.Log("注意: A2CClientから直接OnTcpGripForceCommandReceived()を呼び出してください");
        }
    }
    
    #endregion
    
    #region 🔥 TCP把持力制御
    
    /// <summary>
    /// TCP経由で把持力指令を受信した場合に呼び出される
    /// A2CClientから呼び出されることを想定
    /// </summary>
    public void OnTcpGripForceCommandReceived(float gripForce)
    {
        tcpCommandsReceived++;
        pendingTcpGripForce = gripForce;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔥 TCP把持力指令受信: {gripForce:F2}N");
        }
        
        // 現在TCP指令を待機中の場合、待機を解除
        if (isWaitingForTcpCommand)
        {
            CompleteTcpCommandWait(false);
        }
    }
    
    /// <summary>
    /// TCP指令の処理を更新
    /// </summary>
    void HandleTcpGripForceCommands()
    {
        if (!enableTcpGripForceControl) return;
        
        // TCP指令待機のタイムアウトチェック
        if (isWaitingForTcpCommand)
        {
            float waitTime = Time.time - tcpCommandWaitStartTime;
            if (waitTime > tcpCommandWaitTimeout)
            {
                CompleteTcpCommandWait(true); // タイムアウト
            }
        }
    }
    
    /// <summary>
    /// TCP指令の待機を開始
    /// </summary>
    void StartTcpCommandWait()
    {
        if (!enableTcpGripForceControl || !waitForTcpCommandBeforeStart) return;
        
        currentState = EpisodeState.WaitingForTcp;
        isWaitingForTcpCommand = true;
        tcpCommandWaitStartTime = Time.time;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔥 TCP把持力指令を待機中... (タイムアウト: {tcpCommandWaitTimeout}秒)");
        }
    }
    
    /// <summary>
    /// TCP指令待機の完了
    /// </summary>
    void CompleteTcpCommandWait(bool wasTimeout)
    {
        isWaitingForTcpCommand = false;
        
        if (wasTimeout)
        {
            tcpTimeouts++;
            if (enableDebugLogs)
            {
                Debug.LogWarning($"⏰ TCP指令待機タイムアウト (統計: {tcpTimeouts}回目)");
            }
        }
        else
        {
            if (enableDebugLogs)
            {
                Debug.Log($"✅ TCP指令受信完了");
            }
        }
    }
    
    /// <summary>
    /// 把持力を決定して適用
    /// TCP指令 > ランダム > デフォルトの優先順位
    /// </summary>
    void DetermineAndApplyGripForce()
    {
        float targetGripForce = 0f;
        GripForceSource source = GripForceSource.Default;
        
        // 🔥 TCP指令が利用可能な場合は優先使用
        if (enableTcpGripForceControl && useTcpForceWhenAvailable && pendingTcpGripForce.HasValue)
        {
            targetGripForce = pendingTcpGripForce.Value;
            source = GripForceSource.Tcp;
            tcpCommandsUsed++;
            
            // 使用済みTCP指令をクリア
            pendingTcpGripForce = null;
        }
        // ランダム把持力が有効な場合
        else if (enableRandomGripForce && gripForceController != null)
        {
            targetGripForce = Random.Range(minGripForce, maxGripForce);
            source = GripForceSource.Random;
        }
        // デフォルト把持力
        else if (gripForceController != null)
        {
            targetGripForce = gripForceController.baseGripForce;
            source = GripForceSource.Default;
        }
        
        // 把持力の適用
        if (gripForceController != null && targetGripForce > 0)
        {
            currentEpisodeGripForce = targetGripForce;
            gripForceController.baseGripForce = targetGripForce;
            gripForceController.forceVariability = 0f; // 実験用
            
            // 統計に記録
            usedGripForces.Add(targetGripForce);
            gripForceSources.Add(source);
            
            // イベント発火
            OnGripForceApplied?.Invoke(targetGripForce, source);
            
            if (logGripForceChanges)
            {
                string sourceText = source switch
                {
                    GripForceSource.Tcp => "🔥TCP指令",
                    GripForceSource.Random => "🎲ランダム",
                    GripForceSource.Default => "⚙️デフォルト",
                    _ => "❓不明"
                };
                
                Debug.Log($"{sourceText} 把持力設定: {targetGripForce:F2}N");
                
                // アルミ缶の変形閾値と比較
                if (aluminumCan != null)
                {
                    float threshold = aluminumCan.deformationThreshold;
                    bool willCrush = targetGripForce > threshold;
                    Debug.Log($"   変形閾値: {threshold:F2}N -> {(willCrush ? "⚠️つぶれる可能性" : "✅安全範囲")}");
                }
            }
        }
        
        currentGripForceSource = source;
    }
    
    #endregion
    
    #region エピソード制御
    
    public void StartAutoEpisodes()
    {
        if (episodeLoopCoroutine != null) return;
        if (!enableAutoEpisodes) return;
        
        currentState = EpisodeState.Starting;
        currentEpisodeNumber = 0;
        successfulEpisodes = 0;
        failedEpisodes = 0;
        totalEpisodeTime = 0f;
        
        // 🔥 TCP統計のリセット
        tcpCommandsReceived = 0;
        tcpCommandsUsed = 0;
        tcpTimeouts = 0;
        
        if (enableDebugLogs)
        {
            Debug.Log("🚀 自動エピソード開始！");
            Debug.Log($"最大エピソード数: {maxEpisodesPerSession}");
            Debug.Log($"エピソード時間: {episodeDuration}秒");
            Debug.Log($"🔥 TCP把持力制御: {(enableTcpGripForceControl ? "有効" : "無効")}");
        }
        episodeLoopCoroutine = StartCoroutine(ExecuteEpisodeLoop());
    }

    public void StopAutoEpisodes()
    {
        enableAutoEpisodes = false;
        currentState = EpisodeState.Finished;
        isWaitingForTcpCommand = false;

        if (episodeLoopCoroutine != null)
        {
            StopCoroutine(episodeLoopCoroutine);
            episodeLoopCoroutine = null;
        }
        
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
            // 🔥 TCP指令待機フェーズ（オプション）
            if (enableTcpGripForceControl && waitForTcpCommandBeforeStart)
            {
                yield return StartCoroutine(WaitForTcpCommand());
            }
            
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
        episodeLoopCoroutine = null;
    }
    
    // 🔥 TCP指令待機のコルーチン
    IEnumerator WaitForTcpCommand()
    {
        StartTcpCommandWait();
        
        while (isWaitingForTcpCommand && enableAutoEpisodes)
        {
            yield return new WaitForSeconds(0.1f);
        }
        
        if (enableDebugLogs)
        {
            bool receivedCommand = pendingTcpGripForce.HasValue;
            Debug.Log($"🔥 TCP指令待機完了: {(receivedCommand ? "指令受信" : "タイムアウト")}");
        }
    }
    
    IEnumerator StartNewEpisode()
    {
        pendingEpisodeResult = null;
        currentState = EpisodeState.Starting;
        currentEpisodeNumber++;
        episodeStartTime = Time.time;
        lastMovementTime = Time.time;

        if (aluminumCan != null)
        {
            initialCanPosition = aluminumCan.transform.position;
        }
        
        // 🔥 把持力の決定と適用（TCP指令を優先）
        DetermineAndApplyGripForce();
        
        if (enableDebugLogs)
        {
            Debug.Log($"📋 エピソード {currentEpisodeNumber} 開始");
            string sourceText = currentGripForceSource switch
            {
                GripForceSource.Tcp => "🔥TCP",
                GripForceSource.Random => "🎲ランダム",
                GripForceSource.Default => "⚙️デフォルト",
                _ => "❓不明"
            };
            Debug.Log($"{sourceText} 把持力: {currentEpisodeGripForce:F2}N");
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
            trajectoryPlanner.PublishJointAlminumCan();
        }
        
        OnEpisodeStarted?.Invoke(currentEpisodeNumber);
        
        yield return new WaitForSeconds(0.5f);
    }
    
    IEnumerator RunEpisode()
    {
        currentState = EpisodeState.Running;
        
        float episodeTime = 0f;
        bool episodeEnded = false;
        
        while (episodeTime < episodeDuration && !episodeEnded)
        {
            episodeTime = Time.time - episodeStartTime;
            
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
        
        bool wasSuccessful = DetermineEpisodeSuccess();
        
        if (wasSuccessful)
            successfulEpisodes++;
        else
            failedEpisodes++;
        

        // リセット直前に送信するため結果を保存
        pendingEpisodeResult = wasSuccessful;

        
        if (enableDebugLogs)
        {
            float successRate = (float)successfulEpisodes / currentEpisodeNumber * 100f;
            string sourceText = currentGripForceSource switch
            {
                GripForceSource.Tcp => "🔥TCP",
                GripForceSource.Random => "🎲ランダム",
                GripForceSource.Default => "⚙️デフォルト",
                _ => "❓不明"
            };
            
            Debug.Log($"🏁 エピソード {currentEpisodeNumber} 完了");
            Debug.Log($"   結果: {(wasSuccessful ? "✅成功" : "❌失敗")}");
            Debug.Log($"   時間: {episodeTime:F2}秒");
            Debug.Log($"   把持力: {currentEpisodeGripForce:F2}N ({sourceText})");
            Debug.Log($"   成功率: {successRate:F1}% ({successfulEpisodes}/{currentEpisodeNumber})");
        }
        
        OnEpisodeCompleted?.Invoke(currentEpisodeNumber, wasSuccessful);
        
        yield return new WaitForSeconds(0.5f);
    }
    
    IEnumerator ResetForNextEpisode()
    {
        currentState = EpisodeState.Resetting;

        // 🔥 リセットの直前にエピソード結果を送信
        if (a2cClient != null && pendingEpisodeResult.HasValue)
        {
            a2cClient.SendEpisodeResult(pendingEpisodeResult.Value);
            a2cClient.SendEpisodeEnd();
            pendingEpisodeResult = null;
        }

        if (enableDebugLogs)
        {
            Debug.Log("🔄 次のエピソードに向けてリセット中...");
        }
        
        if (trajectoryPlanner != null)
        {
            trajectoryPlanner.ResetToInitialPositions();
        }
        
        if (aluminumCan != null)
        {
            aluminumCan.ResetCan();
        }
        
        yield return new WaitForSeconds(resetDelay);
        
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
    
    #region 状態判定（既存のメソッドは変更なし）
    
    void UpdateEpisodeState()
    {
        // ExecuteEpisodeLoop で管理
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
        if (aluminumCan != null && aluminumCan.IsBroken)
        {
            return true;
        }
        
        if (!isRobotMoving && (Time.time - lastMovementTime) > stoppedTimeThreshold)
        {
            return true;
        }
        
        return false;
    }
        
    public bool DetermineEpisodeSuccess()
    {
        if (aluminumCan == null || aluminumCan.IsBroken)
        {
            return false;
        }
        
        bool hasValidContact = false;
        if (gripperInterface != null)
        {
            hasValidContact = gripperInterface.HasValidContact();
        }
        
        bool isLifted = false;
        if (aluminumCan != null)
        {
            float liftHeight = aluminumCan.transform.position.y - initialCanPosition.y;
            isLifted = liftHeight > 0.08f;
        }
        
        bool notFalling = true;
        Rigidbody canRigidbody = aluminumCan.GetComponent<Rigidbody>();
        if (canRigidbody != null)
        {
            notFalling = canRigidbody.velocity.y > -0.3f;
        }
        
        // 缶が破損していないかの追加チェック（既存のGetCurrentStateを使用）
        bool notDeformed = true;
        if (aluminumCan != null)
        {
            var canState = aluminumCan.GetCurrentState();
            notDeformed = !canState.isBroken && canState.deformation < 0.5f; // 変形が50%未満
        }
        
        bool success = !aluminumCan.IsBroken && hasValidContact && isLifted && notFalling && notDeformed;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔍 成功判定詳細:");
            Debug.Log($"   つぶれていない: {!aluminumCan.IsBroken}");
            Debug.Log($"   接触維持: {hasValidContact}");
            Debug.Log($"   持ち上げ完了: {isLifted}");
            Debug.Log($"   落下していない: {notFalling}");
            Debug.Log($"   変形許容範囲: {notDeformed}");
            Debug.Log($"   最終判定: {(success ? "✅成功" : "❌失敗")}");
        }
        
        return success;
    }
    #endregion
    
    #region UI・デバッグ
    
    void UpdateDebugDisplay()
    {
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
        
        // 🔥 TCP把持力統計
        if (enableTcpGripForceControl)
        {
            float tcpUsageRate = tcpCommandsReceived > 0 ? (float)tcpCommandsUsed / tcpCommandsReceived * 100f : 0f;
            Debug.Log($"🔥 TCP把持力統計:");
            Debug.Log($"- 受信指令数: {tcpCommandsReceived}");
            Debug.Log($"- 使用指令数: {tcpCommandsUsed}");
            Debug.Log($"- 使用率: {tcpUsageRate:F1}%");
            Debug.Log($"- タイムアウト回数: {tcpTimeouts}");
        }
        
        // 把持力統計
        if (usedGripForces.Count > 0)
        {
            var gripStats = GetGripForceStatistics();
            Debug.Log($"把持力統計:");
            Debug.Log($"- 平均把持力: {gripStats.averageForce:F2}N");
            Debug.Log($"- 使用範囲: {gripStats.minUsedForce:F2}N - {gripStats.maxUsedForce:F2}N");
            Debug.Log($"- 設定回数: {gripStats.totalForceSettings}回");
            
            // 🔥 ソース別統計
            var sourceStats = GetGripForceSourceStatistics();
            Debug.Log($"- TCP指令使用: {sourceStats.tcpCount}回 ({sourceStats.tcpPercentage:F1}%)");
            Debug.Log($"- ランダム使用: {sourceStats.randomCount}回 ({sourceStats.randomPercentage:F1}%)");
            Debug.Log($"- デフォルト使用: {sourceStats.defaultCount}回 ({sourceStats.defaultPercentage:F1}%)");
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
        
        // 🔥 TCP指令待機状態の表示
        if (currentState == EpisodeState.WaitingForTcp)
        {
            float waitTime = Time.time - tcpCommandWaitStartTime;
            GUI.Label(new Rect(10, y, 400, lineHeight), $"🔥 TCP指令待機中: {waitTime:F1}s / {tcpCommandWaitTimeout:F1}s", style);
            y += lineHeight;
        }
        
        if (currentEpisodeNumber > 0)
        {
            float successRate = (float)successfulEpisodes / currentEpisodeNumber * 100f;
            GUI.Label(new Rect(10, y, 400, lineHeight), $"成功率: {successRate:F1}% ({successfulEpisodes}/{currentEpisodeNumber})", style);
            y += lineHeight;
        }
        
        // 把持力情報
        if (currentEpisodeGripForce > 0)
        {
            string sourceText = currentGripForceSource switch
            {
                GripForceSource.Tcp => "🔥TCP",
                GripForceSource.Random => "🎲ランダム",
                GripForceSource.Default => "⚙️デフォルト",
                _ => "❓"
            };
            GUI.Label(new Rect(10, y, 400, lineHeight), $"把持力: {currentEpisodeGripForce:F1}N ({sourceText})", style);
            y += lineHeight;
        }
        
        // 🔥 TCP統計表示
        if (enableTcpGripForceControl)
        {
            GUI.Label(new Rect(10, y, 400, lineHeight), $"🔥 TCP: 受信{tcpCommandsReceived} / 使用{tcpCommandsUsed} / TO{tcpTimeouts}", style);
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
    
    #region 把持力制御（既存メソッドの拡張）
    
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
    /// 🔥 新規追加：把持力ソース別統計の取得
    /// </summary>
    public GripForceSourceStatistics GetGripForceSourceStatistics()
    {
        if (gripForceSources.Count == 0)
        {
            return new GripForceSourceStatistics();
        }
        
        int tcpCount = 0;
        int randomCount = 0;
        int defaultCount = 0;
        
        foreach (var source in gripForceSources)
        {
            switch (source)
            {
                case GripForceSource.Tcp:
                    tcpCount++;
                    break;
                case GripForceSource.Random:
                    randomCount++;
                    break;
                case GripForceSource.Default:
                    defaultCount++;
                    break;
            }
        }
        
        int total = gripForceSources.Count;
        
        return new GripForceSourceStatistics
        {
            tcpCount = tcpCount,
            randomCount = randomCount,
            defaultCount = defaultCount,
            tcpPercentage = (float)tcpCount / total * 100f,
            randomPercentage = (float)randomCount / total * 100f,
            defaultPercentage = (float)defaultCount / total * 100f,
            totalCount = total
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
    
    /// <summary>
    /// 🔥 新規追加：TCP制御設定の動的変更
    /// </summary>
    public void SetTcpGripForceControlEnabled(bool enabled)
    {
        enableTcpGripForceControl = enabled;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔥 TCP把持力制御: {(enabled ? "有効化" : "無効化")}");
        }
    }
    
    /// <summary>
    /// 🔥 新規追加：TCP待機タイムアウトの設定
    /// </summary>
    public void SetTcpCommandWaitTimeout(float timeoutSeconds)
    {
        tcpCommandWaitTimeout = Mathf.Clamp(timeoutSeconds, 1f, 60f);
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔥 TCP待機タイムアウト設定: {tcpCommandWaitTimeout:F1}秒");
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
            if (episodeLoopCoroutine != null)
            {
                StopCoroutine(episodeLoopCoroutine);
                episodeLoopCoroutine = null;
            }
            StartCoroutine(CompleteEpisode());
        }
    }
    
    /// <summary>
    /// 🔥 新規追加：TCP指令待機を強制終了
    /// </summary>
    public void ForceEndTcpWait()
    {
        if (isWaitingForTcpCommand)
        {
            CompleteTcpCommandWait(true);
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
    
    /// <summary>
    /// 🔥 新規追加：TCP統計情報の取得
    /// </summary>
    public TcpStatistics GetTcpStatistics()
    {
        return new TcpStatistics
        {
            commandsReceived = tcpCommandsReceived,
            commandsUsed = tcpCommandsUsed,
            timeouts = tcpTimeouts,
            usageRate = tcpCommandsReceived > 0 ? (float)tcpCommandsUsed / tcpCommandsReceived * 100f : 0f,
            isEnabled = enableTcpGripForceControl,
            hasPendingCommand = pendingTcpGripForce.HasValue,
            pendingForceValue = pendingTcpGripForce ?? 0f
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

/// <summary>
/// 🔥 新規追加：把持力ソース別統計
/// </summary>
[System.Serializable]
public class GripForceSourceStatistics
{
    public int tcpCount;
    public int randomCount;
    public int defaultCount;
    public float tcpPercentage;
    public float randomPercentage;
    public float defaultPercentage;
    public int totalCount;
}

/// <summary>
/// 🔥 新規追加：TCP通信統計
/// </summary>
[System.Serializable]
public class TcpStatistics
{
    public int commandsReceived;
    public int commandsUsed;
    public int timeouts;
    public float usageRate;
    public bool isEnabled;
    public bool hasPendingCommand;
    public float pendingForceValue;
}