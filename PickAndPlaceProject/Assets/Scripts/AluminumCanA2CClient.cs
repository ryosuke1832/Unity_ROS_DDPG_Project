using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using System.Collections;
using System.Collections.Generic;

[System.Serializable]
public class CanStateMessage
{
    public string type = "can_state";
    public bool is_crushed;
    public string grasp_result;
    public float current_force;
    public float accumulated_force;
    public float timestamp;
}

[System.Serializable]
public class SimpleMessage
{
    public string type;
    public float timestamp;
}

[System.Serializable]
public class A2CResponse
{
    public string type;
    public float recommended_force;
    public float calculated_reward;
    public float timestamp;
    public string message;
}

[System.Serializable]
public class GripForceCommand
{
    public string type;
    public float target_force;
    public string execution_mode;
    public float duration;
    public float timestamp;
}

public class AluminumCanA2CClient : MonoBehaviour
{
    [Header("接続設定")]
    public string serverHost = "localhost";
    public int serverPort = 12345;
    
    [Header("コンポーネント参照")]
    public IntegratedAluminumCan aluminumCan;
    
    [Header("送信設定")]
    [Range(0.1f, 2.0f)]
    public float sendInterval = 0.5f;
    
    [Header("把持力指令待機設定")]
    public bool waitForGripForceCommand = true;  // 新規追加：把持力指令待機を有効にするか
    [Range(1f, 60f)]
    public float commandWaitTimeout = 30f;       // 新規追加：待機タイムアウト時間
    public bool showWaitingStatus = true;        // 新規追加：待機状態をGUIに表示するか
    
    [Header("デバッグ")]
    public bool enableDebugLogs = true;
    public bool enableVerboseReceiveLog = true;
    
    [Header("GUI表示設定")]
    public bool showGripForceGUI = true;
    public Vector2 guiPosition = new Vector2(10, 10);
    public Vector2 guiSize = new Vector2(350, 200);  // サイズを少し大きく
    
    // 通信関連
    private TcpClient tcpClient;
    private NetworkStream stream;
    private Thread communicationThread;
    private bool isConnected = false;
    private bool shouldStop = false;
    
    // メインスレッドでの処理用キュー
    private Queue<string> messageQueue = new Queue<string>();
    private readonly object queueLock = new object();
    
    // 状態管理
    private bool lastCrushedState = false;
    private float lastSendTime = 0f;
    private bool hasEvaluatedThisEpisode = false;
    private bool isEpisodeActive = false;
    private float lastForce = 0f;
    
    // 🔥 新規追加：把持力指令待機システム
    private bool isWaitingForGripForceCommand = false;
    private float waitStartTime = 0f;
    private bool hasReceivedCommandThisEpisode = false;
    private int currentEpisodeNumber = 0;
    
    // 把持力表示用の変数
    private float receivedGripForce = 0f;
    private string lastExecutionMode = "";
    private float lastDuration = 0f;
    private string lastReceivedTime = "";
    private int totalCommandsReceived = 0;
    private bool hasReceivedCommand = false;
    
    // デバッグ用統計情報
    private int totalMessagesReceived = 0;
    private int gripForceCommandsReceived = 0;
    private int episodesWithCommands = 0;
    private int episodesWithTimeout = 0;
    
    // イベント
    public event System.Action<bool> OnConnectionChanged;
    public event System.Action<float> OnRecommendedForceReceived;
    public event System.Action<float> OnRewardReceived;
    public event System.Action<float> OnGripForceCommandReceived;
    public event System.Action OnGripForceCommandWaitStarted;    // 新規追加
    public event System.Action OnGripForceCommandWaitCompleted; // 新規追加
    public event System.Action OnGripForceCommandTimeout;       // 新規追加
    
    void Start()
    {
        if (aluminumCan == null)
        {
            aluminumCan = FindObjectOfType<IntegratedAluminumCan>();
        }
        
        if (aluminumCan == null)
        {
            Debug.LogError("❌ IntegratedAluminumCanが見つかりません！");
            return;
        }
        
        ConnectToA2CServer();
    }
    
    void Update()
    {
        // メインスレッドでキューからメッセージを処理
        ProcessMessageQueue();
        
        // 🔥 新規追加：把持力指令待機のタイムアウトチェック
        CheckGripForceCommandTimeout();
        
        // エピソードがアクティブでない場合は送信しない
        if (!isEpisodeActive)
        {
            return;
        }
        
        // 評価済みの場合は送信停止
        if (hasEvaluatedThisEpisode)
        {
            return;
        }
        
        if (isConnected && Time.time - lastSendTime >= sendInterval)
        {
            SendCanState();
            lastSendTime = Time.time;
        }
    }
    
    // 🔥 新規追加：把持力指令待機のタイムアウトチェック
    void CheckGripForceCommandTimeout()
    {
        if (!isWaitingForGripForceCommand) return;
        
        if (Time.time - waitStartTime > commandWaitTimeout)
        {
            Debug.LogWarning($"⏰ 把持力指令待機タイムアウト（{commandWaitTimeout}秒）- エピソードを続行");
            CompleteGripForceCommandWait(true);
        }
    }
    
    // メインスレッドでのメッセージ処理
    void ProcessMessageQueue()
    {
        lock (queueLock)
        {
            while (messageQueue.Count > 0)
            {
                string message = messageQueue.Dequeue();
                ProcessA2CResponse(message);
            }
        }
    }
    
    #region GUI表示
    
    void OnGUI()
    {
        if (!showGripForceGUI) return;
        
        // GUI領域の設定
        GUILayout.BeginArea(new Rect(guiPosition.x, guiPosition.y, guiSize.x, guiSize.y));
        
        // 背景ボックス
        GUI.Box(new Rect(0, 0, guiSize.x, guiSize.y), "");
        
        GUILayout.BeginVertical();
        
        // タイトル
        GUILayout.Label("🎯 Python把持力指令", new GUIStyle(GUI.skin.label) 
        { 
            fontSize = 16, 
            fontStyle = FontStyle.Bold,
            normal = { textColor = Color.white }
        });
        
        GUILayout.Space(5);
        
        // 接続状態
        string connectionStatus = isConnected ? "✅ 接続中" : "❌ 切断";
        Color connectionColor = isConnected ? Color.green : Color.red;
        GUILayout.Label($"接続状態: {connectionStatus}", new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = connectionColor }
        });
        
        // 🔥 新規追加：待機状態の表示
        if (showWaitingStatus && waitForGripForceCommand)
        {
            if (isWaitingForGripForceCommand)
            {
                float elapsedTime = Time.time - waitStartTime;
                float remainingTime = commandWaitTimeout - elapsedTime;
                
                GUILayout.Label($"⏳ 把持力指令待機中... ({remainingTime:F1}秒)", new GUIStyle(GUI.skin.label) 
                { 
                    normal = { textColor = Color.yellow },
                    fontStyle = FontStyle.Bold
                });
            }
            else
            {
                GUILayout.Label("✅ 指令受信済み", new GUIStyle(GUI.skin.label) 
                { 
                    normal = { textColor = Color.green }
                });
            }
        }
        
        // 統計情報
        GUILayout.Label($"総受信: {totalMessagesReceived} / 把持力: {gripForceCommandsReceived}");
        GUILayout.Label($"エピソード: {currentEpisodeNumber} / 指令有: {episodesWithCommands} / タイムアウト: {episodesWithTimeout}");
        
        GUILayout.Space(5);
        
        // 受信した把持力の表示
        if (hasReceivedCommand)
        {
            GUILayout.Label($"受信把持力: {receivedGripForce:F1} N", new GUIStyle(GUI.skin.label) 
            { 
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                normal = { textColor = Color.cyan }
            });
            
            GUILayout.Label($"実行モード: {lastExecutionMode}");
            
            if (lastExecutionMode == "gradual")
            {
                GUILayout.Label($"継続時間: {lastDuration:F1} 秒");
            }
            
            GUILayout.Label($"受信時刻: {lastReceivedTime}");
        }
        else
        {
            GUILayout.Label("把持力指令待機中...", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = Color.yellow }
            });
        }
        
        GUILayout.Space(5);
        
        // 現在の缶の状態（参考用）
        if (aluminumCan != null)
        {
            var state = aluminumCan.GetCurrentState();
            GUILayout.Label($"現在力: {state.appliedForce:F1} N");
            
            string statusText = state.isBroken ? "🔴 つぶれ" : "🟢 正常";
            Color statusColor = state.isBroken ? Color.red : Color.green;
            GUILayout.Label($"缶状態: {statusText}", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = statusColor }
            });
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
    
    #endregion
    
    #region 接続管理
    
    void ConnectToA2CServer()
    {
        try
        {
            tcpClient = new TcpClient(serverHost, serverPort);
            stream = tcpClient.GetStream();
            isConnected = true;
            
            OnConnectionChanged?.Invoke(true);
            
            if (enableDebugLogs)
                Debug.Log("✅ A2Cサーバーに接続しました");
            
            // 通信スレッド開始
            communicationThread = new Thread(CommunicationLoop);
            communicationThread.Start();
            
            // 接続テスト
            SendPing();
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ A2Cサーバー接続失敗: {e.Message}");
            OnConnectionChanged?.Invoke(false);
        }
    }
    
    void CommunicationLoop()
    {
        byte[] buffer = new byte[4096];
        
        while (isConnected && !shouldStop)
        {
            try
            {
                if (stream.DataAvailable)
                {
                    int bytes = stream.Read(buffer, 0, buffer.Length);
                    string response = Encoding.UTF8.GetString(buffer, 0, bytes);
                    
                    if (enableVerboseReceiveLog)
                    {
                        Debug.Log($"🔍 RAW受信データ（{bytes}バイト）: {response}");
                    }
                    
                    lock (queueLock)
                    {
                        messageQueue.Enqueue(response);
                    }
                }
                
                Thread.Sleep(10);
            }
            catch (Exception e)
            {
                if (enableDebugLogs)
                    Debug.LogError($"❌ 通信エラー: {e.Message}");
                break;
            }
        }
        
        Debug.Log("🔌 CommunicationLoop終了");
    }
    
    #endregion
    
    #region 🔥 新規追加：把持力指令待機システム
    
    /// <summary>
    /// 把持力指令の待機を開始
    /// </summary>
    public void StartWaitingForGripForceCommand()
    {
        if (!waitForGripForceCommand) return;
        
        isWaitingForGripForceCommand = true;
        hasReceivedCommandThisEpisode = false;
        waitStartTime = Time.time;
        
        OnGripForceCommandWaitStarted?.Invoke();
        
        if (enableDebugLogs)
        {
            Debug.Log($"⏳ 把持力指令待機開始 - タイムアウト: {commandWaitTimeout}秒");
        }
    }
    
    /// <summary>
    /// 把持力指令待機の完了
    /// </summary>
    /// <param name="isTimeout">タイムアウトによる完了かどうか</param>
    private void CompleteGripForceCommandWait(bool isTimeout = false)
    {
        if (!isWaitingForGripForceCommand) return;
        
        isWaitingForGripForceCommand = false;
        
        if (isTimeout)
        {
            episodesWithTimeout++;
            OnGripForceCommandTimeout?.Invoke();
            
            if (enableDebugLogs)
            {
                Debug.LogWarning($"⏰ 把持力指令待機タイムアウト - エピソード {currentEpisodeNumber}");
            }
        }
        else
        {
            episodesWithCommands++;
            OnGripForceCommandWaitCompleted?.Invoke();
            
            if (enableDebugLogs)
            {
                Debug.Log($"✅ 把持力指令受信完了 - エピソード {currentEpisodeNumber}");
            }
        }
    }
    
    /// <summary>
    /// 現在把持力指令を待機中かどうか
    /// </summary>
    public bool IsWaitingForGripForceCommand()
    {
        return isWaitingForGripForceCommand;
    }
    
    /// <summary>
    /// 現在のエピソードで把持力指令を受信したかどうか
    /// </summary>
    public bool HasReceivedCommandThisEpisode()
    {
        return hasReceivedCommandThisEpisode;
    }
    
    /// <summary>
    /// 次のエピソードに進んでも良いかチェック
    /// </summary>
    public bool CanProceedToNextEpisode()
    {
        if (!waitForGripForceCommand) return true;  // 待機機能が無効なら常にOK
        
        return !isWaitingForGripForceCommand;  // 待機中でなければOK
    }
    
    #endregion
    
    #region メッセージ送信
    
    void SendCanState()
    {
        if (!isConnected || aluminumCan == null) return;
        
        var state = aluminumCan.GetCurrentState();
        
        var message = new CanStateMessage
        {
            is_crushed = state.isBroken,
            grasp_result = DetermineGraspResult(),
            current_force = state.appliedForce,
            accumulated_force = aluminumCan.GetAccumulatedForce(),
            timestamp = Time.time
        };
        
        // 送信条件を厳格化
        bool shouldSend = false;
        
        if (message.is_crushed && !lastCrushedState)
        {
            shouldSend = true;
            hasEvaluatedThisEpisode = true;
        }
        else if (!message.is_crushed && Math.Abs(message.current_force - lastForce) > 0.1f)
        {
            shouldSend = true;
        }
        else if (!message.is_crushed && message.current_force > 0.1f && lastForce <= 0.1f)
        {
            shouldSend = true;
        }
        
        if (shouldSend)
        {
            SendMessage(message);
        }
        
        // 状態更新
        lastCrushedState = message.is_crushed;
        lastForce = message.current_force;
    }

    private string DetermineGraspResult()
    {
        if (aluminumCan.IsBroken)
            return "overgrip";
        
        var state = aluminumCan.GetCurrentState();
        if (state.appliedForce > 0.1f && state.appliedForce < 15f && !state.isBroken)
            return "success";
        
        return "undergrip";
    }
    
    void SendPing()
    {
        var pingMessage = new SimpleMessage { type = "ping", timestamp = Time.time };
        SendMessage(pingMessage);
        
        if (enableDebugLogs)
            Debug.Log("🏓 Ping送信");
    }
    
    public void SendEpisodeEnd()
    {
        var endMessage = new SimpleMessage { type = "episode_end", timestamp = Time.time };
        SendMessage(endMessage);
        
        isEpisodeActive = false;
        
        if (enableDebugLogs)
            Debug.Log("📋 エピソード終了通知を送信");
    }
    
    public void SendReset()
    {
        hasEvaluatedThisEpisode = false;
        lastCrushedState = false;
        lastForce = 0f;
        isEpisodeActive = true;
        currentEpisodeNumber++;  // 🔥 新規追加：エピソード番号をカウント
        
        // 🔥 新規追加：把持力指令待機を開始
        if (waitForGripForceCommand)
        {
            StartWaitingForGripForceCommand();
        }
        
        var resetMessage = new SimpleMessage { type = "reset", timestamp = Time.time };
        SendMessage(resetMessage);
        
        if (enableDebugLogs)
            Debug.Log($"🔄 リセット通知を送信 - エピソード {currentEpisodeNumber}");
    }
    
    public void OnNewEpisodeStarted()
    {
        hasEvaluatedThisEpisode = false;
        lastCrushedState = false;
        lastForce = 0f;
        isEpisodeActive = true;
        
        if (enableDebugLogs)
            Debug.Log("🆕 新エピソード開始");
    }
    
    public void OnEpisodeCompleted()
    {
        isEpisodeActive = false;
        
        if (enableDebugLogs)
            Debug.Log("🏁 エピソード完了");
    }
    
    void SendMessage(object message)
    {
        if (!isConnected || stream == null) return;
        
        try
        {
            string json = JsonUtility.ToJson(message);
            byte[] data = Encoding.UTF8.GetBytes(json + "\n");
            stream.Write(data, 0, data.Length);
            
            if (enableDebugLogs)
            {
                Debug.Log($"📤 送信データ: {json}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ メッセージ送信エラー: {e.Message}");
        }
    }
    
    #endregion
    
    #region メッセージ受信処理
    
    void ProcessA2CResponse(string response)
    {
        if (string.IsNullOrEmpty(response))
        {
            if (enableVerboseReceiveLog)
                Debug.LogWarning("⚠️ 空の応答を受信");
            return;
        }
        
        totalMessagesReceived++;
        
        try
        {
            if (enableVerboseReceiveLog)
            {
                Debug.Log($"📥 メッセージ処理開始: {response}");
            }
            
            string[] messages = response.Split(new char[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
            
            foreach (string message in messages)
            {
                if (string.IsNullOrWhiteSpace(message)) continue;
                
                ProcessSingleMessage(message.Trim());
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ 応答処理エラー: {e.Message}");
            if (enableDebugLogs)
                Debug.Log($"問題のあるデータ: '{response}'");
        }
    }
    
    void ProcessSingleMessage(string jsonMessage)
    {
        try
        {
            string type = ExtractTypeFromJson(jsonMessage);
            
            if (enableVerboseReceiveLog)
                Debug.Log($"📋 メッセージタイプ: '{type}' - データ: {jsonMessage}");
            
            switch (type)
            {
                case "pong":
                    if (enableDebugLogs)
                        Debug.Log("🏓 Pong受信 - 接続正常");
                    break;
                    
                case "ack":
                    if (enableDebugLogs)
                        Debug.Log("✅ ACK受信 - メッセージ確認");
                    break;
                    
                case "action_response":
                    if (enableDebugLogs)
                        Debug.Log("🎯 A2C応答受信");
                    break;
                    
                case "episode_complete":
                    if (enableDebugLogs)
                        Debug.Log("🏁 エピソード完了受信");
                    break;
                    
                case "reset_complete":
                    if (enableDebugLogs)
                        Debug.Log("🔄 リセット完了受信");
                    break;
                    
                case "grip_force_command":
                    Debug.Log($"🎯🎯🎯 把持力コマンド検出！メッセージ: {jsonMessage}");
                    ProcessGripForceCommand(jsonMessage);
                    break;
                    
                default:
                    if (enableDebugLogs)
                        Debug.Log($"❓ 不明な応答タイプ: '{type}' - データ: {jsonMessage}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ 単一メッセージ処理エラー: {e.Message}");
            Debug.LogError($"問題のメッセージ: '{jsonMessage}'");
        }
    }
    
    void ProcessGripForceCommand(string jsonMessage)
    {
        Debug.Log($"🔥 ProcessGripForceCommand開始 - JSON: {jsonMessage}");
        
        try
        {
            var command = JsonUtility.FromJson<GripForceCommand>(jsonMessage);
            
            Debug.Log($"🔥 JSON解析成功 - target_force: {command.target_force}");
            
            // 受信データを保存
            receivedGripForce = command.target_force;
            lastExecutionMode = command.execution_mode;
            lastDuration = command.duration;
            lastReceivedTime = DateTime.Now.ToString("HH:mm:ss");
            totalCommandsReceived++;
            gripForceCommandsReceived++;
            hasReceivedCommand = true;
            hasReceivedCommandThisEpisode = true;  // 🔥 新規追加
            
            // 🔥 新規追加：待機完了
            CompleteGripForceCommandWait(false);
            
            // イベント発火
            OnGripForceCommandReceived?.Invoke(command.target_force);
            
            // 確実にログ出力
            Debug.Log($"🎯✅ 把持力コマンド受信成功！");
            Debug.Log($"  └ 把持力: {command.target_force:F1}N");
            Debug.Log($"  └ モード: {command.execution_mode}");
            Debug.Log($"  └ 時間: {command.duration:F1}秒");
            Debug.Log($"  └ 受信時刻: {lastReceivedTime}");
            Debug.Log($"  └ エピソード: {currentEpisodeNumber}");
            Debug.Log($"  └ 待機完了: ✅");
            
            // 応答を送信
            SendGripForceResponse(command.target_force, "received");
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ 把持力コマンド処理エラー: {e.Message}");
            Debug.LogError($"失敗したJSON: '{jsonMessage}'");
        }
    }
    
    void SendGripForceResponse(float targetForce, string status)
    {
        var response = new
        {
            type = "grip_force_response",
            target_force = targetForce,
            status = status,
            timestamp = Time.time
        };
        
        SendMessage(response);
        
        if (enableDebugLogs)
        {
            Debug.Log($"📤 把持力応答送信: {status} - 力: {targetForce:F1}N");
        }
    }
    
    string ExtractTypeFromJson(string json)
    {
        try
        {
            int typeStart = json.IndexOf("\"type\"");
            if (typeStart == -1) return "unknown";
            
            int valueStart = json.IndexOf(":", typeStart) + 1;
            int quoteStart = json.IndexOf("\"", valueStart) + 1;
            int quoteEnd = json.IndexOf("\"", quoteStart);
            
            if (quoteStart > 0 && quoteEnd > quoteStart)
            {
                return json.Substring(quoteStart, quoteEnd - quoteStart);
            }
        }
        catch
        {
            // 解析失敗時はunknownを返す
        }
        
        return "unknown";
    }
    
    #endregion
    
    #region 公開API
    
    public float GetReceivedGripForce()
    {
        return receivedGripForce;
    }
    
    public bool HasReceivedGripForceCommand()
    {
        return hasReceivedCommand;
    }
    
    public bool IsConnected()
    {
        return isConnected;
    }
    
    public void ClearGripForceDisplay()
    {
        hasReceivedCommand = false;
        receivedGripForce = 0f;
        lastExecutionMode = "";
        lastDuration = 0f;
        lastReceivedTime = "";
    }
    
    // 🔥 新規追加：待機システム制御API
    public void SetWaitForGripForceCommand(bool enable)
    {
        waitForGripForceCommand = enable;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔧 把持力指令待機: {(enable ? "有効" : "無効")}");
        }
    }
    
    public void SetCommandWaitTimeout(float timeoutSeconds)
    {
        commandWaitTimeout = timeoutSeconds;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔧 待機タイムアウト設定: {timeoutSeconds}秒");
        }
    }
    
    [ContextMenu("統計リセット")]
    public void ResetStatistics()
    {
        totalMessagesReceived = 0;
        gripForceCommandsReceived = 0;
        totalCommandsReceived = 0;
        episodesWithCommands = 0;
        episodesWithTimeout = 0;
        currentEpisodeNumber = 0;
        Debug.Log("📊 統計情報をリセットしました");
    }
    
    #endregion
    
    #region ライフサイクル
    
    void OnDestroy()
    {
        Disconnect();
    }
    
    void OnApplicationQuit()
    {
        Disconnect();
    }
    
    void Disconnect()
    {
        shouldStop = true;
        isConnected = false;
        
        if (communicationThread != null && communicationThread.IsAlive)
        {
            communicationThread.Join(1000);
        }
        
        if (stream != null) stream.Close();
        if (tcpClient != null) tcpClient.Close();
        
        OnConnectionChanged?.Invoke(false);
        
        if (enableDebugLogs)
            Debug.Log("🔌 A2Cサーバーから切断");
    }
    
    #endregion
}