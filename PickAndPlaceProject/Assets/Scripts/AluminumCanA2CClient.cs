// AluminumCanA2CClient.cs の改良版
// TCP受信の行分割問題を修正し、ポート分離に対応

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;


/// <summary>
/// A2C強化学習サーバーとの通信クライアント
/// TCP受信の行分割処理を修正し、ポート分離に対応
/// </summary>
public class AluminumCanA2CClient : MonoBehaviour
{
    [Header("🔗 接続設定")]
    public string serverHost = "127.0.0.1";
    public int serverPort = 12345;
    public bool autoConnect = true;
    public bool autoReconnect = true;
    [Range(1f, 10f)]
    public float reconnectInterval = 3f;
    
    [Header("📡 通信設定")]
    [Range(0.1f, 5f)]
    public float sendInterval = 1f;
    public bool enableCompression = false;
    public int maxRetries = 3;
    
    [Header("🔥 把持力指令設定")]
    public bool enableGripForceReceiving = true;
    public bool enableGripForceForwarding = true;
    [Range(1f, 30f)]
    public float maxGripForceValue = 30f;
    [Range(0.1f, 5f)]
    public float minGripForceValue = 0.1f;
    
    [Header("🎯 AutoEpisodeManager 連携")]
    public AutoEpisodeManager episodeManager;
    public bool autoFindEpisodeManager = true;
    
    [Header("🔍 デバッグ")]
    public bool enableDebugLogs = true;
    public bool enableVerboseReceiveLog = false;
    public bool showGripForceGUI = true;
    public Vector2 guiPosition = new Vector2(10, 300);
    public Vector2 guiSize = new Vector2(300, 200);
    
    [Header("🎮 ゲームオブジェクト参照")]
    public IntegratedAluminumCan aluminumCan;
    public SimpleGripForceController gripForceController;
    public GripperTargetInterface gripperInterface;


    [Header("📸 送信モード")]
    public bool sendOnlyOnFirstContact = true;   

    private bool firstContactSentThisEpisode = false; 
    private bool prevContactState = false;       


    
    // 通信関連
    private TcpClient tcpClient;
    private NetworkStream stream;
    private Thread communicationThread;
    private bool isConnected = false;
    private bool shouldStop = false;
    private float lastSendTime = 0f;
    private int retryCount = 0;
    
    // 🔥 TCP受信バッファ（行分割対応）
    private StringBuilder receiveBuffer = new StringBuilder();
    
    // メッセージキュー（スレッドセーフ）
    private Queue<string> messageQueue = new Queue<string>();
    private readonly object queueLock = new object();
    
    // エピソード制御
    private bool isEpisodeActive = false;
    private bool hasEvaluatedThisEpisode = false;
    private int currentEpisodeNumber = 0;
    private bool episodeResultSent = false;
    
    // 🔥 把持力指令関連
    private float? pendingGripForceCommand = null;
    private readonly object gripForceQueueLock = new object();
    private float? lastReceivedGripForce = null;
    private DateTime lastGripForceReceiveTime = DateTime.MinValue;
    
    // 統計
    private int totalMessagesReceived = 0;
    private int gripForceCommandsReceived = 0;
    private int gripForceCommandsForwarded = 0;
    private int invalidGripForceCommands = 0;
    private int totalMessagesSent = 0;
    private int connectionAttempts = 0;
    private int lineParsingErrors = 0; // 🔥 新規追加

    // イベント
    public System.Action<bool> OnConnectionChanged;
    public System.Action<float> OnGripForceCommandReceived;
    public System.Action<string> OnMessageReceived;
    public System.Action<int> OnEpisodeStateChanged;

    private bool eventsHooked = false;
    
    void Start()
    {
        InitializeComponents();
        
        if (autoConnect)
        {
            ConnectToA2CServer();
        }
    }
    
    void Update()
    {
        ProcessMessageQueue();
        ProcessGripForceCommands();
        
        if (!isEpisodeActive || hasEvaluatedThisEpisode)
        {
            return;
        }

        if (sendOnlyOnFirstContact)
        {
            bool nowContact = (gripperInterface != null) && gripperInterface.HasValidContact();


            if (nowContact && !prevContactState && !firstContactSentThisEpisode)
            {
                SendCanState();                       
                firstContactSentThisEpisode = true;   
                hasEvaluatedThisEpisode = true;       

                if (enableDebugLogs)
                    Debug.Log("📸 接触立ち上がりで状態を1回だけ送信しました");
            }

            prevContactState = nowContact;
        }
        else
        {

            if (isConnected && Time.time - lastSendTime >= sendInterval)
            {
                SendCanState();
                lastSendTime = Time.time;
            }
        }

        if (!isConnected && autoReconnect && Time.time - lastSendTime > reconnectInterval)
        {
            AttemptReconnection();
        }

        
        if (!isConnected && autoReconnect && Time.time - lastSendTime > reconnectInterval)
        {
            AttemptReconnection();
        }
    }
    
    #region 初期化
    
    void InitializeComponents()
    {
        if (aluminumCan == null)
            aluminumCan = FindObjectOfType<IntegratedAluminumCan>();
            
        if (gripForceController == null)
            gripForceController = FindObjectOfType<SimpleGripForceController>();
            
        if (gripperInterface == null)
            gripperInterface = FindObjectOfType<GripperTargetInterface>();
        
        if (autoFindEpisodeManager && episodeManager == null)
        {
            episodeManager = FindObjectOfType<AutoEpisodeManager>();
        }
        
        SetupEpisodeManagerIntegration();
        
        if (enableDebugLogs)
        {
            Debug.Log("=== AluminumCanA2CClient 初期化 ===");
            Debug.Log($"接続先: {serverHost}:{serverPort}");
            Debug.Log($"AluminumCan: {(aluminumCan != null ? "✅" : "❌")}");
            Debug.Log($"GripForceController: {(gripForceController != null ? "✅" : "❌")}");
            Debug.Log($"GripperInterface: {(gripperInterface != null ? "✅" : "❌")}");
            Debug.Log($"🔥 EpisodeManager: {(episodeManager != null ? "✅連携設定" : "❌未設定")}");
            Debug.Log($"🔥 把持力指令受信: {(enableGripForceReceiving ? "有効" : "無効")}");
            Debug.Log($"🔥 把持力指令転送: {(enableGripForceForwarding ? "有効" : "無効")}");
        }
    }
    
    void SetupEpisodeManagerIntegration()
    {
        if (eventsHooked) return;
        if (episodeManager == null) return;

        episodeManager.OnEpisodeStarted += OnEpisodeStarted;
        episodeManager.OnEpisodeCompleted += OnEpisodeCompleted;
        episodeManager.OnSessionCompleted += OnSessionCompleted;
        eventsHooked = true;

        if (enableDebugLogs)
        {
            Debug.Log("🔥 AutoEpisodeManagerとの連携を設定しました");
        }
    }
    
    #endregion
    
    #region エピソード連携イベント
    
    void OnEpisodeStarted(int episodeNumber)
    {
        currentEpisodeNumber = episodeNumber;
        isEpisodeActive = true;
        hasEvaluatedThisEpisode = false;

        firstContactSentThisEpisode = false;
        prevContactState = false;
        
        OnEpisodeStateChanged?.Invoke(episodeNumber);
        
        if (enableDebugLogs)
        {
            Debug.Log($"📋 エピソード {episodeNumber} 開始通知受信");
        }
    }
    
    void OnEpisodeCompleted(int episodeNumber, bool wasSuccessful)
    {
        isEpisodeActive = false;
        hasEvaluatedThisEpisode = true;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🏁 エピソード {episodeNumber} 完了通知受信: {(wasSuccessful ? "成功" : "失敗")}");
        }
    }
    
    void OnSessionCompleted()
    {
        isEpisodeActive = false;
        
        if (enableDebugLogs)
        {
            Debug.Log("🏆 セッション完了通知受信");
            ShowStatistics();
        }
    }
    
    #endregion
    
    #region 🔥 把持力指令処理
    
    void ProcessGripForceCommands()
    {
        if (!enableGripForceReceiving) return;
        
        lock (gripForceQueueLock)
        {
            if (pendingGripForceCommand.HasValue)
            {
                float gripForce = pendingGripForceCommand.Value;
                pendingGripForceCommand = null;
                ProcessGripForceCommand(gripForce);
            }
        }
    }
    
    void ProcessGripForceCommand(float gripForce)
    {
        if (gripForce < minGripForceValue || gripForce > maxGripForceValue)
        {
            invalidGripForceCommands++;
            return;
        }
        
        lastReceivedGripForce = gripForce;
        lastGripForceReceiveTime = DateTime.Now;
        gripForceCommandsReceived++;

        if (enableGripForceForwarding)
        {
            if (OnGripForceCommandReceived != null)
            {
                OnGripForceCommandReceived.Invoke(gripForce);
            }
            else if (episodeManager != null)
            {
                episodeManager.OnTcpGripForceCommandReceived(gripForce);
                gripForceCommandsForwarded++;
            }
        }
    }
    
    bool TryParseGripForceCommand(string message, out float gripForce)
    {
        gripForce = 0f;
        
        // JSON形式: {"type": "grip_force_command", "target_force": 10.0, ...}
        try
        {
            if (message.Contains("grip_force_command") && message.Contains("target_force"))
            {
                int targetForceIndex = message.IndexOf("target_force");
                if (targetForceIndex >= 0)
                {
                    int colonIndex = message.IndexOf(":", targetForceIndex);
                    if (colonIndex >= 0)
                    {
                        string remaining = message.Substring(colonIndex + 1);
                        int endIndex = remaining.IndexOfAny(new char[] { ',', '}' });
                        if (endIndex >= 0)
                        {
                            string valueStr = remaining.Substring(0, endIndex).Trim();
                            if (float.TryParse(valueStr, out gripForce))
                            {
                                if (enableDebugLogs)
                                {
                                    Debug.Log($"🔥 JSON形式の把持力指令解析成功: {gripForce:F2}N");
                                }
                                return true;
                            }
                        }
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (enableDebugLogs)
            {
                Debug.LogWarning($"JSON把持力指令解析エラー: {ex.Message}");
            }
        }
        
        // テキスト形式: "GRIP_FORCE:15.5"
        string[] patterns = { "GRIP_FORCE:", "grip_force:", "GripForce:", "gripforce:" };
        
        foreach (string pattern in patterns)
        {
            if (message.StartsWith(pattern, StringComparison.OrdinalIgnoreCase))
            {
                string valueStr = message.Substring(pattern.Length).Trim();
                
                if (float.TryParse(valueStr, out gripForce))
                {
                    if (enableDebugLogs)
                    {
                        Debug.Log($"🔥 テキスト形式の把持力指令解析成功: {gripForce:F2}N");
                    }
                    return true;
                }
            }
        }
        
        return false;
    }
    
    #endregion
    
    #region 🔥 修正済み通信処理（行分割対応）
    
    void ConnectToA2CServer()
    {
        try
        {
            connectionAttempts++;
            tcpClient = new TcpClient(serverHost, serverPort);
            stream = tcpClient.GetStream();
            isConnected = true;
            retryCount = 0;
            
            // 🔥 受信バッファをクリア
            receiveBuffer.Clear();
            
            OnConnectionChanged?.Invoke(true);
            
            if (enableDebugLogs)
                Debug.Log($"✅ A2Cサーバーに接続しました: {serverHost}:{serverPort} (試行回数: {connectionAttempts})");
            
            communicationThread = new Thread(CommunicationLoop);
            communicationThread.Start();
            
            SendPing();
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ A2Cサーバー接続失敗: {e.Message}");
            OnConnectionChanged?.Invoke(false);
            
            if (autoReconnect && retryCount < maxRetries)
            {
                retryCount++;
                Invoke(nameof(AttemptReconnection), reconnectInterval);
            }
        }
    }
    
    void AttemptReconnection()
    {
        if (isConnected || shouldStop) return;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔄 再接続試行 ({retryCount + 1}/{maxRetries})");
        }
        
        ConnectToA2CServer();
    }
    
    // 🔥 修正された通信ループ（行分割対応）
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
                    string newData = Encoding.UTF8.GetString(buffer, 0, bytes);
                    
                    // 🔥 受信バッファに追記
                    receiveBuffer.Append(newData);
                    
                    // 🔥 完全な行を抽出してキューに追加
                    ProcessReceiveBuffer();
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
    
    // 🔥 新規追加：受信バッファから完全な行を抽出
    void ProcessReceiveBuffer()
    {
        string bufferContent = receiveBuffer.ToString();
        int newlineIndex;
        
        // \n で区切られた完全な行を抽出
        while ((newlineIndex = bufferContent.IndexOf('\n')) >= 0)
        {
            try
            {
                // 完全な1行を取得
                string completeLine = bufferContent.Substring(0, newlineIndex).Trim();
                
                // バッファから処理済み部分を削除
                receiveBuffer.Remove(0, newlineIndex + 1);
                bufferContent = receiveBuffer.ToString();
                
                // 空行でなければキューに追加
                if (!string.IsNullOrEmpty(completeLine))
                {
                    lock (queueLock)
                    {
                        messageQueue.Enqueue(completeLine);
                    }
                    
                    if (enableVerboseReceiveLog)
                    {
                        Debug.Log($"🔍 完全な行を抽出: {completeLine}");
                    }
                }
            }
            catch (Exception ex)
            {
                lineParsingErrors++;
                if (enableDebugLogs)
                {
                    Debug.LogError($"❌ 行解析エラー: {ex.Message}");
                }
                break;
            }
        }
    }
    
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
    
    void ProcessA2CResponse(string message)
    {
        if (string.IsNullOrEmpty(message)) return;
        
        totalMessagesReceived++;

        if (enableVerboseReceiveLog)
        {
            Debug.Log($"📨 メッセージ処理: {message}");
        }
        
        // 🔥 把持力指令の解析と処理
        if (enableGripForceReceiving && TryParseGripForceCommand(message, out float gripForce))
        {
            lock (gripForceQueueLock)
            {
                pendingGripForceCommand = gripForce;
            }

            Debug.Log($"🔥 把持力指令を検出: {gripForce:F2}N");
        }
        
        OnMessageReceived?.Invoke(message);
    }
    
    #endregion
    
    #region データ送信
    
    public void SendCanState()
    {
        if (!isConnected || aluminumCan == null) return;
        
        try
        {
            var state = CollectCanStateData();
            string jsonData = CreateStateJson(state);
            
            SendMessage(jsonData);
            totalMessagesSent++;
            
            if (enableVerboseReceiveLog)
            {
                Debug.Log($"📤 状態送信: {jsonData}");
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ 状態送信エラー: {e.Message}");
        }
    }
    
    private CanStateData CollectCanStateData()
    {
        var state = new CanStateData();
        
        if (aluminumCan != null)
        {
            state.position = aluminumCan.transform.position;
            state.rotation = aluminumCan.transform.rotation;
            state.isBroken = aluminumCan.IsBroken;
            
            var canState = aluminumCan.GetCurrentState();
            state.deformationLevel = canState.deformation;
            
            var rb = aluminumCan.GetComponent<Rigidbody>();
            if (rb != null)
            {
                state.velocity = rb.velocity;
                state.angularVelocity = rb.angularVelocity;
            }
        }
        
        if (gripperInterface != null)
        {
            state.hasContact = gripperInterface.HasValidContact();
            state.contactForce = 0f;
        }
        
        if (gripForceController != null)
        {
            state.currentGripForce = gripForceController.baseGripForce;
            state.actualGripForce = gripForceController.baseGripForce;
        }
        
        state.episodeNumber = currentEpisodeNumber;
        state.episodeActive = isEpisodeActive;
        state.timestamp = Time.time;
        
        state.lastTcpGripForce = lastReceivedGripForce ?? 0f;
        state.hasTcpCommand = lastReceivedGripForce.HasValue;
        state.tcpCommandAge = lastReceivedGripForce.HasValue ? 
            (float)(DateTime.Now - lastGripForceReceiveTime).TotalSeconds : -1f;
        
        return state;
    }
    
    private string CreateStateJson(CanStateData state)
    {
        var json = new StringBuilder();
        json.Append("{");
        
        json.Append($"\"episode\":{state.episodeNumber},");
        json.Append($"\"active\":{state.episodeActive.ToString().ToLower()},");
        json.Append($"\"timestamp\":{state.timestamp:F3},");
        
        json.Append($"\"position\":[{state.position.x:F3},{state.position.y:F3},{state.position.z:F3}],");
        json.Append($"\"velocity\":[{state.velocity.x:F3},{state.velocity.y:F3},{state.velocity.z:F3}],");
        json.Append($"\"broken\":{state.isBroken.ToString().ToLower()},");
        json.Append($"\"deformation\":{state.deformationLevel:F3},");
        
        json.Append($"\"contact\":{state.hasContact.ToString().ToLower()},");
        json.Append($"\"contact_force\":{state.contactForce:F3},");
        json.Append($"\"grip_force\":{state.currentGripForce:F3},");
        json.Append($"\"actual_grip_force\":{state.actualGripForce:F3},");
        
        json.Append($"\"tcp_grip_force\":{state.lastTcpGripForce:F3},");
        json.Append($"\"has_tcp_command\":{state.hasTcpCommand.ToString().ToLower()},");
        json.Append($"\"tcp_command_age\":{state.tcpCommandAge:F3}");
        
        json.Append("}");
        
        return json.ToString();
    }
    
    public void SendMessage(string message)
    {
        if (!isConnected || stream == null) return;
        
        try
        {
            // 🔥 必ず改行を付加
            byte[] data = Encoding.UTF8.GetBytes(message + "\n");
            stream.Write(data, 0, data.Length);
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ メッセージ送信エラー: {e.Message}");
            isConnected = false;
        }
    }
    
    public void SendPing()
    {
        SendMessage("PING");
    }
    
    public void SendReset()
    {
        SendMessage("RESET");
        hasEvaluatedThisEpisode = false;
        episodeResultSent = false;
    }

    public void SendEpisodeEnd()
    {
        SendMessage("EPISODE_END");
        hasEvaluatedThisEpisode = true;
    }

    public void SendEpisodeResult(bool wasSuccessful)
    {
        if (episodeResultSent) return;

        string resultMessage = wasSuccessful ? "RESULT_SUCCESS" : "RESULT_FAIL";
        SendMessage(resultMessage);
        episodeResultSent = true;
        
        if (enableDebugLogs)
        {
            Debug.Log($"📤 エピソード結果送信: {resultMessage}");
        }
    }

    public void SendGripForceRequest()
    {
        if (!isConnected)
        {
            if (enableDebugLogs)
            {
                Debug.LogWarning("❌ 把持力リクエスト送信失敗: サーバーに接続されていません");
            }
            return;
        }

        SendMessage("REQUEST_GRIP_FORCE");

        if (enableDebugLogs)
        {
            Debug.Log($"📡 把持力リクエスト送信 -> {serverHost}:{serverPort}");
        }
    }
    
    #endregion
    
    #region GUI表示
    
    void OnGUI()
    {
        if (!showGripForceGUI) return;
        
        GUILayout.BeginArea(new Rect(guiPosition.x, guiPosition.y, guiSize.x, guiSize.y));
        
        GUI.Box(new Rect(0, 0, guiSize.x, guiSize.y), "");
        
        GUILayout.BeginVertical();
        
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label) 
        { 
            fontSize = 16, 
            fontStyle = FontStyle.Bold,
            normal = { textColor = Color.white }
        };
        GUILayout.Label($"🔥 TCP把持力制御 ({serverPort})", titleStyle);
        
        GUILayout.Space(5);
        
        string connectionStatus = isConnected ? "✅ 接続中" : "❌ 切断";
        Color connectionColor = isConnected ? Color.green : Color.red;
        GUILayout.Label(connectionStatus, new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = connectionColor }
        });
        
        if (isEpisodeActive)
        {
            GUILayout.Label($"📋 エピソード: {currentEpisodeNumber}", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = Color.cyan }
            });
        }
        
        GUILayout.Space(5);
        
        if (lastReceivedGripForce.HasValue)
        {
            float age = (float)(DateTime.Now - lastGripForceReceiveTime).TotalSeconds;
            Color forceColor = age < 5f ? Color.green : Color.yellow;
            
            GUILayout.Label($"🔥 最新指令: {lastReceivedGripForce.Value:F1}N", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = forceColor }
            });
            GUILayout.Label($"   受信: {age:F1}秒前", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = Color.gray }
            });
        }
        else
        {
            GUILayout.Label("🔥 指令: 未受信", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = Color.gray }
            });
        }
        
        GUILayout.Space(5);
        
        GUILayout.Label($"📊 統計:", new GUIStyle(GUI.skin.label) 
        { 
            fontStyle = FontStyle.Bold,
            normal = { textColor = Color.white }
        });
        GUILayout.Label($"  受信: {totalMessagesReceived}", new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = Color.white }
        });
        GUILayout.Label($"  把持力: {gripForceCommandsReceived}", new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = Color.white }
        });
        GUILayout.Label($"  転送: {gripForceCommandsForwarded}", new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = Color.white }
        });
        // 🔥 行解析エラー統計も表示
        if (lineParsingErrors > 0)
        {
            GUILayout.Label($"  解析エラー: {lineParsingErrors}", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = Color.red }
            });
        }
        
        if (aluminumCan != null)
        {
            string statusText = aluminumCan.IsBroken ? "🔴 つぶれ" : "🟢 正常";
            Color statusColor = aluminumCan.IsBroken ? Color.red : Color.green;
            GUILayout.Label($"缶状態: {statusText}", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = statusColor }
            });
        }
        
        GUILayout.EndVertical();
        GUILayout.EndArea();
    }
    
    #endregion
    
    #region 統計・ユーティリティ
    
    void ShowStatistics()
    {
        float tcpUsageRate = totalMessagesReceived > 0 ? 
            (float)gripForceCommandsReceived / totalMessagesReceived * 100f : 0f;
        float forwardingRate = gripForceCommandsReceived > 0 ? 
            (float)gripForceCommandsForwarded / gripForceCommandsReceived * 100f : 0f;
        
        Debug.Log("=== AluminumCanA2CClient 統計 ===");
        Debug.Log($"接続先: {serverHost}:{serverPort}");
        Debug.Log($"接続試行回数: {connectionAttempts}");
        Debug.Log($"総受信メッセージ: {totalMessagesReceived}");
        Debug.Log($"総送信メッセージ: {totalMessagesSent}");
        Debug.Log($"🔥 把持力指令受信: {gripForceCommandsReceived} ({tcpUsageRate:F1}%)");
        Debug.Log($"🔥 把持力指令転送: {gripForceCommandsForwarded} ({forwardingRate:F1}%)");
        Debug.Log($"🔥 無効指令: {invalidGripForceCommands}");
        Debug.Log($"🔥 行解析エラー: {lineParsingErrors}");
        Debug.Log($"現在接続状態: {(isConnected ? "接続中" : "切断")}");
    }
    
    [ContextMenu("テスト把持力指令送信")]
    public void SendTestGripForceCommand()
    {
        float testForce = UnityEngine.Random.Range(minGripForceValue, maxGripForceValue);
        OnTcpGripForceCommandReceived(testForce);
        
        if (enableDebugLogs)
        {
            Debug.Log($"🧪 テスト把持力指令送信: {testForce:F2}N");
        }
    }
    
    public void OnTcpGripForceCommandReceived(float gripForce)
    {
        lock (gripForceQueueLock)
        {
            pendingGripForceCommand = gripForce;
        }
    }
    
    public A2CClientStatistics GetStatistics()
    {
        return new A2CClientStatistics
        {
            isConnected = isConnected,
            connectionAttempts = connectionAttempts,
            totalMessagesReceived = totalMessagesReceived,
            totalMessagesSent = totalMessagesSent,
            gripForceCommandsReceived = gripForceCommandsReceived,
            gripForceCommandsForwarded = gripForceCommandsForwarded,
            invalidGripForceCommands = invalidGripForceCommands,
            lastGripForceValue = lastReceivedGripForce ?? 0f,
            hasRecentGripForce = lastReceivedGripForce.HasValue && 
                (DateTime.Now - lastGripForceReceiveTime).TotalSeconds < 10f
        };
    }
    
    public void SetGripForceReceivingEnabled(bool enabled)
    {
        enableGripForceReceiving = enabled;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔥 把持力指令受信: {(enabled ? "有効化" : "無効化")}");
        }
    }
    
    public void SetGripForceForwardingEnabled(bool enabled)
    {
        enableGripForceForwarding = enabled;
        
        if (enableDebugLogs)
        {
            Debug.Log($"🔥 把持力指令転送: {(enabled ? "有効化" : "無効化")}");
        }
    }
    
    #endregion
    
    #region ライフサイクル
    
    void OnDestroy()
    {
        Disconnect();

        if (episodeManager != null && eventsHooked)
        {
            episodeManager.OnEpisodeStarted -= OnEpisodeStarted;
            episodeManager.OnEpisodeCompleted -= OnEpisodeCompleted;
            episodeManager.OnSessionCompleted -= OnSessionCompleted;
        }
        eventsHooked = false;
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
            Debug.Log($"🔌 A2Cサーバーから切断: {serverHost}:{serverPort}");
    }
    
    #endregion
}

[System.Serializable]
public class CanStateData
{
    public Vector3 position;
    public Quaternion rotation;
    public Vector3 velocity;
    public Vector3 angularVelocity;
    public bool isBroken;
    public float deformationLevel;
    public bool hasContact;
    public float contactForce;
    public float currentGripForce;
    public float actualGripForce;
    public int episodeNumber;
    public bool episodeActive;
    public float timestamp;
    
    public float lastTcpGripForce;
    public bool hasTcpCommand;
    public float tcpCommandAge;
}

[System.Serializable]
public class A2CClientStatistics
{
    public bool isConnected;
    public int connectionAttempts;
    public int totalMessagesReceived;
    public int totalMessagesSent;
    public int gripForceCommandsReceived;
    public int gripForceCommandsForwarded;
    public int invalidGripForceCommands;
    public float lastGripForceValue;
    public bool hasRecentGripForce;
}