// AluminumCanA2CClient.cs の改良版
// AutoEpisodeManagerとの連携を強化し、把持力指令の受信・転送機能を追加

using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// A2C強化学習サーバーとの通信クライアント
/// AutoEpisodeManagerとの連携により把持力指令を受信・転送
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
    
    // 通信関連
    private TcpClient tcpClient;
    private NetworkStream stream;
    private Thread communicationThread;
    private bool isConnected = false;
    private bool shouldStop = false;
    private float lastSendTime = 0f;
    private int retryCount = 0;
    
    // メッセージキュー（スレッドセーフ）
    private Queue<string> messageQueue = new Queue<string>();
    private readonly object queueLock = new object();
    
    // エピソード制御
    private bool isEpisodeActive = false;
    private bool hasEvaluatedThisEpisode = false;
    private int currentEpisodeNumber = 0;
    // 一回のエピソードで結果を送信したかのフラグ
    private bool episodeResultSent = false;
    
    // 🔥 把持力指令関連（キューの上限は1つ）
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
    
    // イベント
    public System.Action<bool> OnConnectionChanged;
    public System.Action<float> OnGripForceCommandReceived; // 🔥 新規追加
    public System.Action<string> OnMessageReceived;
    public System.Action<int> OnEpisodeStateChanged;
    
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
        ProcessGripForceCommands(); // 🔥 新規追加
        
        if (!isEpisodeActive || hasEvaluatedThisEpisode)
        {
            return;
        }
        
        if (isConnected && Time.time - lastSendTime >= sendInterval)
        {
            SendCanState();
            lastSendTime = Time.time;
        }
        
        // 自動再接続
        if (!isConnected && autoReconnect && Time.time - lastSendTime > reconnectInterval)
        {
            AttemptReconnection();
        }
    }
    
    #region 初期化
    
    void InitializeComponents()
    {
        // コンポーネントの自動検索
        if (aluminumCan == null)
            aluminumCan = FindObjectOfType<IntegratedAluminumCan>();
            
        if (gripForceController == null)
            gripForceController = FindObjectOfType<SimpleGripForceController>();
            
        if (gripperInterface == null)
            gripperInterface = FindObjectOfType<GripperTargetInterface>();
        
        // 🔥 AutoEpisodeManagerの自動検索と連携設定
        if (autoFindEpisodeManager && episodeManager == null)
        {
            episodeManager = FindObjectOfType<AutoEpisodeManager>();
        }
        
        SetupEpisodeManagerIntegration();
        
        if (enableDebugLogs)
        {
            Debug.Log("=== AluminumCanA2CClient 初期化 ===");
            Debug.Log($"AluminumCan: {(aluminumCan != null ? "✅" : "❌")}");
            Debug.Log($"GripForceController: {(gripForceController != null ? "✅" : "❌")}");
            Debug.Log($"GripperInterface: {(gripperInterface != null ? "✅" : "❌")}");
            Debug.Log($"🔥 EpisodeManager: {(episodeManager != null ? "✅連携設定" : "❌未設定")}");
            Debug.Log($"🔥 把持力指令受信: {(enableGripForceReceiving ? "有効" : "無効")}");
            Debug.Log($"🔥 把持力指令転送: {(enableGripForceForwarding ? "有効" : "無効")}");
        }
    }
    
    // 🔥 AutoEpisodeManagerとの連携設定
    void SetupEpisodeManagerIntegration()
    {
        if (episodeManager == null) return;
        
        // エピソード開始/終了イベントの購読
        episodeManager.OnEpisodeStarted += OnEpisodeStarted;
        episodeManager.OnEpisodeCompleted += OnEpisodeCompleted;
        episodeManager.OnSessionCompleted += OnSessionCompleted;
        
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
    
    /// <summary>
    /// 把持力指令ストックの処理（常に最新1件のみ）
    /// </summary>
    void ProcessGripForceCommands()
    {
        if (!enableGripForceReceiving) return;
        
        lock (gripForceQueueLock)
        {
            if (pendingGripForceCommand.HasValue)
            {
                float gripForce = pendingGripForceCommand.Value;
                pendingGripForceCommand = null; // ストックを空にする
                ProcessGripForceCommand(gripForce);
            }
        }
    }
    
    /// <summary>
    /// 個別の把持力指令を処理
    /// </summary>
    void ProcessGripForceCommand(float gripForce)
    {
        Debug.Log($"🔥 把持力指令処理開始: {gripForce:F2}N");
        
        // 値の妥当性チェック
        if (gripForce < minGripForceValue || gripForce > maxGripForceValue)
        {
            invalidGripForceCommands++;
            
            Debug.LogWarning($"⚠️ 無効な把持力指令: {gripForce:F2}N (範囲: {minGripForceValue:F1}-{maxGripForceValue:F1}N)");
            return;
        }
        
        lastReceivedGripForce = gripForce;
        lastGripForceReceiveTime = DateTime.Now;
        gripForceCommandsReceived++;
        
        Debug.Log($"🔥 把持力指令受信完了: {gripForce:F2}N (受信数: {gripForceCommandsReceived})");

        if (enableGripForceForwarding)
        {
            if (OnGripForceCommandReceived != null)
            {
                OnGripForceCommandReceived.Invoke(gripForce);
                Debug.Log($"🔥 イベント発火完了");
            }
            else if (episodeManager != null)
            {
                episodeManager.OnTcpGripForceCommandReceived(gripForce);
                gripForceCommandsForwarded++;
                Debug.Log($"🔥 把持力指令転送完了: {gripForce:F2}N -> AutoEpisodeManager (転送数: {gripForceCommandsForwarded})");
            }
            else
            {
                Debug.LogWarning($"⚠️ EpisodeManagerが設定されていません");
            }
        }
        else
        {
            Debug.LogWarning($"⚠️ 把持力転送が無効化されています");
        }

        Debug.Log($"🔥 把持力指令処理完了: {gripForce:F2}N");
    }
    
    /// <summary>
    /// 受信したメッセージから把持力指令を抽出
    /// </summary>
    bool TryParseGripForceCommand(string message, out float gripForce)
    {
        gripForce = 0f;
        
        // 🔥 新しいJSON形式への対応: {"type": "grip_force_command", "target_force": 10.0, ...}
        try
        {
            if (message.Contains("grip_force_command") && message.Contains("target_force"))
            {
                // target_forceの値を抽出
                int targetForceIndex = message.IndexOf("target_force");
                if (targetForceIndex >= 0)
                {
                    // "target_force": の後の値を取得
                    int colonIndex = message.IndexOf(":", targetForceIndex);
                    if (colonIndex >= 0)
                    {
                        // コロンの後から次のカンマまたは}まで
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
        
        // 従来のテキスト形式への対応: "GRIP_FORCE:15.5" または "grip_force:15.5"
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
        
        // 旧JSON形式の試行: {"grip_force": 15.5}
        try
        {
            if (message.Contains("grip_force") && message.Contains("{") && message.Contains("}"))
            {
                // 簡易JSON解析（JsonUtilityは使用しないで手動解析）
                int startIndex = message.IndexOf("grip_force") + "grip_force".Length;
                string remaining = message.Substring(startIndex);
                
                int colonIndex = remaining.IndexOf(':');
                if (colonIndex >= 0)
                {
                    string valueStr = remaining.Substring(colonIndex + 1);
                    valueStr = valueStr.Trim().TrimStart('"').TrimEnd('"', '}', ',', ' ');
                    
                    if (float.TryParse(valueStr, out gripForce))
                    {
                        if (enableDebugLogs)
                        {
                            Debug.Log($"🔥 旧JSON形式の把持力指令解析成功: {gripForce:F2}N");
                        }
                        return true;
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (enableDebugLogs)
            {
                Debug.LogWarning($"旧JSON解析エラー: {ex.Message}");
            }
        }
        
        return false;
    }
    
    #endregion
    
    #region 通信処理
    
    void ConnectToA2CServer()
    {
        try
        {
            connectionAttempts++;
            tcpClient = new TcpClient(serverHost, serverPort);
            stream = tcpClient.GetStream();
            isConnected = true;
            retryCount = 0;
            
            OnConnectionChanged?.Invoke(true);
            
            if (enableDebugLogs)
                Debug.Log($"✅ A2Cサーバーに接続しました (試行回数: {connectionAttempts})");
            
            // 通信スレッド開始
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
            Debug.Log($"📨 処理開始: {message}");
        }
        
        // 🔥 把持力指令の解析と処理
        if (enableGripForceReceiving && TryParseGripForceCommand(message, out float gripForce))
        {
            lock (gripForceQueueLock)
            {
                pendingGripForceCommand = gripForce; // ストックは常に1つだけ保持
            }

            Debug.Log($"🔥 把持力指令を検出してストックを更新: {gripForce:F2}N");
        }
        else
        {
            // 把持力指令でない場合のデバッグ
            if (enableGripForceReceiving && (message.Contains("grip_force") || message.Contains("target_force")))
            {
                Debug.LogWarning($"⚠️ 把持力関連メッセージの解析に失敗: {message.Substring(0, Math.Min(100, message.Length))}...");
            }
        }
        
        // その他のメッセージ処理
        OnMessageReceived?.Invoke(message);
        
        if (enableVerboseReceiveLog)
        {
            Debug.Log($"📨 メッセージ処理完了: {message}");
        }
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
        
        // アルミ缶の状態
        if (aluminumCan != null)
        {
            state.position = aluminumCan.transform.position;
            state.rotation = aluminumCan.transform.rotation;
            state.isBroken = aluminumCan.IsBroken;
            
            // 既存のGetCurrentState()メソッドを使用
            var canState = aluminumCan.GetCurrentState();
            state.deformationLevel = canState.deformation;
            
            var rb = aluminumCan.GetComponent<Rigidbody>();
            if (rb != null)
            {
                state.velocity = rb.velocity;
                state.angularVelocity = rb.angularVelocity;
            }
        }
        
        // グリッパーの状態
        if (gripperInterface != null)
        {
            state.hasContact = gripperInterface.HasValidContact();
            
            // 既存のメソッドがないため、デフォルト値を使用
            state.contactForce = 0f; // TODO: 実際の接触力の取得方法を実装
        }
        
        // 把持力の状態
        if (gripForceController != null)
        {
            state.currentGripForce = gripForceController.baseGripForce;
            
            // GetCurrentForce()がないため、代替手段を使用
            state.actualGripForce = gripForceController.baseGripForce; // TODO: 実際の現在力の取得
        }
        
        // エピソード情報
        state.episodeNumber = currentEpisodeNumber;
        state.episodeActive = isEpisodeActive;
        state.timestamp = Time.time;
        
        // 🔥 TCP把持力情報
        state.lastTcpGripForce = lastReceivedGripForce ?? 0f;
        state.hasTcpCommand = lastReceivedGripForce.HasValue;
        state.tcpCommandAge = lastReceivedGripForce.HasValue ? 
            (float)(DateTime.Now - lastGripForceReceiveTime).TotalSeconds : -1f;
        
        return state;
    }
    
    private string CreateStateJson(CanStateData state)
    {
        // 手動でJSON文字列を作成（JsonUtilityを使わない場合）
        var json = new StringBuilder();
        json.Append("{");
        
        // 基本状態
        json.Append($"\"episode\":{state.episodeNumber},");
        json.Append($"\"active\":{state.episodeActive.ToString().ToLower()},");
        json.Append($"\"timestamp\":{state.timestamp:F3},");
        
        // アルミ缶状態
        json.Append($"\"position\":[{state.position.x:F3},{state.position.y:F3},{state.position.z:F3}],");
        json.Append($"\"velocity\":[{state.velocity.x:F3},{state.velocity.y:F3},{state.velocity.z:F3}],");
        json.Append($"\"broken\":{state.isBroken.ToString().ToLower()},");
        json.Append($"\"deformation\":{state.deformationLevel:F3},");
        
        // グリッパー状態
        json.Append($"\"contact\":{state.hasContact.ToString().ToLower()},");
        json.Append($"\"contact_force\":{state.contactForce:F3},");
        json.Append($"\"grip_force\":{state.currentGripForce:F3},");
        json.Append($"\"actual_grip_force\":{state.actualGripForce:F3},");
        
        // 🔥 TCP把持力情報
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
        // 次のエピソードのために結果送信フラグをリセット
        episodeResultSent = false;
    }

    public void SendEpisodeEnd()
    {
        SendMessage("EPISODE_END");
        hasEvaluatedThisEpisode = true;
    }

    /// <summary>
    /// エピソードの成功/失敗結果を送信
    /// </summary>
    /// <param name="wasSuccessful">成功した場合は true</param>
    public void SendEpisodeResult(bool wasSuccessful)
    {
        if (episodeResultSent) return;

        string resultMessage = wasSuccessful ? "RESULT_SUCCESS" : "RESULT_FAIL";
        SendMessage(resultMessage);
        Debug.Log($"📤 エピソード結果送信: {resultMessage}");
        episodeResultSent = true;
    }
    
    #endregion
    
    #region GUI表示
    
    void OnGUI()
    {
        if (!showGripForceGUI) return;
        
        GUILayout.BeginArea(new Rect(guiPosition.x, guiPosition.y, guiSize.x, guiSize.y));
        
        // 背景ボックス
        GUI.Box(new Rect(0, 0, guiSize.x, guiSize.y), "");
        
        GUILayout.BeginVertical();
        
        // タイトル
        GUIStyle titleStyle = new GUIStyle(GUI.skin.label) 
        { 
            fontSize = 16, 
            fontStyle = FontStyle.Bold,
            normal = { textColor = Color.white }
        };
        GUILayout.Label("🔥 TCP把持力制御", titleStyle);
        
        GUILayout.Space(5);
        
        // 接続状態
        string connectionStatus = isConnected ? "✅ 接続中" : "❌ 切断";
        Color connectionColor = isConnected ? Color.green : Color.red;
        GUILayout.Label(connectionStatus, new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = connectionColor }
        });
        
        // エピソード状態
        if (isEpisodeActive)
        {
            GUILayout.Label($"📋 エピソード: {currentEpisodeNumber}", new GUIStyle(GUI.skin.label) 
            { 
                normal = { textColor = Color.cyan }
            });
        }
        
        GUILayout.Space(5);
        
        // 🔥 把持力指令情報
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
        
        // 統計情報
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
        GUILayout.Label($"  無効: {invalidGripForceCommands}", new GUIStyle(GUI.skin.label) 
        { 
            normal = { textColor = Color.red }
        });
        
        // アルミ缶状態
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
        Debug.Log($"接続試行回数: {connectionAttempts}");
        Debug.Log($"総受信メッセージ: {totalMessagesReceived}");
        Debug.Log($"総送信メッセージ: {totalMessagesSent}");
        Debug.Log($"🔥 把持力指令受信: {gripForceCommandsReceived} ({tcpUsageRate:F1}%)");
        Debug.Log($"🔥 把持力指令転送: {gripForceCommandsForwarded} ({forwardingRate:F1}%)");
        Debug.Log($"🔥 無効指令: {invalidGripForceCommands}");
        Debug.Log($"現在接続状態: {(isConnected ? "接続中" : "切断")}");
    }
    
    /// <summary>
    /// 手動で把持力指令をテスト送信
    /// </summary>
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
    
    /// <summary>
    /// 外部から呼び出し可能な把持力指令受信メソッド
    /// </summary>
    public void OnTcpGripForceCommandReceived(float gripForce)
    {
        lock (gripForceQueueLock)
        {
            pendingGripForceCommand = gripForce; // 外部からの指令も1件のみ保持
        }
    }
    
    /// <summary>
    /// 統計情報の取得
    /// </summary>
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
    
    /// <summary>
    /// 設定の動的変更
    /// </summary>
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
        
        // イベントの解除
        if (episodeManager != null)
        {
            episodeManager.OnEpisodeStarted -= OnEpisodeStarted;
            episodeManager.OnEpisodeCompleted -= OnEpisodeCompleted;
            episodeManager.OnSessionCompleted -= OnSessionCompleted;
        }
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

/// <summary>
/// アルミ缶の状態データ
/// </summary>
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
    
    // 🔥 TCP把持力関連
    public float lastTcpGripForce;
    public bool hasTcpCommand;
    public float tcpCommandAge;
}

/// <summary>
/// A2CClient統計情報
/// </summary>
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