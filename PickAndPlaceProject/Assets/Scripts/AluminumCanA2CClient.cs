using System;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading;
using System.Collections;

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
    
    [Header("デバッグ")]
    public bool enableDebugLogs = true;
    
    // 通信関連
    private TcpClient tcpClient;
    private NetworkStream stream;
    private Thread communicationThread;
    private bool isConnected = false;
    private bool shouldStop = false;
    
    // 🔥 状態管理（修正版）
    private bool lastCrushedState = false;
    private float lastSendTime = 0f;
    private bool hasEvaluatedThisEpisode = false; // エピソードごとの評価フラグ
    private bool isEpisodeActive = false; // エピソードアクティブ状態
    private float lastForce = 0f; // 前回の力の値
    
    // イベント
    public event System.Action<bool> OnConnectionChanged;
    public event System.Action<float> OnRecommendedForceReceived;
    public event System.Action<float> OnRewardReceived;
    
    void Start()
    {
        // アルミ缶コンポーネント自動検索
        if (aluminumCan == null)
        {
            aluminumCan = FindObjectOfType<IntegratedAluminumCan>();
        }
        
        if (aluminumCan == null)
        {
            Debug.LogError("❌ IntegratedAluminumCanが見つかりません！");
            return;
        }
        
        // A2Cサーバーに接続
        ConnectToA2CServer();
    }
    
    void Update()
    {
        // 🔥 修正：エピソードがアクティブでない場合は送信しない
        if (!isEpisodeActive)
        {
            if (enableDebugLogs)
                // Debug.Log("⏸️ エピソード非アクティブのため送信停止");
            return;
        }
        
        // 評価済みの場合は送信停止
        if (hasEvaluatedThisEpisode)
        {
            if (enableDebugLogs)
                Debug.Log("⏸️ 評価済みのため送信停止中");
            return;
        }
        
        if (isConnected && Time.time - lastSendTime >= sendInterval)
        {
            if (enableDebugLogs)
                Debug.Log("🔄 SendCanState()を呼び出し");
            SendCanState();
            lastSendTime = Time.time;
        }
    }
    
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
                    
                    // メインスレッドで処理
                    UnityMainThreadDispatcher.Instance().Enqueue(() => {
                        ProcessA2CResponse(response);
                    });
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
    }
    
    #endregion
    
    #region メッセージ送信
    
    void SendCanState()
    {
        if (enableDebugLogs)
            Debug.Log("🔍 SendCanState開始");

        if (!isConnected || aluminumCan == null) {
            if (enableDebugLogs)
                Debug.Log("❌ 送信条件未満：接続=" + isConnected + " アルミ缶=" + (aluminumCan != null));
            return;
        }
        
        var state = aluminumCan.GetCurrentState();
        if (enableDebugLogs)
            Debug.Log($"🔍 缶の状態取得：潰れ={state.isBroken} 力={state.appliedForce}");
        
        var message = new CanStateMessage
        {
            is_crushed = state.isBroken,
            grasp_result = DetermineGraspResult(),
            current_force = state.appliedForce,
            accumulated_force = aluminumCan.GetAccumulatedForce(),
            timestamp = Time.time
        };
        
        // 🔥 修正：送信条件を厳格化
        bool shouldSend = false;
        string sendReason = "";
        
        if (message.is_crushed && !lastCrushedState)
        {
            // つぶれた瞬間 - 一度だけ送信
            shouldSend = true;
            sendReason = "つぶれた瞬間";
            hasEvaluatedThisEpisode = true; // 評価完了フラグ
        }
        else if (!message.is_crushed && Math.Abs(message.current_force - lastForce) > 0.1f)
        {
            // 力に有意な変化がある場合のみ送信（0.1N以上の変化）
            shouldSend = true;
            sendReason = "力の変化";
        }
        else if (!message.is_crushed && message.current_force > 0.1f && lastForce <= 0.1f)
        {
            // 把持開始時（力が0から増加した瞬間）
            shouldSend = true;
            sendReason = "把持開始";
        }
        
        if (shouldSend)
        {
            if (enableDebugLogs)
                Debug.Log($"🔥 {sendReason}の送信");
            SendMessage(message);
        }
        else
        {
            if (enableDebugLogs)
                Debug.Log("🔥 送信条件に該当せず（変化なし）");
        }
        
        // 状態更新
        lastCrushedState = message.is_crushed;
        lastForce = message.current_force;
    }

    private string DetermineGraspResult()
    {
        if (aluminumCan.IsBroken)
            return "overgrip";
        
        // AutoEpisodeManagerの成功判定ロジックを使用
        var episodeManager = FindObjectOfType<AutoEpisodeManager>();
        if (episodeManager != null)
        {
            bool isSuccess = episodeManager.DetermineEpisodeSuccess(); // publicにする必要あり
            return isSuccess ? "success" : "undergrip";
        }
        
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
        
        // 🔥 修正：エピソード終了時に送信停止
        isEpisodeActive = false;
        
        if (enableDebugLogs)
            Debug.Log("📋 エピソード終了通知を送信");
    }
    
    public void SendReset()
    {
        // 🔥 修正：リセット時にすべてのフラグをリセット
        hasEvaluatedThisEpisode = false;
        lastCrushedState = false;
        lastForce = 0f;
        isEpisodeActive = true; // エピソード開始
        
        var resetMessage = new SimpleMessage { type = "reset", timestamp = Time.time };
        SendMessage(resetMessage);
        
        if (enableDebugLogs)
            Debug.Log("🔄 リセット通知を送信（評価フラグもリセット）");
    }
    
    // 🔥 新しいメソッド：外部からエピソード状態を制御
    public void OnNewEpisodeStarted()
    {
        hasEvaluatedThisEpisode = false;
        lastCrushedState = false;
        lastForce = 0f;
        isEpisodeActive = true;
        
        if (enableDebugLogs)
            Debug.Log("🆕 新エピソード開始 - 評価フラグをリセット");
    }
    
    public void OnEpisodeCompleted()
    {
        isEpisodeActive = false;
        
        if (enableDebugLogs)
            Debug.Log("🏁 エピソード完了 - 送信を停止");
    }
    
    void SendMessage(object message)
    {
        if (!isConnected || stream == null) return;
        
        try
        {
            string json = JsonUtility.ToJson(message);
            byte[] data = Encoding.UTF8.GetBytes(json);
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
        try
        {
            // 簡単なJSON解析（タイプのみ抽出）
            string type = ExtractTypeFromJson(response);
            
            if (enableDebugLogs)
                Debug.Log($"📥 受信: タイプ={type}");
            
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
                    // 将来のA2C応答処理用
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
                    
                default:
                    if (enableDebugLogs)
                        Debug.Log($"❓ 不明な応答: {response}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ 応答処理エラー: {e.Message}");
            if (enableDebugLogs)
                Debug.Log($"受信データ: {response}");
        }
    }
    
    // 簡単なJSON解析（typeフィールドのみ抽出）
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