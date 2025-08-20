// TcpGripForceConnector.cs
// AutoEpisodeManagerとAluminumCanA2CClientを連携させるためのコネクター

using UnityEngine;

/// <summary>
/// AutoEpisodeManagerとAluminumCanA2CClientの間でTCP把持力指令を仲介するコネクター
/// 既存のコードを最小限の変更で連携させるために作成
/// </summary>
public class TcpGripForceConnector : MonoBehaviour
{
    [Header("🔗 連携コンポーネント")]
    public AutoEpisodeManager episodeManager;
    public AluminumCanA2CClient a2cClient;
    
    [Header("⚙️ 設定")]
    public bool enableAutoConnection = true;
    public bool enableDebugLogs = true;
    
    void Start()
    {
        if (enableAutoConnection)
        {
            SetupConnections();
        }
    }
    
    /// <summary>
    /// 自動的に連携を設定
    /// </summary>
    void SetupConnections()
    {
        // コンポーネントの自動検索
        if (episodeManager == null)
            episodeManager = FindObjectOfType<AutoEpisodeManager>();
            
        if (a2cClient == null)
            a2cClient = FindObjectOfType<AluminumCanA2CClient>();
        
        if (episodeManager == null || a2cClient == null)
        {
            Debug.LogError("❌ AutoEpisodeManager または AluminumCanA2CClient が見つかりません");
            return;
        }
        
        // A2CClientの把持力指令受信イベントをAutoEpisodeManagerに接続
        if (a2cClient.OnGripForceCommandReceived == null)
            a2cClient.OnGripForceCommandReceived = new System.Action<float>(OnGripForceReceived);
        else
            a2cClient.OnGripForceCommandReceived += OnGripForceReceived;
        
        // 相互参照の設定
        if (a2cClient.episodeManager == null)
            a2cClient.episodeManager = episodeManager;
            
        if (episodeManager.a2cClient == null)
            episodeManager.a2cClient = a2cClient;

        // 🔥 TCP待機タイムアウトを2秒に設定
        if (episodeManager != null)
        {
            episodeManager.SetTcpCommandWaitTimeout(2.0f);
        }
        
        if (enableDebugLogs)
        {
            Debug.Log("✅ TCP把持力連携が設定されました");
            Debug.Log($"   AutoEpisodeManager: {episodeManager.name}");
            Debug.Log($"   AluminumCanA2CClient: {a2cClient.name}");
        }
    }
    
    /// <summary>
    /// A2CClientから把持力指令を受信した際の処理
    /// </summary>
    void OnGripForceReceived(float gripForce)
    {
        if (episodeManager != null)
        {
            episodeManager.OnTcpGripForceCommandReceived(gripForce);
            
            if (enableDebugLogs)
            {
                Debug.Log($"🔥 把持力指令を転送: {gripForce:F2}N (A2CClient → EpisodeManager)");
            }
        }
    }
    
    /// <summary>
    /// 手動で連携を設定
    /// </summary>
    [ContextMenu("手動連携設定")]
    public void ManualSetupConnections()
    {
        SetupConnections();
    }
    
    /// <summary>
    /// 連携をテスト
    /// </summary>
    [ContextMenu("連携テスト")]
    public void TestConnection()
    {
        if (episodeManager == null || a2cClient == null)
        {
            Debug.LogError("❌ 連携コンポーネントが設定されていません");
            return;
        }
        
        float testForce = Random.Range(5f, 25f);
        
        if (enableDebugLogs)
        {
            Debug.Log($"🧪 連携テスト開始: テスト把持力 {testForce:F2}N");
        }
        
        // A2CClientを通じてテスト指令を送信
        a2cClient.OnTcpGripForceCommandReceived(testForce);
        
        if (enableDebugLogs)
        {
            Debug.Log("✅ 連携テスト完了");
        }
    }
    
    /// <summary>
    /// 現在の連携状態を確認
    /// </summary>
    [ContextMenu("連携状態確認")]
    public void CheckConnectionStatus()
    {
        Debug.Log("=== 🔍 TCP把持力連携状態 ===");
        
        // コンポーネント存在確認
        Debug.Log($"AutoEpisodeManager: {(episodeManager != null ? "✅" : "❌")}");
        Debug.Log($"AluminumCanA2CClient: {(a2cClient != null ? "✅" : "❌")}");
        
        if (episodeManager != null)
        {
            // AutoEpisodeManagerの設定確認（リフレクションを使用）
            var tcpEnabledField = episodeManager.GetType().GetField("enableTcpGripForceControl", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            
            if (tcpEnabledField != null)
            {
                bool tcpEnabled = (bool)tcpEnabledField.GetValue(episodeManager);
                Debug.Log($"EpisodeManager TCP制御: {(tcpEnabled ? "✅有効" : "❌無効")}");
            }
        }
        
        if (a2cClient != null)
        {
            Debug.Log($"A2CClient 把持力受信: {(a2cClient.enableGripForceReceiving ? "✅有効" : "❌無効")}");
            Debug.Log($"A2CClient 把持力転送: {(a2cClient.enableGripForceForwarding ? "✅有効" : "❌無効")}");
            Debug.Log($"A2CClient 接続状態: {(a2cClient.GetStatistics().isConnected ? "✅接続中" : "❌切断")}");
        }
        
        // 相互参照確認
        bool crossReferenceOK = episodeManager != null && a2cClient != null && 
                               episodeManager.a2cClient == a2cClient && 
                               a2cClient.episodeManager == episodeManager;
        
        Debug.Log($"相互参照: {(crossReferenceOK ? "✅正常" : "❌設定不完全")}");
        
        // イベント接続確認
        bool eventConnected = a2cClient != null && a2cClient.OnGripForceCommandReceived != null;
        Debug.Log($"イベント接続: {(eventConnected ? "✅設定済み" : "❌未設定")}");
        
        Debug.Log("========================");
    }
    
    void OnDestroy()
    {
        // イベントの解除
        if (a2cClient != null && a2cClient.OnGripForceCommandReceived != null)
        {
            a2cClient.OnGripForceCommandReceived -= OnGripForceReceived;
        }
    }
}