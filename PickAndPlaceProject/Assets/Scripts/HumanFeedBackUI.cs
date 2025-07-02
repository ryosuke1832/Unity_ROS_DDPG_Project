using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System;

/// <summary>
/// 人間フィードバック収集UI
/// 研究用途：強化学習での人間フィードバック収集
/// </summary>
public class HumanFeedbackUI : MonoBehaviour
{
    [Header("UI要素")]
    [SerializeField] private Button tooSoftButton;
    [SerializeField] private Button perfectButton;
    [SerializeField] private Button tooHardButton;
    [SerializeField] private Button resetButton;
    
    [Header("スライダー制御")]
    [SerializeField] private Slider forceSlider;
    [SerializeField] private Slider softnessSlider;
    [SerializeField] private Text forceValueText;
    [SerializeField] private Text softnessValueText;
    
    [Header("情報表示")]
    [SerializeField] private Text statusText;
    [SerializeField] private Text currentForceText;
    [SerializeField] private Text feedbackCountText;
    
    [Header("参照")]
    [SerializeField] private GripperForceController gripperController;
    
    // フィードバック管理
    private int feedbackCount = 0;
    private List<FeedbackData> feedbackHistory = new List<FeedbackData>();
    
    // イベント
    public event Action<float, GraspingState> OnHumanFeedback;
    
    void Start()
    {
        InitializeUI();
        SetupButtonListeners();
        
        // グリッパーコントローラーの自動検出
        if (gripperController == null)
        {
            gripperController = FindObjectOfType<GripperForceController>();
        }
    }
    
    void Update()
    {
        UpdateUI();
    }
    
    /// <summary>
    /// UI初期化
    /// </summary>
    private void InitializeUI()
    {
        // スライダー初期値設定
        if (forceSlider != null)
        {
            forceSlider.minValue = 0.1f;
            forceSlider.maxValue = 50f;
            forceSlider.value = 10f;
        }
        
        if (softnessSlider != null)
        {
            softnessSlider.minValue = 0f;
            softnessSlider.maxValue = 1f;
            softnessSlider.value = 0.5f;
        }
        
        UpdateStatusText("待機中...");
    }
    
    /// <summary>
    /// ボタンリスナー設定
    /// </summary>
    private void SetupButtonListeners()
    {
        if (tooSoftButton != null)
            tooSoftButton.onClick.AddListener(() => SendFeedback(-1f, "力が弱すぎます"));
        
        if (perfectButton != null)
            perfectButton.onClick.AddListener(() => SendFeedback(1f, "完璧な把持です"));
        
        if (tooHardButton != null)
            tooHardButton.onClick.AddListener(() => SendFeedback(-0.5f, "力が強すぎます"));
        
        if (resetButton != null)
            resetButton.onClick.AddListener(ResetSystem);
        
        // スライダーリスナー
        if (forceSlider != null)
            forceSlider.onValueChanged.AddListener(OnForceSliderChanged);
        
        if (softnessSlider != null)
            softnessSlider.onValueChanged.AddListener(OnSoftnessSliderChanged);
    }
    
    /// <summary>
    /// UI情報更新
    /// </summary>
    private void UpdateUI()
    {
        if (gripperController == null) return;
        
        var state = gripperController.GetGraspingState();
        
        // 現在の力表示
        if (currentForceText != null)
        {
            currentForceText.text = $"現在の力: {state.currentForce:F2} N";
        }
        
        // フィードバック回数表示
        if (feedbackCountText != null)
        {
            feedbackCountText.text = $"フィードバック回数: {feedbackCount}";
        }
        
        // ボタンの有効/無効制御
        bool canGiveFeedback = state.isGrasping;
        if (tooSoftButton != null) tooSoftButton.interactable = canGiveFeedback;
        if (perfectButton != null) perfectButton.interactable = canGiveFeedback;
        if (tooHardButton != null) tooHardButton.interactable = canGiveFeedback;
    }
    
    /// <summary>
    /// 力スライダー変更時
    /// </summary>
    private void OnForceSliderChanged(float value)
    {
        if (gripperController != null)
        {
            gripperController.SetTargetGripForce(value);
        }
        
        if (forceValueText != null)
        {
            forceValueText.text = $"目標力: {value:F1} N";
        }
    }
    
    /// <summary>
    /// 柔軟性スライダー変更時
    /// </summary>
    private void OnSoftnessSliderChanged(float value)
    {
        if (gripperController != null)
        {
            gripperController.SetSoftness(value);
        }
        
        if (softnessValueText != null)
        {
            string softnessDesc = value < 0.3f ? "硬い" : value > 0.7f ? "柔らかい" : "普通";
            softnessValueText.text = $"柔軟性: {value:F2} ({softnessDesc})";
        }
    }
    
    /// <summary>
    /// 人間フィードバック送信
    /// </summary>
    private void SendFeedback(float reward, string description)
    {
        if (gripperController == null)
        {
            Debug.LogWarning("GripperForceControllerが見つかりません");
            return;
        }
        
        var currentState = gripperController.GetGraspingState();
        
        // フィードバックデータ記録
        var feedbackData = new FeedbackData
        {
            timestamp = Time.time,
            reward = reward,
            description = description,
            graspState = currentState
        };
        
        feedbackHistory.Add(feedbackData);
        feedbackCount++;
        
        // イベント発火
        OnHumanFeedback?.Invoke(reward, currentState);
        
        // ログ出力
        Debug.Log($"フィードバック: {description} (報酬: {reward}, 力: {currentState.currentForce:F2}N)");
        
        // ステータス更新
        UpdateStatusText($"フィードバック: {description}");
        
        // 視覚フィードバック
        StartCoroutine(ShowFeedbackEffect(reward > 0));
    }
    
    /// <summary>
    /// フィードバック効果表示
    /// </summary>
    private IEnumerator ShowFeedbackEffect(bool isPositive)
    {
        Color originalColor = Color.white;
        Color feedbackColor = isPositive ? Color.green : Color.red;
        
        // ボタンの色を一時的に変更
        Button[] buttons = { tooSoftButton, perfectButton, tooHardButton };
        
        foreach (var button in buttons)
        {
            if (button != null)
            {
                var colors = button.colors;
                originalColor = colors.normalColor;
                colors.normalColor = feedbackColor;
                button.colors = colors;
            }
        }
        
        yield return new WaitForSeconds(0.3f);
        
        // 元の色に戻す
        foreach (var button in buttons)
        {
            if (button != null)
            {
                var colors = button.colors;
                colors.normalColor = originalColor;
                button.colors = colors;
            }
        }
    }
    
    /// <summary>
    /// システムリセット
    /// </summary>
    private void ResetSystem()
    {
        feedbackCount = 0;
        feedbackHistory.Clear();
        
        if (gripperController != null)
        {
            gripperController.StopGrasping();
        }
        
        UpdateStatusText("システムをリセットしました");
        
        // スライダーを初期値に戻す
        if (forceSlider != null) forceSlider.value = 10f;
        if (softnessSlider != null) softnessSlider.value = 0.5f;
    }
    
    /// <summary>
    /// ステータステキスト更新
    /// </summary>
    private void UpdateStatusText(string message)
    {
        if (statusText != null)
        {
            statusText.text = $"[{DateTime.Now:HH:mm:ss}] {message}";
        }
    }
    
    /// <summary>
    /// 把持開始（UIから）
    /// </summary>
    public void StartGraspingFromUI()
    {
        if (gripperController != null)
        {
            float targetForce = forceSlider != null ? forceSlider.value : 10f;
            gripperController.StartGrasping(targetForce, true);
            UpdateStatusText("把持を開始しました");
        }
    }
    
    /// <summary>
    /// 把持停止（UIから）
    /// </summary>
    public void StopGraspingFromUI()
    {
        if (gripperController != null)
        {
            gripperController.StopGrasping();
            UpdateStatusText("把持を停止しました");
        }
    }
    
    /// <summary>
    /// フィードバック履歴の取得
    /// </summary>
    public List<FeedbackData> GetFeedbackHistory()
    {
        return new List<FeedbackData>(feedbackHistory);
    }
    
    /// <summary>
    /// フィードバック統計の取得
    /// </summary>
    public FeedbackStatistics GetFeedbackStatistics()
    {
        if (feedbackHistory.Count == 0)
        {
            return new FeedbackStatistics();
        }
        
        float totalReward = 0f;
        int positiveCount = 0;
        int negativeCount = 0;
        
        foreach (var feedback in feedbackHistory)
        {
            totalReward += feedback.reward;
            if (feedback.reward > 0) positiveCount++;
            else if (feedback.reward < 0) negativeCount++;
        }
        
        return new FeedbackStatistics
        {
            totalFeedbacks = feedbackHistory.Count,
            averageReward = totalReward / feedbackHistory.Count,
            positiveRatio = (float)positiveCount / feedbackHistory.Count,
            negativeRatio = (float)negativeCount / feedbackHistory.Count
        };
    }
    
    // デバッグ用UI表示
    void OnGUI()
    {
        if (Input.GetKey(KeyCode.F1)) // F1キーでデバッグ情報表示
        {
            var stats = GetFeedbackStatistics();
            
            GUILayout.BeginArea(new Rect(Screen.width - 300, 10, 290, 150));
            GUILayout.Label("=== フィードバック統計 ===");
            GUILayout.Label($"総フィードバック数: {stats.totalFeedbacks}");
            GUILayout.Label($"平均報酬: {stats.averageReward:F2}");
            GUILayout.Label($"良い評価率: {stats.positiveRatio * 100:F1}%");
            GUILayout.Label($"悪い評価率: {stats.negativeRatio * 100:F1}%");
            GUILayout.EndArea();
        }
    }
}

/// <summary>
/// フィードバックデータ構造体
/// </summary>
[System.Serializable]
public struct FeedbackData
{
    public float timestamp;
    public float reward;
    public string description;
    public GraspingState graspState;
}

/// <summary>
/// フィードバック統計構造体
/// </summary>
[System.Serializable]
public struct FeedbackStatistics
{
    public int totalFeedbacks;
    public float averageReward;
    public float positiveRatio;
    public float negativeRatio;
}