using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// 把持力制御研究に最適化されたUI構造を自動構築（重なり修正版）
/// </summary>
public class OptimizedUIStructure : MonoBehaviour
{
    [Header("UI自動構築設定")]
    [SerializeField] private bool buildOnStart = true;
    [SerializeField] private bool replaceExistingUI = false;
    
    [Header("デザイン設定")]
    [SerializeField] private Color panelColor = new Color(0.2f, 0.2f, 0.2f, 0.8f);
    [SerializeField] private Color buttonColor = new Color(0.4f, 0.6f, 0.8f, 1f);
    [SerializeField] private Font customFont;
    
    void Start()
    {
        if (buildOnStart)
        {
            BuildOptimizedUI();
        }
    }
    
    /// <summary>
    /// Build optimized UI structure
    /// </summary>
    [ContextMenu("Build Optimized UI")]
    public void BuildOptimizedUI()
    {
        Debug.Log("Building optimized UI structure...");
        
        // Handle existing UI
        if (replaceExistingUI)
        {
            DestroyExistingUI();
        }
        
        // Build new UI structure
        GameObject mainCanvas = CreateMainCanvas();
        CreateControlPanel(mainCanvas);
        CreateFeedbackPanel(mainCanvas);
        CreateDataPanel(mainCanvas);
        CreateDebugPanel(mainCanvas);
        
        Debug.Log("UI structure build completed!");
    }
    
    /// <summary>
    /// Destroy existing UI
    /// </summary>
    private void DestroyExistingUI()
    {
        Canvas[] canvases = FindObjectsOfType<Canvas>();
        foreach (Canvas canvas in canvases)
        {
            if (canvas.name == "Canvas" || canvas.name == "MainCanvas") 
            {
                DestroyImmediate(canvas.gameObject);
            }
        }
    }
    
    /// <summary>
    /// Create main Canvas
    /// </summary>
    private GameObject CreateMainCanvas()
    {
        GameObject canvasObj = new GameObject("MainCanvas");
        Canvas canvas = canvasObj.AddComponent<Canvas>();
        canvas.renderMode = RenderMode.ScreenSpaceOverlay;
        canvas.sortingOrder = 0;
        
        CanvasScaler scaler = canvasObj.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);
        scaler.matchWidthOrHeight = 0.5f;
        
        canvasObj.AddComponent<GraphicRaycaster>();
        
        // Ensure EventSystem exists
        if (FindObjectOfType<UnityEngine.EventSystems.EventSystem>() == null)
        {
            GameObject eventSystem = new GameObject("EventSystem");
            eventSystem.AddComponent<UnityEngine.EventSystems.EventSystem>();
            eventSystem.AddComponent<UnityEngine.EventSystems.StandaloneInputModule>();
        }
        
        return canvasObj;
    }
    
    /// <summary>
    /// Create control panel (左上)
    /// </summary>
    private void CreateControlPanel(GameObject parent)
    {
        // 左上アンカーで配置
        GameObject panel = CreatePanelWithAnchor("ControlPanel", parent, 
                                                 new Vector2(0f, 1f), new Vector2(0f, 1f),
                                                 new Vector2(185f, -135f), new Vector2(350, 250));
        
        // Title
        CreateText("ControlTitle", panel, "Grasp Force Control", new Vector2(0, 100), 18, FontStyle.Bold);
        
        // Force control slider
        CreateText("ForceLabel", panel, "Target Force (N):", new Vector2(-100, 60), 14);
        GameObject forceSlider = CreateSlider("ForceSlider", panel, new Vector2(50, 60), 0.1f, 50f, 10f);
        CreateText("ForceValue", panel, "10.0 N", new Vector2(150, 60), 12);
        
        // Softness slider
        CreateText("SoftnessLabel", panel, "Softness:", new Vector2(-100, 20), 14);
        GameObject softnessSlider = CreateSlider("SoftnessSlider", panel, new Vector2(50, 20), 0f, 1f, 0.5f);
        CreateText("SoftnessValue", panel, "0.5", new Vector2(150, 20), 12);
        
        // Control buttons
        CreateButton("StartButton", panel, "Start Grasp", new Vector2(-80, -40), new Vector2(120, 30), Color.green);
        CreateButton("StopButton", panel, "Stop Grasp", new Vector2(80, -40), new Vector2(120, 30), Color.red);
        
        // Reset button
        CreateButton("ResetButton", panel, "Reset", new Vector2(0, -80), new Vector2(160, 30), Color.gray);
    }
    
    /// <summary>
    /// Create feedback panel (右上)
    /// </summary>
    private void CreateFeedbackPanel(GameObject parent)
    {
        // 右上アンカーで配置
        GameObject panel = CreatePanelWithAnchor("FeedbackPanel", parent,
                                                 new Vector2(1f, 1f), new Vector2(1f, 1f),
                                                 new Vector2(-185f, -135f), new Vector2(350, 250));
        
        // Title
        CreateText("FeedbackTitle", panel, "Human Feedback", new Vector2(0, 100), 18, FontStyle.Bold);
        
        // Feedback buttons
        CreateButton("TooSoftButton", panel, "Too Soft", new Vector2(0, 40), new Vector2(200, 35), Color.blue);
        CreateButton("PerfectButton", panel, "Perfect", new Vector2(0, 0), new Vector2(200, 35), Color.green);
        CreateButton("TooHardButton", panel, "Too Hard", new Vector2(0, -40), new Vector2(200, 35), Color.red);
        
        // Statistics display
        CreateText("FeedbackCount", panel, "Feedback Count: 0", new Vector2(0, -80), 12);
    }
    
    /// <summary>
    /// Create data panel (左下)
    /// </summary>
    private void CreateDataPanel(GameObject parent)
    {
        // 左下アンカーで配置
        GameObject panel = CreatePanelWithAnchor("DataPanel", parent,
                                                 new Vector2(0f, 0f), new Vector2(0f, 0f),
                                                 new Vector2(185f, 135f), new Vector2(350, 250));
        
        // Title
        CreateText("DataTitle", panel, "Real-time Data", new Vector2(0, 100), 18, FontStyle.Bold);
        
        // Data display - 縦に整列
        CreateText("CurrentForce", panel, "Current Force: 0.0 N", new Vector2(0, 60), 14);
        CreateText("GripperPosition", panel, "Gripper Position: 0.000", new Vector2(0, 30), 14);
        CreateText("GraspStatus", panel, "Grasp Status: Standby", new Vector2(0, 0), 14);
        CreateText("ObjectInfo", panel, "Object: Not Detected", new Vector2(0, -30), 14);
        CreateText("SuccessRate", panel, "Success Rate: 0%", new Vector2(0, -60), 14);
    }
    
    /// <summary>
    /// Create debug panel (右下)
    /// </summary>
    private void CreateDebugPanel(GameObject parent)
    {
        // 右下アンカーで配置
        GameObject panel = CreatePanelWithAnchor("DebugPanel", parent,
                                                 new Vector2(1f, 0f), new Vector2(1f, 0f),
                                                 new Vector2(-185f, 135f), new Vector2(350, 250));
        
        // Title
        CreateText("DebugTitle", panel, "Debug Information", new Vector2(0, 100), 18, FontStyle.Bold);
        
        // Debug information - 縦に整列して重なりを回避
        CreateText("LeftForce", panel, "Left Force: 0.0 N", new Vector2(-80, 60), 12);
        CreateText("RightForce", panel, "Right Force: 0.0 N", new Vector2(80, 60), 12);
        CreateText("ForceBalance", panel, "Force Balance: 0.0", new Vector2(0, 30), 12);
        CreateText("PIDOutput", panel, "PID Output: 0.0", new Vector2(0, 0), 12);
        CreateText("ErrorValue", panel, "Error: 0.0", new Vector2(0, -30), 12);
        
        // Debug toggle
        CreateToggle("ShowDebug", panel, "Show Details", new Vector2(0, -70), true);
    }
    
    /// <summary>
    /// アンカー指定パネル作成ヘルパー（修正版）
    /// </summary>
    private GameObject CreatePanelWithAnchor(string name, GameObject parent, 
                                           Vector2 anchorMin, Vector2 anchorMax,
                                           Vector2 anchoredPosition, Vector2 size)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent.transform, false);
        
        RectTransform rect = panel.AddComponent<RectTransform>();
        rect.anchorMin = anchorMin;
        rect.anchorMax = anchorMax;
        rect.anchoredPosition = anchoredPosition;
        rect.sizeDelta = size;
        
        Image image = panel.AddComponent<Image>();
        image.color = panelColor;
        
        return panel;
    }
    
    /// <summary>
    /// パネル作成ヘルパー（従来版 - 互換性のため残す）
    /// </summary>
    private GameObject CreatePanel(string name, GameObject parent, Vector2 position, Vector2 size)
    {
        GameObject panel = new GameObject(name);
        panel.transform.SetParent(parent.transform, false);
        
        RectTransform rect = panel.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        
        Image image = panel.AddComponent<Image>();
        image.color = panelColor;
        
        return panel;
    }
    
    /// <summary>
    /// Text creation helper with improved font handling
    /// </summary>
    private GameObject CreateText(string name, GameObject parent, string text, Vector2 position, int fontSize, FontStyle fontStyle = FontStyle.Normal)
    {
        GameObject textObj = new GameObject(name);
        textObj.transform.SetParent(parent.transform, false);
        
        RectTransform rect = textObj.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(300, 30); // Increased width for better text display
        
        Text textComponent = textObj.AddComponent<Text>();
        textComponent.text = text;
        
        // Improved font handling
        if (customFont != null)
        {
            textComponent.font = customFont;
        }
        else
        {
            // Try to use LegacyRuntime.ttf or Arial as fallback
            Font defaultFont = Resources.Load<Font>("LegacyRuntime");
            if (defaultFont == null)
            {
                defaultFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            }
            if (defaultFont == null)
            {
                defaultFont = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
            textComponent.font = defaultFont;
        }
        
        textComponent.fontSize = fontSize;
        textComponent.fontStyle = fontStyle;
        textComponent.color = Color.white;
        textComponent.alignment = TextAnchor.MiddleCenter;
        
        // Add outline for better visibility
        Outline outline = textObj.AddComponent<Outline>();
        outline.effectColor = Color.black;
        outline.effectDistance = new Vector2(1, 1);
        
        return textObj;
    }
    
    /// <summary>
    /// Button creation helper with improved text handling
    /// </summary>
    private GameObject CreateButton(string name, GameObject parent, string text, Vector2 position, Vector2 size, Color color)
    {
        GameObject buttonObj = new GameObject(name);
        buttonObj.transform.SetParent(parent.transform, false);
        
        RectTransform rect = buttonObj.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = size;
        
        Image image = buttonObj.AddComponent<Image>();
        image.color = color;
        
        Button button = buttonObj.AddComponent<Button>();
        
        // Button text with improved font handling
        GameObject textObj = new GameObject("Text");
        textObj.transform.SetParent(buttonObj.transform, false);
        
        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;
        
        Text textComponent = textObj.AddComponent<Text>();
        textComponent.text = text;
        
        // Improved font handling for buttons
        if (customFont != null)
        {
            textComponent.font = customFont;
        }
        else
        {
            Font defaultFont = Resources.Load<Font>("LegacyRuntime");
            if (defaultFont == null)
            {
                defaultFont = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            }
            if (defaultFont == null)
            {
                defaultFont = Resources.GetBuiltinResource<Font>("Arial.ttf");
            }
            textComponent.font = defaultFont;
        }
        
        textComponent.fontSize = 14;
        textComponent.color = Color.white;
        textComponent.alignment = TextAnchor.MiddleCenter;
        textComponent.fontStyle = FontStyle.Bold;
        
        // Add outline for better visibility
        Outline outline = textObj.AddComponent<Outline>();
        outline.effectColor = Color.black;
        outline.effectDistance = new Vector2(1, 1);
        
        return buttonObj;
    }
    
    /// <summary>
    /// スライダー作成ヘルパー
    /// </summary>
    private GameObject CreateSlider(string name, GameObject parent, Vector2 position, float min, float max, float value)
    {
        GameObject sliderObj = new GameObject(name);
        sliderObj.transform.SetParent(parent.transform, false);
        
        RectTransform rect = sliderObj.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(120, 20);
        
        Slider slider = sliderObj.AddComponent<Slider>();
        slider.minValue = min;
        slider.maxValue = max;
        slider.value = value;
        
        // 背景
        GameObject background = new GameObject("Background");
        background.transform.SetParent(sliderObj.transform, false);
        RectTransform bgRect = background.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;
        Image bgImage = background.AddComponent<Image>();
        bgImage.color = Color.gray;
        
        // ハンドル
        GameObject handle = new GameObject("Handle");
        handle.transform.SetParent(sliderObj.transform, false);
        RectTransform handleRect = handle.AddComponent<RectTransform>();
        handleRect.sizeDelta = new Vector2(20, 20);
        Image handleImage = handle.AddComponent<Image>();
        handleImage.color = Color.white;
        
        slider.targetGraphic = handleImage;
        slider.handleRect = handleRect;
        
        return sliderObj;
    }
    
    /// <summary>
    /// トグル作成ヘルパー
    /// </summary>
    private GameObject CreateToggle(string name, GameObject parent, string text, Vector2 position, bool isOn)
    {
        GameObject toggleObj = new GameObject(name);
        toggleObj.transform.SetParent(parent.transform, false);
        
        RectTransform rect = toggleObj.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0.5f, 0.5f);
        rect.anchorMax = new Vector2(0.5f, 0.5f);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(150, 20);
        
        Toggle toggle = toggleObj.AddComponent<Toggle>();
        toggle.isOn = isOn;
        
        // チェックボックス
        GameObject checkbox = new GameObject("Checkbox");
        checkbox.transform.SetParent(toggleObj.transform, false);
        RectTransform cbRect = checkbox.AddComponent<RectTransform>();
        cbRect.anchorMin = new Vector2(0, 0.5f);
        cbRect.anchorMax = new Vector2(0, 0.5f);
        cbRect.anchoredPosition = new Vector2(10, 0);
        cbRect.sizeDelta = new Vector2(20, 20);
        Image cbImage = checkbox.AddComponent<Image>();
        cbImage.color = Color.white;
        
        // チェックマーク
        GameObject checkmark = new GameObject("Checkmark");
        checkmark.transform.SetParent(checkbox.transform, false);
        RectTransform cmRect = checkmark.AddComponent<RectTransform>();
        cmRect.anchorMin = Vector2.zero;
        cmRect.anchorMax = Vector2.one;
        cmRect.offsetMin = Vector2.zero;
        cmRect.offsetMax = Vector2.zero;
        Image cmImage = checkmark.AddComponent<Image>();
        cmImage.color = Color.green;
        
        // テキスト
        GameObject textObj = new GameObject("Text");
        textObj.transform.SetParent(toggleObj.transform, false);
        RectTransform textRect = textObj.AddComponent<RectTransform>();
        textRect.anchorMin = new Vector2(0, 0);
        textRect.anchorMax = new Vector2(1, 1);
        textRect.offsetMin = new Vector2(25, 0);
        textRect.offsetMax = Vector2.zero;
        Text textComponent = textObj.AddComponent<Text>();
        textComponent.text = text;
        textComponent.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        textComponent.fontSize = 12;
        textComponent.color = Color.white;
        textComponent.alignment = TextAnchor.MiddleLeft;
        
        toggle.targetGraphic = cbImage;
        toggle.graphic = cmImage;
        
        return toggleObj;
    }
}