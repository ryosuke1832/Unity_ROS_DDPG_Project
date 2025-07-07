using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 変形テスト管理システム
/// </summary>
public class DeformationTestManager : MonoBehaviour
{
    [Header("=== 参照 ===")]
    [SerializeField] private EnhancedGripperForceController gripperController;
    [SerializeField] private SimpleGripForceController simpleGripController;
    [SerializeField] private Transform targetSpawnPoint;
    
    [Header("=== テスト用ターゲットプリファブ ===")]
    [SerializeField] private GameObject[] deformableTargetPrefabs;
    
    [Header("=== UI要素 ===")]
    [SerializeField] private Canvas testUI;
    [SerializeField] private Button startGraspButton;
    [SerializeField] private Button stopGraspButton;
    [SerializeField] private Button spawnTargetButton;
    [SerializeField] private Slider forceSlider;
    [SerializeField] private Slider softnessSlider;
    [SerializeField] private Dropdown deformationTypeDropdown;
    [SerializeField] private Text statusText;
    
    [Header("=== テストパラメータ ===")]
    [SerializeField] private bool autoCreateUI = true;
    [SerializeField] private bool enableDataLogging = true;
    
    // 内部変数
    private List<DeformableTarget> spawnedTargets = new List<DeformableTarget>();
    private DeformableTarget currentTestTarget = null;
    private List<TestDataEntry> testData = new List<TestDataEntry>();
    
    [System.Serializable]
    public struct TestDataEntry
    {
        public float timestamp;
        public float appliedForce;
        public float deformation;
        public string deformationType;
        public float targetSoftness;
    }
    
    void Start()
    {
        InitializeTestManager();
        if (autoCreateUI) CreateTestUI();
        SetupEventHandlers();
    }
    
    void Update()
    {
        UpdateStatus();
        if (enableDataLogging) LogTestData();
    }
    
    /// <summary>
    /// テストマネージャーの初期化
    /// </summary>
    private void InitializeTestManager()
    {
        // コントローラーの自動検出
        if (gripperController == null)
            gripperController = FindObjectOfType<EnhancedGripperForceController>();
        
        if (simpleGripController == null)
            simpleGripController = FindObjectOfType<SimpleGripForceController>();
        
        // スポーン地点の設定
        if (targetSpawnPoint == null)
        {
            GameObject spawnGO = new GameObject("TargetSpawnPoint");
            targetSpawnPoint = spawnGO.transform;
            targetSpawnPoint.position = new Vector3(0f, 0.65f, 0f);
        }
        
        Debug.Log("DeformationTestManager初期化完了");
    }
    
    /// <summary>
    /// テスト用UIの自動作成
    /// </summary>
    private void CreateTestUI()
    {
        if (testUI != null) return;
        
        // Canvas作成
        GameObject canvasGO = new GameObject("DeformationTestUI");
        testUI = canvasGO.AddComponent<Canvas>();
        testUI.renderMode = RenderMode.ScreenSpaceOverlay;
        canvasGO.AddComponent<CanvasScaler>();
        canvasGO.AddComponent<GraphicRaycaster>();
        
        // パネル作成
        GameObject panel = new GameObject("TestPanel");
        panel.transform.SetParent(testUI.transform, false);
        RectTransform panelRect = panel.AddComponent<RectTransform>();
        panelRect.anchorMin = new Vector2(0, 0.7f);
        panelRect.anchorMax = new Vector2(0.4f, 1f);
        panelRect.offsetMin = Vector2.zero;
        panelRect.offsetMax = Vector2.zero;
        
        Image panelImage = panel.AddComponent<Image>();
        panelImage.color = new Color(0, 0, 0, 0.7f);
        
        // UI要素の作成
        CreateButton("把持開始", new Vector2(10, -30), panel.transform, OnStartGrasp);
        CreateButton("把持停止", new Vector2(120, -30), panel.transform, OnStopGrasp);
        CreateButton("ターゲット生成", new Vector2(230, -30), panel.transform, OnSpawnTarget);
        
        forceSlider = CreateSlider("力スライダー", new Vector2(10, -70), panel.transform, 1f, 50f, 10f);
        CreateLabel("把持力", new Vector2(10, -50), panel.transform);
        
        softnessSlider = CreateSlider("柔軟性スライダー", new Vector2(10, -110), panel.transform, 0f, 1f, 0.5f);
        CreateLabel("柔軟性", new Vector2(10, -90), panel.transform);
        
        // 変形タイプドロップダウン
        deformationTypeDropdown = CreateDropdown("変形タイプ", new Vector2(10, -150), panel.transform,
            new string[] { "Squeeze", "Bend", "Stretch", "Soft" });
        CreateLabel("変形タイプ", new Vector2(10, -130), panel.transform);
        
        // ステータステキスト
        statusText = CreateLabel("ステータス: 待機中", new Vector2(10, -180), panel.transform);
    }
    
    /// <summary>
    /// ボタン作成ヘルパー
    /// </summary>
    private Button CreateButton(string text, Vector2 position, Transform parent, System.Action onClick)
    {
        GameObject buttonGO = new GameObject(text + "Button");
        buttonGO.transform.SetParent(parent, false);
        
        RectTransform rect = buttonGO.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(100, 30);
        
        Image image = buttonGO.AddComponent<Image>();
        image.color = Color.white;
        
        Button button = buttonGO.AddComponent<Button>();
        button.onClick.AddListener(() => onClick?.Invoke());
        
        // テキスト
        GameObject textGO = new GameObject("Text");
        textGO.transform.SetParent(buttonGO.transform, false);
        RectTransform textRect = textGO.AddComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero;
        textRect.anchorMax = Vector2.one;
        textRect.offsetMin = Vector2.zero;
        textRect.offsetMax = Vector2.zero;
        
        Text textComponent = textGO.AddComponent<Text>();
        textComponent.text = text;
        textComponent.color = Color.black;
        textComponent.alignment = TextAnchor.MiddleCenter;
        textComponent.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        
        return button;
    }
    
    /// <summary>
    /// スライダー作成ヘルパー
    /// </summary>
    private Slider CreateSlider(string name, Vector2 position, Transform parent, float min, float max, float value)
    {
        GameObject sliderGO = new GameObject(name);
        sliderGO.transform.SetParent(parent, false);
        
        RectTransform rect = sliderGO.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(200, 20);
        
        Slider slider = sliderGO.AddComponent<Slider>();
        slider.minValue = min;
        slider.maxValue = max;
        slider.value = value;
        
        // 背景
        GameObject background = new GameObject("Background");
        background.transform.SetParent(sliderGO.transform, false);
        RectTransform bgRect = background.AddComponent<RectTransform>();
        bgRect.anchorMin = Vector2.zero;
        bgRect.anchorMax = Vector2.one;
        bgRect.offsetMin = Vector2.zero;
        bgRect.offsetMax = Vector2.zero;
        Image bgImage = background.AddComponent<Image>();
        bgImage.color = Color.gray;
        
        // ハンドル
        GameObject handle = new GameObject("Handle");
        handle.transform.SetParent(sliderGO.transform, false);
        RectTransform handleRect = handle.AddComponent<RectTransform>();
        handleRect.sizeDelta = new Vector2(20, 20);
        Image handleImage = handle.AddComponent<Image>();
        handleImage.color = Color.white;
        
        slider.targetGraphic = handleImage;
        slider.handleRect = handleRect;
        
        return slider;
    }
    
    /// <summary>
    /// ドロップダウン作成ヘルパー
    /// </summary>
    private Dropdown CreateDropdown(string name, Vector2 position, Transform parent, string[] options)
    {
        GameObject dropdownGO = new GameObject(name);
        dropdownGO.transform.SetParent(parent, false);
        
        RectTransform rect = dropdownGO.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(150, 25);
        
        Image image = dropdownGO.AddComponent<Image>();
        image.color = Color.white;
        
        Dropdown dropdown = dropdownGO.AddComponent<Dropdown>();
        dropdown.options.Clear();
        
        foreach (string option in options)
        {
            dropdown.options.Add(new Dropdown.OptionData(option));
        }
        
        return dropdown;
    }
    
    /// <summary>
    /// ラベル作成ヘルパー
    /// </summary>
    private Text CreateLabel(string text, Vector2 position, Transform parent)
    {
        GameObject labelGO = new GameObject(text + "Label");
        labelGO.transform.SetParent(parent, false);
        
        RectTransform rect = labelGO.AddComponent<RectTransform>();
        rect.anchorMin = new Vector2(0, 1);
        rect.anchorMax = new Vector2(0, 1);
        rect.anchoredPosition = position;
        rect.sizeDelta = new Vector2(200, 20);
        
        Text textComponent = labelGO.AddComponent<Text>();
        textComponent.text = text;
        textComponent.color = Color.white;
        textComponent.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        
        return textComponent;
    }
    
    /// <summary>
    /// イベントハンドラーの設定
    /// </summary>
    private void SetupEventHandlers()
    {
        if (forceSlider != null)
        {
            forceSlider.onValueChanged.AddListener(OnForceChanged);
        }
        
        if (softnessSlider != null)
        {
            softnessSlider.onValueChanged.AddListener(OnSoftnessChanged);
        }
        
        if (deformationTypeDropdown != null)
        {
            deformationTypeDropdown.onValueChanged.AddListener(OnDeformationTypeChanged);
        }
    }
    
    /// <summary>
    /// 把持開始ボタンハンドラー
    /// </summary>
    private void OnStartGrasp()
    {
        if (gripperController != null)
        {
            float targetForce = forceSlider != null ? forceSlider.value : 10f;
            gripperController.StartGrasping(targetForce);
        }
        else if (simpleGripController != null)
        {
            // SimpleGripForceControllerを使用する場合
            simpleGripController.enabled = true;
        }
        
        Debug.Log("把持開始");
    }
    
    /// <summary>
    /// 把持停止ボタンハンドラー
    /// </summary>
    private void OnStopGrasp()
    {
        if (gripperController != null)
        {
            gripperController.StopGrasping();
        }
        
        Debug.Log("把持停止");
    }
    
    /// <summary>
    /// ターゲット生成ボタンハンドラー
    /// </summary>
    private void OnSpawnTarget()
    {
        if (deformableTargetPrefabs.Length == 0)
        {
            // デフォルトターゲット生成
            CreateDefaultDeformableTarget();
        }
        else
        {
            // プリファブからランダム選択
            int randomIndex = Random.Range(0, deformableTargetPrefabs.Length);
            GameObject spawnedGO = Instantiate(deformableTargetPrefabs[randomIndex], targetSpawnPoint.position, Quaternion.identity);
            
            DeformableTarget target = spawnedGO.GetComponent<DeformableTarget>();
            if (target != null)
            {
                spawnedTargets.Add(target);
                currentTestTarget = target;
                ApplyCurrentSettings(target);
            }
        }
        
        Debug.Log("ターゲット生成");
    }
    
    /// <summary>
    /// デフォルト変形ターゲットの作成
    /// </summary>
    private void CreateDefaultDeformableTarget()
    {
        GameObject targetGO = GameObject.CreatePrimitive(PrimitiveType.Cube);
        targetGO.name = "DeformableTarget";
        targetGO.transform.position = targetSpawnPoint.position;
        targetGO.transform.localScale = Vector3.one * 0.05f;
        
        // 物理コンポーネント
        Rigidbody rb = targetGO.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = 0.1f;
        }
        
        // 変形コンポーネント追加
        DeformableTarget target = targetGO.AddComponent<DeformableTarget>();
        spawnedTargets.Add(target);
        currentTestTarget = target;
        
        ApplyCurrentSettings(target);
    }
    
    /// <summary>
    /// 現在の設定をターゲットに適用
    /// </summary>
    private void ApplyCurrentSettings(DeformableTarget target)
    {
        if (target == null) return;
        
        if (softnessSlider != null)
        {
            target.SetSoftness(softnessSlider.value);
        }
        
        if (deformationTypeDropdown != null)
        {
            DeformableTarget.DeformationType type = (DeformableTarget.DeformationType)deformationTypeDropdown.value;
            target.SetDeformationType(type);
        }
    }
    
    /// <summary>
    /// 力スライダー変更ハンドラー
    /// </summary>
    private void OnForceChanged(float value)
    {
        if (gripperController != null)
        {
            gripperController.SetTargetGripForce(value);
        }
    }
    
    /// <summary>
    /// 柔軟性スライダー変更ハンドラー
    /// </summary>
    private void OnSoftnessChanged(float value)
    {
        if (currentTestTarget != null)
        {
            currentTestTarget.SetSoftness(value);
        }
        
        // 全ての生成済みターゲットに適用
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                target.SetSoftness(value);
            }
        }
    }
    
    /// <summary>
    /// 変形タイプ変更ハンドラー
    /// </summary>
    private void OnDeformationTypeChanged(int value)
    {
        DeformableTarget.DeformationType type = (DeformableTarget.DeformationType)value;
        
        if (currentTestTarget != null)
        {
            currentTestTarget.SetDeformationType(type);
        }
        
        // 全ての生成済みターゲットに適用
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                target.SetDeformationType(type);
            }
        }
    }
    
    /// <summary>
    /// ステータス更新
    /// </summary>
    private void UpdateStatus()
    {
        if (statusText == null) return;
        
        string status = "待機中";
        
        if (gripperController != null)
        {
            var graspState = gripperController.GetGraspingState();
            if (graspState.isGrasping)
            {
                status = $"把持中 - 力: {graspState.currentForce:F1}N";
                if (currentTestTarget != null)
                {
                    status += $", 変形: {currentTestTarget.CurrentDeformation:F2}";
                }
            }
        }
        
        statusText.text = $"ステータス: {status}";
    }
    
    /// <summary>
    /// テストデータのログ記録
    /// </summary>
    private void LogTestData()
    {
        if (currentTestTarget == null || !currentTestTarget.IsDeformed) return;
        
        var dataEntry = new TestDataEntry
        {
            timestamp = Time.time,
            appliedForce = currentTestTarget.CurrentForce,
            deformation = currentTestTarget.CurrentDeformation,
            deformationType = currentTestTarget.GetType().Name,
            targetSoftness = currentTestTarget.Softness
        };
        
        testData.Add(dataEntry);
        
        // データが多くなりすぎた場合の制限
        if (testData.Count > 1000)
        {
            testData.RemoveAt(0);
        }
    }
    
    /// <summary>
    /// テストデータのエクスポート
    /// </summary>
    public void ExportTestData()
    {
        if (testData.Count == 0)
        {
            Debug.Log("エクスポートするデータがありません");
            return;
        }
        
        string csvData = "Timestamp,AppliedForce,Deformation,DeformationType,TargetSoftness\n";
        
        foreach (var entry in testData)
        {
            csvData += $"{entry.timestamp},{entry.appliedForce},{entry.deformation},{entry.deformationType},{entry.targetSoftness}\n";
        }
        
        System.IO.File.WriteAllText(Application.persistentDataPath + "/deformation_test_data.csv", csvData);
        Debug.Log($"テストデータをエクスポートしました: {Application.persistentDataPath}/deformation_test_data.csv");
    }
    
    /// <summary>
    /// 生成されたターゲットのクリーンアップ
    /// </summary>
    public void ClearAllTargets()
    {
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                DestroyImmediate(target.gameObject);
            }
        }
        
        spawnedTargets.Clear();
        currentTestTarget = null;
        
        Debug.Log("全てのターゲットをクリアしました");
    }
    
    void OnDestroy()
    {
        ClearAllTargets();
    }
}