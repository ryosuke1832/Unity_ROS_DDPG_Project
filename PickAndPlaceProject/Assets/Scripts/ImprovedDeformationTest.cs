using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 改良版変形テストマネージャー
/// UIレイアウトと操作性を改善
/// </summary>
public class ImprovedDeformationTest : MonoBehaviour
{
    [Header("=== 参照 ===")]
    [SerializeField] private EnhancedGripperForceController enhancedGripperController;
    [SerializeField] private GripperForceController gripperController;
    [SerializeField] private SimpleGripForceController simpleGripController;
    [SerializeField] private Transform targetSpawnPoint;
    
    [Header("=== テスト設定 ===")]
    [SerializeField] private GameObject targetPrefab;
    [SerializeField] private bool showDebugGUI = true;
    [SerializeField] private bool autoSpawnOnStart = true;
    
    // 内部変数
    private List<DeformableTarget> spawnedTargets = new List<DeformableTarget>();
    private DeformableTarget currentTarget = null;
    
    // テスト用パラメータ
    private float testForce = 15f;
    private float testSoftness = 0.6f;
    private int deformationType = 0;
    
    // UI制御用
    private bool showAdvancedSettings = false;
    private bool isGrasping = false;
    
    void Start()
    {
        InitializeTest();
        
        if (autoSpawnOnStart)
        {
            Invoke(nameof(SpawnTestTarget), 1f); // 1秒後に生成
        }
    }
    
    void Update()
    {
        HandleKeyboardInput();
        UpdateCurrentTarget();
        UpdateGraspingState();
    }
    
    /// <summary>
    /// テスト初期化
    /// </summary>
    private void InitializeTest()
    {
        // コントローラーの自動検出（優先順位順）
        if (enhancedGripperController == null)
            enhancedGripperController = FindObjectOfType<EnhancedGripperForceController>();
            
        if (gripperController == null)
            gripperController = FindObjectOfType<GripperForceController>();
        
        if (simpleGripController == null)
            simpleGripController = FindObjectOfType<SimpleGripForceController>();
        
        // スポーン地点の設定
        if (targetSpawnPoint == null)
        {
            GameObject spawnGO = new GameObject("TargetSpawnPoint");
            targetSpawnPoint = spawnGO.transform;
            targetSpawnPoint.position = new Vector3(0f, 0.65f, -0.1f); // ロボットの前に配置
        }
        
        Debug.Log("ImprovedDeformationTest初期化完了");
        Debug.Log("=== 操作方法 ===");
        Debug.Log("T: ターゲット生成");
        Debug.Log("G: 把持開始");
        Debug.Log("S: 把持停止");
        Debug.Log("C: 全ターゲットクリア");
        Debug.Log("R: 設定適用");
    }
    
    /// <summary>
    /// キーボード入力処理
    /// </summary>
    private void HandleKeyboardInput()
    {
        if (Input.GetKeyDown(KeyCode.G))
        {
            StartGrasping();
        }
        
        if (Input.GetKeyDown(KeyCode.S))
        {
            StopGrasping();
        }
        
        if (Input.GetKeyDown(KeyCode.T))
        {
            SpawnTestTarget();
        }
        
        if (Input.GetKeyDown(KeyCode.C))
        {
            ClearTargets();
        }
        
        if (Input.GetKeyDown(KeyCode.R))
        {
            ApplySettingsToAllTargets();
        }
    }
    
    /// <summary>
    /// 把持状態の更新
    /// </summary>
    private void UpdateGraspingState()
    {
        if (enhancedGripperController != null)
        {
            var graspState = enhancedGripperController.GetGraspingState();
            isGrasping = graspState.isGrasping;
        }
        else if (gripperController != null)
        {
            var graspState = gripperController.GetGraspingState();
            isGrasping = graspState.isGrasping;
        }
    }
    
    /// <summary>
    /// 現在のターゲット更新
    /// </summary>
    private void UpdateCurrentTarget()
    {
        // EnhancedGripperControllerから現在のターゲットを取得
        if (enhancedGripperController != null)
        {
            currentTarget = enhancedGripperController.GetCurrentTarget();
        }
        
        // フォールバック：最後に生成されたターゲットを使用
        if (currentTarget == null && spawnedTargets.Count > 0)
        {
            for (int i = spawnedTargets.Count - 1; i >= 0; i--)
            {
                if (spawnedTargets[i] != null)
                {
                    currentTarget = spawnedTargets[i];
                    break;
                }
            }
        }
    }
    
    /// <summary>
    /// 把持開始
    /// </summary>
    public void StartGrasping()
    {
        if (enhancedGripperController != null)
        {
            enhancedGripperController.StartGrasping(testForce);
        }
        else if (gripperController != null)
        {
            gripperController.StartGrasping(testForce);
        }
        
        Debug.Log($"把持開始 - 力: {testForce:F1}N, 柔軟性: {testSoftness:F2}");
    }
    
    /// <summary>
    /// 把持停止
    /// </summary>
    public void StopGrasping()
    {
        if (enhancedGripperController != null)
        {
            enhancedGripperController.StopGrasping();
        }
        else if (gripperController != null)
        {
            gripperController.StopGrasping();
        }
        
        Debug.Log("把持停止");
    }
    
    /// <summary>
    /// テスト用ターゲット生成
    /// </summary>
    public void SpawnTestTarget()
    {
        GameObject targetGO = CreateDeformableTarget();
        
        // DeformableTargetコンポーネントを追加
        DeformableTarget deformable = targetGO.AddComponent<DeformableTarget>();
        
        // 設定を適用
        ApplyCurrentSettings(deformable);
        
        spawnedTargets.Add(deformable);
        currentTarget = deformable;
        
        Debug.Log($"変形可能ターゲット生成 - 柔軟性: {testSoftness:F2}, タイプ: {(DeformableTarget.DeformationType)deformationType}");
    }
    
    /// <summary>
    /// 変形可能ターゲットの作成
    /// </summary>
    private GameObject CreateDeformableTarget()
    {
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = "DeformableTarget";
        cube.transform.position = targetSpawnPoint.position;
        cube.transform.localScale = Vector3.one * 0.06f; // 少し大きめ
        
        // 物理設定
        Rigidbody rb = cube.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = 0.1f;
            rb.drag = 0.5f;
        }
        
        // 視覚的に分かりやすいマテリアル
        MeshRenderer renderer = cube.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            Material mat = new Material(Shader.Find("Standard"));
            mat.color = new Color(0.3f, 0.7f, 1f, 1f); // 明るい青
            mat.metallic = 0.2f;
            mat.smoothness = 0.8f;
            renderer.material = mat;
        }
        
        return cube;
    }
    
    /// <summary>
    /// 設定を特定のターゲットに適用
    /// </summary>
    private void ApplyCurrentSettings(DeformableTarget target)
    {
        if (target == null) return;
        
        target.SetSoftness(testSoftness);
        target.SetDeformationType((DeformableTarget.DeformationType)deformationType);
    }
    
    /// <summary>
    /// 設定を全ターゲットに適用
    /// </summary>
    public void ApplySettingsToAllTargets()
    {
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                ApplyCurrentSettings(target);
            }
        }
        
        // 把持力も更新
        if (enhancedGripperController != null)
        {
            enhancedGripperController.SetTargetGripForce(testForce);
        }
        else if (gripperController != null)
        {
            gripperController.SetTargetGripForce(testForce);
        }
        
        Debug.Log("全ターゲットに設定を適用しました");
    }
    
    /// <summary>
    /// 全ターゲットクリア
    /// </summary>
    public void ClearTargets()
    {
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                DestroyImmediate(target.gameObject);
            }
        }
        
        spawnedTargets.Clear();
        currentTarget = null;
        
        Debug.Log("全ターゲットをクリアしました");
    }
    
    /// <summary>
    /// 改良版デバッグGUI
    /// </summary>
    void OnGUI()
    {
        if (!showDebugGUI) return;
        
        // メインコントロールパネル（右側に移動）
        GUILayout.BeginArea(new Rect(Screen.width - 320, 10, 300, 400));
        
        // タイトル
        GUI.color = Color.yellow;
        GUILayout.Label("=== 変形テスト制御 ===", GUI.skin.box);
        GUI.color = Color.white;
        
        GUILayout.Space(5);
        
        // 把持力設定
        GUILayout.Label($"把持力: {testForce:F1}N");
        testForce = GUILayout.HorizontalSlider(testForce, 5f, 50f);
        
        // 柔軟性設定
        GUILayout.Label($"柔軟性: {testSoftness:F2}");
        testSoftness = GUILayout.HorizontalSlider(testSoftness, 0f, 1f);
        
        GUILayout.Space(5);
        
        // 変形タイプ
        GUILayout.Label("変形タイプ:");
        string[] typeNames = { "Squeeze", "Bend", "Stretch", "Soft" };
        int newDeformationType = GUILayout.SelectionGrid(deformationType, typeNames, 2);
        if (newDeformationType != deformationType)
        {
            deformationType = newDeformationType;
            ApplySettingsToAllTargets(); // 即座に適用
        }
        
        GUILayout.Space(10);
        
        // メインボタン
        GUI.color = isGrasping ? Color.red : Color.green;
        if (GUILayout.Button(isGrasping ? "把持停止 (S)" : "把持開始 (G)", GUILayout.Height(30)))
        {
            if (isGrasping) StopGrasping();
            else StartGrasping();
        }
        GUI.color = Color.white;
        
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("ターゲット生成 (T)"))
        {
            SpawnTestTarget();
        }
        if (GUILayout.Button("クリア (C)"))
        {
            ClearTargets();
        }
        GUILayout.EndHorizontal();
        
        if (GUILayout.Button("設定適用 (R)"))
        {
            ApplySettingsToAllTargets();
        }
        
        GUILayout.Space(10);
        
        // ステータス情報
        GUI.color = Color.cyan;
        GUILayout.Label("=== ステータス ===", GUI.skin.box);
        GUI.color = Color.white;
        
        // コントローラー情報
        bool hasController = false;
        float currentForce = 0f;
        
        if (enhancedGripperController != null)
        {
            var graspState = enhancedGripperController.GetGraspingState();
            currentForce = graspState.currentForce;
            hasController = true;
            GUILayout.Label($"コントローラー: Enhanced");
        }
        else if (gripperController != null)
        {
            var graspState = gripperController.GetGraspingState();
            currentForce = graspState.currentForce;
            hasController = true;
            GUILayout.Label($"コントローラー: Standard");
        }
        else
        {
            GUI.color = Color.red;
            GUILayout.Label($"コントローラー: 未検出");
            GUI.color = Color.white;
        }
        
        if (hasController)
        {
            GUILayout.Label($"把持状態: {(isGrasping ? "把持中" : "待機中")}");
            GUILayout.Label($"現在力: {currentForce:F2}N");
        }
        
        // ターゲット情報
        if (currentTarget != null)
        {
            GUI.color = Color.green;
            GUILayout.Label($"アクティブターゲット: 検出");
            GUI.color = Color.white;
            GUILayout.Label($"変形度: {currentTarget.CurrentDeformation:F3}");
            
            // 変形度のプログレスバー
            Rect progressRect = GUILayoutUtility.GetRect(250, 15);
            GUI.Box(progressRect, "");
            Rect fillRect = new Rect(progressRect.x + 1, progressRect.y + 1, 
                                   (progressRect.width - 2) * currentTarget.CurrentDeformation, 
                                   progressRect.height - 2);
            GUI.color = Color.Lerp(Color.green, Color.red, currentTarget.CurrentDeformation);
            GUI.DrawTexture(fillRect, Texture2D.whiteTexture);
            GUI.color = Color.white;
            
            GUILayout.Label($"変形中: {(currentTarget.IsDeformed ? "はい" : "いいえ")}");
        }
        else
        {
            GUI.color = Color.yellow;
            GUILayout.Label($"アクティブターゲット: なし");
            GUI.color = Color.white;
        }
        
        GUILayout.Label($"生成済み: {spawnedTargets.Count}個");
        
        GUILayout.EndArea();
        
        // 操作説明（左下）
        GUILayout.BeginArea(new Rect(10, Screen.height - 120, 300, 110));
        GUI.color = Color.yellow;
        GUILayout.Label("=== 操作方法 ===", GUI.skin.box);
        GUI.color = Color.white;
        GUILayout.Label("T: ターゲット生成");
        GUILayout.Label("G: 把持開始 / S: 把持停止");
        GUILayout.Label("C: 全クリア / R: 設定適用");
        GUILayout.Label("スライダーで力と柔軟性を調整");
        GUILayout.EndArea();
    }
    
    void OnDrawGizmos()
    {
        // スポーン地点の表示
        if (targetSpawnPoint != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireCube(targetSpawnPoint.position, Vector3.one * 0.08f);
            Gizmos.DrawRay(targetSpawnPoint.position, Vector3.up * 0.05f);
        }
        
        // 生成されたターゲットの表示
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                Gizmos.color = target.IsDeformed ? Color.red : Color.blue;
                Gizmos.DrawWireSphere(target.transform.position, 0.04f);
            }
        }
    }
    
    void OnDestroy()
    {
        ClearTargets();
    }
}