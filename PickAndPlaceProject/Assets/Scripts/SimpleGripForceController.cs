using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// 更新された変形テストマネージャー
/// 新しいDeformableTargetクラスに対応
/// </summary>
public class UpdatedSimpleDeformationTest : MonoBehaviour
{
    [Header("=== 参照 ===")]
    [SerializeField] private EnhancedGripperForceController enhancedGripperController;
    [SerializeField] private GripperForceController gripperController;
    [SerializeField] private SimpleGripForceController simpleGripController;
    [SerializeField] private Transform targetSpawnPoint;
    
    [Header("=== テスト設定 ===")]
    [SerializeField] private GameObject targetPrefab;
    [SerializeField] private bool showDebugGUI = true;
    [SerializeField] private bool autoSpawnOnStart = false;
    
    // 内部変数
    private List<DeformableTarget> spawnedTargets = new List<DeformableTarget>();
    private DeformableTarget currentTarget = null;
    
    // テスト用パラメータ
    private float testForce = 10f;
    private float testSoftness = 0.5f;
    private int deformationType = 0;
    
    // UI制御用
    private bool showAdvancedSettings = false;
    
    void Start()
    {
        InitializeTest();
        
        if (autoSpawnOnStart)
        {
            SpawnTestTarget();
        }
    }
    
    void Update()
    {
        // キーボード制御
        HandleKeyboardInput();
        
        // 現在のターゲット追跡
        UpdateCurrentTarget();
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
            targetSpawnPoint.position = new Vector3(0f, 0.65f, 0f);
        }
        
        Debug.Log("UpdatedSimpleDeformationTest初期化完了");
        Debug.Log("操作方法: G=把持開始, S=把持停止, T=ターゲット生成, C=クリア, R=設定適用");
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
        
        if (Input.GetKeyDown(KeyCode.Tab))
        {
            showAdvancedSettings = !showAdvancedSettings;
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
        // 優先順位に従ってコントローラーを使用
        if (enhancedGripperController != null)
        {
            enhancedGripperController.StartGrasping(testForce);
        }
        else if (gripperController != null)
        {
            gripperController.StartGrasping(testForce);
        }
        
        Debug.Log($"把持開始 - 力: {testForce}N");
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
        GameObject targetGO;
        
        if (targetPrefab != null)
        {
            targetGO = Instantiate(targetPrefab, targetSpawnPoint.position, Quaternion.identity);
        }
        else
        {
            targetGO = CreateDefaultTarget();
        }
        
        // DeformableTargetコンポーネントを追加/取得
        DeformableTarget deformable = targetGO.GetComponent<DeformableTarget>();
        if (deformable == null)
        {
            deformable = targetGO.AddComponent<DeformableTarget>();
        }
        
        // 設定を適用
        ApplyCurrentSettings(deformable);
        
        spawnedTargets.Add(deformable);
        currentTarget = deformable;
        
        Debug.Log($"ターゲット生成 - 柔軟性: {testSoftness}, タイプ: {(DeformableTarget.DeformationType)deformationType}");
    }
    
    /// <summary>
    /// デフォルトターゲットの作成
    /// </summary>
    private GameObject CreateDefaultTarget()
    {
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.name = "DeformableTarget";
        cube.transform.position = targetSpawnPoint.position;
        cube.transform.localScale = Vector3.one * 0.05f;
        
        // 物理設定
        Rigidbody rb = cube.GetComponent<Rigidbody>();
        if (rb != null)
        {
            rb.mass = 0.1f;
            rb.drag = 0.5f; // 安定性のために抵抗を追加
        }
        
        // マテリアル設定（オプション）
        MeshRenderer renderer = cube.GetComponent<MeshRenderer>();
        if (renderer != null)
        {
            // 基本的な青いマテリアル
            Material mat = new Material(Shader.Find("Standard"));
            mat.color = Color.blue;
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
    /// 複数ターゲット生成（テスト用）
    /// </summary>
    public void SpawnMultipleTargets(int count = 3)
    {
        for (int i = 0; i < count; i++)
        {
            Vector3 offset = new Vector3(
                Random.Range(-0.1f, 0.1f),
                Random.Range(0f, 0.05f),
                Random.Range(-0.1f, 0.1f)
            );
            
            Vector3 spawnPos = targetSpawnPoint.position + offset;
            
            GameObject targetGO = CreateDefaultTarget();
            targetGO.transform.position = spawnPos;
            
            DeformableTarget deformable = targetGO.AddComponent<DeformableTarget>();
            
            // ランダムな設定を適用
            deformable.SetSoftness(Random.Range(0.2f, 0.8f));
            deformable.SetDeformationType((DeformableTarget.DeformationType)Random.Range(0, 4));
            
            spawnedTargets.Add(deformable);
        }
        
        Debug.Log($"{count}個のターゲットを生成しました");
    }
    
    /// <summary>
    /// デバッグGUI
    /// </summary>
    void OnGUI()
    {
        if (!showDebugGUI) return;
        
        GUILayout.BeginArea(new Rect(10, 400, 350, 500));
        GUILayout.Label("=== 変形テスト制御 ===");
        
        // 基本制御
        GUILayout.Label($"把持力: {testForce:F1}N");
        testForce = GUILayout.HorizontalSlider(testForce, 1f, 50f);
        
        GUILayout.Label($"柔軟性: {testSoftness:F2}");
        testSoftness = GUILayout.HorizontalSlider(testSoftness, 0f, 1f);
        
        // 変形タイプ
        GUILayout.Label("変形タイプ:");
        string[] typeNames = { "Squeeze", "Bend", "Stretch", "Soft" };
        deformationType = GUILayout.SelectionGrid(deformationType, typeNames, 2);
        
        GUILayout.Space(10);
        
        // 基本ボタン
        GUILayout.BeginHorizontal();
        if (GUILayout.Button("把持開始 (G)"))
        {
            StartGrasping();
        }
        if (GUILayout.Button("把持停止 (S)"))
        {
            StopGrasping();
        }
        GUILayout.EndHorizontal();
        
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
        
        // 高度な機能
        if (GUILayout.Button(showAdvancedSettings ? "高度な設定を隠す" : "高度な設定を表示 (Tab)"))
        {
            showAdvancedSettings = !showAdvancedSettings;
        }
        
        if (showAdvancedSettings)
        {
            GUILayout.Label("=== 高度な機能 ===");
            
            if (GUILayout.Button("複数ターゲット生成"))
            {
                SpawnMultipleTargets();
            }
            
            if (targetSpawnPoint != null)
            {
                GUILayout.Label("スポーン位置:");
                GUILayout.BeginHorizontal();
                GUILayout.Label("Y:");
                targetSpawnPoint.position = new Vector3(
                    targetSpawnPoint.position.x,
                    GUILayout.HorizontalSlider(targetSpawnPoint.position.y, 0.5f, 1.0f),
                    targetSpawnPoint.position.z
                );
                GUILayout.EndHorizontal();
            }
        }
        
        GUILayout.Space(10);
        
        // ステータス情報
        GUILayout.Label("=== ステータス ===");
        
        // コントローラー状態
        bool isGrasping = false;
        float currentForce = 0f;
        
        if (enhancedGripperController != null)
        {
            var graspState = enhancedGripperController.GetGraspingState();
            isGrasping = graspState.isGrasping;
            currentForce = graspState.currentForce;
            GUILayout.Label($"コントローラー: Enhanced");
        }
        else if (gripperController != null)
        {
            var graspState = gripperController.GetGraspingState();
            isGrasping = graspState.isGrasping;
            currentForce = graspState.currentForce;
            GUILayout.Label($"コントローラー: Standard");
        }
        else
        {
            GUILayout.Label($"コントローラー: 未検出");
        }
        
        GUILayout.Label($"把持中: {(isGrasping ? "はい" : "いいえ")}");
        GUILayout.Label($"現在力: {currentForce:F2}N");
        
        // ターゲット情報
        if (currentTarget != null)
        {
            GUILayout.Label($"アクティブターゲット: {currentTarget.name}");
            GUILayout.Label($"変形度: {currentTarget.CurrentDeformation:F3}");
            GUILayout.Label($"変形中: {(currentTarget.IsDeformed ? "はい" : "いいえ")}");
            
            // プログレスバー風の変形度表示
            Rect progressRect = GUILayoutUtility.GetRect(200, 20);
            GUI.Box(progressRect, "");
            Rect fillRect = new Rect(progressRect.x, progressRect.y, 
                                   progressRect.width * currentTarget.CurrentDeformation, 
                                   progressRect.height);
            GUI.color = Color.Lerp(Color.green, Color.red, currentTarget.CurrentDeformation);
            GUI.Box(fillRect, "");
            GUI.color = Color.white;
        }
        else
        {
            GUILayout.Label($"アクティブターゲット: なし");
        }
        
        GUILayout.Label($"生成済み: {spawnedTargets.Count}個");
        
        GUILayout.EndArea();
    }
    
    /// <summary>
    /// シーン上でのギズモ表示
    /// </summary>
    void OnDrawGizmos()
    {
        // スポーン地点の表示
        if (targetSpawnPoint != null)
        {
            Gizmos.color = Color.green;
            Gizmos.DrawWireCube(targetSpawnPoint.position, Vector3.one * 0.1f);
            Gizmos.DrawRay(targetSpawnPoint.position, Vector3.up * 0.05f);
        }
        
        // 生成されたターゲットの表示
        foreach (var target in spawnedTargets)
        {
            if (target != null)
            {
                Gizmos.color = target.IsDeformed ? Color.red : Color.blue;
                Gizmos.DrawWireSphere(target.transform.position, 0.03f);
            }
        }
    }
    
    void OnDestroy()
    {
        ClearTargets();
    }
}