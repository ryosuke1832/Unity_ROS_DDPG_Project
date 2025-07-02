using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;

/// <summary>
/// シンプルな変形テストマネージャー
/// 既存のファイルとの競合を避けるため、最小限の機能で実装
/// </summary>
public class SimpleDeformationTest : MonoBehaviour
{
    [Header("=== 参照 ===")]
    [SerializeField] private GripperForceController gripperController;
    [SerializeField] private SimpleGripForceController simpleGripController;
    [SerializeField] private Transform targetSpawnPoint;
    
    [Header("=== テスト設定 ===")]
    [SerializeField] private GameObject targetPrefab;
    [SerializeField] private bool showDebugGUI = true;
    
    // 内部変数
    private List<DeformableTarget> spawnedTargets = new List<DeformableTarget>();
    private DeformableTarget currentTarget = null;
    
    // テスト用パラメータ
    private float testForce = 10f;
    private float testSoftness = 0.5f;
    private int deformationType = 0;
    
    void Start()
    {
        InitializeTest();
    }
    
    void Update()
    {
        // 簡単なキーボード制御
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
    }
    
    /// <summary>
    /// テスト初期化
    /// </summary>
    private void InitializeTest()
    {
        // コントローラーの自動検出
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
        
        Debug.Log("SimpleDeformationTest初期化完了");
        Debug.Log("操作方法: G=把持開始, S=把持停止, T=ターゲット生成, C=クリア");
    }
    
    /// <summary>
    /// 把持開始
    /// </summary>
    public void StartGrasping()
    {
        if (gripperController != null)
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
        if (gripperController != null)
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
            // デフォルトキューブを作成
            targetGO = CreateDefaultTarget();
        }
        
        // DeformableTargetコンポーネントを追加
        DeformableTarget deformable = targetGO.GetComponent<DeformableTarget>();
        if (deformable == null)
        {
            deformable = targetGO.AddComponent<DeformableTarget>();
        }
        
        // 設定を適用
        deformable.SetSoftness(testSoftness);
        deformable.SetDeformationType((DeformableTarget.DeformationType)deformationType);
        
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
        }
        
        return cube;
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
    /// デバッグGUI
    /// </summary>
    void OnGUI()
    {
        if (!showDebugGUI) return;
        
        GUILayout.BeginArea(new Rect(10, 300, 300, 400));
        GUILayout.Label("=== 変形テスト制御 ===");
        
        // 力制御
        GUILayout.Label($"把持力: {testForce:F1}N");
        testForce = GUILayout.HorizontalSlider(testForce, 1f, 50f);
        
        // 柔軟性制御
        GUILayout.Label($"柔軟性: {testSoftness:F2}");
        testSoftness = GUILayout.HorizontalSlider(testSoftness, 0f, 1f);
        
        // 変形タイプ
        GUILayout.Label("変形タイプ:");
        string[] typeNames = { "Squeeze", "Bend", "Stretch", "Soft" };
        deformationType = GUILayout.SelectionGrid(deformationType, typeNames, 2);
        
        GUILayout.Space(10);
        
        // ボタン
        if (GUILayout.Button("把持開始 (G)"))
        {
            StartGrasping();
        }
        
        if (GUILayout.Button("把持停止 (S)"))
        {
            StopGrasping();
        }
        
        if (GUILayout.Button("ターゲット生成 (T)"))
        {
            SpawnTestTarget();
        }
        
        if (GUILayout.Button("クリア (C)"))
        {
            ClearTargets();
        }
        
        GUILayout.Space(10);
        
        // 現在の状態
        if (gripperController != null)
        {
            var graspState = gripperController.GetGraspingState();
            GUILayout.Label($"把持中: {graspState.isGrasping}");
            GUILayout.Label($"現在力: {graspState.currentForce:F2}N");
        }
        
        if (currentTarget != null)
        {
            GUILayout.Label($"変形度: {currentTarget.CurrentDeformation:F3}");
            GUILayout.Label($"変形中: {currentTarget.IsDeformed}");
        }
        
        GUILayout.Label($"生成済み: {spawnedTargets.Count}個");
        
        GUILayout.EndArea();
    }
    
    /// <summary>
    /// 設定を現在のターゲットに適用
    /// </summary>
    public void ApplySettingsToCurrentTarget()
    {
        if (currentTarget != null)
        {
            currentTarget.SetSoftness(testSoftness);
            currentTarget.SetDeformationType((DeformableTarget.DeformationType)deformationType);
        }
        
        // 把持力も更新
        if (gripperController != null)
        {
            gripperController.SetTargetGripForce(testForce);
        }
    }
    
    void OnDestroy()
    {
        ClearTargets();
    }
}