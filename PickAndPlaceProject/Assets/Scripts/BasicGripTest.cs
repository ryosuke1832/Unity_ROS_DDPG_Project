using UnityEngine;

/// <summary>
/// 基本的な把持テスト（変形機能なし）
/// 競合を避けるために最小限の機能で実装
/// </summary>
public class BasicGripTest : MonoBehaviour
{
    [Header("=== 基本設定 ===")]
    [SerializeField] private GripperForceController gripperController;
    [SerializeField] private SimpleGripForceController simpleGripController;
    
    [Header("=== テスト設定 ===")]
    [SerializeField] private float testForce = 10f;
    [SerializeField] private bool showGUI = true;
    
    void Start()
    {
        // コントローラーの自動検出
        if (gripperController == null)
            gripperController = FindObjectOfType<GripperForceController>();
            
        if (simpleGripController == null)
            simpleGripController = FindObjectOfType<SimpleGripForceController>();
            
        Debug.Log("BasicGripTest初期化完了");
        Debug.Log("キー操作: G=把持開始, S=把持停止");
    }
    
    void Update()
    {
        // キーボード制御
        if (Input.GetKeyDown(KeyCode.G))
        {
            StartGrip();
        }
        
        if (Input.GetKeyDown(KeyCode.S))
        {
            StopGrip();
        }
    }
    
    public void StartGrip()
    {
        if (gripperController != null)
        {
            gripperController.StartGrasping(testForce);
            Debug.Log($"把持開始 - 力: {testForce}N");
        }
    }
    
    public void StopGrip()
    {
        if (gripperController != null)
        {
            gripperController.StopGrasping();
            Debug.Log("把持停止");
        }
    }
    
    void OnGUI()
    {
        if (!showGUI) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 200, 150));
        GUILayout.Label("=== 基本把持テスト ===");
        
        GUILayout.Label($"把持力: {testForce:F1}N");
        testForce = GUILayout.HorizontalSlider(testForce, 1f, 50f);
        
        if (GUILayout.Button("把持開始 (G)"))
        {
            StartGrip();
        }
        
        if (GUILayout.Button("把持停止 (S)"))
        {
            StopGrip();
        }
        
        if (gripperController != null)
        {
            var state = gripperController.GetGraspingState();
            GUILayout.Label($"状態: {(state.isGrasping ? "把持中" : "待機中")}");
            GUILayout.Label($"現在力: {state.currentForce:F2}N");
        }
        
        GUILayout.EndArea();
    }
}