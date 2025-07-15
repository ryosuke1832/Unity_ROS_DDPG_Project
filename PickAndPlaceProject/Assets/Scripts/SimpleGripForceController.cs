using UnityEngine;

public class SimpleGripForceController : MonoBehaviour
{
    [Header("基本把持力設定")]
    [Range(0.1f, 100f)]
    public float baseGripForce = 10f;
    
    [Range(0f, 1f)]
    public float forceVariability = 0.1f;
    
    [Range(0.1f, 5f)]
    public float forceChangeRate = 1f;
    
    [Header("デバッグ")]
    public bool showDebugInfo = true;
    
    // 内部変数
    private float currentTargetForce;
    
    void Start()
    {
        currentTargetForce = baseGripForce;
        Debug.Log("SimpleGripForceController initialized");
    }
    
    void Update()
    {
        // 基本的な力制御
        float targetForce = baseGripForce + Random.Range(-forceVariability, forceVariability) * baseGripForce;
        currentTargetForce = Mathf.Lerp(currentTargetForce, targetForce, forceChangeRate * Time.deltaTime);
    }
    
    public float GetCurrentTargetForce()
    {
        return currentTargetForce;
    }
    
    public GraspingState GetGraspingStateForInterface()
    {
        return new GraspingState
        {
            isGrasping = currentTargetForce > 1f,
            currentForce = currentTargetForce,
            targetForce = baseGripForce,
            gripperPosition = 0f,
            isSuccessful = currentTargetForce > 1f && currentTargetForce < 50f,
            softness = 0.5f
        };
    }
    
    public void SetForceControlEnabled(bool enabled)
    {
        this.enabled = enabled;
        Debug.Log($"Force control {(enabled ? "enabled" : "disabled")}");
    }
    
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(10, 10, 300, 150));
        GUILayout.Label("=== Simple Grip Force Controller ===");
        GUILayout.Label($"基本力: {baseGripForce:F1} N");
        GUILayout.Label($"目標力: {currentTargetForce:F1} N");
        GUILayout.Label($"変動: {forceVariability:F2}");
        GUILayout.EndArea();
    }

    
}