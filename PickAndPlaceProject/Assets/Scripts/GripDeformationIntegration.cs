using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 把持力制御と物体変形を統合するシステム
/// </summary>
public class GripDeformationIntegration : MonoBehaviour
{
    [Header("=== 統合設定 ===")]
    [SerializeField, Tooltip("グリッパー制御コンポーネント")]
    private GripperForceController gripperController;
    
    [SerializeField, Tooltip("変形可能な物体リスト")]
    private List<DeformableTarget> deformableTargets = new List<DeformableTarget>();
    
    [SerializeField, Tooltip("自動で変形可能物体を検索")]
    private bool autoFindDeformableTargets = true;
    
    [Header("=== 把持検出設定 ===")]
    [SerializeField, Range(0.1f, 5f), Tooltip("把持検出の距離閾値")]
    private float graspDetectionDistance = 0.5f;
    
    [SerializeField, Tooltip("グリッパーの左フィンガー")]
    private Transform leftGripperFinger;
    
    [SerializeField, Tooltip("グリッパーの右フィンガー")]
    private Transform rightGripperFinger;
    
    [Header("=== フィードバック設定 ===")]
    [SerializeField, Tooltip("変形に応じて把持力をフィードバック")]
    private bool enableDeformationFeedback = true;
    
    [SerializeField, Range(0f, 1f), Tooltip("フィードバックの強さ")]
    private float feedbackStrength = 0.3f;
    
    [SerializeField, Tooltip("HumanFeedbackUI参照")]
    private HumanFeedBackUI humanFeedbackUI;
    
    [Header("=== デバッグ ===")]
    [SerializeField, Tooltip("デバッグ情報表示")]
    private bool showDebugInfo = true;
    
    // 内部変数
    private DeformableTarget currentTarget;
    private bool isCurrentlyGrasping = false;
    private float lastUpdateTime;
    private Dictionary<DeformableTarget, float> targetDistances = new Dictionary<DeformableTarget, float>();
    
    private void Start()
    {
        InitializeComponents();
        FindDeformableTargets();
    }
    
    private void InitializeComponents()
    {
        if (gripperController == null)
        {
            gripperController = FindObjectOfType<GripperForceController>();
        }
        
        if (humanFeedbackUI == null)
        {
            humanFeedbackUI = FindObjectOfType<HumanFeedBackUI>();
        }
        
        // グリッパーのフィンガーが設定されていない場合の自動検索
        if (leftGripperFinger == null || rightGripperFinger == null)
        {
            FindGripperFingers();
        }
    }
    
    private void FindGripperFingers()
    {
        // グリッパーのフィンガーを自動検索
        // 実際のプロジェクト構造に応じて調整が必要
        GameObject[] allObjects = FindObjectsOfType<GameObject>();
        
        foreach (GameObject obj in allObjects)
        {
            if (obj.name.Contains("LeftFinger") || obj.name.Contains("Left_Finger"))
            {
                leftGripperFinger = obj.transform;
            }
            else if (obj.name.Contains("RightFinger") || obj.name.Contains("Right_Finger"))
            {
                rightGripperFinger = obj.transform;
            }
        }
    }
    
    private void FindDeformableTargets()
    {
        if (!autoFindDeformableTargets) return;
        
        DeformableTarget[] targets = FindObjectsOfType<DeformableTarget>();
        foreach (var target in targets)
        {
            if (!deformableTargets.Contains(target))
            {
                deformableTargets.Add(target);
            }
        }
        
        Debug.Log($"発見された変形可能物体: {deformableTargets.Count}個");
    }
    
    private void Update()
    {
        if (gripperController == null) return;
        
        UpdateGraspDetection();
        UpdateForceTransmission();
        
        if (enableDeformationFeedback)
        {
            UpdateDeformationFeedback();
        }
        
        if (showDebugInfo && Time.time - lastUpdateTime > 1f)
        {
            LogDebugInfo();
            lastUpdateTime = Time.time;
        }
    }
    
    /// <summary>
    /// 把持検出の更新
    /// </summary>
    private void UpdateGraspDetection()
    {
        var graspState = gripperController.GetGraspingState();
        isCurrentlyGrasping = graspState.isGrasping;
        
        if (!isCurrentlyGrasping)
        {
            // 把持していない場合、現在のターゲットをクリア
            if (currentTarget != null)
            {
                currentTarget.SetGripForce(0f, false);
                currentTarget = null;
            }
            return;
        }
        
        // 最も近い変形可能物体を検索
        DeformableTarget closestTarget = FindClosestDeformableTarget();
        
        if (closestTarget != currentTarget)
        {
            // ターゲットが変わった場合
            if (currentTarget != null)
            {
                currentTarget.SetGripForce(0f, false);
            }
            currentTarget = closestTarget;
        }
    }
    
    /// <summary>
    /// 最も近い変形可能物体を検索
    /// </summary>
    private DeformableTarget FindClosestDeformableTarget()
    {
        if (leftGripperFinger == null || rightGripperFinger == null)
            return null;
        
        Vector3 gripperCenter = (leftGripperFinger.position + rightGripperFinger.position) / 2f;
        DeformableTarget closest = null;
        float closestDistance = float.MaxValue;
        
        foreach (var target in deformableTargets)
        {
            if (target == null) continue;
            
            float distance = Vector3.Distance(gripperCenter, target.transform.position);
            targetDistances[target] = distance;
            
            if (distance < graspDetectionDistance && distance < closestDistance)
            {
                closest = target;
                closestDistance = distance;
            }
        }
        
        return closest;
    }
    
    /// <summary>
    /// 把持力の伝達
    /// </summary>
    private void UpdateForceTransmission()
    {
        if (currentTarget == null || !isCurrentlyGrasping) return;
        
        var graspState = gripperController.GetGraspingState();
        currentTarget.SetGripForce(graspState.currentForce, true);
    }
    
    /// <summary>
    /// 変形フィードバックの更新
    /// </summary>
    private void UpdateDeformationFeedback()
    {
        if (currentTarget == null || !isCurrentlyGrasping) return;
        
        var deformState = currentTarget.GetCurrentState();
        
        // 変形レベルに応じて把持力を調整
        if (deformState.deformationLevel > 0.1f)
        {
            float forceReduction = deformState.deformationLevel * feedbackStrength;
            var currentGraspState = gripperController.GetGraspingState();
            
            float adjustedForce = currentGraspState.targetForce * (1f - forceReduction);
            adjustedForce = Mathf.Max(adjustedForce, 0.5f); // 最小力を保持
            
            gripperController.SetTargetGripForce(adjustedForce);
            
            // 自動フィードバック送信（変形が激しい場合）
            if (deformState.deformationLevel > 0.7f && humanFeedbackUI != null)
            {
                // "Too Hard"フィードバックを自動送信
                SendAutomaticFeedback(-0.5f, "変形による自動フィードバック");
            }
        }
        
        // 物体が破損した場合の処理
        if (deformState.isDestroyed)
        {
            gripperController.SetTargetGripForce(0f);
            if (humanFeedbackUI != null)
            {
                SendAutomaticFeedback(-1f, "物体破損による自動フィードバック");
            }
        }
    }
    
    /// <summary>
    /// 自動フィードバック送信
    /// </summary>
    private void SendAutomaticFeedback(float reward, string description)
    {
        if (humanFeedbackUI == null) return;
        
        // HumanFeedBackUIのフィードバック機能を利用
        var graspState = gripperController.GetGraspingState();
        
        // リフレクションを使用してプライベートメソッドを呼び出し
        var method = humanFeedbackUI.GetType().GetMethod("SendFeedback", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        
        if (method != null)
        {
            method.Invoke(humanFeedbackUI, new object[] { reward, description });
        }
    }
    
    /// <summary>
    /// 変形可能物体を追加
    /// </summary>
    public void AddDeformableTarget(DeformableTarget target)
    {
        if (target != null && !deformableTargets.Contains(target))
        {
            deformableTargets.Add(target);
        }
    }
    
    /// <summary>
    /// 変形可能物体を削除
    /// </summary>
    public void RemoveDeformableTarget(DeformableTarget target)
    {
        if (deformableTargets.Contains(target))
        {
            deformableTargets.Remove(target);
            if (currentTarget == target)
            {
                currentTarget = null;
            }
        }
    }
    
    /// <summary>
    /// 全ての変形可能物体をリセット
    /// </summary>
    public void ResetAllTargets()
    {
        foreach (var target in deformableTargets)
        {
            if (target != null)
            {
                target.ResetObject();
            }
        }
        currentTarget = null;
    }
    
    /// <summary>
    /// デバッグ情報の記録
    /// </summary>
    private void LogDebugInfo()
    {
        if (currentTarget != null)
        {
            var graspState = gripperController.GetGraspingState();
            var deformState = currentTarget.GetCurrentState();
            
            Debug.Log($"[統合システム] ターゲット: {currentTarget.name}, " +
                     $"把持力: {graspState.currentForce:F2}N, " +
                     $"変形度: {deformState.deformationLevel:F2}");
        }
    }
    
    /// <summary>
    /// 現在の統合状態を取得
    /// </summary>
    public IntegrationState GetCurrentState()
    {
        return new IntegrationState
        {
            currentTarget = currentTarget,
            isGrasping = isCurrentlyGrasping,
            targetCount = deformableTargets.Count,
            graspState = gripperController?.GetGraspingState() ?? new GraspingState(),
            deformationState = currentTarget?.GetCurrentState() ?? new DeformationState()
        };
    }
    
    // OnGUI デバッグ表示
    private void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(630, 10, 300, 250));
        GUILayout.Label("=== 統合システム ===");
        GUILayout.Label($"変形可能物体数: {deformableTargets.Count}");
        GUILayout.Label($"現在のターゲット: {(currentTarget != null ? currentTarget.name : "なし")}");
        GUILayout.Label($"把持中: {isCurrentlyGrasping}");
        GUILayout.Label($"フィードバック: {enableDeformationFeedback}");
        
        if (currentTarget != null)
        {
            var distance = targetDistances.ContainsKey(currentTarget) ? 
                targetDistances[currentTarget] : 0f;
            GUILayout.Label($"距離: {distance:F3}m");
        }
        
        GUILayout.Space(10);
        
        if (GUILayout.Button("全ターゲットリセット"))
        {
            ResetAllTargets();
        }
        
        if (GUILayout.Button("ターゲット再検索"))
        {
            FindDeformableTargets();
        }
        
        GUILayout.EndArea();
    }
}

/// <summary>
/// 統合システムの状態
/// </summary>
[System.Serializable]
public struct IntegrationState
{
    public DeformableTarget currentTarget;
    public bool isGrasping;
    public int targetCount;
    public GraspingState graspState;
    public DeformationState deformationState;
}