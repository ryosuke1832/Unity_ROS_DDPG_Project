using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 変形システムの詳細診断用スクリプト
/// PublisherオブジェクトまたはTargetオブジェクトにアタッチして使用
/// </summary>
public class DeformationDiagnostics : MonoBehaviour
{
    [Header("診断設定")]
    public bool enableContinuousDiagnostics = true;
    public float diagnosticInterval = 2f;
    
    [Header("参照")]
    public TrajectoryPlannerDeform trajectoryPlannerDeform;
    public DeformableTarget target;
    public SimpleGripForceController forceController;
    public GripperTargetInterface gripperInterface;
    
    private float lastDiagnosticTime;
    
    void Start()
    {
        Debug.Log("=== 変形システム診断開始 ===");
        
        // 自動検索
        if (trajectoryPlannerDeform == null)
            trajectoryPlannerDeform = FindObjectOfType<TrajectoryPlannerDeform>();
            
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
            
        if (forceController == null)
            forceController = FindObjectOfType<SimpleGripForceController>();
            
        if (gripperInterface == null)
            gripperInterface = FindObjectOfType<GripperTargetInterface>();
        
        PerformFullDiagnostic();
    }
    
    void Update()
    {
        if (enableContinuousDiagnostics && Time.time - lastDiagnosticTime > diagnosticInterval)
        {
            PerformRuntimeDiagnostic();
            lastDiagnosticTime = Time.time;
        }
    }
    
    [ContextMenu("Full System Diagnostic")]
    public void PerformFullDiagnostic()
    {
        Debug.Log("=== フル診断実行 ===");
        
        // コンポーネント存在チェック
        Debug.Log($"TrajectoryPlannerDeform: {(trajectoryPlannerDeform != null ? "✓" : "✗")}");
        Debug.Log($"DeformableTarget: {(target != null ? "✓" : "✗")}");
        Debug.Log($"SimpleGripForceController: {(forceController != null ? "✓" : "✗")}");
        Debug.Log($"GripperTargetInterface: {(gripperInterface != null ? "✓" : "✗")}");
        
        // 各コンポーネントの詳細チェック
        if (trajectoryPlannerDeform != null)
            DiagnoseTrajectoryPlannerDeform();
            
        if (target != null)
            DiagnoseDeformableTarget();
            
        if (forceController != null)
            DiagnoseForceController();
            
        if (gripperInterface != null)
            DiagnoseGripperInterface();
    }
    
    [ContextMenu("Runtime Diagnostic")]
    public void PerformRuntimeDiagnostic()
    {
        if (target != null)
        {
            var state = target.GetCurrentState();
            Debug.Log($"Runtime Status - Deformation: {state.deformation:F3}, " +
                     $"Force: {state.appliedForce:F2}N, " +
                     $"Grasped: {state.isBeingGrasped}, " +
                     $"Broken: {state.isBroken}");
        }
        
        if (forceController != null)
        {
            Debug.Log($"Force Controller - Enabled: {forceController.enabled}");
        }
    }
    
    private void DiagnoseTrajectoryPlannerDeform()
    {
        Debug.Log("--- TrajectoryPlannerDeform診断 ---");
        Debug.Log($"  Enabled: {trajectoryPlannerDeform.enabled}");
        Debug.Log($"  EnableDeformationLogging: {trajectoryPlannerDeform.enableDeformationLogging}");
        Debug.Log($"  GraspEvaluationDelay: {trajectoryPlannerDeform.graspEvaluationDelay}");
        
        // 参照チェック
        var gripperInterfaceRef = trajectoryPlannerDeform.gripperInterface;
        var targetRef = trajectoryPlannerDeform.target;
        var forceControllerRef = trajectoryPlannerDeform.forceController;
        
        Debug.Log($"  GripperInterface参照: {(gripperInterfaceRef != null ? "✓" : "✗")}");
        Debug.Log($"  Target参照: {(targetRef != null ? "✓" : "✗")}");
        Debug.Log($"  ForceController参照: {(forceControllerRef != null ? "✓" : "✗")}");
    }
    
    private void DiagnoseDeformableTarget()
    {
        Debug.Log("--- DeformableTarget診断 ---");
        Debug.Log($"  Enabled: {target.enabled}");
        Debug.Log($"  MaterialType: {target.materialType}");
        Debug.Log($"  Softness: {target.softness}");
        Debug.Log($"  EnableVisualDeformation: {target.enableVisualDeformation}");
        Debug.Log($"  EnableDebugLogs: {target.enableDebugLogs}");
        
        var state = target.GetCurrentState();
        Debug.Log($"  Current State - Deformation: {state.deformation:F3}");
        Debug.Log($"  Current State - AppliedForce: {state.appliedForce:F2}N");
        Debug.Log($"  Current State - IsBeingGrasped: {state.isBeingGrasped}");
        Debug.Log($"  Current State - IsBroken: {state.isBroken}");
    }
    
    private void DiagnoseForceController()
    {
        Debug.Log("--- SimpleGripForceController診断 ---");
        Debug.Log($"  Enabled: {forceController.enabled}");
        Debug.Log($"  BaseGripForce: {forceController.baseGripForce}");
        Debug.Log($"  ControlMode: {forceController.controlMode}");
        Debug.Log($"  EnableAdaptiveControl: {forceController.enableAdaptiveControl}");
    }
    
    private void DiagnoseGripperInterface()
    {
        Debug.Log("--- GripperTargetInterface診断 ---");
        Debug.Log($"  Enabled: {gripperInterface.enabled}");
        Debug.Log($"  ContactDetectionRadius: {gripperInterface.contactDetectionRadius}");
        Debug.Log($"  ForceTransferRate: {gripperInterface.forceTransferRate}");
        Debug.Log($"  ShowForceGizmos: {gripperInterface.showForceGizmos}");
        Debug.Log($"  EnableForceLogging: {gripperInterface.enableForceLogging}");
        
        // 参照チェック
        var simpleGripperController = gripperInterface.simpleGripperController;
        var targetRef = gripperInterface.target;
        
        Debug.Log($"  SimpleGripperController参照: {(simpleGripperController != null ? "✓" : "✗")}");
        Debug.Log($"  Target参照: {(targetRef != null ? "✓" : "✗")}");
    }
    
    [ContextMenu("Test Force Application")]
    public void TestForceApplication()
    {
        if (target == null)
        {
            Debug.LogError("Targetが設定されていません");
            return;
        }
        
        Debug.Log("=== 力適用テスト ===");
        
        // テスト用の力を直接適用
        Vector3 testPosition = target.transform.position;
        float testForce = 15f;
        
        Debug.Log($"テスト力 {testForce}N を位置 {testPosition} に適用");
        target.ApplyGripperForce(testForce, testPosition);
        
        // 0.5秒後に状態確認
        Invoke(nameof(CheckTestResult), 0.5f);
    }
    
    private void CheckTestResult()
    {
        if (target != null)
        {
            var state = target.GetCurrentState();
            Debug.Log($"テスト結果 - Deformation: {state.deformation:F3}, " +
                     $"Force: {state.appliedForce:F2}N, " +
                     $"Grasped: {state.isBeingGrasped}");
        }
    }
    
    [ContextMenu("Enable All Debug Logs")]
    public void EnableAllDebugLogs()
    {
        if (trajectoryPlannerDeform != null)
            trajectoryPlannerDeform.enableDeformationLogging = true;
            
        if (target != null)
            target.enableDebugLogs = true;
            
        if (gripperInterface != null)
            gripperInterface.enableForceLogging = true;
            
        Debug.Log("すべてのデバッグログを有効化しました");
    }
}