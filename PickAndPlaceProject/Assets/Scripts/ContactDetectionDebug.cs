using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// 接触検出の詳細デバッグ用スクリプト
/// </summary>
public class ContactDetectionDebug : MonoBehaviour
{
    [Header("参照")]
    public GripperTargetInterface gripperInterface;
    public DeformableTarget target;
    
    [Header("デバッグ設定")]
    public bool enableContinuousLogging = true;
    public float loggingInterval = 1f;
    
    private float lastLogTime;
    
    void Start()
    {
        if (gripperInterface == null)
            gripperInterface = FindObjectOfType<GripperTargetInterface>();
        if (target == null)
            target = FindObjectOfType<DeformableTarget>();
    }
    
    void Update()
    {
        if (enableContinuousLogging && Time.time - lastLogTime > loggingInterval)
        {
            LogDistanceInfo();
            lastLogTime = Time.time;
        }
    }
    
    [ContextMenu("Log Distance Info")]
    public void LogDistanceInfo()
    {
        if (gripperInterface == null || target == null)
        {
            Debug.LogError("必要なコンポーネントが見つかりません");
            return;
        }
        
        Vector3 gripperPos = gripperInterface.transform.position;
        Vector3 targetPos = target.transform.position;
        float distance = Vector3.Distance(gripperPos, targetPos);
        float detectionRadius = gripperInterface.contactDetectionRadius;
        
        Debug.Log("=== 接触検出詳細情報 ===");
        Debug.Log($"グリッパー位置: {gripperPos}");
        Debug.Log($"ターゲット位置: {targetPos}");
        Debug.Log($"現在の距離: {distance:F3}m");
        Debug.Log($"検出半径: {detectionRadius:F3}m");
        Debug.Log($"接触判定: {(distance <= detectionRadius ? "接触中" : "範囲外")}");
        Debug.Log($"距離差: {(distance - detectionRadius):F3}m");
        
        if (distance > detectionRadius)
        {
            Debug.LogWarning($"⚠️ 距離が検出範囲外です！{(distance - detectionRadius):F3}m 離れています");
            Debug.Log("対策: Contact Detection Radius を最低でも " + (distance + 0.1f).ToString("F2") + " に設定してください");
        }
    }
    
    [ContextMenu("Force Contact Test")]
    public void ForceContactTest()
    {
        if (gripperInterface == null || target == null)
        {
            Debug.LogError("必要なコンポーネントが見つかりません");
            return;
        }
        
        // 強制的に接触範囲を拡大してテスト
        float originalRadius = gripperInterface.contactDetectionRadius;
        float distance = Vector3.Distance(gripperInterface.transform.position, target.transform.position);
        
        Debug.Log($"元の検出半径: {originalRadius:F3}m");
        Debug.Log($"現在の距離: {distance:F3}m");
        
        // 距離より少し大きな値に設定
        gripperInterface.contactDetectionRadius = distance + 0.2f;
        
        Debug.Log($"検出半径を {gripperInterface.contactDetectionRadius:F3}m に変更しました");
        Debug.Log("これで接触検出が成功するはずです");
    }
    
    [ContextMenu("Move Gripper to Target")]
    public void MoveGripperToTarget()
    {
        if (gripperInterface == null || target == null)
        {
            Debug.LogError("必要なコンポーネントが見つかりません");
            return;
        }
        
        // グリッパーをターゲットの近くに移動
        Vector3 targetPos = target.transform.position;
        Vector3 newGripperPos = targetPos + Vector3.up * 0.05f; // ターゲットの少し上
        
        gripperInterface.transform.position = newGripperPos;
        
        Debug.Log($"グリッパーを {newGripperPos} に移動しました");
        
        // 移動後の距離確認
        LogDistanceInfo();
    }
    
    void OnDrawGizmos()
    {
        if (gripperInterface == null || target == null) return;
        
        // 接続線を描画
        Gizmos.color = Color.blue;
        Gizmos.DrawLine(gripperInterface.transform.position, target.transform.position);
        
        // 距離をテキスト表示（Scene Viewで確認）
        Vector3 midPoint = (gripperInterface.transform.position + target.transform.position) / 2f;
        float distance = Vector3.Distance(gripperInterface.transform.position, target.transform.position);
        
        #if UNITY_EDITOR
        UnityEditor.Handles.Label(midPoint, $"Distance: {distance:F3}m");
        #endif
    }
}
