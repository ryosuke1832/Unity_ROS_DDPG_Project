using UnityEngine;

/// <summary>
/// グリッパーの物理的接触を検出するコンポーネント
/// 各グリッパーの先端に追加して使用
/// </summary>
public class GripperContactDetector : MonoBehaviour
{
    private GripperTargetInterface parentInterface;
    private bool isLeftGripper;
    private bool isInitialized = false;
    
    [Header("デバッグ")]
    public bool enableContactLogs = false;
    
    /// <summary>
    /// 初期化（GripperTargetInterfaceから呼び出し）
    /// </summary>
    public void Initialize(GripperTargetInterface targetInterface, bool isLeft)
    {
        parentInterface = targetInterface;
        isLeftGripper = isLeft;
        isInitialized = true;
        
        if (enableContactLogs)
        {
            Debug.Log($"GripperContactDetector initialized for {(isLeft ? "LEFT" : "RIGHT")} gripper");
        }
    }
    
    /// <summary>
    /// 物理的な衝突開始
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        if (!isInitialized || parentInterface == null) return;
        
        if (enableContactLogs)
        {
            Debug.Log($"{(isLeftGripper ? "LEFT" : "RIGHT")} gripper collision ENTER with {collision.gameObject.name}");
        }
        
        parentInterface.OnGripperContactEnter(collision, isLeftGripper);
    }
    
    /// <summary>
    /// 物理的な衝突終了
    /// </summary>
    void OnCollisionExit(Collision collision)
    {
        if (!isInitialized || parentInterface == null) return;
        
        if (enableContactLogs)
        {
            Debug.Log($"{(isLeftGripper ? "LEFT" : "RIGHT")} gripper collision EXIT with {collision.gameObject.name}");
        }
        
        parentInterface.OnGripperContactExit(collision, isLeftGripper);
    }
    
    /// <summary>
    /// トリガー接触開始（代替検出方法）
    /// </summary>
    void OnTriggerEnter(Collider other)
    {
        if (!isInitialized || parentInterface == null) return;
        
        if (enableContactLogs)
        {
            Debug.Log($"{(isLeftGripper ? "LEFT" : "RIGHT")} gripper trigger ENTER with {other.gameObject.name}");
        }
        
        // 簡易的な接触情報を作成
        Vector3 contactPoint = other.ClosestPoint(transform.position);
        Vector3 contactNormal = (transform.position - other.transform.position).normalized;
        
        // 直接parentInterfaceのメソッドを呼び出し
        parentInterface.OnGripperContactWithCollider(other, isLeftGripper, contactPoint, contactNormal);
    }
    
    /// <summary>
    /// トリガー接触終了
    /// </summary>
    void OnTriggerExit(Collider other)
    {
        if (!isInitialized || parentInterface == null) return;
        
        if (enableContactLogs)
        {
            Debug.Log($"{(isLeftGripper ? "LEFT" : "RIGHT")} gripper trigger EXIT with {other.gameObject.name}");
        }
        
        parentInterface.OnGripperContactExitWithCollider(other, isLeftGripper);
    }
}