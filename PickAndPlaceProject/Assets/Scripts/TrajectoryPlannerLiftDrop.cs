using System.Collections;
using UnityEngine;

/// <summary>
/// TrajectoryPlannerに持ち上げ動作を追加するコンポーネント
/// 既存のTrajectoryPlannerと同じGameObjectにアタッチして使用
/// TrajectoryPlannerDeformと同じパターンで実装
/// </summary>
public class TrajectoryPlannerLiftDrop : MonoBehaviour
{
    [Header("持ち上げ設定")]
    public float liftHeight = 0.15f; // 持ち上げる高さ
    public float holdDuration = 2.0f; // 保持時間
    public bool enableLiftLogging = true;
    
    // 元のTrajectoryPlannerへの参照
    private TrajectoryPlanner originalTrajectoryPlanner;
    
    // 持ち上げ状態管理
    private bool isCurrentlyLifting = false;
    
    void Start()
    {
        // 同じGameObjectのTrajectoryPlannerを取得
        originalTrajectoryPlanner = GetComponent<TrajectoryPlanner>();
        if (originalTrajectoryPlanner == null)
        {
            Debug.LogError("TrajectoryPlannerLiftDrop: 同じGameObjectにTrajectoryPlannerが見つかりません");
            enabled = false;
            return;
        }
        
        if (enableLiftLogging)
            Debug.Log("TrajectoryPlannerLiftDrop初期化完了");
    }
    
    /// <summary>
    /// 持ち上げ動作を実行
    /// 外部から呼び出し可能（例：UI、ボタンイベント）
    /// </summary>
    public void ExecuteLiftAndDrop()
    {
        if (isCurrentlyLifting)
        {
            if (enableLiftLogging)
                Debug.LogWarning("既に持ち上げ動作中です");
            return;
        }
        
        if (enableLiftLogging)
            Debug.Log("持ち上げ動作開始");
        
        StartCoroutine(LiftAndDropSequence());
    }
    
    /// <summary>
    /// 持ち上げ動作のシーケンス
    /// </summary>
    private IEnumerator LiftAndDropSequence()
    {
        isCurrentlyLifting = true;
        
        // 1. 通常のピック動作を実行
        if (enableLiftLogging)
            Debug.Log("ステップ1: 通常のピック動作");
        
        originalTrajectoryPlanner.PublishJoints();
        
        // ピック動作完了まで待機（適当な時間）
        yield return new WaitForSeconds(8f);
        
        // 2. 持ち上げ動作
        if (enableLiftLogging)
            Debug.Log($"ステップ2: {liftHeight}m持ち上げ");
        
        yield return StartCoroutine(LiftTarget());
        
        // 3. 保持
        if (enableLiftLogging)
            Debug.Log($"ステップ3: {holdDuration}秒保持");
        
        yield return new WaitForSeconds(holdDuration);
        
        // 4. 離す
        if (enableLiftLogging)
            Debug.Log("ステップ4: 物体を離す");
        
        yield return StartCoroutine(DropTarget());
        
        isCurrentlyLifting = false;
        
        if (enableLiftLogging)
            Debug.Log("持ち上げ動作完了");
    }
    
    /// <summary>
    /// ターゲットを持ち上げる
    /// </summary>
    private IEnumerator LiftTarget()
    {
        if (originalTrajectoryPlanner.Target == null)
        {
            Debug.LogError("ターゲットが設定されていません");
            yield break;
        }
        
        // 現在のターゲット位置から上に移動
        Vector3 currentPos = originalTrajectoryPlanner.Target.transform.position;
        Vector3 liftedPos = currentPos + Vector3.up * liftHeight;
        
        // 簡易実装：ターゲットを直接移動（実際にはロボットアームが移動する）
        float moveTime = 2f;
        float elapsed = 0f;
        
        while (elapsed < moveTime)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / moveTime;
            
            Vector3 newPos = Vector3.Lerp(currentPos, liftedPos, t);
            originalTrajectoryPlanner.Target.transform.position = newPos;
            
            yield return null;
        }
        
        if (enableLiftLogging)
            Debug.Log($"ターゲットを{liftHeight}m持ち上げました");
    }
    
    /// <summary>
    /// ターゲットを離す
    /// </summary>
    private IEnumerator DropTarget()
    {
        // グリッパーを開く（元のメソッドを使用）
        // TrajectoryPlannerのOpenGripperは非公開なので、リフレクションか別の方法が必要
        // 簡易実装として、ターゲットを少し下に落とす
        
        if (originalTrajectoryPlanner.Target == null) yield break;
        
        Vector3 currentPos = originalTrajectoryPlanner.Target.transform.position;
        Vector3 dropPos = currentPos + Vector3.down * (liftHeight * 0.5f);
        
        float dropTime = 1f;
        float elapsed = 0f;
        
        while (elapsed < dropTime)
        {
            elapsed += Time.deltaTime;
            float t = elapsed / dropTime;
            
            Vector3 newPos = Vector3.Lerp(currentPos, dropPos, t);
            originalTrajectoryPlanner.Target.transform.position = newPos;
            
            yield return null;
        }
        
        if (enableLiftLogging)
            Debug.Log("ターゲットを離しました");
    }
    
    /// <summary>
    /// 手動テスト用
    /// </summary>
    [ContextMenu("Execute Lift and Drop")]
    public void TestLiftAndDrop()
    {
        ExecuteLiftAndDrop();
    }
    
    /// <summary>
    /// 持ち上げパラメータの設定
    /// </summary>
    public void SetLiftParameters(float height, float duration)
    {
        liftHeight = height;
        holdDuration = duration;
        
        if (enableLiftLogging)
            Debug.Log($"持ち上げパラメータ更新: 高さ={height}m, 保持時間={duration}秒");
    }
    
    /// <summary>
    /// 現在の状態確認
    /// </summary>
    public void ShowCurrentStatus()
    {
        Debug.Log("=== 持ち上げシステム状態 ===");
        Debug.Log($"持ち上げ中: {isCurrentlyLifting}");
        Debug.Log($"持ち上げ高さ: {liftHeight}m");
        Debug.Log($"保持時間: {holdDuration}秒");
        Debug.Log($"ターゲット: {(originalTrajectoryPlanner?.Target != null ? originalTrajectoryPlanner.Target.name : "未設定")}");
    }
}