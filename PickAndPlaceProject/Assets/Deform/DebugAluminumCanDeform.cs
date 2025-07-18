using UnityEngine;
using Deform;
using System.Collections;

/// <summary>
/// デバッグ用アルミ缶変形テスト - 変形しない問題を診断・修正
/// </summary>
public class DebugAluminumCanDeform : MonoBehaviour
{
    [Header("缶の基本設定")]
    [Range(0.1f, 0.3f)]
    public float canRadius = 0.066f;
    [Range(0.08f, 0.25f)]
    public float canHeight = 0.123f;
    
    [Header("変形設定")]
    [Range(1f, 50f)]
    public float deformationThreshold = 8.7f; // UI値と同じ
    [Range(0.1f, 5f)]
    public float deformationSpeed = 2f;
    [Range(0f, 1f)]
    public float maxDeformationFactor = 0.8f;
    public bool instantDeformation = true;
    
    [Header("Deformコンポーネント")]
    public SquashAndStretchDeformer squashDeformer;
    public Deformable deformableComponent;
    
    [Header("デバッグ設定")]
    public bool enableDebugLogs = true;
    public bool autoTestOnStart = true;
    public bool showDetailedInfo = true;
    
    // 内部状態
    private Rigidbody canRigidbody;
    private MeshCollider meshCollider;
    private MeshRenderer meshRenderer;
    private bool isSetupComplete = false;
    private float currentDeformationLevel = 0f;
    
    void Start()
    {
        Debug.Log("=== DebugAluminumCanDeform 開始 ===");
        StartCoroutine(InitializeWithDelay());
    }
    
    IEnumerator InitializeWithDelay()
    {
        yield return new WaitForEndOfFrame();
        
        SetupComponents();
        ValidateSetup();
        
        if (autoTestOnStart)
        {
            yield return new WaitForSeconds(1f);
            TestDeformation();
        }
    }
    
    /// <summary>
    /// コンポーネントのセットアップ
    /// </summary>
    void SetupComponents()
    {
        Debug.Log("🔧 コンポーネントセットアップ開始");
        
        // Deformableコンポーネントの設定
        if (deformableComponent == null)
        {
            deformableComponent = GetComponent<Deformable>();
            if (deformableComponent == null)
            {
                deformableComponent = gameObject.AddComponent<Deformable>();
                Debug.Log("➕ Deformableコンポーネントを追加");
            }
        }
        
        // SquashAndStretchDeformerの設定
        if (squashDeformer == null)
        {
            squashDeformer = GetComponent<SquashAndStretchDeformer>();
            if (squashDeformer == null)
            {
                squashDeformer = gameObject.AddComponent<SquashAndStretchDeformer>();
                Debug.Log("➕ SquashAndStretchDeformerを追加");
            }
        }
        
        // Deformerの初期設定
        SetupSquashDeformer();
        
        // 他のコンポーネント取得
        canRigidbody = GetComponent<Rigidbody>();
        meshCollider = GetComponent<MeshCollider>();
        meshRenderer = GetComponent<MeshRenderer>();
        
        Debug.Log("✅ コンポーネントセットアップ完了");
    }
    
    /// <summary>
    /// SquashAndStretchDeformerの詳細設定
    /// </summary>
    void SetupSquashDeformer()
    {
        if (squashDeformer == null) return;
        
        try
        {
            // 基本的なプロパティ設定
            squashDeformer.Factor = 0f;
            squashDeformer.Top = canHeight * 0.5f;     // 缶の上端
            squashDeformer.Bottom = -canHeight * 0.5f; // 缶の下端
            squashDeformer.Curvature = 1f;             // カーブ設定
            squashDeformer.Axis = transform;           // 軸設定
            
            Debug.Log($"✅ SquashDeformer設定完了:");
            Debug.Log($"   Factor: {squashDeformer.Factor}");
            Debug.Log($"   Top: {squashDeformer.Top}");
            Debug.Log($"   Bottom: {squashDeformer.Bottom}");
            Debug.Log($"   Curvature: {squashDeformer.Curvature}");
            
        }
        catch (System.Exception e)
        {
            Debug.LogError($"❌ SquashDeformer設定エラー: {e.Message}");
        }
    }
    
    /// <summary>
    /// セットアップの検証
    /// </summary>
    void ValidateSetup()
    {
        Debug.Log("🔍 セットアップ検証開始");
        
        bool allValid = true;
        
        // MeshFilterとMeshの確認
        var meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null || meshFilter.sharedMesh == null)
        {
            Debug.LogError("❌ MeshFilterまたはMeshが見つかりません");
            allValid = false;
        }
        else
        {
            Debug.Log($"✅ Mesh: {meshFilter.sharedMesh.name} (vertices: {meshFilter.sharedMesh.vertexCount})");
        }
        
        // Deformableの確認
        if (deformableComponent == null)
        {
            Debug.LogError("❌ Deformableコンポーネントが見つかりません");
            allValid = false;
        }
        else
        {
            Debug.Log("✅ Deformableコンポーネント");
        }
        
        // SquashAndStretchDeformerの確認
        if (squashDeformer == null)
        {
            Debug.LogError("❌ SquashAndStretchDeformerが見つかりません");
            allValid = false;
        }
        else
        {
            Debug.Log("✅ SquashAndStretchDeformer");
        }
        
        isSetupComplete = allValid;
        Debug.Log($"🏁 セットアップ検証結果: {(allValid ? "成功" : "失敗")}");
    }
    
    /// <summary>
    /// 手動変形テスト
    /// </summary>
    [ContextMenu("Test Deformation")]
    public void TestDeformation()
    {
        if (!isSetupComplete)
        {
            Debug.LogWarning("⚠️ セットアップが完了していません");
            return;
        }
        
        Debug.Log("🧪 変形テスト開始");
        StartCoroutine(DeformationTestCoroutine());
    }
    
    IEnumerator DeformationTestCoroutine()
    {
        if (squashDeformer == null)
        {
            Debug.LogError("❌ SquashDeformerが見つかりません");
            yield break;
        }
        
        // 変形適用
        float targetFactor = 0.5f; // 50%の変形
        float currentFactor = 0f;
        
        Debug.Log($"🔄 変形適用中... 目標: {targetFactor}");
        
        while (currentFactor < targetFactor)
        {
            currentFactor += Time.deltaTime * deformationSpeed;
            currentFactor = Mathf.Min(currentFactor, targetFactor);
            
            squashDeformer.Factor = currentFactor;
            currentDeformationLevel = currentFactor;
            
            if (enableDebugLogs)
            {
                Debug.Log($"   現在のFactor: {currentFactor:F3}");
            }
            
            yield return null;
        }
        
        Debug.Log($"✅ 変形完了: Factor = {squashDeformer.Factor}");
        
        // 1秒待機
        yield return new WaitForSeconds(2f);
        
        // 元に戻す
        Debug.Log("🔄 変形を元に戻しています...");
        
        while (currentFactor > 0f)
        {
            currentFactor -= Time.deltaTime * deformationSpeed;
            currentFactor = Mathf.Max(currentFactor, 0f);
            
            squashDeformer.Factor = currentFactor;
            currentDeformationLevel = currentFactor;
            
            yield return null;
        }
        
        Debug.Log("✅ 変形リセット完了");
    }
    
    /// <summary>
    /// 衝突による変形
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        if (!isSetupComplete) return;
        
        float impactForce = collision.impulse.magnitude / Time.fixedDeltaTime;
        
        if (enableDebugLogs)
        {
            Debug.Log($"💥 衝突検出: 力 = {impactForce:F1}N, 閾値 = {deformationThreshold}N");
        }
        
        if (impactForce >= deformationThreshold)
        {
            ApplyCollisionDeformation(impactForce);
        }
    }
    
    /// <summary>
    /// 衝突変形の適用
    /// </summary>
    void ApplyCollisionDeformation(float force)
    {
        if (squashDeformer == null) return;
        
        float deformationFactor = Mathf.Clamp01(force / (deformationThreshold * 3f));
        deformationFactor *= maxDeformationFactor;
        
        if (instantDeformation)
        {
            squashDeformer.Factor = Mathf.Max(squashDeformer.Factor, deformationFactor);
            currentDeformationLevel = squashDeformer.Factor;
            
            if (enableDebugLogs)
            {
                Debug.Log($"⚡ 即座変形適用: Factor = {squashDeformer.Factor:F3}");
            }
        }
        else
        {
            // 段階的変形
            StartCoroutine(GradualDeformation(deformationFactor));
        }
    }
    
    IEnumerator GradualDeformation(float targetFactor)
    {
        float startFactor = squashDeformer.Factor;
        float elapsedTime = 0f;
        float duration = 1f / deformationSpeed;
        
        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            float currentFactor = Mathf.Lerp(startFactor, targetFactor, elapsedTime / duration);
            
            squashDeformer.Factor = currentFactor;
            currentDeformationLevel = currentFactor;
            
            yield return null;
        }
        
        squashDeformer.Factor = targetFactor;
        currentDeformationLevel = targetFactor;
    }
    
    /// <summary>
    /// 缶のリセット
    /// </summary>
    [ContextMenu("Reset Can")]
    public void ResetCan()
    {
        if (squashDeformer != null)
        {
            squashDeformer.Factor = 0f;
            currentDeformationLevel = 0f;
            Debug.Log("🔄 缶をリセットしました");
        }
    }
    
    /// <summary>
    /// 手動Factor設定（デバッグ用）
    /// </summary>
    [ContextMenu("Set Factor 0.3")]
    public void SetFactor03()
    {
        SetManualFactor(0.3f);
    }
    
    [ContextMenu("Set Factor 0.5")]
    public void SetFactor05()
    {
        SetManualFactor(0.5f);
    }
    
    [ContextMenu("Set Factor 0.8")]
    public void SetFactor08()
    {
        SetManualFactor(0.8f);
    }
    
    void SetManualFactor(float factor)
    {
        if (squashDeformer != null)
        {
            squashDeformer.Factor = factor;
            currentDeformationLevel = factor;
            Debug.Log($"🎛️ 手動でFactor設定: {factor}");
        }
        else
        {
            Debug.LogError("❌ SquashDeformerが見つかりません");
        }
    }
    
    /// <summary>
    /// 詳細情報の表示
    /// </summary>
    [ContextMenu("Show Detailed Info")]
    public void ShowDetailedInfo()
    {
        Debug.Log("=== 詳細情報 ===");
        
        if (squashDeformer != null)
        {
            Debug.Log($"Factor: {squashDeformer.Factor}");
            Debug.Log($"Top: {squashDeformer.Top}");
            Debug.Log($"Bottom: {squashDeformer.Bottom}");
            Debug.Log($"Curvature: {squashDeformer.Curvature}");
            Debug.Log($"Axis: {squashDeformer.Axis?.name}");
        }
        
        if (deformableComponent != null)
        {
            Debug.Log($"Deformable enabled: {deformableComponent.enabled}");
        }
        
        var meshFilter = GetComponent<MeshFilter>();
        if (meshFilter?.sharedMesh != null)
        {
            Debug.Log($"Mesh vertices: {meshFilter.sharedMesh.vertexCount}");
            Debug.Log($"Mesh bounds: {meshFilter.sharedMesh.bounds}");
        }
    }
    
    void Update()
    {
        if (showDetailedInfo && isSetupComplete && Time.frameCount % 60 == 0)
        {
            // 1秒ごとに状態をログ
            if (squashDeformer != null && currentDeformationLevel > 0.01f)
            {
                Debug.Log($"📊 現在のFactor: {squashDeformer.Factor:F3}, 変形レベル: {currentDeformationLevel:F3}");
            }
        }
    }
    
    void OnDrawGizmos()
    {
        if (!isSetupComplete) return;
        
        // 変形の視覚化
        if (currentDeformationLevel > 0.01f)
        {
            Gizmos.color = Color.red;
            Vector3 center = transform.position;
            Vector3 size = new Vector3(canRadius * 2f, canHeight * (1f - currentDeformationLevel * 0.5f), canRadius * 2f);
            Gizmos.DrawWireCube(center, size);
        }
        
        // 軸の表示
        if (squashDeformer != null && squashDeformer.Axis != null)
        {
            Gizmos.color = Color.blue;
            Vector3 axisStart = squashDeformer.Axis.position + Vector3.up * squashDeformer.Bottom;
            Vector3 axisEnd = squashDeformer.Axis.position + Vector3.up * squashDeformer.Top;
            Gizmos.DrawLine(axisStart, axisEnd);
        }
    }
}