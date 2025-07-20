using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 缶の左右圧力による変形を実装するカスタムスクリプト
/// </summary>
public class CustomCanDeformation : MonoBehaviour
{
    [Header("変形設定")]
    [Range(0f, 1f)]
    public float deformationFactor = 0f;
    
    [Range(0.1f, 0.9f)]
    public float maxCompressionRatio = 0.7f; // 最大圧縮率（0.7 = 30%圧縮）
    
    [Range(0f, 0.5f)]
    public float verticalExpansion = 0.1f; // 縦方向への微細な膨張
    
    [Header("アニメーション")]
    public bool animate = false;
    [Range(0.1f, 5f)]
    public float animationSpeed = 1f;
    
    [Header("物理設定")]
    [Range(1f, 50f)]
    public float forceThreshold = 10f; // 変形開始の力の閾値
    
    [Range(0.1f, 2f)]
    public float deformationSpeed = 1f; // 変形速度
    
    [Header("デバッグ")]
    public bool showDebugInfo = true;
    public bool enableAutoTest = false;
    
    // 内部状態
    private MeshFilter meshFilter;
    private Mesh originalMesh;
    private Mesh deformMesh;
    private Vector3[] originalVertices;
    private Vector3[] deformVertices;
    private Bounds originalBounds;
    
    private float currentDeformation = 0f;
    private bool isInitialized = false;
    
    void Start()
    {
        InitializeMesh();
        
        if (enableAutoTest)
        {
            StartCoroutine(AutoTestDeformation());
        }
    }
    
    void Update()
    {
        if (!isInitialized) return;
        
        if (animate)
        {
            // 自動アニメーション
            float time = Time.time * animationSpeed;
            deformationFactor = (Mathf.Sin(time) + 1f) * 0.5f;
        }
        
        // 変形の適用
        ApplyDeformation(deformationFactor);
        
        if (showDebugInfo)
        {
            ShowDebugInfo();
        }
    }
    
    /// <summary>
    /// メッシュの初期化
    /// </summary>
    void InitializeMesh()
    {
        meshFilter = GetComponent<MeshFilter>();
        if (meshFilter == null || meshFilter.sharedMesh == null)
        {
            Debug.LogError("❌ MeshFilterまたはMeshが見つかりません");
            return;
        }
        
        // オリジナルメッシュのコピーを作成
        originalMesh = meshFilter.sharedMesh;
        deformMesh = Instantiate(originalMesh);
        deformMesh.name = originalMesh.name + "_Deformed";
        
        // 頂点データをコピー
        originalVertices = originalMesh.vertices;
        deformVertices = new Vector3[originalVertices.Length];
        originalBounds = originalMesh.bounds;
        
        // 変形用メッシュを設定
        meshFilter.mesh = deformMesh;
        
        isInitialized = true;
        Debug.Log($"✅ メッシュ初期化完了: {originalVertices.Length} vertices");
    }
    
    /// <summary>
    /// 変形を適用
    /// </summary>
    void ApplyDeformation(float factor)
    {
        if (!isInitialized) return;
        
        factor = Mathf.Clamp01(factor);
        currentDeformation = factor;
        
        for (int i = 0; i < originalVertices.Length; i++)
        {
            Vector3 vertex = originalVertices[i];
            
            // 缶の中心位置
            Vector3 center = originalBounds.center;
            
            // 中心からの相対位置
            Vector3 relativePos = vertex - center;
            
            // X軸方向の圧縮（一方向から押される）
            float compressionX = 1f - (factor * (1f - maxCompressionRatio));
            relativePos.x *= compressionX;
            
            // Z軸方向は逆に膨らむ（押されて膨らむ効果）
            float expansionZ = 1f + (factor * 0.3f); // 30%まで膨らむ
            relativePos.z *= expansionZ;
            
            // Y軸（高さ）は変更しない
            // relativePos.y はそのまま
            
            // 中央付近での追加の変形（より自然な潰れ）
            float heightFactor = 1f - (Mathf.Abs(relativePos.y) / (originalBounds.size.y * 0.5f));
            float additionalCompression = factor * 0.2f * heightFactor;
            relativePos.x *= (1f - additionalCompression);
            
            // 変形後の位置を計算
            vertex = center + relativePos;
            
            deformVertices[i] = vertex;
        }
        
        // メッシュに変形を適用
        deformMesh.vertices = deformVertices;
        deformMesh.RecalculateNormals();
        deformMesh.RecalculateBounds();
    }
    
    /// <summary>
    /// 外部から変形を適用（衝突時など）
    /// </summary>
    public void ApplyForce(float force)
    {
        if (force >= forceThreshold)
        {
            float targetDeformation = Mathf.Clamp01(force / (forceThreshold * 3f));
            StartCoroutine(SmoothDeformation(targetDeformation));
        }
    }
    
    /// <summary>
    /// スムーズな変形アニメーション
    /// </summary>
    System.Collections.IEnumerator SmoothDeformation(float targetDeformation)
    {
        float startDeformation = deformationFactor;
        float elapsedTime = 0f;
        float duration = 1f / deformationSpeed;
        
        while (elapsedTime < duration)
        {
            elapsedTime += Time.deltaTime;
            deformationFactor = Mathf.Lerp(startDeformation, targetDeformation, elapsedTime / duration);
            yield return null;
        }
        
        deformationFactor = targetDeformation;
    }
    
    /// <summary>
    /// 変形をリセット
    /// </summary>
    [ContextMenu("Reset Deformation")]
    public void ResetDeformation()
    {
        deformationFactor = 0f;
        ApplyDeformation(0f);
        Debug.Log("🔄 変形をリセットしました");
    }
    
    /// <summary>
    /// 最大変形を適用
    /// </summary>
    [ContextMenu("Apply Max Deformation")]
    public void ApplyMaxDeformation()
    {
        deformationFactor = 1f;
        ApplyDeformation(1f);
        Debug.Log("⚡ 最大変形を適用しました");
    }
    
    /// <summary>
    /// 自動テスト
    /// </summary>
    System.Collections.IEnumerator AutoTestDeformation()
    {
        yield return new WaitForSeconds(2f);
        
        Debug.Log("🧪 自動変形テスト開始");
        
        // 変形適用
        yield return SmoothDeformation(0.8f);
        yield return new WaitForSeconds(1f);
        
        // リセット
        yield return SmoothDeformation(0f);
        
        Debug.Log("✅ 自動変形テスト完了");
    }
    
    /// <summary>
    /// 衝突検出
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        float impactForce = collision.impulse.magnitude / Time.fixedDeltaTime;
        
        if (showDebugInfo)
        {
            Debug.Log($"💥 衝突検出: 力 = {impactForce:F1}N");
        }
        
        ApplyForce(impactForce);
    }
    
    /// <summary>
    /// デバッグ情報表示
    /// </summary>
    void ShowDebugInfo()
    {
        if (Time.frameCount % 60 == 0) // 1秒ごと
        {
            if (currentDeformation > 0.01f)
            {
                Debug.Log($"📊 変形率: {currentDeformation:F3} ({currentDeformation * 100:F1}%)");
            }
        }
    }
    
    /// <summary>
    /// Gizmo描画
    /// </summary>
    void OnDrawGizmos()
    {
        if (!isInitialized || currentDeformation < 0.01f) return;
        
        Gizmos.color = Color.red;
        Vector3 center = transform.position + originalBounds.center;
        
        // 変形後のサイズを計算
        Vector3 size = originalBounds.size;
        
        // X軸方向に圧縮
        float compressionX = 1f - (currentDeformation * (1f - maxCompressionRatio));
        size.x *= compressionX;
        
        // Z軸方向に膨張
        float expansionZ = 1f + (currentDeformation * 0.3f);
        size.z *= expansionZ;
        
        // Y軸（高さ）は変更しない
        
        Gizmos.DrawWireCube(center, size);
        
        // X軸方向の圧力を示す（左右から押される）
        Gizmos.color = Color.yellow;
        float arrowLength = currentDeformation * 0.4f;
        float offsetX = size.x * 0.5f + 0.1f;
        
        // 左右からの圧力
        Gizmos.DrawRay(center + Vector3.right * offsetX, Vector3.left * arrowLength);
        Gizmos.DrawRay(center + Vector3.left * offsetX, Vector3.right * arrowLength);
        
        // Z軸方向の膨張を示す
        Gizmos.color = Color.green;
        float offsetZ = size.z * 0.5f;
        Gizmos.DrawRay(center, Vector3.forward * offsetZ);
        Gizmos.DrawRay(center, Vector3.back * offsetZ);
        
        // 缶の軸（Y軸）を示す
        Gizmos.color = Color.blue;
        float height = originalBounds.size.y * 0.6f;
        Gizmos.DrawLine(center + Vector3.up * height, center + Vector3.down * height);
        
        // 変形の説明テキスト
        Gizmos.color = Color.white;
        // デバッグ情報表示用の線
        if (currentDeformation > 0.1f)
        {
            Gizmos.DrawWireSphere(center, 0.01f);
        }
    }
    
    /// <summary>
    /// クリーンアップ
    /// </summary>
    void OnDestroy()
    {
        if (deformMesh != null)
        {
            DestroyImmediate(deformMesh);
        }
    }
}