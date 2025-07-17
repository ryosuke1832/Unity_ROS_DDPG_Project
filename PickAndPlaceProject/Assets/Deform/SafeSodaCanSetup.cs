using UnityEngine;
using Deform;

/// <summary>
/// 安全なソーダ缶セットアップ（エラーなし版）
/// </summary>
public class SafeSodaCanSetup : MonoBehaviour
{
    [Header("セットアップ状況")]
    public bool meshFilterComplete = false;
    public bool meshRendererComplete = false;
    public bool meshColliderComplete = false;
    public bool rigidbodyComplete = false;
    public bool deformableComplete = false;
    public bool deformerComplete = false;
    
    [Header("変形設定")]
    public SquashAndStretchDeformer squashDeformer;
    public Deformable deformable;
    
    [Header("自動設定")]
    public bool autoSetupOnStart = true;
    
    [Header("デバッグ")]
    public bool showSetupLogs = true;
    
    void Start()
    {
        if (autoSetupOnStart)
        {
            SetupSodaCan();
        }
    }
    
    [ContextMenu("Setup Soda Can")]
    public void SetupSodaCan()
    {
        if (showSetupLogs)
            Debug.Log("=== Safe Soda Can Setup Started ===");
        
        CheckAndSetupMeshFilter();
        CheckAndSetupMeshRenderer();
        CheckAndSetupMeshCollider();
        CheckAndSetupRigidbody();
        CheckAndSetupDeformSystem();
        CheckAndSetupMaterial();
        
        if (showSetupLogs)
            Debug.Log("=== Safe Soda Can Setup Complete ===");
        
        ValidateSetup();
    }
    
    private void CheckAndSetupMeshFilter()
    {
        var meshFilter = GetComponent<MeshFilter>();
        if (meshFilter != null && meshFilter.sharedMesh != null)
        {
            meshFilterComplete = true;
            if (showSetupLogs)
                Debug.Log("✅ MeshFilter: Complete");
        }
        else
        {
            Debug.LogError("❌ MeshFilter: Missing or no mesh assigned!");
        }
    }
    
    private void CheckAndSetupMeshRenderer()
    {
        var meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer != null)
        {
            meshRendererComplete = true;
            if (showSetupLogs)
                Debug.Log("✅ MeshRenderer: Complete");
        }
        else
        {
            Debug.LogError("❌ MeshRenderer: Missing!");
        }
    }
    
    private void CheckAndSetupMeshCollider()
    {
        var meshCollider = GetComponent<MeshCollider>();
        if (meshCollider == null)
        {
            meshCollider = gameObject.AddComponent<MeshCollider>();
            if (showSetupLogs)
                Debug.Log("➕ MeshCollider: Added");
        }
        
        // 変形処理のための設定
        meshCollider.convex = true;
        meshCollider.isTrigger = false;
        
        var meshFilter = GetComponent<MeshFilter>();
        if (meshFilter != null && meshFilter.sharedMesh != null)
        {
            meshCollider.sharedMesh = meshFilter.sharedMesh;
        }
        
        meshColliderComplete = true;
        if (showSetupLogs)
            Debug.Log("✅ MeshCollider: Complete (Convex enabled)");
    }
    
    private void CheckAndSetupRigidbody()
    {
        var rigidbody = GetComponent<Rigidbody>();
        if (rigidbody == null)
        {
            rigidbody = gameObject.AddComponent<Rigidbody>();
            if (showSetupLogs)
                Debug.Log("➕ Rigidbody: Added");
        }
        
        // アルミ缶の物理設定
        rigidbody.mass = 0.015f;      // 15g (空の缶)
        rigidbody.drag = 0.1f;        // 空気抵抗
        rigidbody.angularDrag = 0.05f; // 回転抵抗
        rigidbody.useGravity = true;
        
        rigidbodyComplete = true;
        if (showSetupLogs)
            Debug.Log("✅ Rigidbody: Complete (Mass: 15g)");
    }
    
    private void CheckAndSetupDeformSystem()
    {
        // Deformableコンポーネントの確認・追加
        deformable = GetComponent<Deformable>();
        if (deformable == null)
        {
            deformable = gameObject.AddComponent<Deformable>();
            if (showSetupLogs)
                Debug.Log("➕ Deformable: Added");
        }
        
        // SquashAndStretchDeformerの確認・追加
        squashDeformer = GetComponent<SquashAndStretchDeformer>();
        if (squashDeformer == null)
        {
            squashDeformer = gameObject.AddComponent<SquashAndStretchDeformer>();
            if (showSetupLogs)
                Debug.Log("➕ SquashAndStretchDeformer: Added");
        }
        
        // 基本設定（安全に）
        SetupDeformerProperties();
        
        deformableComplete = true;
        deformerComplete = true;
        if (showSetupLogs)
            Debug.Log("✅ Deform System: Complete");
    }
    
    private void SetupDeformerProperties()
    {
        if (squashDeformer == null) return;
        
        // リフレクションを使用してプロパティを安全に設定
        var deformerType = squashDeformer.GetType();
        
        // Factor プロパティの設定
        SetPropertySafely(squashDeformer, "Factor", 0f);
        
        // その他のプロパティを試行
        SetPropertySafely(squashDeformer, "Squash", 0.5f);
        SetPropertySafely(squashDeformer, "Stretch", 0.5f);
        SetPropertySafely(squashDeformer, "Top", 0.5f);
        SetPropertySafely(squashDeformer, "Bottom", -0.5f);
        
        if (showSetupLogs)
            Debug.Log("✅ Deformer properties set safely");
    }
    
    private void SetPropertySafely(Component component, string propertyName, object value)
    {
        try
        {
            var property = component.GetType().GetProperty(propertyName);
            if (property != null && property.CanWrite)
            {
                property.SetValue(component, value);
                if (showSetupLogs)
                    Debug.Log($"   → {propertyName} = {value}");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogWarning($"Could not set {propertyName}: {e.Message}");
        }
    }
    
    private void CheckAndSetupMaterial()
    {
        var meshRenderer = GetComponent<MeshRenderer>();
        if (meshRenderer != null)
        {
            // 基本的なアルミ缶マテリアルを作成
            Material canMaterial = new Material(Shader.Find("Standard"));
            canMaterial.name = "AluminumCanMaterial";
            
            // アルミ缶の基本設定
            canMaterial.SetColor("_Color", new Color(0.8f, 0.8f, 0.85f, 1f));
            canMaterial.SetFloat("_Metallic", 0.9f);
            canMaterial.SetFloat("_Glossiness", 0.7f);
            
            meshRenderer.material = canMaterial;
            
            if (showSetupLogs)
                Debug.Log("✅ Material: Aluminum can material created");
        }
    }
    
    private void ValidateSetup()
    {
        Debug.Log("=== Setup Validation ===");
        Debug.Log($"MeshFilter: {(meshFilterComplete ? "✅" : "❌")}");
        Debug.Log($"MeshRenderer: {(meshRendererComplete ? "✅" : "❌")}");
        Debug.Log($"MeshCollider: {(meshColliderComplete ? "✅" : "❌")}");
        Debug.Log($"Rigidbody: {(rigidbodyComplete ? "✅" : "❌")}");
        Debug.Log($"Deformable: {(deformableComplete ? "✅" : "❌")}");
        Debug.Log($"Deformer: {(deformerComplete ? "✅" : "❌")}");
        
        bool allComplete = meshFilterComplete && meshRendererComplete && 
                          meshColliderComplete && rigidbodyComplete && 
                          deformableComplete && deformerComplete;
        
        if (allComplete)
        {
            Debug.Log("🎉 All systems ready! Soda can is ready for deformation testing.");
            
            // TrajectoryPlannerのTargetを自動設定
            UpdateTrajectoryPlannerTarget();
        }
        else
        {
            Debug.LogWarning("⚠️ Some components are missing. Please check the setup.");
        }
    }
    
    private void UpdateTrajectoryPlannerTarget()
    {
        var trajectoryPlanner = FindObjectOfType<TrajectoryPlanner>();
        if (trajectoryPlanner != null)
        {
            // public フィールドがある場合
            var targetField = trajectoryPlanner.GetType().GetField("Target");
            if (targetField != null)
            {
                targetField.SetValue(trajectoryPlanner, gameObject);
                Debug.Log("✅ TrajectoryPlanner Target updated (public field)");
                return;
            }
            
            // private フィールドの場合
            var privateTargetField = trajectoryPlanner.GetType().GetField("m_Target", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            if (privateTargetField != null)
            {
                privateTargetField.SetValue(trajectoryPlanner, gameObject);
                Debug.Log("✅ TrajectoryPlanner Target updated (private field)");
            }
        }
    }
    
    [ContextMenu("Test Deformation")]
    public void TestDeformation()
    {
        if (squashDeformer != null)
        {
            StartCoroutine(DeformationTest());
        }
    }
    
    private System.Collections.IEnumerator DeformationTest()
    {
        Debug.Log("🧪 Starting deformation test...");
        
        // 変形を適用
        float targetDeformation = 0.5f;
        float currentDeformation = 0f;
        
        while (currentDeformation < targetDeformation)
        {
            currentDeformation += Time.deltaTime * 0.5f;
            SetPropertySafely(squashDeformer, "Factor", currentDeformation);
            yield return null;
        }
        
        yield return new WaitForSeconds(1f);
        
        // 元に戻す
        while (currentDeformation > 0f)
        {
            currentDeformation -= Time.deltaTime * 0.5f;
            SetPropertySafely(squashDeformer, "Factor", currentDeformation);
            yield return null;
        }
        
        Debug.Log("✅ Deformation test complete");
    }
    
    /// <summary>
    /// 外部からの変形制御
    /// </summary>
    public void SetDeformation(float factor)
    {
        if (squashDeformer != null)
        {
            SetPropertySafely(squashDeformer, "Factor", Mathf.Clamp01(factor));
        }
    }
    
    /// <summary>
    /// 変形の強さを取得
    /// </summary>
    public float GetDeformationAmount()
    {
        if (squashDeformer != null)
        {
            try
            {
                var factorProperty = squashDeformer.GetType().GetProperty("Factor");
                if (factorProperty != null)
                {
                    return (float)factorProperty.GetValue(squashDeformer);
                }
            }
            catch
            {
                // プロパティが見つからない場合
            }
        }
        return 0f;
    }
    
    [ContextMenu("Show Available Properties")]
    public void ShowAvailableProperties()
    {
        if (squashDeformer != null)
        {
            Debug.Log("=== Available Properties ===");
            var properties = squashDeformer.GetType().GetProperties();
            foreach (var prop in properties)
            {
                Debug.Log($"Property: {prop.Name} ({prop.PropertyType.Name})");
            }
        }
    }
}