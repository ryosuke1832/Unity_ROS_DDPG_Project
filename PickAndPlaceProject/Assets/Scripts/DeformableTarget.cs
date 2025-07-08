using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 新規作成：変形可能なターゲットオブジェクト
/// 既存のTarget.prefabとは独立した新しいターゲット
/// </summary>
public class DeformableTarget : MonoBehaviour
{
    [Header("変形設定")]
    [Range(0f, 1f)]
    public float softness = 0.5f;              // 0=硬い, 1=柔らかい
    public float maxDeformation = 0.3f;         // 最大変形量
    public float deformationSpeed = 2f;         // 変形速度
    public bool enableVisualDeformation = true; // 視覚的変形
    
    [Header("物理特性")]
    public float breakingForce = 50f;           // 破損力閾値
    public float compressionResistance = 15f;   // 圧縮抵抗
    public float mass = 1f;                     // 質量
    
    [Header("材質プリセット")]
    public MaterialType materialType = MaterialType.Medium;
    
    [Header("視覚設定")]
    public Color originalColor = Color.white;
    public Color stressColor = Color.red;
    public Color brokenColor = Color.gray;
    
    [Header("デバッグ")]
    public bool enableDebugLogs = false;
    public bool showContactGizmos = true;
    
    // 内部状態
    private Vector3 originalScale;
    private float currentDeformation = 0f;
    private bool isBroken = false;
    private Renderer objectRenderer;
    private Rigidbody rb;
    private Collider col;
    
    // グリッパー接触状態
    private bool isBeingGrasped = false;
    private float appliedForce = 0f;
    private List<ContactInfo> contactPoints = new List<ContactInfo>();
    
    public enum MaterialType
    {
        Soft,     // スポンジ・ゴム系
        Medium,   // プラスチック・木材
        Hard,     // 金属・石
        Fragile   // ガラス・卵
    }
    
    [System.Serializable]
    public struct ContactInfo
    {
        public Vector3 position;
        public float force;
        public float timestamp;
    }
    
    // プロパティ
    public bool IsBroken => isBroken;
    public bool IsBeingGrasped => isBeingGrasped;
    public float CurrentDeformation => currentDeformation;
    public float AppliedForce => appliedForce;
    
    void Start()
    {
        InitializeDeformableObject();
    }
    
    void Update()
    {
        UpdateDeformation();
        UpdateVisualFeedback();
        CleanupOldContactPoints();
    }
    
    private void InitializeDeformableObject()
    {
        // 基本コンポーネントの初期化
        SetupComponents();
        
        // 初期状態の保存
        originalScale = transform.localScale;
        
        // 材質に応じた初期設定
        ApplyMaterialPreset(materialType);
        
        if (enableDebugLogs)
            Debug.Log($"DeformableTarget initialized - Material: {materialType}");
    }
    
    private void SetupComponents()
    {
        // Renderer の取得/作成
        objectRenderer = GetComponent<Renderer>();
        if (objectRenderer == null)
        {
            objectRenderer = gameObject.AddComponent<MeshRenderer>();
            var meshFilter = GetComponent<MeshFilter>();
            if (meshFilter == null)
            {
                meshFilter = gameObject.AddComponent<MeshFilter>();
                meshFilter.mesh = CreateDefaultMesh();
            }
        }
        
        // Material の設定
        if (objectRenderer.material == null)
        {
            objectRenderer.material = CreateDefaultMaterial();
        }
        
        // Rigidbody の取得/作成
        rb = GetComponent<Rigidbody>();
        if (rb == null)
        {
            rb = gameObject.AddComponent<Rigidbody>();
        }
        rb.mass = mass;
        rb.useGravity = true;
        
        // Collider の取得/作成
        col = GetComponent<Collider>();
        if (col == null)
        {
            col = gameObject.AddComponent<BoxCollider>();
        }
    }
    
    private Mesh CreateDefaultMesh()
    {
        // 基本的なキューブメッシュを作成
        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        Mesh mesh = cube.GetComponent<MeshFilter>().mesh;
        DestroyImmediate(cube);
        return mesh;
    }
    
    private Material CreateDefaultMaterial()
    {
        Material mat = new Material(Shader.Find("Standard"));
        mat.color = originalColor;
        return mat;
    }
    
    /// <summary>
    /// グリッパーからの力を受け取る
    /// </summary>
    public void ApplyGripperForce(float force, Vector3 contactPosition)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        // 接触情報の記録
        var contact = new ContactInfo
        {
            position = contactPosition,
            force = force,
            timestamp = Time.time
        };
        contactPoints.Add(contact);
        
        // 破損チェック
        if (force > breakingForce)
        {
            BreakObject();
        }
        
        // 変形の計算
        CalculateDeformation(force);
        
        if (enableDebugLogs)
            Debug.Log($"Force applied: {force:F2}N, Deformation: {currentDeformation:F3}");
    }
    
    private void CalculateDeformation(float force)
    {
        // 材質の柔軟性に基づく変形計算
        float targetDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        
        // 滑らかな変形
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, 
                                      deformationSpeed * Time.deltaTime);
    }
    
    private void UpdateDeformation()
    {
        if (!enableVisualDeformation || isBroken) return;
        
        // 力が加わっていない時は元の形に戻る
        if (!isBeingGrasped)
        {
            currentDeformation = Mathf.Lerp(currentDeformation, 0f, 
                                          deformationSpeed * Time.deltaTime);
        }
        
        // スケール変形（簡単な圧縮効果）
        float compressionFactor = 1f - currentDeformation;
        Vector3 deformedScale = new Vector3(
            originalScale.x * (1f + currentDeformation * 0.2f), // 横に広がる
            originalScale.y * compressionFactor,                 // 縦に圧縮
            originalScale.z * (1f + currentDeformation * 0.2f)   // 奥行きに広がる
        );
        
        transform.localScale = Vector3.Lerp(transform.localScale, deformedScale, 
                                           Time.deltaTime * deformationSpeed);
    }
    
    private void UpdateVisualFeedback()
    {
        if (objectRenderer == null) return;
        
        if (isBroken)
        {
            objectRenderer.material.color = brokenColor;
        }
        else
        {
            // 力に応じた色変化
            Color currentColor = Color.Lerp(originalColor, stressColor, currentDeformation);
            objectRenderer.material.color = currentColor;
        }
    }
    
    private void CleanupOldContactPoints()
    {
        // 古い接触点を削除（5秒以上前のもの）
        for (int i = contactPoints.Count - 1; i >= 0; i--)
        {
            if (Time.time - contactPoints[i].timestamp > 5f)
            {
                contactPoints.RemoveAt(i);
            }
        }
    }
    
    private void ApplyMaterialPreset(MaterialType type)
    {
        switch (type)
        {
            case MaterialType.Soft:
                softness = 0.8f;
                compressionResistance = 5f;
                breakingForce = 30f;
                mass = 0.5f;
                break;
            case MaterialType.Medium:
                softness = 0.5f;
                compressionResistance = 15f;
                breakingForce = 50f;
                mass = 1f;
                break;
            case MaterialType.Hard:
                softness = 0.2f;
                compressionResistance = 30f;
                breakingForce = 80f;
                mass = 2f;
                break;
            case MaterialType.Fragile:
                softness = 0.6f;
                compressionResistance = 8f;
                breakingForce = 20f; // 低い破損閾値
                mass = 0.3f;
                break;
        }
        
        // Rigidbodyの質量を更新
        if (rb != null)
            rb.mass = mass;
    }
    
    private void BreakObject()
    {
        if (isBroken) return;
        
        isBroken = true;
        
        if (enableDebugLogs)
            Debug.Log($"Object broken! Applied force: {appliedForce}N");
        
        // 物理的な効果
        if (rb != null)
        {
            rb.constraints = RigidbodyConstraints.None; // 制約を解除
        }
        
        // 破損エフェクト（後で実装可能）
        OnObjectBroken?.Invoke();
    }
    
    /// <summary>
    /// 材質プリセットを外部から変更
    /// </summary>
    public void SetMaterialType(MaterialType newType)
    {
        materialType = newType;
        ApplyMaterialPreset(materialType);
        
        if (enableDebugLogs)
            Debug.Log($"Material changed to: {materialType}");
    }
    
    /// <summary>
    /// オブジェクトをリセット
    /// </summary>
    public void ResetObject()
    {
        isBroken = false;
        currentDeformation = 0f;
        appliedForce = 0f;
        isBeingGrasped = false;
        contactPoints.Clear();
        
        transform.localScale = originalScale;
        
        if (objectRenderer != null)
            objectRenderer.material.color = originalColor;
            
        if (rb != null)
        {
            rb.velocity = Vector3.zero;
            rb.angularVelocity = Vector3.zero;
            rb.constraints = RigidbodyConstraints.None;
        }
        
        if (enableDebugLogs)
            Debug.Log("Object reset");
    }
    
    /// <summary>
    /// 現在の状態を取得（将来のEEG/DDPG用）
    /// </summary>
    public ObjectState GetCurrentState()
    {
        return new ObjectState
        {
            deformation = currentDeformation,
            appliedForce = appliedForce,
            isBroken = isBroken,
            isBeingGrasped = isBeingGrasped,
            materialType = materialType,
            softness = softness,
            position = transform.position,
            rotation = transform.rotation,
            scale = transform.localScale
        };
    }
    
    // イベント
    public System.Action OnObjectBroken;
    
    [System.Serializable]
    public struct ObjectState
    {
        public float deformation;
        public float appliedForce;
        public bool isBroken;
        public bool isBeingGrasped;
        public MaterialType materialType;
        public float softness;
        public Vector3 position;
        public Quaternion rotation;
        public Vector3 scale;
    }
    
    // デバッグ表示
    void OnDrawGizmos()
    {
        if (!showContactGizmos) return;
        
        // 接触点の表示
        Gizmos.color = Color.yellow;
        foreach (var contact in contactPoints)
        {
            if (Time.time - contact.timestamp < 1f) // 1秒以内の接触点のみ
            {
                float intensity = contact.force / breakingForce;
                Gizmos.color = Color.Lerp(Color.green, Color.red, intensity);
                Gizmos.DrawSphere(contact.position, 0.02f);
            }
        }
        
        // 現在の変形状態の表示
        if (currentDeformation > 0.01f)
        {
            Gizmos.color = Color.blue;
            Gizmos.DrawWireCube(transform.position, transform.localScale);
        }
    }
    
    // Unity Editor用の設定変更時の処理
    void OnValidate()
    {
        if (Application.isPlaying)
        {
            ApplyMaterialPreset(materialType);
        }
    }
}