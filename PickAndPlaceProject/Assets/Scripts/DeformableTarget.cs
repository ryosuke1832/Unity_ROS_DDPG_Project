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
        // ContactInfoにnormalフィールドを追加
    [System.Serializable]
    public struct ContactInfo
    {
        public Vector3 position;
        public Vector3 normal;      // 追加：接触法線
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
    
    // DeformableTarget.cs の ApplyGripperForce メソッドを以下に変更

    /// <summary>
    /// グリッパーからの力を受け取る（修正版：方向情報付き）
    /// </summary>
    public void ApplyGripperForce(float force, Vector3 contactPosition, Vector3 contactNormal)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        // 接触情報の記録（方向情報付き）
        var contact = new ContactInfo
        {
            position = contactPosition,
            normal = contactNormal,
            force = force,
            timestamp = Time.time
        };
        contactPoints.Add(contact);
        
        // 破損チェック
        if (force > breakingForce)
        {
            BreakObject();
        }
        
        // 方向を考慮した変形の計算
        CalculateDirectionalDeformation(force, contactNormal);
        
        if (enableDebugLogs)
            Debug.Log($"Directional force applied: {force:F2}N at {contactPosition}, Normal: {contactNormal}, Deformation: {currentDeformation:F3}");
    }

    /// <summary>
    /// 方向を考慮した変形計算
    /// </summary>
    private void CalculateDirectionalDeformation(float force, Vector3 contactNormal)
    {
        // 材質の柔軟性に基づく変形計算
        float baseDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        
        // 接触法線の方向を考慮した変形係数
        // Y軸（上下）からの圧縮が最も効果的
        float directionFactor = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.up));
        directionFactor = Mathf.Clamp(directionFactor, 0.3f, 1f); // 最小30%の効果は保持
        
        float targetDeformation = baseDeformation * directionFactor;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        
        // 滑らかな変形
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, 
                                    deformationSpeed * Time.deltaTime);
    }

    /// <summary>
    /// 既存のApplyGripperForceメソッドとの互換性維持
    /// </summary>
    public void ApplyGripperForce(float force, Vector3 contactPosition)
    {
        // デフォルトの法線ベクトル（上向き）を使用
        ApplyGripperForce(force, contactPosition, Vector3.up);
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
    
    
    // イベント
    public System.Action OnObjectBroken;

    // DeformableTarget.cs に追加する必要があるメソッドと構造体

/// <summary>
/// オブジェクトの現在状態を表す構造体
/// </summary>
[System.Serializable]
public struct ObjectState
{
    [Header("変形情報")]
    public float deformation;           // 現在の変形量
    public float maxDeformation;        // 最大変形量
    public Vector3 originalScale;       // 元のスケール
    public Vector3 currentScale;        // 現在のスケール
    
    [Header("力の情報")]
    public float appliedForce;          // 適用されている力
    public float lastAppliedForce;      // 最後に適用された力
    public int contactCount;            // 接触点の数
    
    [Header("材質情報")]
    public MaterialType materialType;   // 材質タイプ
    public float softness;              // 柔らかさ
    public float compressionResistance; // 圧縮抵抗
    public float breakingForce;         // 破損力
    
    [Header("状態情報")]
    public bool isBroken;               // 破損状態
    public bool isBeingGrasped;         // 把持されている状態
    public bool isDeforming;            // 変形中かどうか
    
    [Header("タイミング")]
    public float lastContactTime;       // 最後の接触時刻
    public float totalGraspTime;        // 総把持時間
    
    /// <summary>
    /// 健全性スコアを計算（0-1）
    /// </summary>
    public float GetHealthScore()
    {
        if (isBroken) return 0f;
        
        float healthScore = 1f - (deformation / maxDeformation);
        return Mathf.Clamp01(healthScore);
    }
    
    /// <summary>
    /// 変形率を取得（0-1）
    /// </summary>
    public float GetDeformationRatio()
    {
        return Mathf.Clamp01(deformation / maxDeformation);
    }
}

/// <summary>
/// 現在のオブジェクト状態を取得
/// </summary>
public ObjectState GetCurrentState()
{
    ObjectState state = new ObjectState();
    
    // 変形情報
    state.deformation = currentDeformation;
    state.maxDeformation = maxDeformation;
    state.originalScale = originalScale;
    state.currentScale = transform.localScale;
    
    // 力の情報
    state.appliedForce = appliedForce;
    state.contactCount = contactPoints.Count;
    
    // 材質情報
    state.materialType = materialType;
    state.softness = softness;
    state.compressionResistance = compressionResistance;
    state.breakingForce = breakingForce;
    
    // 状態情報
    state.isBroken = isBroken;
    state.isBeingGrasped = isBeingGrasped;
    state.isDeforming = Mathf.Abs(currentDeformation) > 0.001f;
    
    // タイミング情報
    if (contactPoints.Count > 0)
    {
        state.lastContactTime = contactPoints[contactPoints.Count - 1].timestamp;
    }
    
    return state;
}

/// <summary>
/// オブジェクトの状態をリセット
/// </summary>
public void ResetObject()
{
    currentDeformation = 0f;
    appliedForce = 0f;
    isBeingGrasped = false;
    isBroken = false;
    
    // スケールを元に戻す
    transform.localScale = originalScale;
    
    // 色を元に戻す
    if (objectRenderer != null && objectRenderer.material != null)
    {
        objectRenderer.material.color = originalColor;
    }
    
    // 接触点をクリア
    contactPoints.Clear();
    
    if (enableDebugLogs)
    {
        Debug.Log("DeformableTarget reset to original state");
    }
}

/// <summary>
/// 状態の詳細情報を文字列で取得
/// </summary>
public string GetStateInfo()
{
    var state = GetCurrentState();
    return $"Deformation: {state.deformation:F3}, Force: {state.appliedForce:F1}N, " +
           $"Health: {state.GetHealthScore():F2}, Material: {state.materialType}, " +
           $"Broken: {state.isBroken}, Grasped: {state.isBeingGrasped}";
}

/// <summary>
/// 変形の進行度を取得（アニメーション用）
/// </summary>
public float GetDeformationProgress()
{
    return Mathf.Clamp01(currentDeformation / maxDeformation);
}

/// <summary>
/// 材質プリセットを動的に変更
/// </summary>
public void ChangeMaterialType(MaterialType newMaterialType)
{
    materialType = newMaterialType;
    ApplyMaterialPreset(newMaterialType);
    
    if (enableDebugLogs)
    {
        Debug.Log($"Material type changed to: {newMaterialType}");
    }
}

/// <summary>
/// 破損状態を強制的に設定
/// </summary>
public void SetBrokenState(bool broken)
{
    isBroken = broken;
    
    if (broken)
    {
        BreakObject();
    }
    else
    {
        // 破損状態から復旧
        if (objectRenderer != null && objectRenderer.material != null)
        {
            objectRenderer.material.color = originalColor;
        }
    }
}

/// <summary>
/// 接触点の詳細情報を取得
/// </summary>
public ContactInfo[] GetContactPoints()
{
    return contactPoints.ToArray();
}

/// <summary>
/// 最新の接触情報を取得
/// </summary>
public ContactInfo? GetLatestContact()
{
    if (contactPoints.Count == 0) return null;
    return contactPoints[contactPoints.Count - 1];
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

    // DeformableTarget.cs の末尾（最後の } の前）に以下を追加してください

    /// <summary>
    /// 状態の詳細情報を文字列で取得（既存メソッドと名前が重複しないように）
    /// </summary>
    public string GetDetailedStateInfo()
    {
        var state = GetCurrentState();
        return $"Deformation: {state.deformation:F3}, Force: {state.appliedForce:F1}N, " +
               $"Material: {state.materialType}, " +
               $"Broken: {state.isBroken}, Grasped: {state.isBeingGrasped}";
    }

    /// <summary>
    /// 変形の進行度を取得（アニメーション用）
    /// </summary>
    public float GetDeformationProgress()
    {
        return Mathf.Clamp01(currentDeformation / maxDeformation);
    }

    /// <summary>
    /// 材質プリセットを動的に変更
    /// </summary>
    public void ChangeMaterialType(MaterialType newMaterialType)
    {
        materialType = newMaterialType;
        ApplyMaterialPreset(newMaterialType);
        
        if (enableDebugLogs)
        {
            Debug.Log($"Material type changed to: {newMaterialType}");
        }
    }

    /// <summary>
    /// 接触点の詳細情報を取得（既存メソッドと名前が重複しないように）
    /// </summary>
    public ContactInfo[] GetAllContactPoints()
    {
        return contactPoints.ToArray();
    }

    /// <summary>
    /// 最新の接触情報を取得（既存メソッドと名前が重複しないように）
    /// </summary>
    public ContactInfo? GetLatestContactInfo()
    {
        if (contactPoints.Count == 0) return null;
        return contactPoints[contactPoints.Count - 1];
    }

    /// <summary>
    /// 方向を考慮した力適用メソッド（オーバーロード）
    /// </summary>
    public void ApplyGripperForceWithDirection(float force, Vector3 contactPosition, Vector3 contactNormal)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        
        // 既存のContactInfoを使用
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
        
        // 方向を考慮した変形の計算
        CalculateDirectionalDeformation(force, contactNormal);
        
        if (enableDebugLogs)
            Debug.Log($"Directional force applied: {force:F2}N at {contactPosition}, Normal: {contactNormal}, Deformation: {currentDeformation:F3}");
    }

    /// <summary>
    /// 方向を考慮した変形計算（新規メソッド）
    /// </summary>
    private void CalculateDirectionalDeformation(float force, Vector3 contactNormal)
    {
        // 材質の柔軟性に基づく変形計算
        float baseDeformation = Mathf.Clamp01(force / compressionResistance) * softness;
        
        // 接触法線の方向を考慮した変形係数
        // Y軸（上下）からの圧縮が最も効果的
        float directionFactor = Mathf.Abs(Vector3.Dot(contactNormal, Vector3.up));
        directionFactor = Mathf.Clamp(directionFactor, 0.3f, 1f); // 最小30%の効果は保持
        
        float targetDeformation = baseDeformation * directionFactor;
        targetDeformation = Mathf.Clamp(targetDeformation, 0f, maxDeformation);
        
        // 滑らかな変形
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, 
                                      deformationSpeed * Time.deltaTime);
    }

    /// <summary>
    /// 破損状態を強制的に設定
    /// </summary>
    public void SetBrokenState(bool broken)
    {
        isBroken = broken;
        
        if (broken)
        {
            BreakObject();
        }
        else
        {
            // 破損状態から復旧
            if (objectRenderer != null && objectRenderer.material != null)
            {
                objectRenderer.material.color = originalColor;
            }
        }
    }

// また、既存のContactInfoが拡張できない場合のため、以下の新しい構造体を追加
/// <summary>
/// 法線情報付きの接触情報（ContactInfoの拡張版）
/// </summary>
[System.Serializable]
public struct ExtendedContactInfo
{
    public Vector3 position;
    public Vector3 normal;      // 新規追加：法線
    public float force;
    public float timestamp;
    
    /// <summary>
    /// 既存のContactInfoから変換
    /// </summary>
    public static ExtendedContactInfo FromContactInfo(ContactInfo original, Vector3 normal)
    {
        return new ExtendedContactInfo
        {
            position = original.position,
            normal = normal,
            force = original.force,
            timestamp = original.timestamp
        };
    }
}

}