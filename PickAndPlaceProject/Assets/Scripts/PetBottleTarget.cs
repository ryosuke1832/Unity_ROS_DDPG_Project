using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// ペットボトル形状の変形可能なターゲット
/// 円筒形状で側面がへこみやすく、リアルなペットボトルの変形を再現
/// </summary>
public class PetBottleTarget : MonoBehaviour
{
    [Header("ペットボトル設定")]
    [Range(0.1f, 2f)]
    public float bottleHeight = 1.5f;
    [Range(0.1f, 1f)]
    public float bottleRadius = 0.25f;
    [Range(0.1f, 0.5f)]
    public float neckHeight = 0.3f;
    [Range(0.05f, 0.2f)]
    public float neckRadius = 0.08f;
    
    [Header("変形特性")]
    [Range(0f, 1f)]
    public float sideWallFlexibility = 0.8f;  // 側面の柔軟性
    [Range(0f, 1f)]
    public float topBottomRigidity = 0.9f;    // 上下の硬さ
    [Range(0f, 1f)]
    public float neckRigidity = 0.95f;        // 首部分の硬さ
    public float maxSideDeformation = 0.4f;    // 最大側面変形
    public float deformationSpeed = 3f;
    
    [Header("物理特性")]
    public float plasticThickness = 0.002f;   // プラスチック厚み（メートル）
    public float compressionResistance = 8f;
    public float breakingForce = 35f;
    public bool enableCrackingSound = true;
    
    [Header("視覚設定")]
    public Color originalColor = new Color(0.8f, 0.9f, 1f, 0.6f); // 透明なブルー
    public Color stressColor = Color.red;
    public Material bottleMaterial;
    
    [Header("デバッグ")]
    public bool enableDebugLogs = false;
    public bool showDeformationGizmos = false;
    
    // 内部状態
    private Vector3 originalScale;
    private Mesh originalMesh;
    private Mesh deformedMesh;
    private Vector3[] originalVertices;
    private Vector3[] deformedVertices;
    private MeshFilter meshFilter;
    private Renderer objectRenderer;
    private AudioSource audioSource;
    
    // 変形状態
    private bool isBroken = false;
    private float currentDeformation = 0f;
    private bool isBeingGrasped = false;
    private float appliedForce = 0f;
    private Vector3 lastContactPoint;
    private Vector3 lastContactNormal;
    private List<DeformationPoint> activeDeformations = new List<DeformationPoint>();
    
    [System.Serializable]
    public struct DeformationPoint
    {
        public Vector3 localPosition;
        public Vector3 contactNormal;
        public float intensity;
        public float radius;
        public float timestamp;
    }
    
    [System.Serializable]
    public struct BottleState
    {
        public float deformation;
        public float appliedForce;
        public bool isBroken;
        public bool isBeingGrasped;
        public int activeDeformationCount;
        public float sideFlexibility;
    }
    
    void Start()
    {
        InitializeBottleStructure();
        SetupMeshDeformation();
        SetupMaterials();
        SetupAudio();
        
        Debug.Log("PetBottleTarget initialized");
    }
    
    void Update()
    {
        CleanupOldDeformations();
        UpdateMeshDeformation();
        UpdateVisualFeedback();
        UpdatePhysicalProperties();
    }
    
    private void InitializeBottleStructure()
    {
        originalScale = transform.localScale;
        meshFilter = GetComponent<MeshFilter>();
        objectRenderer = GetComponent<Renderer>();
        
        if (meshFilter == null)
        {
            meshFilter = gameObject.AddComponent<MeshFilter>();
        }
        
        // ペットボトル形状のメッシュを生成
        CreateBottleMesh();
    }
    
    private void CreateBottleMesh()
    {
        // 簡単な円筒形ペットボトルメッシュを生成
        int segments = 16; // 円周分割数
        int heightSegments = 20; // 高さ分割数
        
        List<Vector3> vertices = new List<Vector3>();
        List<int> triangles = new List<int>();
        List<Vector2> uvs = new List<Vector2>();
        
        // ボトル本体の頂点を生成
        for (int h = 0; h <= heightSegments; h++)
        {
            float normalizedHeight = (float)h / heightSegments;
            float currentY = normalizedHeight * bottleHeight;
            float currentRadius = CalculateRadiusAtHeight(normalizedHeight);
            
            for (int s = 0; s <= segments; s++)
            {
                float angle = (float)s / segments * Mathf.PI * 2f;
                float x = Mathf.Cos(angle) * currentRadius;
                float z = Mathf.Sin(angle) * currentRadius;
                
                vertices.Add(new Vector3(x, currentY, z));
                uvs.Add(new Vector2((float)s / segments, normalizedHeight));
            }
        }
        
        // 三角形を生成
        for (int h = 0; h < heightSegments; h++)
        {
            for (int s = 0; s < segments; s++)
            {
                int current = h * (segments + 1) + s;
                int next = current + 1;
                int below = (h + 1) * (segments + 1) + s;
                int belowNext = below + 1;
                
                // 2つの三角形で四角形を作成
                triangles.Add(current);
                triangles.Add(below);
                triangles.Add(next);
                
                triangles.Add(next);
                triangles.Add(below);
                triangles.Add(belowNext);
            }
        }
        
        // メッシュを作成
        originalMesh = new Mesh();
        originalMesh.vertices = vertices.ToArray();
        originalMesh.triangles = triangles.ToArray();
        originalMesh.uv = uvs.ToArray();
        originalMesh.RecalculateNormals();
        originalMesh.RecalculateBounds();
        
        meshFilter.mesh = originalMesh;
        originalVertices = originalMesh.vertices;
        
        // 変形用メッシュのコピーを作成
        deformedMesh = Instantiate(originalMesh);
        deformedVertices = new Vector3[originalVertices.Length];
        System.Array.Copy(originalVertices, deformedVertices, originalVertices.Length);
    }
    
    private float CalculateRadiusAtHeight(float normalizedHeight)
    {
        if (normalizedHeight > (bottleHeight - neckHeight) / bottleHeight)
        {
            // 首部分
            float neckProgress = (normalizedHeight - (bottleHeight - neckHeight) / bottleHeight) / (neckHeight / bottleHeight);
            return Mathf.Lerp(bottleRadius, neckRadius, neckProgress);
        }
        else
        {
            // ボトル本体
            return bottleRadius;
        }
    }
    
    private void SetupMeshDeformation()
    {
        if (originalMesh == null)
        {
            Debug.LogError("Original mesh not found!");
            return;
        }
    }
    
    private void SetupMaterials()
    {
        if (objectRenderer != null)
        {
            if (bottleMaterial != null)
            {
                objectRenderer.material = bottleMaterial;
            }
            objectRenderer.material.color = originalColor;
        }
    }
    
    private void SetupAudio()
    {
        audioSource = GetComponent<AudioSource>();
        if (audioSource == null && enableCrackingSound)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.volume = 0.5f;
            audioSource.spatialBlend = 1f; // 3D音声
        }
    }
    
    public void ApplyGripperForce(float force, Vector3 contactPosition)
    {
        ApplyGripperForceWithDirection(force, contactPosition, Vector3.zero);
    }
    
    public void ApplyGripperForceWithDirection(float force, Vector3 contactPosition, Vector3 contactNormal)
    {
        if (isBroken) return;
        
        appliedForce = force;
        isBeingGrasped = force > 0.1f;
        lastContactPoint = transform.InverseTransformPoint(contactPosition);
        lastContactNormal = transform.InverseTransformDirection(contactNormal);
        
        // 破損チェック
        if (force > breakingForce)
        {
            BreakBottle();
            return;
        }
        
        // 接触点での変形を追加
        if (isBeingGrasped)
        {
            AddDeformationPoint(lastContactPoint, lastContactNormal, force);
        }
        
        if (enableDebugLogs)
            Debug.Log($"Bottle force applied: {force:F2}N at {contactPosition}");
    }
    
    private void AddDeformationPoint(Vector3 localPosition, Vector3 normal, float force)
    {
        // 高さに基づく柔軟性の計算
        float normalizedHeight = localPosition.y / bottleHeight;
        float flexibility = CalculateFlexibilityAtHeight(normalizedHeight);
        
        // 変形強度の計算
        float intensity = Mathf.Clamp01(force / compressionResistance) * flexibility;
        float radius = bottleRadius * 0.3f; // 変形影響半径
        
        DeformationPoint newDeformation = new DeformationPoint
        {
            localPosition = localPosition,
            contactNormal = normal,
            intensity = intensity,
            radius = radius,
            timestamp = Time.time
        };
        
        activeDeformations.Add(newDeformation);
        
        // 音声効果
        PlayDeformationSound(intensity);
    }
    
    private float CalculateFlexibilityAtHeight(float normalizedHeight)
    {
        if (normalizedHeight > (bottleHeight - neckHeight) / bottleHeight)
        {
            // 首部分 - 硬い
            return 1f - neckRigidity;
        }
        else if (normalizedHeight < 0.1f || normalizedHeight > 0.9f)
        {
            // 上下部分 - 比較的硬い
            return 1f - topBottomRigidity;
        }
        else
        {
            // 側面部分 - 柔らかい
            return sideWallFlexibility;
        }
    }
    
    private void UpdateMeshDeformation()
    {
        if (originalVertices == null || deformedVertices == null) return;
        
        // 元の頂点をコピー
        System.Array.Copy(originalVertices, deformedVertices, originalVertices.Length);
        
        // 各変形点の影響を適用
        foreach (var deformation in activeDeformations)
        {
            ApplyDeformationToVertices(deformation);
        }
        
        // 変形が無い場合は元の形に戻る
        if (activeDeformations.Count == 0 && !isBeingGrasped)
        {
            for (int i = 0; i < deformedVertices.Length; i++)
            {
                deformedVertices[i] = Vector3.Lerp(deformedVertices[i], originalVertices[i], 
                                                 deformationSpeed * Time.deltaTime);
            }
        }
        
        // メッシュを更新
        deformedMesh.vertices = deformedVertices;
        deformedMesh.RecalculateNormals();
        meshFilter.mesh = deformedMesh;
    }
    
    private void ApplyDeformationToVertices(DeformationPoint deformation)
    {
        for (int i = 0; i < deformedVertices.Length; i++)
        {
            Vector3 vertex = originalVertices[i];
            float distance = Vector3.Distance(vertex, deformation.localPosition);
            
            if (distance < deformation.radius)
            {
                // 距離に基づく影響の減衰
                float influence = 1f - (distance / deformation.radius);
                influence = Mathf.Pow(influence, 2f); // 二次的減衰
                
                // 変形方向の計算
                Vector3 deformationDirection;
                if (deformation.contactNormal != Vector3.zero)
                {
                    deformationDirection = deformation.contactNormal;
                }
                else
                {
                    // 法線が無い場合は中心向きに変形
                    deformationDirection = (deformation.localPosition - vertex).normalized;
                }
                
                // 変形を適用
                float deformationAmount = deformation.intensity * influence * maxSideDeformation;
                Vector3 deformationVector = deformationDirection * deformationAmount;
                
                deformedVertices[i] = Vector3.Lerp(deformedVertices[i], 
                                                 originalVertices[i] + deformationVector,
                                                 deformationSpeed * Time.deltaTime);
            }
        }
    }
    
    private void UpdateVisualFeedback()
    {
        if (objectRenderer == null) return;
        
        // 変形強度に基づく色変化
        float totalDeformation = 0f;
        foreach (var deformation in activeDeformations)
        {
            totalDeformation += deformation.intensity;
        }
        totalDeformation = Mathf.Clamp01(totalDeformation);
        
        Color currentColor = Color.Lerp(originalColor, stressColor, totalDeformation);
        objectRenderer.material.color = currentColor;
        
        // 破損時の視覚効果
        if (isBroken)
        {
            objectRenderer.material.color = Color.gray;
        }
    }
    
    private void UpdatePhysicalProperties()
    {
        // リアルタイムで物理特性を更新
        currentDeformation = 0f;
        foreach (var deformation in activeDeformations)
        {
            currentDeformation += deformation.intensity;
        }
        currentDeformation = Mathf.Clamp01(currentDeformation);
    }
    
    private void CleanupOldDeformations()
    {
        // 古い変形を除去（1秒以上前のもの）
        activeDeformations.RemoveAll(deformation => 
            Time.time - deformation.timestamp > 1f);
    }
    
    private void PlayDeformationSound(float intensity)
    {
        if (audioSource != null && enableCrackingSound)
        {
            audioSource.pitch = 0.8f + intensity * 0.4f;
            audioSource.volume = intensity * 0.3f;
            // ここでプラスチック変形音のクリップを再生
            // audioSource.PlayOneShot(plasticDeformationClip);
        }
    }
    
    private void BreakBottle()
    {
        isBroken = true;
        Debug.Log($"Bottle broken! Applied force: {appliedForce}N");
        
        // 破損音効果
        if (audioSource != null && enableCrackingSound)
        {
            audioSource.pitch = 0.5f;
            audioSource.volume = 0.8f;
            // audioSource.PlayOneShot(bottleBreakClip);
        }
        
        // 破損エフェクト（パーティクルシステムなど）
        // 将来的にプラスチック片の物理シミュレーションを追加
    }
    
    public BottleState GetCurrentState()
    {
        return new BottleState
        {
            deformation = currentDeformation,
            appliedForce = appliedForce,
            isBroken = isBroken,
            isBeingGrasped = isBeingGrasped,
            activeDeformationCount = activeDeformations.Count,
            sideFlexibility = sideWallFlexibility
        };
    }
    
    public void ResetBottle()
    {
        isBroken = false;
        currentDeformation = 0f;
        appliedForce = 0f;
        isBeingGrasped = false;
        activeDeformations.Clear();
        
        // メッシュを元に戻す
        if (originalVertices != null && deformedVertices != null)
        {
            System.Array.Copy(originalVertices, deformedVertices, originalVertices.Length);
            deformedMesh.vertices = deformedVertices;
            deformedMesh.RecalculateNormals();
            meshFilter.mesh = deformedMesh;
        }
        
        transform.localScale = originalScale;
        
        if (objectRenderer != null)
            objectRenderer.material.color = originalColor;
        
        Debug.Log("Bottle reset");
    }
    
    void OnDrawGizmos()
    {
        if (!showDeformationGizmos) return;
        
        // 変形点を可視化
        Gizmos.color = Color.yellow;
        foreach (var deformation in activeDeformations)
        {
            Vector3 worldPos = transform.TransformPoint(deformation.localPosition);
            Gizmos.DrawWireSphere(worldPos, deformation.radius);
            
            Gizmos.color = Color.red;
            Vector3 worldNormal = transform.TransformDirection(deformation.contactNormal);
            Gizmos.DrawRay(worldPos, worldNormal * 0.1f);
        }
        
        // ボトル形状の概要を表示
        Gizmos.color = Color.blue;
        Gizmos.DrawWireCube(transform.position + Vector3.up * bottleHeight/2, 
                           new Vector3(bottleRadius*2, bottleHeight, bottleRadius*2));
    }
}