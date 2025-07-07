using UnityEngine;
using System.Collections.Generic;

/// <summary>
/// 把持力に応じて変形するターゲットオブジェクト
/// </summary>
public class DeformableTarget : MonoBehaviour
{
    [Header("=== 変形パラメータ ===")]
    [SerializeField, Range(0f, 1f), Tooltip("物体の柔軟性（0=硬い、1=柔らかい）")]
    private float softness = 0.5f;
    
    [SerializeField, Range(0.1f, 2f), Tooltip("最大変形度合い")]
    private float maxDeformation = 0.3f;
    
    [SerializeField, Range(1f, 50f), Tooltip("変形に必要な最小力")]
    private float minForceToDeform = 5f;
    
    [SerializeField, Range(10f, 100f), Tooltip("完全変形に必要な力")]
    private float maxForceToDeform = 30f;
    
    [Header("=== 変形タイプ ===")]
    [SerializeField] private DeformationType deformationType = DeformationType.Squeeze;
    
    [Header("=== ビジュアル設定 ===")]
    [SerializeField] private bool enableMaterialChange = true;
    [SerializeField] private Material normalMaterial;
    [SerializeField] private Material deformedMaterial;
    
    [Header("=== デバッグ ===")]
    [SerializeField] private bool showDebugInfo = true;
    
    // 内部変数
    private Vector3 originalScale;
    private Vector3 originalPosition;
    private Quaternion originalRotation;
    private MeshRenderer meshRenderer;
    private Collider objectCollider;
    private Rigidbody objectRigidbody;
    
    // 変形状態
    private float currentDeformation = 0f;
    private float currentForce = 0f;
    private bool isBeingGrasped = false;
    private Vector3 graspPoint;
    private Vector3 graspDirection;
    
    // 物理シミュレーション用
    private List<Vector3> vertexOffsets = new List<Vector3>();
    private MeshFilter meshFilter;
    private Mesh originalMesh;
    private Vector3[] originalVertices;
    private Vector3[] deformedVertices;
    
    public enum DeformationType
    {
        Squeeze,        // 圧縮変形
        Bend,           // 屈曲変形
        Stretch,        // 伸張変形
        Soft            // ソフトボディ風変形
    }
    
    // プロパティ
    public float CurrentDeformation => currentDeformation;
    public float CurrentForce => currentForce;
    public bool IsDeformed => currentDeformation > 0.1f;
    public float Softness => softness;
    
    void Start()
    {
        InitializeDeformableTarget();
    }
    
    void Update()
    {
        UpdateDeformation();
        
        if (showDebugInfo && isBeingGrasped)
        {
            Debug.DrawRay(graspPoint, graspDirection * 0.1f, Color.red);
        }
    }
    
    /// <summary>
    /// 変形可能ターゲットの初期化
    /// </summary>
    private void InitializeDeformableTarget()
    {
        // 元の状態を保存
        originalScale = transform.localScale;
        originalPosition = transform.localPosition;
        originalRotation = transform.localRotation;
        
        // コンポーネント取得
        meshRenderer = GetComponent<MeshRenderer>();
        objectCollider = GetComponent<Collider>();
        objectRigidbody = GetComponent<Rigidbody>();
        meshFilter = GetComponent<MeshFilter>();
        
        // メッシュ変形の準備
        if (meshFilter != null && meshFilter.mesh != null)
        {
            originalMesh = meshFilter.mesh;
            originalVertices = originalMesh.vertices;
            deformedVertices = new Vector3[originalVertices.Length];
            System.Array.Copy(originalVertices, deformedVertices, originalVertices.Length);
            
            // 頂点オフセットの初期化
            for (int i = 0; i < originalVertices.Length; i++)
            {
                vertexOffsets.Add(Vector3.zero);
            }
        }
        
        // マテリアル設定
        if (normalMaterial == null && meshRenderer != null)
        {
            normalMaterial = meshRenderer.material;
        }
        
        if (showDebugInfo)
        {
            Debug.Log($"DeformableTarget初期化完了 - 柔軟性: {softness}, 最大変形: {maxDeformation}");
        }
    }
    
    /// <summary>
    /// 変形処理の更新
    /// </summary>
    private void UpdateDeformation()
    {
        if (!isBeingGrasped)
        {
            // 把持されていない場合は元の形に戻る
            RestoreOriginalShape();
            return;
        }
        
        // 力に基づく変形度合いの計算
        float forceRatio = Mathf.Clamp01((currentForce - minForceToDeform) / (maxForceToDeform - minForceToDeform));
        float targetDeformation = forceRatio * softness;
        
        // 変形の適用
        currentDeformation = Mathf.Lerp(currentDeformation, targetDeformation, Time.deltaTime * 5f);
        
        // 変形タイプに応じた処理
        switch (deformationType)
        {
            case DeformationType.Squeeze:
                ApplySqueezeDeformation();
                break;
            case DeformationType.Bend:
                ApplyBendDeformation();
                break;
            case DeformationType.Stretch:
                ApplyStretchDeformation();
                break;
            case DeformationType.Soft:
                ApplySoftBodyDeformation();
                break;
        }
        
        // マテリアル変更
        UpdateMaterial();
    }
    
    /// <summary>
    /// 圧縮変形の適用
    /// </summary>
    private void ApplySqueezeDeformation()
    {
        Vector3 deformScale = originalScale;
        
        // 把持方向に圧縮
        if (graspDirection != Vector3.zero)
        {
            Vector3 localGraspDir = transform.InverseTransformDirection(graspDirection);
            deformScale -= localGraspDir * currentDeformation * maxDeformation;
        }
        else
        {
            // デフォルトではY軸方向に圧縮
            deformScale.y *= (1f - currentDeformation * maxDeformation);
        }
        
        transform.localScale = Vector3.Lerp(transform.localScale, deformScale, Time.deltaTime * 10f);
    }
    
    /// <summary>
    /// 屈曲変形の適用
    /// </summary>
    private void ApplyBendDeformation()
    {
        float bendAngle = currentDeformation * maxDeformation * 30f; // 最大30度の傾き
        Vector3 bendRotation = originalRotation.eulerAngles;
        bendRotation.z += bendAngle;
        
        transform.localRotation = Quaternion.Lerp(transform.localRotation, 
                                                Quaternion.Euler(bendRotation), 
                                                Time.deltaTime * 5f);
    }
    
    /// <summary>
    /// 伸張変形の適用
    /// </summary>
    private void ApplyStretchDeformation()
    {
        Vector3 stretchScale = originalScale;
        stretchScale.y *= (1f + currentDeformation * maxDeformation);
        stretchScale.x *= (1f - currentDeformation * maxDeformation * 0.5f);
        stretchScale.z *= (1f - currentDeformation * maxDeformation * 0.5f);
        
        transform.localScale = Vector3.Lerp(transform.localScale, stretchScale, Time.deltaTime * 8f);
    }
    
    /// <summary>
    /// ソフトボディ風変形の適用
    /// </summary>
    private void ApplySoftBodyDeformation()
    {
        if (meshFilter == null || originalVertices == null) return;
        
        // 把持点周辺の頂点を変形
        for (int i = 0; i < originalVertices.Length; i++)
        {
            Vector3 worldVertexPos = transform.TransformPoint(originalVertices[i]);
            float distanceToGrasp = Vector3.Distance(worldVertexPos, graspPoint);
            
            if (distanceToGrasp < 0.1f) // 把持点から0.1m以内
            {
                float influence = 1f - (distanceToGrasp / 0.1f);
                Vector3 deformDirection = graspDirection * currentDeformation * maxDeformation * influence;
                vertexOffsets[i] = Vector3.Lerp(vertexOffsets[i], deformDirection, Time.deltaTime * 3f);
                deformedVertices[i] = originalVertices[i] + transform.InverseTransformDirection(vertexOffsets[i]);
            }
            else
            {
                vertexOffsets[i] = Vector3.Lerp(vertexOffsets[i], Vector3.zero, Time.deltaTime * 2f);
                deformedVertices[i] = Vector3.Lerp(deformedVertices[i], originalVertices[i], Time.deltaTime * 2f);
            }
        }
        
        // メッシュ更新
        Mesh mesh = meshFilter.mesh;
        mesh.vertices = deformedVertices;
        mesh.RecalculateNormals();
        mesh.RecalculateBounds();
    }
    
    /// <summary>
    /// 元の形状に復元
    /// </summary>
    private void RestoreOriginalShape()
    {
        currentDeformation = Mathf.Lerp(currentDeformation, 0f, Time.deltaTime * 3f);
        
        if (currentDeformation < 0.01f)
        {
            currentDeformation = 0f;
            transform.localScale = originalScale;
            transform.localRotation = originalRotation;
            
            // ソフトボディ変形のリセット
            if (meshFilter != null && originalVertices != null)
            {
                Mesh mesh = meshFilter.mesh;
                mesh.vertices = originalVertices;
                mesh.RecalculateNormals();
                mesh.RecalculateBounds();
                
                for (int i = 0; i < vertexOffsets.Count; i++)
                {
                    vertexOffsets[i] = Vector3.zero;
                }
            }
        }
        
        UpdateMaterial();
    }
    
    /// <summary>
    /// マテリアルの更新
    /// </summary>
    private void UpdateMaterial()
    {
        if (!enableMaterialChange || meshRenderer == null) return;
        
        if (currentDeformation > 0.1f && deformedMaterial != null)
        {
            meshRenderer.material = deformedMaterial;
        }
        else if (normalMaterial != null)
        {
            meshRenderer.material = normalMaterial;
        }
    }
    
    /// <summary>
    /// 外部からの力の適用（GripperForceControllerから呼び出される）
    /// </summary>
    /// <param name="force">適用される力</param>
    /// <param name="contactPoint">接触点</param>
    /// <param name="forceDirection">力の方向</param>
    public void ApplyGripForce(float force, Vector3 contactPoint, Vector3 forceDirection)
    {
        currentForce = force;
        isBeingGrasped = force > minForceToDeform;
        graspPoint = contactPoint;
        graspDirection = forceDirection.normalized;
        
        if (showDebugInfo && isBeingGrasped)
        {
            Debug.Log($"把持力適用: {force:F2}N, 変形度: {currentDeformation:F3}, 接触点: {contactPoint}");
        }
    }
    
    /// <summary>
    /// 把持の停止
    /// </summary>
    public void StopGrasping()
    {
        isBeingGrasped = false;
        currentForce = 0f;
        graspPoint = Vector3.zero;
        graspDirection = Vector3.zero;
    }
    
    /// <summary>
    /// 柔軟性パラメータの設定
    /// </summary>
    public void SetSoftness(float newSoftness)
    {
        softness = Mathf.Clamp01(newSoftness);
    }
    
    /// <summary>
    /// 変形タイプの設定
    /// </summary>
    public void SetDeformationType(DeformationType type)
    {
        deformationType = type;
        RestoreOriginalShape();
    }
    
    void OnDrawGizmos()
    {
        if (showDebugInfo && isBeingGrasped)
        {
            Gizmos.color = Color.yellow;
            Gizmos.DrawWireSphere(graspPoint, 0.02f);
            
            Gizmos.color = Color.red;
            Gizmos.DrawRay(graspPoint, graspDirection * 0.1f);
            
            // 変形度合いを表示
            Gizmos.color = Color.Lerp(Color.green, Color.red, currentDeformation);
            Gizmos.DrawWireCube(transform.position, transform.localScale * 1.1f);
        }
    }
}