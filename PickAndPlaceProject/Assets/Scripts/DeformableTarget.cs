using UnityEngine;

/// <summary>
/// Target物体の変形制御コンポーネント
/// 把持力に応じて物体を変形させる
/// </summary>
public class DeformableTarget : MonoBehaviour
{
    [Header("=== 変形パラメーター ===")]
    [SerializeField, Range(0.1f, 50f), Tooltip("変形が始まる最小力(N)")]
    private float deformationThreshold = 8f;
    
    [SerializeField, Range(0.1f, 100f), Tooltip("物体が破損する最大力(N)")]
    private float destructionThreshold = 30f;
    
    [SerializeField, Range(0f, 1f), Tooltip("物体の硬さ（0=柔らかい, 1=硬い）")]
    private float stiffness = 0.7f;
    
    [SerializeField, Range(0f, 1f), Tooltip("変形回復力（0=戻らない, 1=すぐ戻る）")]
    private float elasticity = 0.8f;
    
    [Header("=== 変形設定 ===")]
    [SerializeField, Range(0.5f, 1f), Tooltip("最小スケール倍率")]
    private float minScale = 0.7f;
    
    [SerializeField, Range(0.1f, 5f), Tooltip("変形速度")]
    private float deformationSpeed = 2f;
    
    [SerializeField, Range(0.1f, 5f), Tooltip("回復速度")]
    private float recoverySpeed = 1f;
    
    [Header("=== 視覚効果 ===")]
    [SerializeField, Tooltip("変形時の色変化を有効にする")]
    private bool enableColorChange = true;
    
    [SerializeField, Tooltip("通常時の色")]
    private Color normalColor = Color.white;
    
    [SerializeField, Tooltip("変形時の色")]
    private Color deformedColor = Color.red;
    
    [SerializeField, Tooltip("破損時の色")]
    private Color destroyedColor = Color.black;
    
    [Header("=== 音響効果 ===")]
    [SerializeField, Tooltip("変形開始音")]
    private AudioClip deformationSound;
    
    [SerializeField, Tooltip("破損音")]
    private AudioClip destructionSound;
    
    [Header("=== デバッグ ===")]
    [SerializeField, Tooltip("デバッグ情報を表示")]
    private bool showDebugInfo = true;
    
    // 内部変数
    private Vector3 originalScale;
    private Vector3 currentTargetScale;
    private float currentGripForce;
    private float deformationLevel; // 0-1の変形度合い
    private bool isDestroyed = false;
    private bool isBeingGrasped = false;
    
    // コンポーネント参照
    private Renderer targetRenderer;
    private AudioSource audioSource;
    private Material targetMaterial;
    private Color originalColor;
    
    // 物理設定
    private Rigidbody targetRigidbody;
    private Collider targetCollider;
    
    private void Start()
    {
        InitializeComponents();
        originalScale = transform.localScale;
        currentTargetScale = originalScale;
        
        if (targetRenderer != null && targetRenderer.material != null)
        {
            targetMaterial = targetRenderer.material;
            originalColor = targetMaterial.color;
        }
    }
    
    private void InitializeComponents()
    {
        targetRenderer = GetComponent<Renderer>();
        audioSource = GetComponent<AudioSource>();
        targetRigidbody = GetComponent<Rigidbody>();
        targetCollider = GetComponent<Collider>();
        
        // AudioSourceがない場合は追加
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.volume = 0.5f;
        }
    }
    
    private void Update()
    {
        if (isDestroyed) return;
        
        UpdateDeformation();
        UpdateVisualEffects();
        
        if (showDebugInfo)
        {
            DisplayDebugInfo();
        }
    }
    
    /// <summary>
    /// 把持力を設定する（GripperForceControllerから呼び出される）
    /// </summary>
    public void SetGripForce(float force, bool grasping)
    {
        currentGripForce = force;
        isBeingGrasped = grasping;
        
        if (grasping && force > destructionThreshold)
        {
            DestroyObject();
        }
    }
    
    /// <summary>
    /// 変形処理の更新
    /// </summary>
    private void UpdateDeformation()
    {
        if (isBeingGrasped && currentGripForce > deformationThreshold)
        {
            // 変形レベルの計算
            float forceRatio = (currentGripForce - deformationThreshold) / 
                             (destructionThreshold - deformationThreshold);
            deformationLevel = Mathf.Clamp01(forceRatio);
            
            // 硬さを考慮した変形
            float actualDeformation = deformationLevel * (1f - stiffness);
            
            // 目標スケールの計算
            float scaleMultiplier = Mathf.Lerp(1f, minScale, actualDeformation);
            currentTargetScale = originalScale * scaleMultiplier;
            
            // 変形音の再生
            if (deformationLevel > 0.1f && deformationSound != null && !audioSource.isPlaying)
            {
                audioSource.clip = deformationSound;
                audioSource.pitch = 1f + deformationLevel * 0.5f;
                audioSource.Play();
            }
        }
        else
        {
            // 回復処理
            deformationLevel = Mathf.MoveTowards(deformationLevel, 0f, 
                elasticity * recoverySpeed * Time.deltaTime);
            
            float scaleMultiplier = Mathf.Lerp(1f, minScale, deformationLevel);
            currentTargetScale = originalScale * scaleMultiplier;
        }
        
        // スケールの滑らかな変更
        transform.localScale = Vector3.Lerp(transform.localScale, 
            currentTargetScale, deformationSpeed * Time.deltaTime);
    }
    
    /// <summary>
    /// 視覚効果の更新
    /// </summary>
    private void UpdateVisualEffects()
    {
        if (!enableColorChange || targetMaterial == null) return;
        
        Color targetColor = originalColor;
        
        if (isDestroyed)
        {
            targetColor = destroyedColor;
        }
        else if (deformationLevel > 0.1f)
        {
            targetColor = Color.Lerp(normalColor, deformedColor, deformationLevel);
        }
        else
        {
            targetColor = normalColor;
        }
        
        targetMaterial.color = targetColor;
    }
    
    /// <summary>
    /// 物体の破損処理
    /// </summary>
    private void DestroyObject()
    {
        if (isDestroyed) return;
        
        isDestroyed = true;
        
        // 破損音の再生
        if (destructionSound != null)
        {
            audioSource.clip = destructionSound;
            audioSource.pitch = 1f;
            audioSource.Play();
        }
        
        // 物理的な変化
        if (targetRigidbody != null)
        {
            targetRigidbody.isKinematic = true;
        }
        
        // エフェクト（パーティクルなど）
        CreateDestructionEffect();
        
        Debug.Log($"物体が破損しました！力: {currentGripForce}N");
    }
    
    /// <summary>
    /// 破損エフェクトの生成
    /// </summary>
    private void CreateDestructionEffect()
    {
        // 簡単なパーティクルエフェクト
        GameObject effect = new GameObject("DestructionEffect");
        effect.transform.position = transform.position;
        
        ParticleSystem particles = effect.AddComponent<ParticleSystem>();
        var main = particles.main;
        main.startColor = destroyedColor;
        main.startSize = 0.1f;
        main.startLifetime = 2f;
        main.maxParticles = 50;
        
        var emission = particles.emission;
        emission.rateOverTime = 25f;
        
        // 自動削除
        Destroy(effect, 3f);
    }
    
    /// <summary>
    /// 物体のリセット
    /// </summary>
    public void ResetObject()
    {
        isDestroyed = false;
        isBeingGrasped = false;
        currentGripForce = 0f;
        deformationLevel = 0f;
        transform.localScale = originalScale;
        currentTargetScale = originalScale;
        
        if (targetMaterial != null)
        {
            targetMaterial.color = originalColor;
        }
        
        if (targetRigidbody != null)
        {
            targetRigidbody.isKinematic = false;
        }
    }
    
    /// <summary>
    /// デバッグ情報の表示
    /// </summary>
    private void DisplayDebugInfo()
    {
        if (deformationLevel > 0 || isBeingGrasped)
        {
            Debug.Log($"[DeformableTarget] 力: {currentGripForce:F2}N, 変形度: {deformationLevel:F2}, " +
                     $"把持中: {isBeingGrasped}, 破損: {isDestroyed}");
        }
    }
    
    /// <summary>
    /// 現在の状態を取得
    /// </summary>
    public DeformationState GetCurrentState()
    {
        return new DeformationState
        {
            currentForce = currentGripForce,
            deformationLevel = deformationLevel,
            isDestroyed = isDestroyed,
            isBeingGrasped = isBeingGrasped,
            currentScale = transform.localScale,
            originalScale = originalScale
        };
    }
    
    // Inspector用の設定変更時の処理
    private void OnValidate()
    {
        // パラメータの妥当性チェック
        if (deformationThreshold >= destructionThreshold)
        {
            destructionThreshold = deformationThreshold + 1f;
        }
        
        minScale = Mathf.Clamp(minScale, 0.1f, 1f);
    }
    
    // OnGUI デバッグ表示
    private void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUILayout.BeginArea(new Rect(320, 10, 300, 200));
        GUILayout.Label("=== 変形可能Target ===");
        GUILayout.Label($"把持力: {currentGripForce:F2} N");
        GUILayout.Label($"変形レベル: {deformationLevel:F2}");
        GUILayout.Label($"スケール: {transform.localScale.x:F2}");
        GUILayout.Label($"把持中: {isBeingGrasped}");
        GUILayout.Label($"破損: {isDestroyed}");
        GUILayout.Label($"硬さ: {stiffness:F2}");
        GUILayout.Label($"弾性: {elasticity:F2}");
        
        if (GUILayout.Button("リセット"))
        {
            ResetObject();
        }
        GUILayout.EndArea();
    }
}

/// <summary>
/// 変形状態の構造体
/// </summary>
[System.Serializable]
public struct DeformationState
{
    public float currentForce;
    public float deformationLevel;
    public bool isDestroyed;
    public bool isBeingGrasped;
    public Vector3 currentScale;
    public Vector3 originalScale;
}