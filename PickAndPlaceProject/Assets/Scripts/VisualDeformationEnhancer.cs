using UnityEngine;

/// <summary>
/// 変形効果をより視覚的に分かりやすくするための拡張コンポーネント
/// </summary>
public class VisualDeformationEnhancer : MonoBehaviour
{
    [Header("変形効果の強化")]
    [Range(1f, 5f)]
    public float deformationMultiplier = 2f;  // 変形効果を倍増
    
    [Header("色変化効果")]
    public bool enableColorChange = true;
    public Color normalColor = Color.white;
    public Color compressedColor = Color.red;
    
    [Header("パーティクル効果")]
    public bool enableParticleEffects = true;
    public ParticleSystem compressionParticles;
    
    [Header("音響効果")]
    public bool enableSoundEffects = true;
    public AudioSource audioSource;
    public AudioClip compressionSound;
    
    private DeformableTarget target;
    private Renderer objectRenderer;
    private Vector3 originalScale;
    private Color originalColor;
    private float lastDeformation = 0f;
    
    void Start()
    {
        target = GetComponent<DeformableTarget>();
        objectRenderer = GetComponent<Renderer>();
        originalScale = transform.localScale;
        
        if (objectRenderer != null)
        {
            originalColor = objectRenderer.material.color;
        }
        
        // パーティクルシステムの自動作成
        if (enableParticleEffects && compressionParticles == null)
        {
            CreateCompressionParticles();
        }
        
        // オーディオソースの自動作成
        if (enableSoundEffects && audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.volume = 0.3f;
            audioSource.pitch = 1.2f;
        }
    }
    
    void Update()
    {
        if (target == null) return;
        
        var state = target.GetCurrentState();
        
        // 強化された視覚変形
        UpdateEnhancedDeformation(state.deformation);
        
        // 色変化効果
        if (enableColorChange)
        {
            UpdateColorEffect(state.deformation);
        }
        
        // パーティクル効果
        if (enableParticleEffects && compressionParticles != null)
        {
            UpdateParticleEffect(state.deformation, state.appliedForce);
        }
        
        // 音響効果
        if (enableSoundEffects)
        {
            UpdateSoundEffect(state.deformation);
        }
        
        lastDeformation = state.deformation;
    }
    
    private void UpdateEnhancedDeformation(float deformation)
    {
        // より目立つ変形効果
        float enhancedDeformation = deformation * deformationMultiplier;
        enhancedDeformation = Mathf.Clamp(enhancedDeformation, 0f, 0.8f); // 最大80%まで
        
        // より劇的なスケール変化
        float compressionFactor = 1f - enhancedDeformation;
        Vector3 deformedScale = new Vector3(
            originalScale.x * (1f + enhancedDeformation * 0.5f), // 横に大きく広がる
            originalScale.y * compressionFactor,                 // 縦に圧縮
            originalScale.z * (1f + enhancedDeformation * 0.5f)  // 奥行きに大きく広がる
        );
        
        transform.localScale = Vector3.Lerp(transform.localScale, deformedScale, Time.deltaTime * 5f);
        
        // デバッグ表示
        if (enhancedDeformation > 0.01f)
        {
            Debug.Log($"視覚変形更新: 元変形={deformation:F3}, 強化変形={enhancedDeformation:F3}, スケール={deformedScale}");
        }
    }
    
    private void UpdateColorEffect(float deformation)
    {
        if (objectRenderer == null) return;
        
        // 変形度に応じて色を変化
        Color targetColor = Color.Lerp(normalColor, compressedColor, deformation * 2f);
        objectRenderer.material.color = Color.Lerp(objectRenderer.material.color, targetColor, Time.deltaTime * 3f);
    }
    
    private void UpdateParticleEffect(float deformation, float force)
    {
        if (compressionParticles == null) return;
        
        var emission = compressionParticles.emission;
        
        if (deformation > 0.05f && force > 5f)
        {
            if (!compressionParticles.isPlaying)
            {
                compressionParticles.Play();
            }
            
            // 変形度に応じてパーティクル量を調整
            emission.rateOverTime = deformation * 50f;
        }
        else
        {
            if (compressionParticles.isPlaying)
            {
                compressionParticles.Stop();
            }
        }
    }
    
    private void UpdateSoundEffect(float deformation)
    {
        if (audioSource == null || compressionSound == null) return;
        
        // 変形が急激に増加した時に音を再生
        float deformationIncrease = deformation - lastDeformation;
        if (deformationIncrease > 0.05f && !audioSource.isPlaying)
        {
            audioSource.pitch = 1f + deformation; // 変形度に応じてピッチ変更
            audioSource.PlayOneShot(compressionSound);
        }
    }
    
    private void CreateCompressionParticles()
    {
        GameObject particleObject = new GameObject("CompressionParticles");
        particleObject.transform.SetParent(transform);
        particleObject.transform.localPosition = Vector3.zero;
        
        compressionParticles = particleObject.AddComponent<ParticleSystem>();
        
        var main = compressionParticles.main;
        main.startLifetime = 0.5f;
        main.startSpeed = 2f;
        main.startSize = 0.02f;
        main.startColor = Color.yellow;
        main.maxParticles = 100;
        
        var emission = compressionParticles.emission;
        emission.rateOverTime = 0f; // 手動制御
        
        var shape = compressionParticles.shape;
        shape.shapeType = ParticleSystemShapeType.Sphere;
        shape.radius = 0.1f;
        
        Debug.Log("圧縮パーティクルシステムを作成しました");
    }
    
    /// <summary>
    /// 変形効果をリセット
    /// </summary>
    public void ResetVisualEffects()
    {
        transform.localScale = originalScale;
        
        if (objectRenderer != null)
        {
            objectRenderer.material.color = originalColor;
        }
        
        if (compressionParticles != null && compressionParticles.isPlaying)
        {
            compressionParticles.Stop();
        }
        
        lastDeformation = 0f;
    }
    
    void OnDisable()
    {
        ResetVisualEffects();
    }
}