using UnityEngine;

/// <summary>
/// 統合されたアルミ缶変形システム
/// GripperTargetInterface からの力を受け取り、モデルスワップを実行
/// </summary>
public class IntegratedAluminumCan : MonoBehaviour
{
    [Header("3Dモデル設定")]
    [Tooltip("正常なアルミ缶のGameObject")]
    public GameObject normalCanModel;

        [Header("制御システム連携")]
    public SimpleGripperForceController simpleGripperController; // ★ publicに変更してInspectorで設定可能に
    
    [Tooltip("つぶれたアルミ缶のGameObject")]
    public GameObject crushedCanModel;
    
    [Header("変形設定")]
    [Range(1f, 100f)]
    [Tooltip("変形が発生する力の閾値（N）")]
    public float deformationThreshold = 15f;
    
    [Range(0f, 2f)]
    [Tooltip("変形速度")]
    public float deformationSpeed = 1f;
    
    [Header("音響効果")]
    [Tooltip("つぶれる音のAudioClip")]
    public AudioClip crushSound;
    
    [Tooltip("AudioSource")]
    public AudioSource audioSource;
    
    [Header("物理設定")]
    [Range(0.01f, 0.1f)]
    [Tooltip("アルミ缶の重さ（kg）")]
    public float canMass = 0.015f; // 15g
    
    [Header("デバッグ設定")]
    public bool showDebugInfo = true;
    public bool showForceGizmos = true;
    public bool enableCrushAnimation = true;
    

    
    // 内部状態
    private bool isCrushed = false;
    private float appliedForce = 0f;
    private Vector3 lastContactPoint = Vector3.zero;
    private Vector3 lastContactNormal = Vector3.up;
    private Rigidbody canRigidbody;
    private Vector3 originalScale;
    
    // プロパティ（BasicTypes.csとの互換性用）
    public bool IsBroken => isCrushed;
    public float CurrentDeformation => isCrushed ? 1f : 0f; // 蓄積力を使わない
    public MaterialType MaterialType => MaterialType.Metal;
    public float Softness => 0.1f; // 硬い材質

    /// <summary>
    /// 初期化処理
    /// </summary>
    void Start()
    {
        originalScale = transform.localScale;
        
        // コンポーネント初期化
        InitializeComponents();
        SetupInitialState();
        SetupAntiSlipPhysics();
        
        // SimpleGripperControllerを探す（publicフィールドが設定されていない場合）
        if (simpleGripperController == null)
        {
            simpleGripperController = FindObjectOfType<SimpleGripperForceController>();
        }
        
        if (simpleGripperController == null)
        {
            Debug.LogWarning("SimpleGripperForceController が見つかりません。固定閾値を使用します。");
        }
        else
        {
            Debug.Log($"SimpleGripperForceController 見つかりました。baseGripForce: {simpleGripperController.baseGripForce:F2}N");
        }
        
        Debug.Log("IntegratedAluminumCan 初期化完了");
    }
    
    /// <summary>
    /// 蓄積力システムを廃止したUpdateメソッド
    /// </summary>
    void Update()
    {
        // アニメーション更新のみ保持
        if (isCrushed && enableCrushAnimation)
        {
            UpdateCrushAnimation();
        }
    }

    private void SetupAntiSlipPhysics()
    {
        Collider canCollider = GetComponent<Collider>();
        if (canCollider != null)
        {
            // 高摩擦の物理マテリアルを作成
            PhysicMaterial highFrictionMaterial = new PhysicMaterial("HighFriction");
            highFrictionMaterial.staticFriction = 1.0f;    // 最大静止摩擦
            highFrictionMaterial.dynamicFriction = 0.8f;   // 高い動摩擦
            highFrictionMaterial.bounciness = 0.0f;        // 反発なし
            highFrictionMaterial.frictionCombine = PhysicMaterialCombine.Maximum;
            highFrictionMaterial.bounceCombine = PhysicMaterialCombine.Minimum;
            
            canCollider.material = highFrictionMaterial;
            
            // Rigidbodyの設定も調整
            Rigidbody rb = GetComponent<Rigidbody>();
            if (rb != null)
            {
                rb.drag = 2.0f;        // 空気抵抗を上げる
                rb.angularDrag = 5.0f; // 回転抵抗を上げる
            }
            
            Debug.Log("✅ 滑り防止物理設定完了");
        }
    }
    
    /// <summary>
    /// 初期化処理
    /// </summary>
    void InitializeComponents()
    {
        // Rigidbodyの設定
        canRigidbody = GetComponent<Rigidbody>();
        if (canRigidbody == null)
        {
            canRigidbody = gameObject.AddComponent<Rigidbody>();
        }
        
        canRigidbody.mass = canMass;
        canRigidbody.drag = 0.1f;
        canRigidbody.angularDrag = 0.05f;
        
        // AudioSourceの設定
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.spatialBlend = 1.0f; // 3D sound
            audioSource.volume = 0.7f;
        }
        
        Debug.Log("IntegratedAluminumCan initialized");
    }
    
    /// <summary>
    /// 初期状態の設定
    /// </summary>
    void SetupInitialState()
    {
        if (normalCanModel != null)
        {
            normalCanModel.SetActive(true);
        }
        
        if (crushedCanModel != null)
        {
            crushedCanModel.SetActive(false);
        }
        
        isCrushed = false;
        appliedForce = 0f;
    }
    
    /// <summary>
    /// IGrippableObject インターフェースの実装
    /// グリッパーからの力を受け取る - 修正版
    /// </summary>
    public void ApplyGripperForceWithDirection(float force, Vector3 contactPoint, Vector3 contactNormal)
    {
        if (isCrushed) return;
        
        appliedForce = force;
        lastContactPoint = contactPoint;
        lastContactNormal = contactNormal;
        
        if (showDebugInfo && Time.frameCount % 30 == 0) // 30フレームごとにログ
        {
            Debug.Log($"アルミ缶に力適用: {force:F2}N, 変形閾値計算中...");
        }
        
        // ★ 新しい変形判定: 現在の力が閾値を超えた場合のみ変形
        CheckForceThresholdDirect(force);
    }
    
    /// <summary>
    /// 直接的な力による変形判定 - 修正版
    /// </summary>
    void CheckForceThresholdDirect(float currentForce)
    {
        if (isCrushed) return;
        
        // SimpleGripperControllerのbaseGripForceと比較
        if (simpleGripperController != null)
        {
            float baseForce = simpleGripperController.baseGripForce;
            
            // baseGripForceの1.5倍を超えた場合のみ変形開始
            // 例：baseGripForce=10Nなら15N以上で変形
            float actualThreshold = baseForce * 1.5f;
            
            if (currentForce > actualThreshold)
            {
                if (!isCrushed)
                {
                    Debug.Log($"🔥 アルミ缶変形開始！現在力: {currentForce:F2}N > 閾値: {actualThreshold:F2}N (baseGripForce: {baseForce:F2}N × 1.5)");
                    StartCrushAnimation();
                    isCrushed = true;
                }
            }
        }
        else
        {
            // フォールバック：SimpleGripperControllerが見つからない場合は固定閾値
            if (currentForce > deformationThreshold)
            {
                if (!isCrushed)
                {
                    Debug.Log($"🔥 アルミ缶変形開始！現在力: {currentForce:F2}N > 固定閾値: {deformationThreshold:F2}N");
                    StartCrushAnimation();
                    isCrushed = true;
                }
            }
        }
    }
    
    /// <summary>
    /// 後方互換性のための力適用メソッド
    /// </summary>
    public void ApplyGripperForce(float force, Vector3 contactPoint)
    {
        ApplyGripperForceWithDirection(force, contactPoint, Vector3.up);
    }
    
    /// <summary>
    /// つぶれアニメーション開始
    /// </summary>
    void StartCrushAnimation()
    {
        // モデルの入れ替え
        SwapModels();
        
        // 音効果の再生
        PlayCrushSound();
        
        // 物理特性の調整
        AdjustPhysicsProperties();
    }
    
    /// <summary>
    /// つぶれアニメーションの更新
    /// </summary>
    void UpdateCrushAnimation()
    {
        // 必要に応じてアニメーション処理を追加
    }
    
    /// <summary>
    /// モデルの入れ替え処理
    /// </summary>
    void SwapModels()
    {
        if (normalCanModel != null)
        {
            normalCanModel.SetActive(false);
        }
        
        if (crushedCanModel != null)
        {
            crushedCanModel.SetActive(true);
        }
        
        Debug.Log("✅ モデルを正常な缶からつぶれた缶に切り替えました");
    }
    
    /// <summary>
    /// つぶれる音の再生
    /// </summary>
    void PlayCrushSound()
    {
        if (crushSound != null && audioSource != null)
        {
            audioSource.PlayOneShot(crushSound);
        }
    }
    
    /// <summary>
    /// つぶれた後の物理特性調整
    /// </summary>
    void AdjustPhysicsProperties()
    {
        if (canRigidbody != null)
        {
            // つぶれた缶は少し軽くなり、抵抗が増加
            canRigidbody.mass *= 0.9f;
            canRigidbody.drag *= 1.2f;
        }
    }
    
    /// <summary>
    /// 現在の状態を取得
    /// </summary>
    public ObjectState GetCurrentState()
    {
        return new ObjectState
        {
            appliedForce = this.appliedForce,
            deformation = CurrentDeformation,
            isBroken = this.isCrushed,
            isBeingGrasped = appliedForce > 0f,
            materialType = (int)MaterialType.Metal,
            softness = this.Softness
        };
    }
    
    /// <summary>
    /// デバッグ用GUI表示
    /// </summary>
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUIStyle style = new GUIStyle();
        style.normal.textColor = Color.white;
        style.fontSize = 12;
        
        GUI.Label(new Rect(10, 10, 300, 20), $"缶の状態: {(isCrushed ? "つぶれた" : "正常")}", style);
        GUI.Label(new Rect(10, 30, 300, 20), $"現在の力: {appliedForce:F2}N", style);
        
        if (simpleGripperController != null)
        {
            float baseForce = simpleGripperController.baseGripForce;
            float threshold = baseForce * 1.5f;
            GUI.Label(new Rect(10, 50, 300, 20), $"BaseGripForce: {baseForce:F2}N", style);
            GUI.Label(new Rect(10, 70, 300, 20), $"変形閾値: {threshold:F2}N", style);
            
            // 進行状況バー
            float progress = appliedForce / threshold;
            GUI.Box(new Rect(10, 90, 200, 20), "");
            if (progress > 0)
            {
                GUI.Box(new Rect(10, 90, 200 * Mathf.Min(progress, 1f), 20), "");
            }
            GUI.Label(new Rect(10, 90, 200, 20), $"力レベル: {(progress * 100):F1}%", style);
        }
        else
        {
            GUI.Label(new Rect(10, 50, 300, 20), $"固定閾値: {deformationThreshold:F2}N", style);
        }
    }
    
    /// <summary>
    /// 缶を元の状態にリセット（テスト用）
    /// </summary>
    [ContextMenu("Reset Can")]
    public void ResetCan()
    {
        SetupInitialState();
        
        // 物理特性をリセット
        if (canRigidbody != null)
        {
            canRigidbody.mass = canMass;
            canRigidbody.drag = 0.1f;
            canRigidbody.angularDrag = 0.05f;
        }
        
        Debug.Log("🔄 アルミ缶を初期状態にリセットしました");
    }
    
    /// <summary>
    /// Gizmoの描画（エディタ用）
    /// </summary>
    void OnDrawGizmos()
    {
        if (!showForceGizmos) return;
        
        // 衝突点の可視化
        if (lastContactPoint != Vector3.zero)
        {
            Gizmos.color = isCrushed ? Color.red : Color.yellow;
            Gizmos.DrawWireSphere(lastContactPoint, 0.02f);
        }
        
        // 力の可視化
        if (appliedForce > 0f)
        {
            Gizmos.color = isCrushed ? Color.red : Color.green;
            Gizmos.DrawRay(transform.position, lastContactNormal * (appliedForce * 0.01f));
        }
    }
}

// BasicTypes.csとの互換性のためのenum定義
public enum MaterialType
{
    Soft,
    Medium,
    Hard,
    Metal,
    Fragile
}