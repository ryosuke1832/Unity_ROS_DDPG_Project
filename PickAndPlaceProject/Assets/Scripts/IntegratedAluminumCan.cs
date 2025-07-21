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
    
    // 内部状態
    private bool isCrushed = false;
    private float appliedForce = 0f;
    private float accumulatedForce = 0f;
    private Vector3 lastContactPoint = Vector3.zero;
    private Vector3 lastContactNormal = Vector3.up;
    private Rigidbody canRigidbody;
    
    // プロパティ（BasicTypes.csとの互換性用）
    public bool IsBroken => isCrushed;
    public float CurrentDeformation => isCrushed ? 1f : (accumulatedForce / deformationThreshold);
    public MaterialType MaterialType => MaterialType.Metal;
    public float Softness => 0.1f; // 硬い材質
    
    void Start()
    {
        InitializeComponents();
        SetupInitialState();
    }
    
    void Update()
    {
        UpdateForceDecay();
        CheckForceThreshold();
        
        if (showDebugInfo)
        {
            DisplayDebugInfo();
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
        accumulatedForce = 0f;
    }
    
    /// <summary>
    /// IGrippableObject インターフェースの実装
    /// グリッパーからの力を受け取る
    /// </summary>
    public void ApplyGripperForceWithDirection(float force, Vector3 contactPoint, Vector3 contactNormal)
    {
        if (isCrushed) return;
        
        appliedForce = force;
        lastContactPoint = contactPoint;
        lastContactNormal = contactNormal;
        
        // 力を蓄積（連続的な圧力の効果）
        accumulatedForce += force * Time.deltaTime * deformationSpeed;
        accumulatedForce = Mathf.Min(accumulatedForce, deformationThreshold * 2f);
        
        if (showDebugInfo && Time.frameCount % 30 == 0) // 30フレームごとにログ
        {
            Debug.Log($"アルミ缶に力適用: {force:F2}N, 蓄積力: {accumulatedForce:F2}N, 閾値: {deformationThreshold:F2}N");
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
    /// 力の減衰処理
    /// </summary>
    void UpdateForceDecay()
    {
        if (appliedForce <= 0f && accumulatedForce > 0f)
        {
            // 力が加わっていない時は蓄積力を徐々に減らす
            accumulatedForce -= Time.deltaTime * deformationSpeed * 0.5f;
            accumulatedForce = Mathf.Max(0f, accumulatedForce);
        }
    }
    
    /// <summary>
    /// 変形閾値のチェック
    /// </summary>
    void CheckForceThreshold()
    {
        if (isCrushed) return;
        
        if (accumulatedForce >= deformationThreshold)
        {
            CrushCan();
        }
    }
    
    /// <summary>
    /// アルミ缶をつぶす処理
    /// </summary>
    void CrushCan()
    {
        if (isCrushed) return;
        
        Debug.Log($"🥫 アルミ缶がつぶれました！ 蓄積力: {accumulatedForce:F2}N");
        
        // モデルの入れ替え
        SwapModels();
        
        // 音効果の再生
        PlayCrushSound();
        
        // 物理特性の調整
        AdjustPhysicsProperties();
        
        isCrushed = true;
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
    /// デバッグ情報の表示
    /// </summary>
    void DisplayDebugInfo()
    {
        if (Time.frameCount % 60 == 0) // 1秒ごとに更新
        {
            string status = isCrushed ? "つぶれた" : "正常";
            Debug.Log($"缶の状態: {status}, 現在の力: {appliedForce:F2}N, 蓄積力: {accumulatedForce:F2}N");
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
            materialType = (int)MaterialType.Metal, // 明示的にキャスト
            softness = this.Softness
        };
    }
    
    /// <summary>
    /// 蓄積力を取得（デバッグ用）
    /// </summary>
    public float GetAccumulatedForce()
    {
        return accumulatedForce;
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
    /// 強制的に缶をつぶす（テスト用）
    /// </summary>
    [ContextMenu("Force Crush")]
    public void ForceCrush()
    {
        accumulatedForce = deformationThreshold + 1f;
        CrushCan();
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
            Gizmos.color = accumulatedForce >= deformationThreshold ? Color.red : Color.green;
            Gizmos.DrawRay(transform.position, lastContactNormal * (appliedForce * 0.01f));
        }
        
        // 蓄積力のバー表示
        float barHeight = (accumulatedForce / deformationThreshold) * 0.1f;
        Gizmos.color = Color.blue;
        Gizmos.DrawCube(transform.position + Vector3.up * 0.15f, new Vector3(0.02f, barHeight, 0.02f));
    }
    
    /// <summary>
    /// インスペクター上での情報表示
    /// </summary>
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        GUIStyle style = new GUIStyle();
        style.fontSize = 14;
        style.normal.textColor = Color.white;
        
        GUI.Label(new Rect(10, 10, 300, 20), $"缶の状態: {(isCrushed ? "つぶれた" : "正常")}", style);
        GUI.Label(new Rect(10, 30, 300, 20), $"現在の力: {appliedForce:F2}N", style);
        GUI.Label(new Rect(10, 50, 300, 20), $"蓄積力: {accumulatedForce:F2}N", style);
        GUI.Label(new Rect(10, 70, 300, 20), $"変形閾値: {deformationThreshold:F2}N", style);
        
        // 進行状況バー
        float progress = accumulatedForce / deformationThreshold;
        GUI.Box(new Rect(10, 90, 200, 20), "");
        GUI.Box(new Rect(10, 90, 200 * progress, 20), "");
        GUI.Label(new Rect(10, 90, 200, 20), $"変形進行: {(progress * 100):F1}%", style);
    }
}

// 既存のBasicTypes.csのObjectStateを使用するため、重複定義を削除

public enum MaterialType
{
    Soft,
    Medium,
    Hard,
    Metal,
    Fragile
}