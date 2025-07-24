using UnityEngine;
using System.Collections;

/// <summary>
/// 統合されたアルミ缶変形システム（デバッグ版）
/// deformationThresholdの値の変化を追跡
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
    [SerializeField] private float _deformationThreshold = 15f;

    private const float DEFORMATION_DELAY = 0.2f; // 0.2秒の遅延


    
    
    // 🔍 デバッグ用のプロパティ
    public float deformationThreshold 
    { 
        get => _deformationThreshold; 
        set 
        { 
            if (Mathf.Abs(_deformationThreshold - value) > 0.001f)
            {
                Debug.Log($"[デバッグ] deformationThreshold変更: {_deformationThreshold:F2} → {value:F2}");
                
                // スタックトレースも出力（どこから呼び出されたか）
                Debug.Log($"[デバッグ] 呼び出し元スタック:\n{System.Environment.StackTrace}");
            }
            _deformationThreshold = value; 
        } 
    }
    
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
    // public float CurrentDeformation => isCrushed ? 1f : (accumulatedForce / deformationThreshold);
    public MaterialType MaterialType => MaterialType.Metal;
    public float Softness => 0.1f; // 硬い材質

        /// <summary>
    /// 現在の変形進行度を計算（BaseGripForce基準）
    /// </summary>
    public float CurrentDeformation
    {
        get
        {
            var gripController = FindObjectOfType<SimpleGripForceController>();
            if (gripController != null)
            {
                float baseGripForce = gripController.baseGripForce;
                return Mathf.Clamp01(baseGripForce / deformationThreshold);
            }
            
            // フォールバック: 従来の計算
            return Mathf.Clamp01(appliedForce / deformationThreshold);
        }
    }

    /// <summary>
    /// 🔍 デバッグ: 現在の設定値をすべて表示
    /// </summary>
    [ContextMenu("Debug Show All Values")]
    public void DebugShowAllValues()
    {
        Debug.Log("=== アルミ缶デバッグ情報 ===");
        Debug.Log($"deformationThreshold: {deformationThreshold:F2}N");
        Debug.Log($"isCrushed: {isCrushed}");
        
        var controller = FindObjectOfType<SimpleGripForceController>();
        if (controller != null)
        {
            Debug.Log($"BaseGripForce: {controller.baseGripForce:F2}N");
            Debug.Log($"比較結果: {controller.baseGripForce:F2}N vs {deformationThreshold:F2}N");
            Debug.Log($"変形するか?: {(controller.baseGripForce > deformationThreshold ? "はい" : "いいえ")}");
        }
        else
        {
            Debug.LogError("SimpleGripForceControllerが見つかりません！");
        }
        Debug.Log("=========================");
    }
    
    void Start()
    {
        // 🔍 デバッグ: 初期化開始時の値を記録
        Debug.Log($"[デバッグ] Start()開始時 deformationThreshold: {deformationThreshold:F2}N");
        
        InitializeComponents();
        SetupInitialState();
        
        // 🔍 デバッグ: 初期化完了後の値を確認
        Debug.Log($"[デバッグ] 初期化完了後 deformationThreshold: {deformationThreshold:F2}N");
        
        // 🔍 デバッグ: forceControllerが存在するかチェック
        var controller = FindObjectOfType<SimpleGripForceController>();
        if (controller != null)
        {
            Debug.Log($"[デバッグ] 発見したSimpleGripForceController.baseGripForce: {controller.baseGripForce:F2}N");
            Debug.Log($"[デバッグ] 現在の比率: {deformationThreshold / controller.baseGripForce:F3}倍");
            
            if (Mathf.Abs(deformationThreshold / controller.baseGripForce - 1.5f) < 0.1f)
            {
                Debug.LogWarning("⚠️ [デバッグ] deformationThresholdがbaseGripForceの約1.5倍になっています！");
            }
        }
        
        // 🔍 デバッグ: 他の関連コンポーネントもチェック
        var gripperInterface = FindObjectOfType<GripperTargetInterface>();
        if (gripperInterface != null)
        {
            Debug.Log($"[デバッグ] GripperTargetInterface発見");
        }
        
        var trajectoryPlanner = FindObjectOfType<TrajectoryPlannerDeform>();
        if (trajectoryPlanner != null)
        {
            Debug.Log($"[デバッグ] TrajectoryPlannerDeform発見");
        }
    }
    
    void Awake()
    {
        // 🔍 デバッグ: 最初期の値を記録
        Debug.Log($"[デバッグ] Awake()時 deformationThreshold: {_deformationThreshold:F2}N");
    }
    
    void OnValidate()
    {
        // 🔍 デバッグ: エディタでの値変更を追跡
        Debug.Log($"[デバッグ] OnValidate()で値変更検出: deformationThreshold={_deformationThreshold:F2}N");
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
        // 🔍 デバッグ: InitializeComponents開始時
        Debug.Log($"[デバッグ] InitializeComponents開始時 deformationThreshold: {deformationThreshold:F2}N");
        
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
        
        // 🔍 デバッグ: InitializeComponents完了時
        Debug.Log($"[デバッグ] InitializeComponents完了時 deformationThreshold: {deformationThreshold:F2}N");
    }
    
    /// <summary>
    /// 初期状態の設定
    /// </summary>
    void SetupInitialState()
    {
        // 🔍 デバッグ: SetupInitialState開始時
        Debug.Log($"[デバッグ] SetupInitialState開始時 deformationThreshold: {deformationThreshold:F2}N");
        
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
        
        // 🔍 デバッグ: SetupInitialState完了時
        Debug.Log($"[デバッグ] SetupInitialState完了時 deformationThreshold: {deformationThreshold:F2}N");
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
        
        // シンプルな変形判定：BaseGripForce vs deformationThreshold
        var gripController = FindObjectOfType<SimpleGripForceController>();
        if (gripController != null)
        {
            float baseGripForce = gripController.baseGripForce;
            
            // BaseGripForceがdeformationThresholdを超えたら変形
            if (baseGripForce > deformationThreshold)
            {
                if (!isCrushed)
                {
                    CrushCan();
                    Debug.Log($"🔥 缶が変形しました！ BaseGripForce: {baseGripForce:F2}N > 変形閾値: {deformationThreshold:F2}N");
                }
            }
            
            if (showDebugInfo && Time.frameCount % 30 == 0) // 30フレームごとにログ
            {
                Debug.Log($"BaseGripForce: {baseGripForce:F2}N vs 変形閾値: {deformationThreshold:F2}N");
            }
        }
        else
        {
            Debug.LogWarning("SimpleGripForceControllerが見つかりません。変形判定ができません。");
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
    /// つぶれ処理実行
    /// </summary>
    void CrushCan()
    {
        if (isCrushed) return;
        
        // 0.2秒待ってからつぶす
        StartCoroutine(CrushAfterDelay());
    }

    /// <summary>
    /// 遅延後につぶす
    /// </summary>
    IEnumerator CrushAfterDelay()
    {
        yield return new WaitForSeconds(DEFORMATION_DELAY); // 0.2秒待つ
        
        if (isCrushed) yield break; // 念のため再チェック
        
        isCrushed = true;
        
        // モデルの切り替え
        if (normalCanModel != null)
            normalCanModel.SetActive(false);
            
        if (crushedCanModel != null)
            crushedCanModel.SetActive(true);
        
        // 音響効果
        if (audioSource != null && crushSound != null)
        {
            audioSource.PlayOneShot(crushSound);
        }
        
        Debug.Log($"🥤 アルミ缶がつぶれました！（0.2秒遅延後）");
    }
    
    /// <summary>
    /// デバッグ情報の表示
    /// </summary>
    void DisplayDebugInfo()
    {
        if (Time.frameCount % 60 == 0) // 1秒ごと
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
            materialType = (int)MaterialType.Metal,
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
    
    var gripController = FindObjectOfType<SimpleGripForceController>();
    if (gripController != null)
    {
        GUI.Label(new Rect(10, 10, 300, 20), $"缶の状態: {(isCrushed ? "つぶれた" : "正常")}", style);
        GUI.Label(new Rect(10, 30, 300, 20), $"BaseGripForce: {gripController.baseGripForce:F2}N", style);
        GUI.Label(new Rect(10, 50, 300, 20), $"変形閾値: {deformationThreshold:F2}N", style);
        GUI.Label(new Rect(10, 70, 300, 20), $"変形判定: {(gripController.baseGripForce > deformationThreshold ? "変形" : "正常")}", style);
        
        // 進行状況バー
        float progress = gripController.baseGripForce / deformationThreshold;
        GUI.Box(new Rect(10, 90, 200, 20), "");
        GUI.Box(new Rect(10, 90, 200 * Mathf.Clamp01(progress), 20), "");
        GUI.Label(new Rect(10, 90, 200, 20), $"力の比率: {(progress * 100):F1}%", style);
    }
    else
    {
        GUI.Label(new Rect(10, 10, 300, 20), "SimpleGripForceController not found!", style);
    }
}
}

public enum MaterialType
{
    Soft,
    Medium,
    Hard,
    Metal,
    Fragile
}