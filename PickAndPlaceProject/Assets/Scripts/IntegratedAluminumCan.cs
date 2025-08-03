using UnityEngine;
using System.Collections;

/// <summary>
/// 統合されたアルミ缶変形システム（コライダーサイズ変更版）
/// 力が不足している場合にコライダーを小さくして掴みにくくする + 滑りやすくする
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
    
    [Header("新機能：コライダーサイズ変更設定")]
    [Range(1f, 10f)]
    [Tooltip("この力以下でコライダーが小さくなる（N）")]
    public float minimumGripForce = 5f;
    
    [Range(0.1f, 1f)]
    [Tooltip("力不足時のコライダーサイズ倍率")]
    public float weakGripColliderScale = 0.4f;
    
    [Range(0.1f, 2f)]
    [Tooltip("コライダーサイズ変更の遅延時間")]
    public float colliderChangeDelay = 0.3f;
    
    [Header("摩擦制御設定")]
    [Tooltip("通常時の摩擦材質")]
    public PhysicMaterial normalFriction;
    
    [Tooltip("滑りやすい摩擦材質")]
    public PhysicMaterial slipperyFriction;
    
    [Tooltip("力不足時に滑りやすくするか")]
    public bool enableSlipperyWhenWeak = true;
    
    public bool enableColliderSystem = true;
    public bool showColliderDebug = true;
    
    public float deformationThreshold 
    { 
        get => _deformationThreshold; 
        set 
        { 
            if (Mathf.Abs(_deformationThreshold - value) > 0.001f)
            {
                Debug.Log($"[デバッグ] deformationThreshold変更: {_deformationThreshold:F2} → {value:F2}");
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
    
    // 既存の内部状態
    private bool isCrushed = false;
    private float appliedForce = 0f;
    private float accumulatedForce = 0f;
    private Vector3 lastContactPoint = Vector3.zero;
    private Vector3 lastContactNormal = Vector3.up;
    private Rigidbody canRigidbody;
    
    // 新機能：コライダーサイズ変更システム用の変数
    private BoxCollider canBoxCollider;
    private Vector3 originalColliderSize;
    private Vector3 originalColliderCenter; // 元のセンター位置も保存
    private bool isColliderSmall = false;
    private bool isSlippery = false; // 滑りやすい状態かどうか
    private float lastForceCheckTime = 0f;
    private Coroutine colliderChangeCoroutine;
    
    // プロパティ（BasicTypes.csとの互換性用）
    public bool IsBroken => isCrushed;
    public float CurrentDeformation => isCrushed ? 1f : (accumulatedForce / deformationThreshold);
    public int MaterialTypeInt => (int)MaterialType.Metal; // BasicTypes.csとの互換性
    public float Softness => 0.1f; // 硬い材質

    /// <summary>
    /// BaseGripForce基準の変形進行度
    /// </summary>
    public float CurrentDeformationByBaseForce
    {
        get
        {
            var gripController = FindObjectOfType<SimpleGripForceController>();
            if (gripController != null)
            {
                float baseGripForce = gripController.baseGripForce;
                return Mathf.Clamp01(baseGripForce / deformationThreshold);
            }
            
            return Mathf.Clamp01(appliedForce / deformationThreshold);
        }
    }

    void Start()
    {
        InitializeComponents();
        SetupInitialState();
        InitializeColliderSystem(); // 新機能
        CreatePhysicMaterials(); // 摩擦材質を作成
    }
    
    void OnValidate()
    {
        Debug.Log($"[デバッグ] OnValidate()で値変更検出: deformationThreshold={_deformationThreshold:F2}N");
    }
    
    void Update()
    {
        
        // 新機能：コライダーサイズ変更システムの更新
        if (enableColliderSystem)
        {
            UpdateColliderSystem();
        }
        
        if (showDebugInfo)
        {
            DisplayDebugInfo();
        }
    }
    
    /// <summary>
    /// 新機能：物理材質を作成
    /// </summary>
    private void CreatePhysicMaterials()
    {
        // 通常の摩擦材質（まだ設定されていない場合）
        if (normalFriction == null)
        {
            normalFriction = new PhysicMaterial("NormalGrip");
            normalFriction.dynamicFriction = 0.6f;   // 通常の摩擦
            normalFriction.staticFriction = 0.8f;    // 静止摩擦
            normalFriction.bounciness = 0.1f;
            normalFriction.frictionCombine = PhysicMaterialCombine.Average;
            normalFriction.bounceCombine = PhysicMaterialCombine.Average;
        }
        
        // 滑りやすい材質（まだ設定されていない場合）
        if (slipperyFriction == null)
        {
            slipperyFriction = new PhysicMaterial("SlipperyGrip");
            slipperyFriction.dynamicFriction = 0.05f;  // 非常に滑りやすい
            slipperyFriction.staticFriction = 0.1f;    // 静止摩擦も低い
            slipperyFriction.bounciness = 0.2f;
            slipperyFriction.frictionCombine = PhysicMaterialCombine.Minimum; // 最小値を使用
            slipperyFriction.bounceCombine = PhysicMaterialCombine.Average;
        }
        
        // 初期状態では通常の摩擦を適用
        ApplyNormalFriction();
        
        Debug.Log("物理材質を作成しました");
    }
    
    /// <summary>
    /// 新機能：コライダーシステムの初期化
    /// </summary>
    private void InitializeColliderSystem()
    {
        canBoxCollider = GetComponent<BoxCollider>();
        
        if (canBoxCollider != null)
        {
            originalColliderSize = canBoxCollider.size;
            originalColliderCenter = canBoxCollider.center; // センターも保存
            Debug.Log($"🔧 元のコライダー設定を記録: サイズ={originalColliderSize}, センター={originalColliderCenter}");
        }
        else
        {
            Debug.LogError("❌ BoxColliderが見つかりません！");
        }
    }
    
    /// <summary>
    /// 新機能：コライダーシステムの更新
    /// </summary>
    private void UpdateColliderSystem()
    {
        // 0.5秒間隔で力をチェック
        if (Time.time - lastForceCheckTime > 0.5f)
        {
            lastForceCheckTime = Time.time;
            CheckGripForceAndAdjustCollider();
        }
    }
    
    /// <summary>
    /// 新機能：把持力をチェックしてコライダーを調整
    /// </summary>
    private void CheckGripForceAndAdjustCollider()
    {
        var gripController = FindObjectOfType<SimpleGripForceController>();
        if (gripController == null) return;
        
        float currentGripForce = gripController.GetCurrentTargetForce();
        bool shouldBeShrunk = currentGripForce < minimumGripForce;
        
        if (shouldBeShrunk && !isColliderSmall)
        {
            // コライダーを小さくする（遅延あり）
            if (colliderChangeCoroutine == null)
            {
                colliderChangeCoroutine = StartCoroutine(ShrinkColliderAfterDelay());
            }
        }
        else if (!shouldBeShrunk && isColliderSmall)
        {
            // コライダーを元のサイズに戻す
            if (colliderChangeCoroutine != null)
            {
                StopCoroutine(colliderChangeCoroutine);
                colliderChangeCoroutine = null;
            }
            RestoreColliderSize();
        }
        
        // 摩擦も同時に更新
        if (enableSlipperyWhenWeak)
        {
            if (shouldBeShrunk && !isSlippery)
            {
                ApplySlipperyFriction();
            }
            else if (!shouldBeShrunk && isSlippery)
            {
                ApplyNormalFriction();
            }
        }
        
        if (showColliderDebug && Time.frameCount % 60 == 0) // 1秒ごと
        {
            string frictionStatus = isSlippery ? " + 滑りやすい" : "";
            Debug.Log($"🔍 コライダー判定: 力={currentGripForce:F2}N, 閾値={minimumGripForce:F2}N, 小さい={isColliderSmall}{frictionStatus}");
        }
    }
    
    /// <summary>
    /// 新機能：遅延後にコライダーを小さくする
    /// </summary>
    private IEnumerator ShrinkColliderAfterDelay()
    {
        Debug.Log($"⏰ {colliderChangeDelay}秒後にコライダーを小さくします...");
        
        yield return new WaitForSeconds(colliderChangeDelay);
        
        // まだ力不足状態かチェック
        var gripController = FindObjectOfType<SimpleGripForceController>();
        if (gripController != null && gripController.GetCurrentTargetForce() < minimumGripForce)
        {
            ShrinkCollider();
        }
        
        colliderChangeCoroutine = null;
    }
    
    /// <summary>
    /// 新機能：コライダーを小さくする
    /// </summary>
    private void ShrinkCollider()
    {
        if (canBoxCollider == null) return;
        
        isColliderSmall = true;
        
        // Y軸（高さ）は元のサイズを維持、X軸とZ軸のみを縮小
        Vector3 smallSize = new Vector3(
            originalColliderSize.x,  // X軸を縮小
            originalColliderSize.y,                          // Y軸は元のまま（接地面維持）
            originalColliderSize.z * weakGripColliderScale   // Z軸を縮小
        );
        
        // 重要：サイズ変更時にcenterは変更しない（位置ずれを防ぐ）
        canBoxCollider.size = smallSize;
        canBoxCollider.center = originalColliderCenter; // センターを元の値に固定
        
        Debug.Log($"📦 コライダーを縮小: {originalColliderSize} → {smallSize} (Y軸維持, センター維持: {originalColliderCenter})");
        
        if (showColliderDebug)
        {
            string frictionStatus = enableSlipperyWhenWeak ? " + 滑りやすく" : "";
            Debug.Log($"⚠️ 把持力不足：アルミ缶が掴みにくくなりました（高さは維持{frictionStatus}）");
        }
    }
    
    /// <summary>
    /// 新機能：コライダーサイズを元に戻す
    /// </summary>
    private void RestoreColliderSize()
    {
        if (canBoxCollider == null) return;
        
        isColliderSmall = false;
        canBoxCollider.size = originalColliderSize;
        canBoxCollider.center = originalColliderCenter; // センターも元に戻す
        
        Debug.Log($"📦 コライダーを復元: {originalColliderSize} (センター: {originalColliderCenter})");
        
        if (showColliderDebug)
        {
            Debug.Log("✅ 把持力十分：アルミ缶が掴みやすくなりました（摩擦も正常）");
        }
    }
    
    /// <summary>
    /// 新機能：通常の摩擦を適用
    /// </summary>
    private void ApplyNormalFriction()
    {
        isSlippery = false;
        
        if (canBoxCollider != null)
        {
            canBoxCollider.material = normalFriction;
        }
        
        // Rigidbodyの物理特性も調整
        if (canRigidbody != null)
        {
            canRigidbody.drag = 0.1f; // 通常の空気抵抗
        }
        
        if (showColliderDebug)
        {
            Debug.Log("✅ 通常の摩擦を適用：掴みやすい状態");
        }
    }
    
    /// <summary>
    /// 新機能：滑りやすい摩擦を適用
    /// </summary>
    private void ApplySlipperyFriction()
    {
        isSlippery = true;
        
        if (canBoxCollider != null)
        {
            canBoxCollider.material = slipperyFriction;
        }
        
        // Rigidbodyの物理特性も調整
        if (canRigidbody != null)
        {
            canRigidbody.drag = 0.05f; // 空気抵抗を減らして滑りやすく
        }
        
        if (showColliderDebug)
        {
            Debug.Log("⚠️ 滑りやすい摩擦を適用：物体が滑りやすくなりました");
        }
    }
    
    // ===== 既存メソッドはそのまま維持 =====
    
    void InitializeComponents()
    {
        Debug.Log($"[デバッグ] InitializeComponents開始時 deformationThreshold: {deformationThreshold:F2}N");
        
        canRigidbody = GetComponent<Rigidbody>();
        if (canRigidbody == null)
        {
            canRigidbody = gameObject.AddComponent<Rigidbody>();
        }
        
        canRigidbody.mass = canMass;
        canRigidbody.drag = 0.1f;
        canRigidbody.angularDrag = 0.05f;
        
        if (audioSource == null)
        {
            audioSource = gameObject.AddComponent<AudioSource>();
            audioSource.spatialBlend = 1.0f;
            audioSource.volume = 0.7f;
        }
        
        Debug.Log("IntegratedAluminumCan initialized");
        Debug.Log($"[デバッグ] InitializeComponents完了時 deformationThreshold: {deformationThreshold:F2}N");
    }
    
    void SetupInitialState()
    {
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
        lastContactPoint = Vector3.zero;
        
        Debug.Log($"[デバッグ] SetupInitialState完了時 deformationThreshold: {deformationThreshold:F2}N");
    }
    
        public void ApplyGripperForceWithDirection(float force, Vector3 contactPoint, Vector3 contactNormal)
        {
            if (isCrushed) return;
            
            appliedForce = force;
            lastContactPoint = contactPoint;
            lastContactNormal = contactNormal;
            
            // 蓄積力システムを使わず、現在の把持力で直接判定
            if (appliedForce > deformationThreshold)
            {
                if (!isCrushed)
                {
                    // 遅延なしで即座につぶす
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
                    
                    Debug.Log($"🥤 アルミ缶がつぶれました！ 把持力: {appliedForce:F2}N > 閾値: {deformationThreshold:F2}N");
                }
            }
        }
    
    public void ApplyGripperForce(float force, Vector3 contactPoint)
    {
        ApplyGripperForceWithDirection(force, contactPoint, Vector3.up);
    }
    
    // void UpdateForceDecay()
    // {
    //     if (appliedForce <= 0f && accumulatedForce > 0f)
    //     {
    //         accumulatedForce -= Time.deltaTime * deformationSpeed * 0.5f;
    //         accumulatedForce = Mathf.Max(0f, accumulatedForce);
    //     }
    // }
    
    // void CheckForceThreshold()
    // {
    //     if (isCrushed) return;
        
    //     if (accumulatedForce >= deformationThreshold)
    //     {
    //         CrushCan();
    //     }
    // }
    
    void CrushCan()
    {
        if (isCrushed) return;
        
        StartCoroutine(CrushAfterDelay());
    }

    IEnumerator CrushAfterDelay()
    {
        yield return new WaitForSeconds(DEFORMATION_DELAY);
        
        if (isCrushed) yield break;
        
        isCrushed = true;
        
        if (normalCanModel != null)
            normalCanModel.SetActive(false);
            
        if (crushedCanModel != null)
            crushedCanModel.SetActive(true);
        
        if (audioSource != null && crushSound != null)
        {
            audioSource.PlayOneShot(crushSound);
        }
        
        Debug.Log($"🥤 アルミ缶がつぶれました！（0.2秒遅延後）");
    }
    
    void DisplayDebugInfo()
    {
        if (Time.frameCount % 60 == 0)
        {
            string status = isCrushed ? "つぶれた" : "正常";
            Debug.Log($"缶の状態: {status}, 現在の力: {appliedForce:F2}N, 蓄積力: {accumulatedForce:F2}N");
        }
    }
    
    public ObjectState GetCurrentState()
    {
        return new ObjectState
        {
            appliedForce = this.appliedForce,
            deformation = CurrentDeformation,
            isBroken = this.isCrushed,
            isBeingGrasped = appliedForce > 0f,
            materialType = MaterialTypeInt, // intを使用
            softness = this.Softness
        };
    }
    
    public float GetAccumulatedForce()
    {
        return accumulatedForce;
    }
    
    [ContextMenu("Reset Can")]
    public void ResetCan()
    {
        SetupInitialState();
        
        // コライダーサイズもリセット
        RestoreColliderSize();
        
        // 摩擦も通常に戻す
        ApplyNormalFriction();
        
        if (canRigidbody != null)
        {
            canRigidbody.mass = canMass;
            canRigidbody.drag = 0.1f;
            canRigidbody.angularDrag = 0.05f;
            canRigidbody.velocity = Vector3.zero;
            canRigidbody.angularVelocity = Vector3.zero;
        }
        
        if (colliderChangeCoroutine != null)
        {
            StopCoroutine(colliderChangeCoroutine);
            colliderChangeCoroutine = null;
        }
        
        Debug.Log("🔄 アルミ缶を初期状態にリセットしました");
    }
    
    [ContextMenu("Force Crush")]
    public void ForceCrush()
    {
        accumulatedForce = deformationThreshold + 1f;
        CrushCan();
    }
    
    /// <summary>
    /// 新機能：コライダーサイズの手動テスト
    /// </summary>
    [ContextMenu("Test Collider Size")]
    public void TestColliderSize()
    {
        if (isColliderSmall)
        {
            RestoreColliderSize();
        }
        else
        {
            ShrinkCollider();
        }
    }
    
    /// <summary>
    /// 新機能：摩擦状態の手動テスト
    /// </summary>
    [ContextMenu("Test Friction")]
    public void TestFriction()
    {
        if (isSlippery)
        {
            ApplyNormalFriction();
        }
        else
        {
            ApplySlipperyFriction();
        }
    }
    
    /// <summary>
    /// 新機能：現在のコライダー状況確認
    /// </summary>
    [ContextMenu("Check Collider System")]
    public void CheckColliderSystem()
    {
        var gripController = FindObjectOfType<SimpleGripForceController>();
        if (gripController != null)
        {
            float currentForce = gripController.GetCurrentTargetForce();
            Debug.Log("=== コライダー&摩擦システム状況 ===");
            Debug.Log($"現在の把持力: {currentForce:F2}N");
            Debug.Log($"最小把持力閾値: {minimumGripForce:F2}N");
            Debug.Log($"コライダー縮小状態: {isColliderSmall}");
            Debug.Log($"滑りやすい状態: {isSlippery}");
            Debug.Log($"元のサイズ: {originalColliderSize}");
            Debug.Log($"元のセンター: {originalColliderCenter}");
            Debug.Log($"現在のサイズ: {(canBoxCollider != null ? canBoxCollider.size.ToString() : "なし")}");
            Debug.Log($"現在のセンター: {(canBoxCollider != null ? canBoxCollider.center.ToString() : "なし")}");
            Debug.Log($"現在の材質: {(canBoxCollider != null ? canBoxCollider.material?.name : "なし")}");
            Debug.Log($"縮小倍率: {weakGripColliderScale}");
            Debug.Log($"摩擦制御有効: {enableSlipperyWhenWeak}");
            Debug.Log($"システム有効: {enableColliderSystem}");
        }
        else
        {
            Debug.LogError("SimpleGripForceControllerが見つかりません");
        }
    }
    
    void OnDrawGizmos()
    {
        if (!showForceGizmos) return;
        
        // 既存のGizmo描画
        if (lastContactPoint != Vector3.zero)
        {
            Gizmos.color = isCrushed ? Color.red : Color.yellow;
            Gizmos.DrawWireSphere(lastContactPoint, 0.02f);
        }
        
        if (appliedForce > 0f)
        {
            Gizmos.color = accumulatedForce >= deformationThreshold ? Color.red : Color.green;
            Gizmos.DrawRay(transform.position, lastContactNormal * (appliedForce * 0.01f));
        }
        
        float barHeight = (accumulatedForce / deformationThreshold) * 0.1f;
        Gizmos.color = Color.blue;
        Gizmos.DrawCube(transform.position + Vector3.up * 0.15f, new Vector3(0.02f, barHeight, 0.02f));
        
        // 新機能：コライダーサイズの視覚表示
        if (canBoxCollider != null)
        {
            // 滑りやすい状態なら黄色、そうでなければ通常の色
            Gizmos.color = isSlippery ? Color.yellow : (isColliderSmall ? Color.red : Color.green);
            Gizmos.DrawWireCube(transform.position + originalColliderCenter, canBoxCollider.size);
            
            // 元のサイズも薄く表示
            if (isColliderSmall)
            {
                Gizmos.color = new Color(0, 1, 0, 0.3f); // 薄い緑
                Gizmos.DrawWireCube(transform.position + originalColliderCenter, originalColliderSize);
            }
        }
    }

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
            
            // 新機能：コライダー&摩擦状態の表示
            string colliderState = isColliderSmall ? "小さい" : "通常";
            string frictionState = isSlippery ? " + 滑りやすい" : "";
            GUI.Label(new Rect(10, 90, 350, 20), $"コライダー: {colliderState}{frictionState}", style);
            GUI.Label(new Rect(10, 110, 300, 20), $"最小把持力: {minimumGripForce:F2}N", style);
            
            if (canBoxCollider != null)
            {
                GUI.Label(new Rect(10, 130, 300, 20), $"サイズ: X={canBoxCollider.size.x:F3}, Y={canBoxCollider.size.y:F3}, Z={canBoxCollider.size.z:F3}", style);
            }
            
            float progress = gripController.baseGripForce / deformationThreshold;
            GUI.Box(new Rect(10, 150, 200, 20), "");
            GUI.Box(new Rect(10, 150, 200 * Mathf.Clamp01(progress), 20), "");
            GUI.Label(new Rect(10, 150, 200, 20), $"力の比率: {(progress * 100):F1}%", style);
        }
        else
        {
            GUI.Label(new Rect(10, 10, 300, 20), "SimpleGripForceController not found!", style);
        }
    }
}

// MaterialTypeの定義（既存コードとの互換性）
public enum MaterialType
{
    Soft,
    Medium,
    Hard,
    Metal,
    Fragile
}