using UnityEngine;

/// <summary>
/// アルミ缶のモデル入れ替えシステム
/// 正常な缶とつぶれた缶の3Dモデルを準備して、力に応じて切り替える
/// </summary>
public class AluminumCanModelSwap : MonoBehaviour
{
    [Header("3Dモデル設定")]
    [Tooltip("正常なアルミ缶のGameObject")]
    public GameObject normalCanModel;
    
    [Tooltip("つぶれたアルミ缶のGameObject")]
    public GameObject crushedCanModel;
    
    [Header("力の設定")]
    [Range(1f, 100f)]
    [Tooltip("変形が発生する力の閾値")]
    public float deformationThreshold = 15f;
    
    [Header("音響効果")]
    [Tooltip("つぶれる音のAudioClip")]
    public AudioClip crushSound;
    
    [Tooltip("AudioSource（なければ自動で作成）")]
    public AudioSource audioSource;
    
    [Header("デバッグ設定")]
    [Tooltip("衝突力をコンソールに表示")]
    public bool showForceDebug = true;
    
    [Tooltip("Gizmoで力を可視化")]
    public bool showForceGizmos = true;
    
    // 内部状態
    private bool isCrushed = false;
    private float lastImpactForce = 0f;
    private Vector3 lastContactPoint = Vector3.zero;
    private Rigidbody canRigidbody;
    
    void Start()
    {
        InitializeComponents();
        SetupInitialState();
    }
    
    /// <summary>
    /// 初期化処理
    /// </summary>
    void InitializeComponents()
    {
        // Rigidbodyコンポーネントの取得
        canRigidbody = GetComponent<Rigidbody>();
        if (canRigidbody == null)
        {
            canRigidbody = gameObject.AddComponent<Rigidbody>();
            Debug.Log("Rigidbodyコンポーネントを自動追加しました");
        }
        
        // AudioSourceの設定
        if (audioSource == null)
        {
            audioSource = GetComponent<AudioSource>();
            if (audioSource == null)
            {
                audioSource = gameObject.AddComponent<AudioSource>();
                Debug.Log("AudioSourceコンポーネントを自動追加しました");
            }
        }
        
        // コライダーの確認
        if (GetComponent<Collider>() == null)
        {
            Debug.LogWarning("Colliderコンポーネントが見つかりません。衝突検出のために追加してください。");
        }
    }
    
    /// <summary>
    /// 初期状態の設定
    /// </summary>
    void SetupInitialState()
    {
        // 正常な缶を表示、つぶれた缶を非表示
        if (normalCanModel != null)
        {
            normalCanModel.SetActive(true);
        }
        else
        {
            Debug.LogError("正常なアルミ缶のモデルが設定されていません！");
        }
        
        if (crushedCanModel != null)
        {
            crushedCanModel.SetActive(false);
        }
        else
        {
            Debug.LogError("つぶれたアルミ缶のモデルが設定されていません！");
        }
        
        isCrushed = false;
        
        Debug.Log("アルミ缶モデル入れ替えシステムが初期化されました");
    }
    
    /// <summary>
    /// 衝突検出
    /// </summary>
    void OnCollisionEnter(Collision collision)
    {
        if (isCrushed) return; // 既につぶれている場合は処理しない
        
        // 衝突力の計算
        float impactForce = collision.impulse.magnitude / Time.fixedDeltaTime;
        lastImpactForce = impactForce;
        
        // 衝突点の記録
        if (collision.contacts.Length > 0)
        {
            lastContactPoint = collision.contacts[0].point;
        }
        
        // デバッグ情報の表示
        if (showForceDebug)
        {
            Debug.Log($"衝突検出: 力 = {impactForce:F2}N, 閾値 = {deformationThreshold}N, 衝突相手 = {collision.gameObject.name}");
        }
        
        // 閾値を超えた場合、缶をつぶす
        if (impactForce >= deformationThreshold)
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
        
        Debug.Log($"アルミ缶がつぶれました！ 衝突力: {lastImpactForce:F2}N");
        
        // モデルの入れ替え
        SwapModels();
        
        // 音効果の再生
        PlayCrushSound();
        
        // 物理特性の調整（必要に応じて）
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
        
        Debug.Log("モデルを正常な缶からつぶれた缶に切り替えました");
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
            // つぶれた缶は少し軽くなる
            canRigidbody.mass *= 0.9f;
            
            // 空気抵抗を少し増加
            canRigidbody.drag *= 1.1f;
        }
    }
    
    /// <summary>
    /// 缶を元の状態にリセット（テスト用）
    /// </summary>
    [ContextMenu("Reset Can")]
    public void ResetCan()
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
        lastImpactForce = 0f;
        lastContactPoint = Vector3.zero;
        
        // 物理特性をリセット
        if (canRigidbody != null)
        {
            canRigidbody.mass = 0.015f; // 一般的なアルミ缶の重さ（15g）
            canRigidbody.drag = 0.1f;
            canRigidbody.angularDrag = 0.05f;
        }
        
        Debug.Log("アルミ缶を初期状態にリセットしました");
    }
    
    /// <summary>
    /// 公開メソッド：外部から強制的に缶をつぶす
    /// </summary>
    public void ForceCrush()
    {
        lastImpactForce = deformationThreshold + 1f;
        CrushCan();
    }
    
    /// <summary>
    /// 現在の状態を取得
    /// </summary>
    public bool IsCrushed()
    {
        return isCrushed;
    }
    
    /// <summary>
    /// 最後の衝突力を取得
    /// </summary>
    public float GetLastImpactForce()
    {
        return lastImpactForce;
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
        if (lastImpactForce > 0f)
        {
            Gizmos.color = lastImpactForce >= deformationThreshold ? Color.red : Color.green;
            Vector3 forceDirection = (lastContactPoint - transform.position).normalized;
            Gizmos.DrawRay(transform.position, forceDirection * (lastImpactForce * 0.01f));
        }
    }
    
    /// <summary>
    /// インスペクター上での情報表示
    /// </summary>
    void OnGUI()
    {
        if (!showForceDebug) return;
        
        GUI.Label(new Rect(10, 10, 300, 20), $"缶の状態: {(isCrushed ? "つぶれた" : "正常")}");
        GUI.Label(new Rect(10, 30, 300, 20), $"最後の衝突力: {lastImpactForce:F2}N");
        GUI.Label(new Rect(10, 50, 300, 20), $"変形閾値: {deformationThreshold:F2}N");
    }
}