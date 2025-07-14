using System.Collections;
using System.Collections.Generic;
using UnityEngine;

/// <summary>
/// シンプルな把持力制御システム
/// UI不要、Inspectorで直接パラメータ調整可能
/// </summary>
public class SimpleGripForceController : MonoBehaviour
{
    [Header("=== 基本把持力設定 ===")]
    [SerializeField, Range(0.1f, 100f), Tooltip("基本的な把持力(N)")]
    public float baseGripForce = 10f;
    
    [SerializeField, Range(0f, 1f), Tooltip("把持力のばらつき度合い")]
    private float forceVariability = 0.1f;
    
    [SerializeField, Range(0.1f, 5f), Tooltip("力の変化速度")]
    private float forceChangeRate = 1f;
    
    [Header("=== 高度な制御設定 ===")]
    [SerializeField, Tooltip("適応制御を有効にする")]
    public bool enableAdaptiveControl = true;
    
    [SerializeField, Range(0f, 1f), Tooltip("適応制御の強さ")]
    private float adaptiveGain = 0.5f;
    
    [SerializeField, Range(0f, 2f), Tooltip("制御の安定性")]
    private float dampingFactor = 0.3f;
    
    [Header("=== 実験用設定 ===")]
    [SerializeField, Tooltip("ランダムノイズを追加する")]
    private bool enableForceNoise = false;
    
    [SerializeField, Range(0f, 0.5f), Tooltip("ノイズの強さ")]
    private float noiseStrength = 0.1f;
    
    [SerializeField, Tooltip("制御モード")]
    public GripControlMode controlMode = GripControlMode.Standard;
    
    [Header("=== プリセット選択 ===")]
    [SerializeField, Tooltip("事前定義されたプリセットを使用")]
    private GripForcePresetType presetType = GripForcePresetType.Custom;
    
    [SerializeField, Tooltip("プリセット適用ボタン")]
    private bool applyPreset = false;
    
    [Header("=== デバッグ設定 ===")]
    [SerializeField, Tooltip("デバッグ情報を表示")]
    private bool showDebugInfo = true;
    
    [SerializeField, Tooltip("実験データを記録")]
    private bool recordExperimentData = true;
    
    [Header("=== 参照 ===")]
    [SerializeField] private GripperForceController gripperController;
    
    // 内部変数
    private float currentTargetForce;
    private float previousError;
    private List<ExperimentDataPoint> experimentData = new List<ExperimentDataPoint>();
    private float lastPresetCheckTime;
    
    // プリセット定義
    private static readonly Dictionary<GripForcePresetType, GripForcePreset> Presets = 
        new Dictionary<GripForcePresetType, GripForcePreset>
        {
            [GripForcePresetType.SoftObject] = new GripForcePreset
            {
                baseForce = 5f, variability = 0.05f, changeRate = 0.5f,
                adaptiveEnabled = true, adaptiveGain = 0.8f, damping = 0.5f,
                noiseEnabled = false, noiseStrength = 0.05f
            },
            [GripForcePresetType.MediumObject] = new GripForcePreset
            {
                baseForce = 15f, variability = 0.1f, changeRate = 1f,
                adaptiveEnabled = true, adaptiveGain = 0.5f, damping = 0.3f,
                noiseEnabled = false, noiseStrength = 0.1f
            },
            [GripForcePresetType.HardObject] = new GripForcePreset
            {
                baseForce = 30f, variability = 0.2f, changeRate = 2f,
                adaptiveEnabled = true, adaptiveGain = 0.3f, damping = 0.1f,
                noiseEnabled = true, noiseStrength = 0.15f
            },
            [GripForcePresetType.Experimental] = new GripForcePreset
            {
                baseForce = 20f, variability = 0.3f, changeRate = 1.5f,
                adaptiveEnabled = true, adaptiveGain = 0.7f, damping = 0.4f,
                noiseEnabled = true, noiseStrength = 0.2f
            }
        };
    
    void Start()
    {
        InitializeController();
    }
    
    void Update()
    {
        // プリセット適用チェック
        CheckPresetApplication();
        
        // 把持力制御更新
        UpdateGripForce();
        
        // 実験データ記録
        if (recordExperimentData)
        {
            RecordData();
        }
    }
    
    /// <summary>
    /// コントローラー初期化
    /// </summary>
    private void InitializeController()
    {
        // グリッパーコントローラーの自動検出
        if (gripperController == null)
        {
            gripperController = FindObjectOfType<GripperForceController>();
            if (gripperController == null)
            {
                Debug.LogWarning("GripperForceController が見つかりません");
            }
        }
        
        currentTargetForce = baseGripForce;
        previousError = 0f;
        
        Debug.Log("SimpleGripForceController が初期化されました");
    }
    
    /// <summary>
    /// プリセット適用チェック
    /// </summary>
    private void CheckPresetApplication()
    {
        // applyPreset が true になったらプリセットを適用
        if (applyPreset && presetType != GripForcePresetType.Custom)
        {
            ApplyPreset(presetType);
            applyPreset = false; // 適用後にフラグをリセット
        }
    }
    
    /// <summary>
    /// 把持力制御の更新
    /// </summary>
    private void UpdateGripForce()
    {
        if (gripperController == null) return;
        
        var graspState = gripperController.GetGraspingState();
        
        // 制御モードに応じた処理
        switch (controlMode)
        {
            case GripControlMode.Standard:
                UpdateStandardControl(graspState);
                break;
            case GripControlMode.Adaptive:
                UpdateAdaptiveControl(graspState);
                break;
            case GripControlMode.Experimental:
                UpdateExperimentalControl(graspState);
                break;
        }
        
        // ノイズの追加
        float finalForce = currentTargetForce;
        if (enableForceNoise)
        {
            float noise = Random.Range(-noiseStrength, noiseStrength) * baseGripForce;
            finalForce += noise;
        }
        
        // 最終的な力をグリッパーに設定
        finalForce = Mathf.Clamp(finalForce, 0.1f, 100f);
        gripperController.SetTargetGripForce(finalForce);
    }
    
    /// <summary>
    /// 標準制御
    /// </summary>
    private void UpdateStandardControl(GraspingState graspState)
    {
        // 基本力に変動を加える
        float targetForce = baseGripForce + Random.Range(-forceVariability, forceVariability) * baseGripForce;
        currentTargetForce = Mathf.Lerp(currentTargetForce, targetForce, forceChangeRate * Time.deltaTime);
    }
    
    /// <summary>
    /// 適応制御
    /// </summary>
    private void UpdateAdaptiveControl(GraspingState graspState)
    {
        if (!enableAdaptiveControl || !graspState.isGrasping) 
        {
            UpdateStandardControl(graspState);
            return;
        }
        
        // エラー計算
        float error = graspState.targetForce - graspState.currentForce;
        float derivative = error - previousError;
        
        // PI制御
        float adjustment = adaptiveGain * error + dampingFactor * derivative;
        float targetForce = baseGripForce + adjustment;
        
        // 変動の追加
        if (forceVariability > 0)
        {
            targetForce += Random.Range(-forceVariability, forceVariability) * baseGripForce;
        }
        
        currentTargetForce = Mathf.Lerp(currentTargetForce, targetForce, forceChangeRate * Time.deltaTime);
        previousError = error;
    }
    
    /// <summary>
    /// 実験的制御
    /// </summary>
    private void UpdateExperimentalControl(GraspingState graspState)
    {
        // 時間による力の変動
        float timeVariation = Mathf.Sin(Time.time * 0.5f) * 0.1f;
        float targetForce = baseGripForce * (1f + timeVariation);
        
        // 適応制御も組み合わせ
        if (enableAdaptiveControl && graspState.isGrasping)
        {
            float error = graspState.targetForce - graspState.currentForce;
            targetForce += adaptiveGain * error;
        }
        
        // 変動とノイズ
        targetForce += Random.Range(-forceVariability, forceVariability) * baseGripForce;
        
        currentTargetForce = Mathf.Lerp(currentTargetForce, targetForce, forceChangeRate * Time.deltaTime);
    }
    
    /// <summary>
    /// プリセット適用
    /// </summary>
    private void ApplyPreset(GripForcePresetType preset)
    {
        if (!Presets.ContainsKey(preset)) return;
        
        var presetData = Presets[preset];
        
        baseGripForce = presetData.baseForce;
        forceVariability = presetData.variability;
        forceChangeRate = presetData.changeRate;
        enableAdaptiveControl = presetData.adaptiveEnabled;
        adaptiveGain = presetData.adaptiveGain;
        dampingFactor = presetData.damping;
        enableForceNoise = presetData.noiseEnabled;
        noiseStrength = presetData.noiseStrength;
        
        currentTargetForce = baseGripForce;
        
        Debug.Log($"プリセット '{preset}' を適用しました");
        Debug.Log($"基本力: {baseGripForce}N, 変動: {forceVariability}, 適応制御: {enableAdaptiveControl}");
    }
    
    /// <summary>
    /// 実験データ記録
    /// </summary>
    private void RecordData()
    {
        if (gripperController == null) return;
        
        var graspState = gripperController.GetGraspingState();
        var dataPoint = new ExperimentDataPoint
        {
            timestamp = Time.time,
            targetForce = currentTargetForce,
            actualForce = graspState.currentForce,
            baseForce = baseGripForce,
            controlMode = controlMode.ToString(),
            isGrasping = graspState.isGrasping,
            isSuccessful = graspState.isSuccessful
        };
        
        experimentData.Add(dataPoint);
        
        // データサイズ制限
        if (experimentData.Count > 5000)
        {
            experimentData.RemoveRange(0, 1000);
        }
    }
    
    /// <summary>
    /// データエクスポート（公開メソッド）
    /// </summary>
    public void ExportData()
    {
        if (experimentData.Count == 0)
        {
            Debug.LogWarning("エクスポートするデータがありません");
            return;
        }
        
        string csv = "Time,TargetForce,ActualForce,BaseForce,ControlMode,IsGrasping,IsSuccessful\n";
        foreach (var data in experimentData)
        {
            csv += $"{data.timestamp:F2},{data.targetForce:F2},{data.actualForce:F2},{data.baseForce:F2}," +
                   $"{data.controlMode},{data.isGrasping},{data.isSuccessful}\n";
        }
        
        string filename = $"GripForceData_{System.DateTime.Now:yyyyMMdd_HHmmss}.csv";
        string filepath = System.IO.Path.Combine(Application.persistentDataPath, filename);
        System.IO.File.WriteAllText(filepath, csv);
        
        Debug.Log($"実験データを保存しました: {filepath}");
    }
    
    /// <summary>
    /// パラメータリセット（公開メソッド）
    /// </summary>
    public void ResetToDefaults()
    {
        baseGripForce = 10f;
        forceVariability = 0.1f;
        forceChangeRate = 1f;
        enableAdaptiveControl = true;
        adaptiveGain = 0.5f;
        dampingFactor = 0.3f;
        enableForceNoise = false;
        noiseStrength = 0.1f;
        controlMode = GripControlMode.Standard;
        
        currentTargetForce = baseGripForce;
        previousError = 0f;
        
        Debug.Log("パラメータをデフォルトにリセットしました");
    }
    
    // デバッグ表示
    void OnGUI()
    {
        if (!showDebugInfo) return;
        
        var graspState = gripperController?.GetGraspingState();
        
        GUILayout.BeginArea(new Rect(10, 10, 350, 400));
        GUILayout.Label("=== 把持力制御デバッグ ===");
        GUILayout.Label($"制御モード: {controlMode}");
        GUILayout.Label($"基本把持力: {baseGripForce:F1} N");
        GUILayout.Label($"現在目標力: {currentTargetForce:F2} N");
        GUILayout.Label($"実際の力: {(graspState?.currentForce ?? 0):F2} N");
        GUILayout.Label($"把持中: {(graspState?.isGrasping ?? false)}");
        GUILayout.Label($"成功: {(graspState?.isSuccessful ?? false)}");
        GUILayout.Label($"変動: {forceVariability:F2}");
        GUILayout.Label($"変化速度: {forceChangeRate:F1}");
        GUILayout.Label($"適応制御: {enableAdaptiveControl}");
        GUILayout.Label($"ノイズ: {enableForceNoise}");
        GUILayout.Label($"記録データ数: {experimentData.Count}");
        GUILayout.Space(10);
        
        if (GUILayout.Button("データエクスポート"))
        {
            ExportData();
        }
        if (GUILayout.Button("パラメータリセット"))
        {
            ResetToDefaults();
        }
        
        GUILayout.EndArea();
    }

// SimpleGripForceController.cs に追加する必要があるメソッド（完全版）

/// <summary>
/// 現在のターゲット力を取得（外部アクセス用）
/// </summary>
public float GetCurrentTargetForce()
{
    return currentTargetForce;
}

public GraspingState GetGraspingStateForInterface()
{
    // 既存のGraspingState構造体を使用
    // 上記コードを参照
}

/// <summary>
/// 現在の実際の力を取得
/// </summary>
public float GetCurrentActualForce()
{
    if (gripperController == null) return 0f;
    
    // GripperForceControllerがある場合
    try 
    {
        var graspState = gripperController.GetGraspingState();
        return graspState.currentForce;
    }
    catch
    {
        // フォールバック：基本的な力を返す
        return currentTargetForce;
    }
}

/// <summary>
/// 把持状態を取得（GraspingState形式で）
/// GripperTargetInterfaceとの互換性のため
/// </summary>
public GraspingState GetGraspingState()
{
    GraspingState state = new GraspingState();
    
    state.currentForce = currentTargetForce;
    state.targetForce = baseGripForce;
    state.lastUpdateTime = Time.time;
    
    // グリッパーコントローラーが利用可能な場合、詳細情報を取得
    if (gripperController != null)
    {
        try 
        {
            var gripperState = gripperController.GetGraspingState();
            state.isGrasping = gripperState.isGrasping;
            state.isSuccessful = gripperState.isSuccessful;
            state.leftGripperForce = gripperState.currentForce * 0.5f;  // 仮の分割
            state.rightGripperForce = gripperState.currentForce * 0.5f;
            state.hasContact = gripperState.isGrasping;
        }
        catch
        {
            // GripperForceControllerが利用できない場合のフォールバック
            state.isGrasping = currentTargetForce > 1f;
            state.isSuccessful = state.isGrasping;
            state.hasContact = state.isGrasping;
            state.leftGripperForce = currentTargetForce * 0.5f;
            state.rightGripperForce = currentTargetForce * 0.5f;
        }
    }
    else
    {
        // GripperForceControllerがない場合の簡易判定
        state.isGrasping = currentTargetForce > 1f;
        state.isSuccessful = state.isGrasping && currentTargetForce < 50f;
        state.hasContact = state.isGrasping;
        state.leftGripperForce = currentTargetForce * 0.5f;
        state.rightGripperForce = currentTargetForce * 0.5f;
    }
    
    return state;
}

/// <summary>
/// グリッパーが閉じている状態かどうかを判定
/// </summary>
public bool IsGripperClosed()
{
    if (gripperController == null) 
    {
        // フォールバック：力が一定以上なら閉じていると判定
        return currentTargetForce > 2f;
    }
    
    try 
    {
        var graspState = gripperController.GetGraspingState();
        return graspState.isGrasping;
    }
    catch
    {
        return currentTargetForce > 2f;
    }
}

/// <summary>
/// 力制御を手動で有効/無効にする
/// </summary>
public void SetForceControlEnabled(bool enabled)
{
    this.enabled = enabled;
    
    if (enableDataRecording)
    {
        Debug.Log($"SimpleGripForceController: Force control {(enabled ? "ENABLED" : "DISABLED")}");
    }
}

/// <summary>
/// 外部から基本把持力を設定
/// </summary>
public void SetBaseGripForce(float force)
{
    baseGripForce = Mathf.Clamp(force, 0.1f, 100f);
    currentTargetForce = baseGripForce;
    
    if (enableDataRecording)
    {
        Debug.Log($"Base grip force set to: {baseGripForce:F2}N");
    }
}

/// <summary>
/// 現在の制御モードを取得
/// </summary>
public string GetCurrentControlMode()
{
    return controlMode.ToString();
}

/// <summary>
/// 力制御の統計情報を取得
/// </summary>
public void GetForceStatistics(out float avgForce, out float maxForce, out float minForce)
{
    if (experimentData.Count == 0)
    {
        avgForce = maxForce = minForce = currentTargetForce;
        return;
    }
    
    float sum = 0f;
    maxForce = float.MinValue;
    minForce = float.MaxValue;
    
    foreach (var data in experimentData)
    {
        sum += data.targetForce;
        maxForce = Mathf.Max(maxForce, data.targetForce);
        minForce = Mathf.Min(minForce, data.targetForce);
    }
    
    avgForce = sum / experimentData.Count;
}

/// <summary>
/// 基本把持力を取得（外部アクセス用）
/// </summary>
public float GetBaseGripForce()
{
    return baseGripForce;
}

/// <summary>
/// 把持状態の簡易情報を取得
/// </summary>
public GraspingStateInfo GetGraspingStateInfo()
{
    return new GraspingStateInfo
    {
        currentForce = currentTargetForce,
        baseForce = baseGripForce,
        isEnabled = enabled,
        controlMode = controlMode.ToString()
    };
}

/// <summary>
/// 把持状態の情報構造体
/// </summary>
[System.Serializable]
public struct GraspingStateInfo
{
    public float currentForce;
    public float baseForce;
    public bool isEnabled;
    public string controlMode;
}

// SimpleGripForceController.cs の末尾（最後の } の前）に以下を追加してください

    /// <summary>
    /// 現在のターゲット力を取得（外部アクセス用）
    /// </summary>
    public float GetCurrentTargetForce()
    {
        return currentTargetForce;
    }

    /// <summary>
    /// SimpleGripForceController用の把持状態取得
    /// 既存のGraspingStateと互換性を保つ
    /// </summary>
    public GraspingState GetGraspingStateForInterface()
    {
        GraspingState state = new GraspingState();
        
        if (gripperController != null)
        {
            try 
            {
                // 既存のGripperForceControllerから状態を取得
                var existingState = gripperController.GetGraspingState();
                state.isGrasping = existingState.isGrasping;
                state.currentForce = existingState.currentForce;
                state.targetForce = existingState.targetForce;
                state.gripperPosition = existingState.gripperPosition;
                state.isSuccessful = existingState.isSuccessful;
                state.softness = existingState.softness;
            }
            catch
            {
                // フォールバック
                state.isGrasping = currentTargetForce > 1f;
                state.currentForce = currentTargetForce;
                state.targetForce = baseGripForce;
                state.gripperPosition = 0f;
                state.isSuccessful = state.isGrasping && currentTargetForce < 50f;
                state.softness = 0.5f;
            }
        }
        else
        {
            // GripperForceControllerがない場合の簡易状態
            state.isGrasping = currentTargetForce > 1f;
            state.currentForce = currentTargetForce;
            state.targetForce = baseGripForce;
            state.gripperPosition = 0f;
            state.isSuccessful = state.isGrasping && currentTargetForce < 50f;
            state.softness = 0.5f;
        }
        
        return state;
    }

    /// <summary>
    /// 力制御を手動で有効/無効にする
    /// </summary>
    public void SetForceControlEnabled(bool enabled)
    {
        this.enabled = enabled;
        
        if (enableDataRecording)
        {
            Debug.Log($"SimpleGripForceController: Force control {(enabled ? "ENABLED" : "DISABLED")}");
        }
    }

    /// <summary>
    /// 外部から基本把持力を設定
    /// </summary>
    public void SetBaseGripForce(float force)
    {
        baseGripForce = Mathf.Clamp(force, 0.1f, 100f);
        currentTargetForce = baseGripForce;
        
        if (enableDataRecording)
        {
            Debug.Log($"Base grip force set to: {baseGripForce:F2}N");
        }
    }

    /// <summary>
    /// 現在の制御モードを取得
    /// </summary>
    public string GetCurrentControlMode()
    {
        return controlMode.ToString();
    }

    /// <summary>
    /// グリッパーが閉じている状態かどうかを判定
    /// </summary>
    public bool IsGripperClosed()
    {
        if (gripperController == null) 
        {
            // フォールバック：力が一定以上なら閉じていると判定
            return currentTargetForce > 2f;
        }
        
        try 
        {
            var graspState = gripperController.GetGraspingState();
            return graspState.isGrasping;
        }
        catch
        {
            return currentTargetForce > 2f;
        }
    }
}

/// <summary>
/// 制御モード
/// </summary>
public enum GripControlMode
{
    Standard,    // 標準制御
    Adaptive,    // 適応制御
    Experimental // 実験的制御
}

/// <summary>
/// プリセットタイプ
/// </summary>
public enum GripForcePresetType
{
    Custom,        // カスタム（プリセット未使用）
    SoftObject,    // 柔らかい物体用
    MediumObject,  // 中程度の硬さの物体用
    HardObject,    // 硬い物体用
    Experimental   // 実験用
}

/// <summary>
/// プリセットデータ
/// </summary>
[System.Serializable]
public struct GripForcePreset
{
    public float baseForce;
    public float variability;
    public float changeRate;
    public bool adaptiveEnabled;
    public float adaptiveGain;
    public float damping;
    public bool noiseEnabled;
    public float noiseStrength;
}

/// <summary>
/// 実験データポイント
/// </summary>
[System.Serializable]
public struct ExperimentDataPoint
{
    public float timestamp;
    public float targetForce;
    public float actualForce;
    public float baseForce;
    public string controlMode;
    public bool isGrasping;
    public bool isSuccessful;
}
