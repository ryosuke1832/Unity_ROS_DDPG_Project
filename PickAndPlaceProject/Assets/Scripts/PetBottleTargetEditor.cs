#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

/// <summary>
/// PetBottleTargetのエディター拡張
/// インスペクターでリアルタイムプレビューとプリセット機能を提供
/// </summary>
[CustomEditor(typeof(PetBottleTarget))]
public class PetBottleTargetEditor : Editor
{
    private PetBottleTarget bottleTarget;
    private bool showPresets = false;
    private bool showAdvancedSettings = false;
    
    // プリセット定義
    private static readonly BottlePreset[] bottlePresets = {
        new BottlePreset {
            name = "標準ペットボトル (500ml)",
            height = 1.2f,
            radius = 0.3f,
            neckHeight = 0.25f,
            neckRadius = 0.12f,
            sideFlexibility = 0.7f,
            topBottomRigidity = 0.85f,
            compressionResistance = 8f,
            breakingForce = 30f
        },
        new BottlePreset {
            name = "小型ボトル (250ml)",
            height = 0.8f,
            radius = 0.25f,
            neckHeight = 0.2f,
            neckRadius = 0.1f,
            sideFlexibility = 0.8f,
            topBottomRigidity = 0.8f,
            compressionResistance = 6f,
            breakingForce = 25f
        },
        new BottlePreset {
            name = "大型ボトル (2L)",
            height = 2.0f,
            radius = 0.45f,
            neckHeight = 0.3f,
            neckRadius = 0.15f,
            sideFlexibility = 0.6f,
            topBottomRigidity = 0.9f,
            compressionResistance = 12f,
            breakingForce = 40f
        },
        new BottlePreset {
            name = "薄いペットボトル",
            height = 1.5f,
            radius = 0.35f,
            neckHeight = 0.25f,
            neckRadius = 0.12f,
            sideFlexibility = 0.9f,
            topBottomRigidity = 0.7f,
            compressionResistance = 4f,
            breakingForce = 20f
        },
        new BottlePreset {
            name = "硬質ボトル",
            height = 1.0f,
            radius = 0.28f,
            neckHeight = 0.2f,
            neckRadius = 0.1f,
            sideFlexibility = 0.3f,
            topBottomRigidity = 0.95f,
            compressionResistance = 20f,
            breakingForce = 60f
        }
    };
    
    [System.Serializable]
    private struct BottlePreset
    {
        public string name;
        public float height;
        public float radius;
        public float neckHeight;
        public float neckRadius;
        public float sideFlexibility;
        public float topBottomRigidity;
        public float compressionResistance;
        public float breakingForce;
    }
    
    void OnEnable()
    {
        bottleTarget = (PetBottleTarget)target;
    }
    
    public override void OnInspectorGUI()
    {
        // ヘッダー
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("ペットボトル変形システム", EditorStyles.boldLabel);
        EditorGUILayout.Space();
        
        // プリセットセクション
        DrawPresetSection();
        
        EditorGUILayout.Space();
        
        // 基本設定
        DrawBasicSettings();
        
        EditorGUILayout.Space();
        
        // 詳細設定（折りたたみ可能）
        DrawAdvancedSettings();
        
        EditorGUILayout.Space();
        
        // リアルタイム情報
        DrawRuntimeInfo();
        
        EditorGUILayout.Space();
        
        // ユーティリティボタン
        DrawUtilityButtons();
        
        // 変更を適用
        if (GUI.changed)
        {
            EditorUtility.SetDirty(bottleTarget);
        }
    }
    
    private void DrawPresetSection()
    {
        showPresets = EditorGUILayout.Foldout(showPresets, "ボトルプリセット", true);
        if (showPresets)
        {
            EditorGUI.indentLevel++;
            
            EditorGUILayout.HelpBox("プリセットを選択してボトルの種類を素早く設定できます", MessageType.Info);
            
            foreach (var preset in bottlePresets)
            {
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.LabelField(preset.name);
                if (GUILayout.Button("適用", GUILayout.Width(50)))
                {
                    ApplyPreset(preset);
                }
                EditorGUILayout.EndHorizontal();
            }
            
            EditorGUI.indentLevel--;
        }
    }
    
    private void DrawBasicSettings()
    {
        EditorGUILayout.LabelField("基本形状設定", EditorStyles.boldLabel);
        
        bottleTarget.bottleHeight = EditorGUILayout.Slider("ボトル高さ", bottleTarget.bottleHeight, 0.1f, 3f);
        bottleTarget.bottleRadius = EditorGUILayout.Slider("ボトル半径", bottleTarget.bottleRadius, 0.1f, 1f);
        bottleTarget.neckHeight = EditorGUILayout.Slider("首の高さ", bottleTarget.neckHeight, 0.1f, 0.5f);
        bottleTarget.neckRadius = EditorGUILayout.Slider("首の半径", bottleTarget.neckRadius, 0.05f, 0.3f);
        
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("変形特性", EditorStyles.boldLabel);
        
        bottleTarget.sideWallFlexibility = EditorGUILayout.Slider("側面の柔軟性", bottleTarget.sideWallFlexibility, 0f, 1f);
        bottleTarget.topBottomRigidity = EditorGUILayout.Slider("上下の硬さ", bottleTarget.topBottomRigidity, 0f, 1f);
        bottleTarget.neckRigidity = EditorGUILayout.Slider("首の硬さ", bottleTarget.neckRigidity, 0f, 1f);
        bottleTarget.maxSideDeformation = EditorGUILayout.Slider("最大側面変形", bottleTarget.maxSideDeformation, 0f, 1f);
        
        // リアルタイムプレビュー
        if (GUILayout.Button("メッシュを再生成"))
        {
            // プライベートメソッドにアクセスするためのReflection使用
            var method = typeof(PetBottleTarget).GetMethod("CreateBottleMesh", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            method?.Invoke(bottleTarget, null);
        }
    }
    
    private void DrawAdvancedSettings()
    {
        showAdvancedSettings = EditorGUILayout.Foldout(showAdvancedSettings, "詳細設定", true);
        if (showAdvancedSettings)
        {
            EditorGUI.indentLevel++;
            
            EditorGUILayout.LabelField("物理特性", EditorStyles.boldLabel);
            bottleTarget.plasticThickness = EditorGUILayout.FloatField("プラスチック厚み (m)", bottleTarget.plasticThickness);
            bottleTarget.compressionResistance = EditorGUILayout.FloatField("圧縮抵抗", bottleTarget.compressionResistance);
            bottleTarget.breakingForce = EditorGUILayout.FloatField("破損力", bottleTarget.breakingForce);
            bottleTarget.deformationSpeed = EditorGUILayout.Slider("変形速度", bottleTarget.deformationSpeed, 0.1f, 10f);
            
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("視覚・音声", EditorStyles.boldLabel);
            bottleTarget.originalColor = EditorGUILayout.ColorField("基本色", bottleTarget.originalColor);
            bottleTarget.stressColor = EditorGUILayout.ColorField("ストレス色", bottleTarget.stressColor);
            bottleTarget.bottleMaterial = (Material)EditorGUILayout.ObjectField("ボトル材質", bottleTarget.bottleMaterial, typeof(Material), false);
            bottleTarget.enableCrackingSound = EditorGUILayout.Toggle("破損音有効", bottleTarget.enableCrackingSound);
            
            EditorGUILayout.Space();
            EditorGUILayout.LabelField("デバッグ", EditorStyles.boldLabel);
            bottleTarget.enableDebugLogs = EditorGUILayout.Toggle("デバッグログ", bottleTarget.enableDebugLogs);
            bottleTarget.showDeformationGizmos = EditorGUILayout.Toggle("変形ギズモ表示", bottleTarget.showDeformationGizmos);
            
            EditorGUI.indentLevel--;
        }
    }
    
    private void DrawRuntimeInfo()
    {
        if (!Application.isPlaying) return;
        
        EditorGUILayout.LabelField("実行時情報", EditorStyles.boldLabel);
        EditorGUI.BeginDisabledGroup(true);
        
        var state = bottleTarget.GetCurrentState();
        EditorGUILayout.FloatField("現在の変形", state.deformation);
        EditorGUILayout.FloatField("適用力", state.appliedForce);
        EditorGUILayout.Toggle("把持中", state.isBeingGrasped);
        EditorGUILayout.Toggle("破損", state.isBroken);
        EditorGUILayout.IntField("アクティブ変形数", state.activeDeformationCount);
        
        EditorGUI.EndDisabledGroup();
    }
    
    private void DrawUtilityButtons()
    {
        EditorGUILayout.LabelField("ユーティリティ", EditorStyles.boldLabel);
        
        EditorGUILayout.BeginHorizontal();
        
        if (GUILayout.Button("ボトルをリセット"))
        {
            if (Application.isPlaying)
            {
                bottleTarget.ResetBottle();
            }
            else
            {
                EditorUtility.DisplayDialog("情報", "リセットは実行時のみ有効です", "OK");
            }
        }
        
        if (GUILayout.Button("プリファブを作成"))
        {
            CreateBottlePrefab();
        }
        
        EditorGUILayout.EndHorizontal();
        
        EditorGUILayout.BeginHorizontal();
        
        if (GUILayout.Button("テスト用力を適用"))
        {
            if (Application.isPlaying)
            {
                // テスト用の力を適用
                Vector3 testPosition = bottleTarget.transform.position + Vector3.right * 0.2f;
                Vector3 testNormal = Vector3.left;
                bottleTarget.ApplyGripperForceWithDirection(15f, testPosition, testNormal);
            }
            else
            {
                EditorUtility.DisplayDialog("情報", "力の適用は実行時のみ有効です", "OK");
            }
        }
        
        if (GUILayout.Button("設定をコピー"))
        {
            CopySettingsToClipboard();
        }
        
        EditorGUILayout.EndHorizontal();
    }
    
    private void ApplyPreset(BottlePreset preset)
    {
        Undo.RecordObject(bottleTarget, "Apply Bottle Preset");
        
        bottleTarget.bottleHeight = preset.height;
        bottleTarget.bottleRadius = preset.radius;
        bottleTarget.neckHeight = preset.neckHeight;
        bottleTarget.neckRadius = preset.neckRadius;
        bottleTarget.sideWallFlexibility = preset.sideFlexibility;
        bottleTarget.topBottomRigidity = preset.topBottomRigidity;
        bottleTarget.compressionResistance = preset.compressionResistance;
        bottleTarget.breakingForce = preset.breakingForce;
        
        EditorUtility.SetDirty(bottleTarget);
        
        // メッシュを再生成
        if (Application.isPlaying)
        {
            var method = typeof(PetBottleTarget).GetMethod("CreateBottleMesh", 
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
            method?.Invoke(bottleTarget, null);
        }
        
        Debug.Log($"プリセット '{preset.name}' を適用しました");
    }
    
    private void CreateBottlePrefab()
    {
        string path = EditorUtility.SaveFilePanelInProject(
            "ペットボトルプリファブを保存", 
            "PetBottleTarget", 
            "prefab",
            "プリファブの保存場所を選択してください");
        
        if (!string.IsNullOrEmpty(path))
        {
            GameObject prefab = PrefabUtility.SaveAsPrefabAsset(bottleTarget.gameObject, path);
            EditorGUIUtility.PingObject(prefab);
            Debug.Log($"ペットボトルプリファブを作成しました: {path}");
        }
    }
    
    private void CopySettingsToClipboard()
    {
        string settings = $@"// ペットボトル設定
bottleHeight = {bottleTarget.bottleHeight}f;
bottleRadius = {bottleTarget.bottleRadius}f;
neckHeight = {bottleTarget.neckHeight}f;
neckRadius = {bottleTarget.neckRadius}f;
sideWallFlexibility = {bottleTarget.sideWallFlexibility}f;
topBottomRigidity = {bottleTarget.topBottomRigidity}f;
neckRigidity = {bottleTarget.neckRigidity}f;
maxSideDeformation = {bottleTarget.maxSideDeformation}f;
compressionResistance = {bottleTarget.compressionResistance}f;
breakingForce = {bottleTarget.breakingForce}f;";
        
        EditorGUIUtility.systemCopyBuffer = settings;
        Debug.Log("設定をクリップボードにコピーしました");
    }
    
    // Scene viewでのギズモ描画
    void OnSceneGUI()
    {
        if (bottleTarget == null) return;
        
        // ボトルの概要を描画
        Handles.color = Color.cyan;
        Vector3 center = bottleTarget.transform.position;
        Vector3 size = new Vector3(bottleTarget.bottleRadius * 2, bottleTarget.bottleHeight, bottleTarget.bottleRadius * 2);
        
        // ボトル本体の概要
        Handles.DrawWireCube(center + Vector3.up * bottleTarget.bottleHeight / 2, size);
        
        // 首部分の概要
        Vector3 neckCenter = center + Vector3.up * (bottleTarget.bottleHeight - bottleTarget.neckHeight / 2);
        Vector3 neckSize = new Vector3(bottleTarget.neckRadius * 2, bottleTarget.neckHeight, bottleTarget.neckRadius * 2);
        Handles.color = Color.yellow;
        Handles.DrawWireCube(neckCenter, neckSize);
        
        // ラベル表示
        Handles.Label(center + Vector3.up * bottleTarget.bottleHeight + Vector3.right * 0.3f, 
                     $"ペットボトル\n高さ: {bottleTarget.bottleHeight:F2}m\n半径: {bottleTarget.bottleRadius:F2}m");
    }
}

#endif