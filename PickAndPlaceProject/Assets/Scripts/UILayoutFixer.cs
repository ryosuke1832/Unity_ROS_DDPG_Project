using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// UI要素の重なりを修正するためのスクリプト
/// </summary>
public class UILayoutFixer : MonoBehaviour
{
    [Header("UI修正設定")]
    [SerializeField] private bool autoFixOnStart = true;
    [SerializeField] private float panelSpacing = 20f;
    
    void Start()
    {
        if (autoFixOnStart)
        {
            FixUILayout();
        }
    }
    
    /// <summary>
    /// UI配置を修正
    /// </summary>
    [ContextMenu("Fix UI Layout")]
    public void FixUILayout()
    {
        Debug.Log("UI配置を修正中...");
        
        // 既存のCanvasを見つける
        Canvas[] canvases = FindObjectsOfType<Canvas>();
        foreach (Canvas canvas in canvases)
        {
            FixCanvasLayout(canvas);
        }
        
        Debug.Log("UI配置修正完了");
    }
    
    /// <summary>
    /// Canvas内のレイアウトを修正
    /// </summary>
    private void FixCanvasLayout(Canvas canvas)
    {
        // ROSコントロールパネル（左上）
        FixPanelPosition("ControlPanel", canvas.transform, 
                        new Vector2(0f, 1f), new Vector2(0f, 1f), 
                        new Vector2(175f, -125f));
        
        // フィードバックパネル（右上）
        FixPanelPosition("FeedbackPanel", canvas.transform, 
                        new Vector2(1f, 1f), new Vector2(1f, 1f), 
                        new Vector2(-175f, -125f));
        
        // データパネル（左下）
        FixPanelPosition("DataPanel", canvas.transform, 
                        new Vector2(0f, 0f), new Vector2(0f, 0f), 
                        new Vector2(175f, 125f));
        
        // デバッグパネル（右下）
        FixPanelPosition("DebugPanel", canvas.transform, 
                        new Vector2(1f, 0f), new Vector2(1f, 0f), 
                        new Vector2(-175f, 125f));
                        
        // ROSステータス表示（左側中央）
        FixROSStatusDisplay(canvas.transform);
    }
    
    /// <summary>
    /// パネルの位置を修正
    /// </summary>
    private void FixPanelPosition(string panelName, Transform parent, 
                                 Vector2 anchorMin, Vector2 anchorMax, 
                                 Vector2 anchoredPosition)
    {
        Transform panel = FindChildByName(parent, panelName);
        if (panel != null)
        {
            RectTransform rect = panel.GetComponent<RectTransform>();
            if (rect != null)
            {
                rect.anchorMin = anchorMin;
                rect.anchorMax = anchorMax;
                rect.anchoredPosition = anchoredPosition;
                
                // パネルサイズも調整
                rect.sizeDelta = new Vector2(350, 250);
                
                Debug.Log($"{panelName}の位置を修正しました");
            }
        }
    }
    
    /// <summary>
    /// ROSステータス表示の修正
    /// </summary>
    private void FixROSStatusDisplay(Transform canvasTransform)
    {
        // ROSステータステキストを縦に並べ直す
        string[] rosElements = {
            "ROSConnectionStatus", "ROSIPText", "ForceText", 
            "PositionText", "OrientationText", "StateText"
        };
        
        float startY = 200f;
        float spacing = 30f;
        
        for (int i = 0; i < rosElements.Length; i++)
        {
            Transform element = FindChildByName(canvasTransform, rosElements[i]);
            if (element != null)
            {
                RectTransform rect = element.GetComponent<RectTransform>();
                if (rect != null)
                {
                    // 左上アンカー
                    rect.anchorMin = new Vector2(0f, 1f);
                    rect.anchorMax = new Vector2(0f, 1f);
                    rect.anchoredPosition = new Vector2(150f, -(startY + i * spacing));
                    rect.sizeDelta = new Vector2(280, 25);
                    
                    // テキストの設定を調整
                    Text textComponent = element.GetComponent<Text>();
                    if (textComponent != null)
                    {
                        textComponent.alignment = TextAnchor.MiddleLeft;
                        textComponent.fontSize = 12;
                        
                        // アウトラインを追加して視認性向上
                        if (element.GetComponent<Outline>() == null)
                        {
                            Outline outline = element.gameObject.AddComponent<Outline>();
                            outline.effectColor = Color.black;
                            outline.effectDistance = new Vector2(1, 1);
                        }
                    }
                }
            }
        }
    }
    
    /// <summary>
    /// 名前で子オブジェクトを検索
    /// </summary>
    private Transform FindChildByName(Transform parent, string name)
    {
        // 直接の子を検索
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform child = parent.GetChild(i);
            if (child.name == name)
                return child;
        }
        
        // 再帰的に孫以降も検索
        for (int i = 0; i < parent.childCount; i++)
        {
            Transform found = FindChildByName(parent.GetChild(i), name);
            if (found != null)
                return found;
        }
        
        return null;
    }
    
    /// <summary>
    /// 画面サイズに応じた動的調整
    /// </summary>
    public void AdjustForScreenSize()
    {
        float screenWidth = Screen.width;
        float screenHeight = Screen.height;
        
        // 小さい画面の場合はパネルサイズを縮小
        if (screenWidth < 1200 || screenHeight < 800)
        {
            Canvas[] canvases = FindObjectsOfType<Canvas>();
            foreach (Canvas canvas in canvases)
            {
                CanvasScaler scaler = canvas.GetComponent<CanvasScaler>();
                if (scaler != null)
                {
                    scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
                    scaler.referenceResolution = new Vector2(1920, 1080);
                    scaler.matchWidthOrHeight = 0.5f;
                }
            }
        }
    }
}