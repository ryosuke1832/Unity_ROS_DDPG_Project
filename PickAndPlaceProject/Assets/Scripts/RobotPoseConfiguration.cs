using System.Collections;
using UnityEngine;

/// <summary>
/// ロボットの設定姿勢管理システム
/// 複数の定義済み姿勢とカスタム姿勢をサポート
/// </summary>
[System.Serializable]
public class RobotPoseConfiguration : MonoBehaviour
{
    [Header("ロボット参照")]
    public GameObject niryoOneRobot;
    public TrajectoryPlanner trajectoryPlanner;
    
    [Header("定義済み姿勢")]
    public PredefinedPoses predefinedPose = PredefinedPoses.Home;
    
    [Header("カスタム関節角度 (度)")]
    [Range(-180f, 180f)]
    public float joint1Angle = 0f;
    [Range(-180f, 180f)]
    public float joint2Angle = 0f;
    [Range(-180f, 180f)]
    public float joint3Angle = 0f;
    [Range(-180f, 180f)]
    public float joint4Angle = 0f;
    [Range(-180f, 180f)]
    public float joint5Angle = 0f;
    [Range(-180f, 180f)]
    public float joint6Angle = 0f;
    
    [Header("設定")]
    public bool useCustomAngles = false;
    public float transitionSpeed = 2f;
    public bool enableLogging = true;
    
    // 内部変数
    private ArticulationBody[] jointArticulationBodies;
    private const int NUM_ROBOT_JOINTS = 6;
    
    // 定義済み姿勢の列挙
    public enum PredefinedPoses
    {
        Home,           // ホーム位置
        Ready,          // 準備位置
        PickupReady,    // ピックアップ準備
        Observation,    // 観察位置
        Rest,           // 休息位置
        Calibration     // キャリブレーション位置
    }
    
    void Start()
    {
        InitializeRobotJoints();
    }
    
    /// <summary>
    /// ロボット関節の初期化
    /// </summary>
    private void InitializeRobotJoints()
    {
        if (niryoOneRobot == null && trajectoryPlanner != null)
        {
            niryoOneRobot = trajectoryPlanner.NiryoOne;
        }
        
        if (niryoOneRobot == null)
        {
            Debug.LogError("NiryoOneロボットが見つかりません");
            return;
        }
        
        // 関節のArticulationBodyを取得
        jointArticulationBodies = new ArticulationBody[NUM_ROBOT_JOINTS];
        string linkName = "";
        
        // TrajectoryPlannerと同じ方法で関節を取得
        for (int i = 0; i < NUM_ROBOT_JOINTS; i++)
        {
            linkName += GetLinkName(i);
            Transform jointTransform = niryoOneRobot.transform.Find(linkName);
            
            if (jointTransform != null)
            {
                jointArticulationBodies[i] = jointTransform.GetComponent<ArticulationBody>();
            }
            else
            {
                Debug.LogError($"関節{i + 1}が見つかりません: {linkName}");
            }
        }
        
        if (enableLogging)
        {
            Debug.Log($"ロボット関節初期化完了: {NUM_ROBOT_JOINTS}個の関節");
        }
    }
    
    /// <summary>
    /// リンク名を取得（TrajectoryPlannerのSourceDestinationPublisher.LinkNamesと同等）
    /// </summary>
    private string GetLinkName(int jointIndex)
    {
        string[] linkNames = {
            "/base_link/shoulder_link",
            "/arm_link",
            "/elbow_link",
            "/forearm_link",
            "/wrist_link",
            "/hand_link"
        };
        
        return jointIndex < linkNames.Length ? linkNames[jointIndex] : "";
    }
    
    /// <summary>
    /// 設定姿勢を適用
    /// </summary>
    public void ApplyPose()
    {
        if (jointArticulationBodies == null)
        {
            InitializeRobotJoints();
        }
        
        float[] targetAngles;
        
        if (useCustomAngles)
        {
            // カスタム角度を使用
            targetAngles = GetCustomAngles();
            if (enableLogging)
                Debug.Log("カスタム姿勢を適用中...");
        }
        else
        {
            // 定義済み姿勢を使用
            targetAngles = GetPredefinedPoseAngles(predefinedPose);
            if (enableLogging)
                Debug.Log($"定義済み姿勢を適用中: {predefinedPose}");
        }
        
        StartCoroutine(TransitionToPose(targetAngles));
    }
    
    /// <summary>
    /// カスタム関節角度を取得
    /// </summary>
    private float[] GetCustomAngles()
    {
        return new float[]
        {
            joint1Angle,
            joint2Angle,
            joint3Angle,
            joint4Angle,
            joint5Angle,
            joint6Angle
        };
    }
    
    /// <summary>
    /// 定義済み姿勢の関節角度を取得
    /// </summary>
    private float[] GetPredefinedPoseAngles(PredefinedPoses pose)
    {
        switch (pose)
        {
            case PredefinedPoses.Home:
                return new float[] { 0f, 0f, 0f, 0f, 0f, 0f };
                
            case PredefinedPoses.Ready:
                return new float[] { 0f, -45f, 45f, 0f, 0f, 0f };
                
            case PredefinedPoses.PickupReady:
                return new float[] { 0f, -30f, 60f, 0f, -30f, 0f };
                
            case PredefinedPoses.Observation:
                return new float[] { 0f, -60f, 90f, 0f, -30f, 0f };
                
            case PredefinedPoses.Rest:
                return new float[] { 0f, -90f, 90f, 0f, 0f, 0f };
                
            case PredefinedPoses.Calibration:
                return new float[] { 90f, 0f, 0f, 0f, 90f, 0f };
                
            default:
                return new float[] { 0f, 0f, 0f, 0f, 0f, 0f };
        }
    }
    
    /// <summary>
    /// 指定姿勢への滑らかな遷移
    /// </summary>
    private IEnumerator TransitionToPose(float[] targetAngles)
    {
        if (jointArticulationBodies == null || targetAngles.Length != NUM_ROBOT_JOINTS)
        {
            Debug.LogError("関節設定エラー");
            yield break;
        }
        
        // 現在の関節角度を取得
        float[] startAngles = new float[NUM_ROBOT_JOINTS];
        for (int i = 0; i < NUM_ROBOT_JOINTS; i++)
        {
            if (jointArticulationBodies[i] != null)
            {
                startAngles[i] = jointArticulationBodies[i].jointPosition[0] * Mathf.Rad2Deg;
            }
        }
        
        float elapsedTime = 0f;
        float duration = 1f / transitionSpeed;
        
        while (elapsedTime < duration)
        {
            float progress = elapsedTime / duration;
            progress = Mathf.SmoothStep(0f, 1f, progress); // 滑らかな補間
            
            for (int i = 0; i < NUM_ROBOT_JOINTS; i++)
            {
                if (jointArticulationBodies[i] != null)
                {
                    float currentAngle = Mathf.Lerp(startAngles[i], targetAngles[i], progress);
                    var drive = jointArticulationBodies[i].xDrive;
                    drive.target = currentAngle * Mathf.Deg2Rad; // ラジアンに変換
                    jointArticulationBodies[i].xDrive = drive;
                }
            }
            
            elapsedTime += Time.deltaTime;
            yield return null;
        }
        
        // 最終位置を確実に設定
        for (int i = 0; i < NUM_ROBOT_JOINTS; i++)
        {
            if (jointArticulationBodies[i] != null)
            {
                var drive = jointArticulationBodies[i].xDrive;
                drive.target = targetAngles[i] * Mathf.Deg2Rad;
                jointArticulationBodies[i].xDrive = drive;
            }
        }
        
        if (enableLogging)
        {
            Debug.Log($"姿勢遷移完了: [{string.Join(", ", targetAngles)}]");
        }
    }
    
    /// <summary>
    /// 現在の関節角度を取得
    /// </summary>
    public float[] GetCurrentJointAngles()
    {
        if (jointArticulationBodies == null) return null;
        
        float[] angles = new float[NUM_ROBOT_JOINTS];
        for (int i = 0; i < NUM_ROBOT_JOINTS; i++)
        {
            if (jointArticulationBodies[i] != null)
            {
                angles[i] = jointArticulationBodies[i].jointPosition[0] * Mathf.Rad2Deg;
            }
        }
        return angles;
    }
    
    /// <summary>
    /// 現在の姿勢を保存（カスタム角度として）
    /// </summary>
    public void SaveCurrentPoseAsCustom()
    {
        float[] currentAngles = GetCurrentJointAngles();
        if (currentAngles != null)
        {
            joint1Angle = currentAngles[0];
            joint2Angle = currentAngles[1];
            joint3Angle = currentAngles[2];
            joint4Angle = currentAngles[3];
            joint5Angle = currentAngles[4];
            joint6Angle = currentAngles[5];
            
            useCustomAngles = true;
            
            if (enableLogging)
            {
                Debug.Log($"現在の姿勢をカスタム角度として保存: [{string.Join(", ", currentAngles)}]");
            }
        }
    }
    
    // Unity Editorでのボタン機能（Inspector上で使用可能）
    [ContextMenu("姿勢を適用")]
    public void ApplyPoseFromEditor()
    {
        ApplyPose();
    }
    
    [ContextMenu("現在姿勢を保存")]
    public void SaveCurrentPoseFromEditor()
    {
        SaveCurrentPoseAsCustom();
    }
    
    [ContextMenu("ホーム位置に移動")]
    public void GoToHome()
    {
        predefinedPose = PredefinedPoses.Home;
        useCustomAngles = false;
        ApplyPose();
    }
}