using System;
using System.Collections;
using System.Linq;
using RosMessageTypes.Geometry;
using RosMessageTypes.NiryoMoveit;
using Unity.Robotics.ROSTCPConnector;
using Unity.Robotics.ROSTCPConnector.ROSGeometry;
using UnityEngine;

public class TrajectoryPlanner : MonoBehaviour
{
    // Hardcoded variables
    const int k_NumRobotJoints = 6;
    const float k_JointAssignmentWait = 0.1f;
    const float k_PoseAssignmentWait = 0.5f;

    // Variables required for ROS communication
    [SerializeField]
    string m_RosServiceName = "niryo_moveit";
    public string RosServiceName { get => m_RosServiceName; set => m_RosServiceName = value; }

    [SerializeField]
    GameObject m_NiryoOne;
    public GameObject NiryoOne { get => m_NiryoOne; set => m_NiryoOne = value; }
    [SerializeField]
    GameObject m_Target;
    public GameObject Target { get => m_Target; set => m_Target = value; }
    [SerializeField]
    GameObject m_TargetPlacement;
    public GameObject TargetPlacement { get => m_TargetPlacement; set => m_TargetPlacement = value; }

    // 変形機能用の設定
    [Header("=== 変形機能設定 ===")]
    [SerializeField, Tooltip("変形機能を有効にする")]
    private bool enableDeformation = true;
    [SerializeField, Range(0f, 1f), Tooltip("ターゲットの柔軟性")]
    private float targetSoftness = 0.5f;
    [SerializeField, Tooltip("変形タイプ")]
    private DeformableTarget.DeformationType deformationType = DeformableTarget.DeformationType.Squeeze;
    [SerializeField, Range(5f, 50f), Tooltip("把持力 (N)")]
    private float gripForce = 15f;

    // Assures that the gripper is always positioned above the m_Target cube before grasping.
    readonly Quaternion m_PickOrientation = Quaternion.Euler(90, 90, 0);
    readonly Vector3 m_PickPoseOffset = Vector3.up * 0.1f;

    // Articulation Bodies
    ArticulationBody[] m_JointArticulationBodies;
    ArticulationBody m_LeftGripper;
    ArticulationBody m_RightGripper;

    // ROS Connector
    ROSConnection m_Ros;

    // 変形機能参照
    private DeformableTarget m_DeformableTarget;

    /// <summary>
    ///     Find all robot joints in Start() and add them to the jointArticulationBodies array.
    ///     Find left and right finger joints and assign them to their respective articulation body objects.
    /// </summary>
    void Start()
    {
        // Get ROS connection static instance
        m_Ros = ROSConnection.GetOrCreateInstance();
        m_Ros.RegisterRosService<MoverServiceRequest, MoverServiceResponse>(m_RosServiceName);

        m_JointArticulationBodies = new ArticulationBody[k_NumRobotJoints];

        var linkName = string.Empty;
        for (var i = 0; i < k_NumRobotJoints; i++)
        {
            linkName += SourceDestinationPublisher.LinkNames[i];
            m_JointArticulationBodies[i] = m_NiryoOne.transform.Find(linkName).GetComponent<ArticulationBody>();
        }

        // Find left and right fingers
        var rightGripper = linkName + "/tool_link/gripper_base/servo_head/control_rod_right/right_gripper";
        var leftGripper = linkName + "/tool_link/gripper_base/servo_head/control_rod_left/left_gripper";

        m_RightGripper = m_NiryoOne.transform.Find(rightGripper).GetComponent<ArticulationBody>();
        m_LeftGripper = m_NiryoOne.transform.Find(leftGripper).GetComponent<ArticulationBody>();

        // 変形機能の初期化
        if (enableDeformation)
        {
            InitializeDeformationFeatures();
        }
    }

    /// <summary>
    /// 変形機能の初期化
    /// </summary>
    private void InitializeDeformationFeatures()
    {
        if (m_Target == null) return;

        // ターゲットにDeformableTargetコンポーネントを追加
        m_DeformableTarget = m_Target.GetComponent<DeformableTarget>();
        if (m_DeformableTarget == null)
        {
            m_DeformableTarget = m_Target.AddComponent<DeformableTarget>();
            Debug.Log("DeformableTargetコンポーネントをメインターゲットに追加しました");
        }

        // 設定を適用
        m_DeformableTarget.SetSoftness(targetSoftness);
        m_DeformableTarget.SetDeformationType(deformationType);

        Debug.Log($"変形機能が有効化されました - 柔軟性: {targetSoftness}, タイプ: {deformationType}, 把持力: {gripForce}N");
    }

    /// <summary>
    ///     Close the gripper
    /// </summary>
    void CloseGripper()
    {
        var leftDrive = m_LeftGripper.xDrive;
        var rightDrive = m_RightGripper.xDrive;

        leftDrive.target = -0.01f;
        rightDrive.target = 0.01f;

        m_LeftGripper.xDrive = leftDrive;
        m_RightGripper.xDrive = rightDrive;

        // 変形機能対応の把持開始
        if (enableDeformation && m_DeformableTarget != null)
        {
            // グリッパー中央位置と方向を計算
            Vector3 graspPosition = (m_LeftGripper.transform.position + m_RightGripper.transform.position) / 2f;
            Vector3 graspDirection = (m_RightGripper.transform.position - m_LeftGripper.transform.position).normalized;
            
            // 変形ターゲットに把持開始を通知
            m_DeformableTarget.StartGrasping(gripForce, graspPosition, graspDirection);
            
            Debug.Log($"変形機能付き把持開始 - 力: {gripForce}N");
        }
    }

    /// <summary>
    ///     Open the gripper
    /// </summary>
    void OpenGripper()
    {
        var leftDrive = m_LeftGripper.xDrive;
        var rightDrive = m_RightGripper.xDrive;

        leftDrive.target = 0.01f;
        rightDrive.target = -0.01f;

        m_LeftGripper.xDrive = leftDrive;
        m_RightGripper.xDrive = rightDrive;

        // 変形機能対応の把持停止
        if (enableDeformation && m_DeformableTarget != null)
        {
            // 変形ターゲットに把持停止を通知
            m_DeformableTarget.StopGrasping();
            
            Debug.Log("変形機能付き把持停止");
        }
    }

    /// <summary>
    /// 把持力を動的に設定
    /// </summary>
    public void SetGripForce(float force)
    {
        gripForce = Mathf.Clamp(force, 5f, 50f);
        if (m_DeformableTarget != null)
        {
            m_DeformableTarget.UpdateGraspForce(gripForce);
        }
    }

    /// <summary>
    /// 変形タイプを動的に変更
    /// </summary>
    public void SetDeformationType(DeformableTarget.DeformationType type)
    {
        deformationType = type;
        if (m_DeformableTarget != null)
        {
            m_DeformableTarget.SetDeformationType(type);
        }
    }

    /// <summary>
    /// 柔軟性を動的に変更
    /// </summary>
    public void SetTargetSoftness(float softness)
    {
        targetSoftness = Mathf.Clamp01(softness);
        if (m_DeformableTarget != null)
        {
            m_DeformableTarget.SetSoftness(targetSoftness);
        }
    }

    /// <summary>
    ///     Get the current values of the robot's joint angles.
    /// </summary>
    /// <returns>NiryoMoveitJoints</returns>
    NiryoMoveitJointsMsg CurrentJointConfig()
    {
        var joints = new NiryoMoveitJointsMsg();

        for (var i = 0; i < k_NumRobotJoints; i++)
        {
            joints.joints[i] = m_JointArticulationBodies[i].jointPosition[0];
        }

        return joints;
    }

    /// <summary>
    ///     Create a new MoverServiceRequest with the current values of the robot's joint angles,
    ///     the target cube's current position and rotation, and the targetPlacement position and rotation.
    ///     Call the MoverService using Unity's ServiceConnection.
    /// </summary>
    public void PublishJoints()
    {
        var request = new MoverServiceRequest();
        request.joints_input = CurrentJointConfig();

        // Pick Pose
        request.pick_pose = new PoseMsg
        {
            position = (m_Target.transform.position + m_PickPoseOffset).To<FLU>(),

            // The hardcoded x/z angles assure that the gripper is always positioned above the target cube before grasping.
            orientation = Quaternion.Euler(90, m_Target.transform.eulerAngles.y, 0).To<FLU>()
        };

        // Place Pose
        request.place_pose = new PoseMsg
        {
            position = (m_TargetPlacement.transform.position + m_PickPoseOffset).To<FLU>(),
            orientation = m_PickOrientation.To<FLU>()
        };

        m_Ros.SendServiceMessage<MoverServiceResponse>(m_RosServiceName, request, TrajectoryResponse);
    }

    void TrajectoryResponse(MoverServiceResponse response)
    {
        if (response.trajectories.Length > 0)
        {
            Debug.Log("Trajectory returned.");
            StartCoroutine(ExecuteTrajectories(response));
        }
        else
        {
            Debug.LogError("No trajectory returned from MoverService.");
        }
    }

    /// <summary>
    ///     Execute the returned trajectories from the MoverService.
    ///     The expectation is that the MoverService will return four trajectory plans,
    ///     PreGrasp, Grasp, PickUp, and Place,
    ///     where each plan is an array of robot poses. A robot pose is the joint angle values
    ///     of the six robot joints.
    ///     Executing a single trajectory will iterate through every robot pose in the array while updating the
    ///     joint values on the robot.
    /// </summary>
    /// <param name="response"> MoverServiceResponse received from niryo_moveit mover service running in ROS</param>
    /// <returns></returns>
    private IEnumerator ExecuteTrajectories(MoverServiceResponse response)
    {
        if (response.trajectories != null)
        {
            // For every trajectory plan returned
            for (var poseIndex = 0; poseIndex < response.trajectories.Length; poseIndex++)
            {
                // For every robot pose in trajectory plan
                foreach (var t in response.trajectories[poseIndex].joint_trajectory.points)
                {
                    var jointPositions = t.positions;
                    var result = jointPositions.Select(r => (float)r * Mathf.Rad2Deg).ToArray();

                    // Set the joint values for every joint
                    for (var joint = 0; joint < m_JointArticulationBodies.Length; joint++)
                    {
                        var joint1XDrive = m_JointArticulationBodies[joint].xDrive;
                        joint1XDrive.target = result[joint];
                        m_JointArticulationBodies[joint].xDrive = joint1XDrive;
                    }
                    // Wait for robot to achieve pose for all joint assignments
                    yield return new WaitForSeconds(k_JointAssignmentWait);
                }

                // Close the gripper if completed executing the trajectory for the Grasp pose
                if (poseIndex == (int)Poses.Grasp)
                {
                    CloseGripper();
                }
                // Open the gripper if completed executing the trajectory for the Place pose
                if (poseIndex == (int)Poses.Place)
                {
                    OpenGripper();
                }
                // Wait for the robot to achieve the final pose from joint assignment
                yield return new WaitForSeconds(k_PoseAssignmentWait);
            }
        }
    }

    enum Poses
    {
        PreGrasp,
        Grasp,
        PickUp,
        Place
    }
}