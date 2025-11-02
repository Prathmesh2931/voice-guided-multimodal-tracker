import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# TURTLEBOT3_MODEL = os.environ['TURTLEBOT3_MODEL']
TURTLEBOT3_MODEL = os.environ.get('TURTLEBOT3_MODEL', 'waffle')


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    world_name = LaunchConfiguration('world_name', default='turtlebot3_world')

    launch_file_dir = os.path.join(get_package_share_directory('turtlebot3'), 'launch')
    rviz_config_dir = os.path.join(get_package_share_directory('turtlebot3'), 'rviz', 'rviz.rviz')

    ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=[
            os.path.join("/opt/ros/humble", "share"),
            ":" + os.path.join(get_package_share_directory('turtlebot3'), "models")
        ]
    )

    # Spawn robot
    ignition_spawn_entity = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=[
            '-entity', TURTLEBOT3_MODEL,
            '-name', TURTLEBOT3_MODEL,
            '-file', PathJoinSubstitution([
                get_package_share_directory('turtlebot3'),
                "models", "turtlebot3", "model.sdf"
            ]),
            '-allow_renaming', 'true',
            '-x', '-2.0',
            '-y', '-0.5',
            '-z', '0.01'
        ]
    )

    # Spawn world
    ignition_spawn_world = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=[
            '-file', PathJoinSubstitution([
                get_package_share_directory('turtlebot3'),
                "models", "worlds", "model.sdf"
            ]),
            '-allow_renaming', 'false'
        ]
    )

    world_only = os.path.join(
        get_package_share_directory('turtlebot3'),
        "models", "worlds", "world_only.sdf"
    )

    # YOLO Object Tracker Node
    yolo_tracker_node = Node(
        package='object_tracking',
        executable='yolo_object_tracker.py',
        name='yolo_object_tracker',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    # Voice Command Interface Node
    voice_interface_node = Node(
        package='object_tracking',
        executable='voice_cmd_interface.py',
        name='voice_cmd_interface',
        output='screen'
    )

    # RViz node with delay
    delayed_rviz = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_config_dir],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen'
            )
        ]
    )

    red_box = Node(
            package='ros_ign_gazebo',
            executable='create',
            output='screen',
            arguments=['-entity', 'pickup_box',
                    '-name', 'pickup_box',
                    '-file', PathJoinSubstitution([
                            get_package_share_directory('turtlebot3'),
                            "models","box" ,"pickup_box.sdf"]),
                    '-allow_renaming', 'true',
                    '-x', '-0.0',
                    '-y', '-0.4',
                    '-z', '0.1'],
            )
    blue_box = Node(
        package='ros_ign_gazebo',
        executable='create',
        output='screen',
        arguments=['-entity', 'pickup_box_blue',
                '-name', 'pickup_box_blue',
                '-file', PathJoinSubstitution([
                        get_package_share_directory('turtlebot3'),
                        "models","box" ,"blue_box.sdf"]),
                '-allow_renaming', 'true',
                '-x', '-1.3',
                '-y', '-1.4',
                '-z', '0.1'],
        )
    # CHANGE ONLY THIS NUMBER TO SCALE THE ENTIRE CHAIR!
    SCALE = 1.0  # 0.5 = small, 1.0 = normal, 2.0 = big

    chair = Node(
    package='ros_ign_gazebo',
    executable='create',
    output='screen',
    arguments=['-entity', 'chair',
            '-name', 'chair',
            '-file', PathJoinSubstitution([
                    get_package_share_directory('turtlebot3'),
                    "models","chair" ,"chair.sdf"]),
            '-allow_renaming', 'true',
            '-x', '-1.3',
            '-y', '0.4',
            '-z', '0.1'],
    )
    
    
    # Process XACRO file
    robot_description = Command([
        'xacro ', 
        PathJoinSubstitution([
            FindPackageShare('turtlebot3'),
            'models', 'chair', 'chair.sdf.xacro'
        ]),
        ' scale:=', LaunchConfiguration('scale')
    ])
    
    scale_arg = DeclareLaunchArgument(
        'scale',
        default_value='1.0',
        description='Scale factor for the chair'
    )

    # Dynamically generate SDF from Python
    chair_sdf_content = Command([
        'python3 ',
        PathJoinSubstitution([
            FindPackageShare('turtlebot3'),
            'models', 'chair', 'generate_chair_sdf.py'
        ]),
        ' ', LaunchConfiguration('scale')
    ])

    # Spawn the generated model
    # chair = Node(
    #     package='ros_ign_gazebo',
    #     executable='create',
    #     output='screen',
    #     arguments=[
    #         '-entity', 'chair',
    #         '-name', 'chair',
    #         '-string', chair_sdf_content,
    #         '-allow_renaming', 'true',
    #         '-x', '-1.3',
    #         '-y', '0.4',
    #         '-z', '0.1'
    #     ],
    # )


    return LaunchDescription([
        ign_resource_path,
        ignition_spawn_entity,
        # ignition_spawn_world,
        red_box,
        blue_box,
        scale_arg,
        chair,

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(get_package_share_directory('ros_ign_gazebo'),
                             'launch', 'ign_gazebo.launch.py')
            ]),
            launch_arguments=[('ign_args', [' -r -v 3 ' + world_only])]),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/ros_ign_bridge.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([launch_file_dir, '/robot_state_publisher.launch.py']),
            launch_arguments={'use_sim_time': use_sim_time}.items(),
        ),
       

        yolo_tracker_node,
        # voice_interface_node,
        delayed_rviz,

        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

    ])