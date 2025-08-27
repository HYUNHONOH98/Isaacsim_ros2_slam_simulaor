#!/bin/bash
shopt -s expand_aliases
source ~/.bashrc   # alias 선언이 여기에 있을 때

# Trap Ctrl+C (SIGINT)
cleanup() {
    echo "Caught Ctrl+C, cleaning up..."
    # Kill all background jobs started by this script
    pkill -P $$
    exit 1
}


/workspace/isaaclab/_isaac_sim/python.sh utils/state_pub.py &
ros2 launch utils/state_pub.launch.py &
# python3 source/ros2/isaac-ros/utils/static_node_v2.py &
/workspace/isaaclab/_isaac_sim/python.sh utils/static_node.py &
/workspace/isaaclab/_isaac_sim/python.sh utils/analysis_node.py &
# ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true
ros2 launch fast_lio mapping.launch.py config_file:=ouster64.yaml rviz:=false &
# ros2 run rviz2 rviz2 --ros-args -p use_sim_time:=true


wait