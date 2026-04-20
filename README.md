# Isaac Sim ROS2 SLAM Simulator (G1)

This repository is a project-focused Isaac Lab fork for **G1 locomotion + LiDAR + SLAM** experiments.

- Base: Isaac Lab `v2.1.0`
- Isaac Sim: `4.5.0`
- Main working directory: `source/ros2/isaac-ros`

Primary experiment script:
- `source/ros2/isaac-ros/g1_standalone_v2_14dof_nav_heuristic_test_midsole.py`

## What Is In `source/ros2/isaac-ros`

### Main launch scripts

- `g1_v1.sh`: run `g1_standalone_v1.py` (legacy locomotion policy).
- `g1_v2.sh`: run `g1_standalone_v2.py` (updated locomotion policy).
- `run.sh`: start ROS2-side runtime nodes for state publishing, TF bridging, Fast-LIO launch, and RViz.

### Key standalone simulators

- `g1_standalone_v2.py`: locomotion + sensor publishing (`/points`, `imu/data`, `/g1/odom`).
- `g1_standalone_v2_nav.py`: locomotion + SLAM-feedback navigation policy (`est_pelvis` based).
- `g1_standalone_v2_nav_heuristic*.py`: heuristic/test variants with additional logging.
- `g1_standalone_v2_14dof*.py`: 14-DoF controller variants.
- `g1_standalone_v2_14dof_nav_heuristic_test_midsole.py`: main experiment entrypoint in this project.

### ROS utility nodes (`utils/`)

- `lidar_node.py`: publishes LiDAR point cloud as `PointCloud2` on `points`.
- `imu_node.py`: publishes IMU on `imu/data`.
- `odom_node.py`: publishes odometry on `/g1/odom` and TF `odom -> pelvis`.
- `state_pub.py`: converts `/g1/joint_states` to `joint_states` for `robot_state_publisher`.
- `state_pub.launch.py`: launches `robot_state_publisher` with `g1_29dof_rev_1_0_lidar.urdf`.
- `static_node.py`: publishes required static/initial transforms used by SLAM pipeline.
- `slam_subscribe_node.py`: reads SLAM-estimated transforms from TF.
- `lidar_tf_node.py`: computes/publishes `odom -> est_pelvis` from SLAM frame + kinematic transform.
- `analysis_node.py`: logs SLAM-vs-GT error and saves plots under `g1_data/exp_log/`.

## ROS Interfaces Used By This Project

- Point cloud: `points`
- IMU: `imu/data`
- Robot odom: `/g1/odom`
- Joint states in: `/g1/joint_states`
- Joint states out: `joint_states`
- Sim clock: `/clock`
- Important TF frames: `odom`, `body`, `pelvis`, `mid360_link`, `mid360_link_frame`, `lidar_sensor`, `est_pelvis`

## Development Environment Setup (No Existing ROS2 Assumed)

This section is the recommended path for a new machine: **build Docker container first, then use ROS2 inside it**.

### 1. Host prerequisites

On the host machine, prepare:

- NVIDIA GPU driver
- Docker Engine + Docker Compose plugin
- NVIDIA Container Toolkit
- Access to NVIDIA Isaac Sim container image (`nvcr.io/nvidia/isaac-sim:4.5.0`)

Quick checks:

```bash
docker --version
docker compose version
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 2. Clone and enter repository

```bash
git clone <this-repo-url>
cd Isaacsim_ros2_slam_simulaor
```

### 3. Login to NGC container registry

```bash
docker login nvcr.io
```

Use your NGC key as password (`$oauthtoken` is typically the username).

### 4. Build and start ROS2-enabled container

From repo root:

```bash
python3 docker/container.py start ros2
```

This builds `isaac-lab-ros2` using `docker/Dockerfile.ros2` and runs container `isaac-lab-ros2`.

### 5. Enter container

```bash
python3 docker/container.py enter ros2
```

Inside container, your workspace is `/workspace/isaaclab`.

### 6. Install/repair Python dependencies used by this project

Inside container:

```bash
cd /workspace/isaaclab
isaaclab -i

# Maintainer-proven fix for version conflicts
python -m pip install --no-cache-dir 'lxml<5.0.0' 'numpy==1.26.4'
python -m pip uninstall -y dex-retargeting
```

### 7. Install ROS2 runtime packages used by scripts

`run.sh` depends on tools not always present in minimal `ros-base` profile.

```bash
apt-get update
apt-get install -y \
  ros-humble-rviz2 \
  ros-humble-robot-state-publisher \
  ros-humble-tf-transformations \
  ros-humble-sensor-msgs-py \
  ros-humble-tf2-ros
```

### 8. Install Fast-LIO2 in a ROS2 workspace (external dependency)

`fast_lio` is **not vendored** in this repo, but `run.sh` expects this command to work:

```bash
ros2 launch fast_lio mapping.launch.py config_file:=ouster64.yaml rviz:=false
```

So your ROS2 workspace must provide package `fast_lio` with `mapping.launch.py` and `ouster64.yaml`.

Use this ROS2 repository:
- https://github.com/Ericsii/FAST_LIO_ROS2

Install Fast-LIO2 by following the instructions in that repository.
Typical starting point:

```bash
mkdir -p /workspace/ros2_ws/src
cd /workspace/ros2_ws/src

# Use the upstream ROS2 Fast-LIO2 repository
git clone https://github.com/Ericsii/FAST_LIO_ROS2.git

# Then follow the exact build/dependency instructions in:
# https://github.com/Ericsii/FAST_LIO_ROS2
# (including any required dependencies and workspace steps)

cd /workspace/ros2_ws
source /opt/ros/humble/setup.bash
```

After you finish the upstream build instructions successfully:

```bash
echo 'source /workspace/ros2_ws/install/setup.bash' >> ~/.bashrc
source /workspace/ros2_ws/install/setup.bash
```

## Running LiDAR + SLAM Simulation

Use two terminals **inside the same container**.
Always run the commands from `source/ros2/isaac-ros` because `run.sh` uses relative paths.

### Terminal 1: run simulator (main experiment)

```bash
cd /workspace/isaaclab/source/ros2/isaac-ros
/workspace/isaaclab/_isaac_sim/python.sh g1_standalone_v2_14dof_nav_heuristic_test_midsole.py
```

This runs the project's primary experiment and publishes LiDAR/IMU/odom topics from Isaac Sim.

If you want the simpler baseline locomotion script instead:

```bash
cd /workspace/isaaclab/source/ros2/isaac-ros
bash g1_v2.sh
```

### Terminal 2: run ROS2 + SLAM stack

```bash
cd /workspace/isaaclab/source/ros2/isaac-ros
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
bash run.sh
```

`run.sh` starts:

- `utils/state_pub.py`
- `utils/state_pub.launch.py` (`robot_state_publisher`)
- `utils/static_node.py`
- `utils/analysis_node.py`
- `fast_lio mapping.launch.py`
- `rviz2`

## SLAM-Feedback Navigation Run

If you want SLAM estimate (`est_pelvis`) to drive navigation policy:

### Terminal 1

```bash
cd /workspace/isaaclab/source/ros2/isaac-ros
/workspace/isaaclab/_isaac_sim/python.sh g1_standalone_v2_nav.py
```

### Terminal 2

```bash
cd /workspace/isaaclab/source/ros2/isaac-ros
source /opt/ros/humble/setup.bash
source /workspace/ros2_ws/install/setup.bash
bash run.sh
```

## Useful Verification Commands

Run in any ROS2 terminal in container:

```bash
ros2 topic list
ros2 topic hz /points
ros2 topic hz /imu/data
ros2 topic hz /g1/odom
ros2 run tf2_ros tf2_echo odom body
ros2 run tf2_ros tf2_echo odom est_pelvis
```

## Customizing Scripts

All major simulators support runtime arguments such as:

- `--policy-path`
- `--nav-policy-path` (navigation variants)
- `--physics_dt`
- `--rendering_dt`
- `--period`

Example:

```bash
/workspace/isaaclab/_isaac_sim/python.sh g1_standalone_v2_nav.py \
  --policy-path source/ros2/isaac-ros/assets/r3/exported/policy.pt \
  --nav-policy-path source/ros2/isaac-ros/assets/weights/navigation/0818_nav_policy.pt
```

## Shutdown

- Stop simulator and ROS processes with `Ctrl+C` in each terminal.
- Exit container shells with `exit`.
- On host, stop container:

```bash
python3 docker/container.py stop ros2
```

## Troubleshooting

- `rviz2: command not found`
  - Install `ros-humble-rviz2` inside container.
- `Package 'fast_lio' not found`
  - Ensure you installed `FAST_LIO_ROS2` and followed its instructions completely:
    - https://github.com/Ericsii/FAST_LIO_ROS2
  - Then source your ROS2 workspace (`/workspace/ros2_ws/install/setup.bash`).
- `ModuleNotFoundError` or version conflicts after container start
  - Apply maintainer fix:
    - `python -m pip install --no-cache-dir 'lxml<5.0.0' 'numpy==1.26.4'`
    - `python -m pip uninstall -y dex-retargeting`
- No `odom -> body` TF
  - Fast-LIO is not running correctly or not receiving `/points` + `imu/data`.
- No `odom -> est_pelvis` TF
  - Ensure `g1_standalone_v2_nav.py` is running (it publishes this through `LidarTFPublisher`).

## License

- Core project: [BSD-3-Clause](LICENSE)
- `isaaclab_mimic`: [Apache-2.0](LICENSE-mimic)
