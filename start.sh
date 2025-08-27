#!/bin/bash

# Alias Seeting
echo "alias python3='/workspace/isaaclab/_isaac_sim/python.sh'" >> ~/.bashrc
echo "alias python='/workspace/isaaclab/_isaac_sim/python.sh'" >> ~/.bashrc
echo "alias pip3='/workspace/isaaclab/_isaac_sim/python.sh -m pip'" >> ~/.bashrc
echo "alias pip='/workspace/isaaclab/_isaac_sim/python.sh -m pip'" >> ~/.bashrc
source ~/.bashrc

# python library install
pip install --no-cache-dir 'lxml<5.0.0' 'numpy==1.26.4'
pip install icecream
pip uninstall -y dex-retargeting

# ros2 library install
cd /workspace/isaaclab
apt update
apt install libbrotli1=1.0.9-2build6 -y
apt install libbrotli-dev
apt install ros-humble-rviz2
apt install -y ros-humble-rviz-ogre-vendor
apt install -y libogre-1.12-dev
apt install -y ros-humble-tf-transformations
pip install transforms3d
apt install -y libpcl-dev libeigen3-dev
apt install ros-humble-pcl-conversions

cat <<EOF > /root/DEFAULT_FASTRTPS_PROFILES.xml
<profiles>
  <participant profile_name="default_participant_profile" is_default_profile="true">
  </participant>
</profiles>
EOF

cat <<EOF >> ~/.bashrc
export FASTRTPS_DEFAULT_PROFILES_FILE=/root/DEFAULT_FASTRTPS_PROFILES.xml
EOF
source ~/.bashrc

# Install Livox SDK2
cd /workspace/isaaclab/source/ros2/Livox-SDK2
mkdir build
cd build
cmake .. && make -j
make install

# Install Livox ROS2 Driver2
cd /workspace/isaaclab/source/ros2/ws_livox/src/livox_ros_driver2
./build.sh humble
echo "source /workspace/isaaclab/source/ros2/ws_livox/install/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install Fast-LIO2
cd /workspace/isaaclab/source/ros2/ws_fast_lio/
rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -y
colcon build --symlink-install
echo "source /workspace/isaaclab/source/ros2/ws_fast_lio/install/setup.sh" >> ~/.bashrc
source ~/.bashrc

