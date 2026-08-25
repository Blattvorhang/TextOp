
# Usage
This project uses two Python environments. It's recommended to use conda for environment management.
- `textop` for TextOpRobotMDAR + TextOpDeploy
- `env_isaaclab` for TextOpTracker with IsaacLab

We provide pretrained checkpoints for RobotMDAR and Tracker.

Clone the repository with submodules:

```bash
git clone --recurse-submodules git@github.com:TeleHuman/TextOp.git

```

## System Requirements
We tested the project on:
- Ubuntu 20.04
- ROS2 foxy
- with 1 NVIDIA 4090 GPU



## Preparing Data and Model

Download the descriptions and pretrained models from HuggingFace: [HuggingFace](https://huggingface.co/datasets/Yochish/TextOp-Data) and merge the downloaded folder with this repository.

Please respect the licenses of:
unitree_mujoco, BeyondMimic, AMASS, BABEL-TEACH, LAFAN1.

For details on the dataset composition and data processing pipeline, refer to [DATASET.md](DATASET.md).



## TextOpRobotMDAR

### Installation

```bash
conda create -n textop python=3.11 -y
conda activate textop

pip install -e deps/isaac_utils
pip install -e TextOpRobotMDAR
# Or: pip install -e TextOpRobotMDAR[dev]

pip install git+https://github.com/openai/CLIP.git
```

Set the environment variables:
```bash

cd TextOpRobotMDAR

EXPNAME=ExampleRun
DATADIR=BABEL-AMASS-ROBOT-23dof-FULL-50fps
DATAFLAGS="data.weighted_sample=true data.datadir=./dataset/${DATADIR} data.action_statistics_path=./dataset/RobotMDAR-statistics/action_statistics.json skeleton.asset.assetRoot=./description/robots/g1/"

```

### Inference

We have provided some pretrained checkpoints:
 * `TextOpRobotMDAR/logs/pretrained/checkpoint/ckpt_200000.pth` is the ckpt of dar.
 * `TextOpRobotMDAR/logs/pretrained/checkpoint/vae.pth` is the ckpt of mvae.
 * Note that this set of checkpoints should be run with  `DATADIR=PRIVATE-DATA`, corresponding to the statistics of its training data. It cannot run `vis_mvae` and `vis_dar` since the raw data is not provided.

#### 1. Run Online Motion Generation with DAR:
```bash
robotmdar --config-name=loop_dar ckpt.dar=/path/to/dar/ckpt_200000.pth guidance_scale=5.0 ${DATAFLAGS}
```

#### 2. Run Inference on text-motion pairs from the dataset.

```bash
robotmdar --config-name=vis_mvae ckpt.vae=/path/to/mvae/ckpt_200000.pth ${DATAFLAGS}

robotmdar --config-name=vis_dar ckpt.dar=/path/to/dar/ckpt_200000.pth guidance_scale=5.0 ${DATAFLAGS}

```

### Training MVAE & DAR
- You can shorten training duration by modifying: `train.manager.stages`, which are the number of training steps in multiple stages.

```bash

robotmdar --config-name=train_mvae expname=${EXPNAME} \
${DATAFLAGS} \
train.manager.stages=[100000,50000,50000] \
data.num_primitive=4 \
train.manager.use_rollout=True


robotmdar --config-name=train_dar expname=${EXPNAME} \
${DATAFLAGS} \
train.manager.stages=[100000,100000,100000] \
data.num_primitive=4 \
train.manager.use_rollout=True \
train.manager.use_full_sample=True \
diffusion.num_timesteps=5

```


## TextOpTracker

### Installation

- Install Isaac Lab v2.1.0 by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). Supposed that conda environment `env_isaaclab` is used.

- Then install the tracker:
```bash
cd TextOpTracker
python -m pip install -e source/textop_tracker
```

- Replace `rsl_rl_lib` with the modified version
    > This modified version includes a simplified implementation of [modular norm](https://arxiv.org/abs/2405.14813) for faster network optimization (~15%). If you don't want to use it, remove all `-MNMLP` in the following scripts. The pretrained checkpoint needs it to be evaluated in IsaacLab.

```bash
cd ..
pip uninstall rsl_rl_lib -y
pip install -e deps/rsl_rl-modular-normed/

python -c "import rsl_rl;print(rsl_rl)" # Verify installation
```

### Evaluation

We have provided a pretrained checkpoint:
* `TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/model_75000.pt` is the pretrained policy that can be loaded in IsaacLab.
* `TextOpTracker/logs/rsl_rl/Pretrained/checkpoints/latest.onnx` is exported ONNX version.

#### IsaacLab Evaluation
- Change `/path/to/experiment/model_100000.pt` to be policy checkpoint.
- Change `/path/to/motion` to be your motion name, e.g. `Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1General_c3dA1-Stand_posespkl`.
```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-ProjGravObs-MNMLP-v0 \
    --resume_path=/path/to/experiment/model_100000.pt \
    --motion_file=/path/to/motion \
    env.commands.motion.anchor_body_name="pelvis" \
    env.commands.motion.future_steps=5 \
    agent.policy.actor_hidden_dims=[2048,1024,512] \
    agent.policy.critic_hidden_dims=[2048,1024,512] \
    --num_envs=10 \
    env.commands.motion.enable_adaptive_sampling=True \
    --kit_args "--/log/level=error --/log/outputStreamLevel=error --/log/fileLogLevel=error"
```
- An ONNX file will be exported to: `/path/to/experiment/exported/policy.onnx`.
You can deploy this policy in MuJoCo or on the real robot.

#### Mujoco Evaluation
> This sim-to-sim deployment is implemented in a simplified manner. The sim-to-sim pipeline in TextOpDeploy is much more tightly aligned with the real-robot system, offering a more realistic and system-consistent deployment workflow.


```bash
# `$ pip install mujoco-python-viewer` If `mujoco_viewer` not found.
python scripts/deploy_mujoco.py --motion_path=/path/to/motion.npz --policy_path=/path/to/policy.onnx
```


### Training

Train from scratch
- Change `--num_envs=16384`. `16384` envs need ~40GB GPU Memory.
```bash
python scripts/rsl_rl/train.py --headless --log_project_name TextOpTracker \
--task=Tracking-Flat-G1-ProjGravObs-MNMLP-v0 \
--motion_file=Data10k-open/* \
--run_name base \
agent.experiment_name=ExampleRun \
agent.max_iterations=1000000 \
--num_envs=16384 \
env.commands.motion.anchor_body_name="pelvis" \
env.commands.motion.future_steps=5 \
env.commands.motion.random_static_prob=-1.0 \
env.rewards.feet_slide.params.pfail_threshold=1.0 \
env.rewards.soft_landing.params.pfail_threshold=1.0 \
env.rewards.overspeed.params.pfail_threshold=1.0 \
env.rewards.overeffort.params.pfail_threshold=1.0 \
env.rewards.feet_slide.weight=-0.3 \
env.rewards.soft_landing.weight=-0.0003 \
env.rewards.overspeed.weight=-1.0 \
env.rewards.overeffort.weight=-1.0 \
env.commands.motion.enable_adaptive_sampling=True \
env.commands.motion.ads_type=v2 \
env.commands.motion.adaptive_beta=0.5 \
env.commands.motion.adaptive_alpha=0.1 \
env.commands.motion.adaptive_uniform_ratio=0.1 \
agent.policy.actor_hidden_dims=[2048,1024,512] \
agent.policy.critic_hidden_dims=[2048,1024,512] \
\
--seed=1 \
--device=cuda:0

```


Continue Training or Finetuning from a pretrained checkpoint:
- Ensure the `experiment_name` and configs about network architectures is consistent in pretrained checkpoint and this run. 
- Change `--load_run`, `--checkpoint` according to your pretrained checkpoint.
- Other configs, e.g. for `motion_file`, reward functions, can be freely changed.
```bash
python scripts/rsl_rl/train.py --headless --log_project_name BeyondMimic \
--task=Tracking-Flat-G1-ProjGravObs-MNMLP-v0 \
--motion_file=Data10k-open/* \
--run_name finetune \
agent.experiment_name=ExampleRun \
agent.max_iterations=1000000 \
--num_envs=16384 \
env.commands.motion.anchor_body_name="pelvis" \
env.commands.motion.future_steps=5 \
\
--resume=True \
--load_run=2025-11-01_02-46-39_base-Data10k-open \
--checkpoint=model_131000.pt \
env.commands.motion.enable_adaptive_sampling=True \
env.commands.motion.ads_type=v2 \
env.commands.motion.adaptive_beta=0.5 \
env.commands.motion.adaptive_alpha=0.1 \
env.commands.motion.adaptive_uniform_ratio=0.1 \
agent.policy.actor_hidden_dims=[2048,1024,512] \
agent.policy.critic_hidden_dims=[2048,1024,512] \
\
env.commands.motion.random_static_prob=-1.0 \
env.rewards.feet_slide.params.pfail_threshold=1.0 \
env.rewards.soft_landing.params.pfail_threshold=1.0 \
env.rewards.overspeed.params.pfail_threshold=1.0 \
env.rewards.overeffort.params.pfail_threshold=1.0 \
env.rewards.feet_slide.weight=-0.3 \
env.rewards.soft_landing.weight=-0.0003 \
env.rewards.overspeed.weight=-1.0 \
env.rewards.overeffort.weight=-1.0 \
\
--seed=1 \
--device=cuda:0
```


## TextOpDeploy
TextOpDeploy supports:
- `unitree_mujoco` simulator. A more realistic method for sim2sim validation.
- Real G1 robot deployment

> The `unitree_mujoco` simulator emulates the joystick interface used on the real robot. You can connect an external game joystick (e.g., Xbox or Switch), or use the keyboard to send surrogate joystick commands to the simulator.


### Installation for Sim2Sim

- Use the same Python env as RobotMDAR. Activate it all the time in this section.
```bash
conda activate textop
```
- Install [ros2](https://docs.ros.org/) ([foxy](https://docs.ros.org/en/foxy/index.html) as an example) following the official guide. 
- Install necessary python packages for ros2:
```bash
pip install colcon-common-extensions empy==3.3.4 catkin_pkg lark-parser netifaces transforms3d pyyaml rosdep 
```

- Install `TextOpDeploy/src/unitree_ros2` following [unitree_ros2](https://github.com/unitreerobotics/unitree_ros2). Remember to modify `TextOpDeploy/src/unitree_ros2/setup.sh` to set the net interface correctly.

- Install `unitree_sdk2`, `mujoco` and `TextOpDeploy/src/unitree_mujoco`, following [unitree_mujoco's README](TextOpDeploy/src/unitree_mujoco/readme.md). Only **C++ Simulator** is needed. This version includes communication hooks for `textop_ctrl`.
    > Note: the joystick interface is defined in `TextOpDeploy/src/unitree_mujoco/simulate/src/physics_joystick.h`. 
    > For input configuration, set the device path and type in `TextOpDeploy/src/unitree_mujoco/simulate/config.yaml`: use `/dev/input/js*` as device path for an external joystick, or set the device type to `keyboard` and specify the keyboard device by `/dev/input/event*`. The correct device path can be identified using `sudo evtest` for keyboard or `sudo jstest` for joystick.

- After installing the above packages, you should see the G1's ros2 topics from mujoco simulator.
```bash
export ROS_DOMAIN_ID=10

# Open one terminal, activate your environment
cd TextOpDeploy/src/unitree_mujoco/simulate/build
sudo ./unitree_mujoco -r g1 -n lo -i 10 # Choose the correct net interface and ROS DOMAIN ID
#  `sudo` is required only when using the keyboard joystick

# Open another terminal, activate your environment
ros2 topic list
## Your should find a list of G1-related topic
```

- Install `textop_ctrl`
```bash

# Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases, choose correct platform and version. Extract it to TextOpDeploy/src/textop_ctrl/thirdparty/
# Take `linux-x64-1.22.0` for example. If you use a different one, modify `ONNXRUNTIME_ROOT_DIR` in `TextOpDeploy/src/textop_ctrl/CMakeLists.txt`
mkdir TextOpDeploy/src/textop_ctrl/thirdparty && cd TextOpDeploy/src/textop_ctrl/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz 
tar -xvf onnxruntime-linux-x64-1.22.0.tgz 

# Set the environment variable
export LD_LIBRARY_PATH=$(pwd)/src/textop_ctrl/thirdparty/onnxruntime-linux-x64-1.22.0/lib:$LD_LIBRARY_PATH

# Build
cd ../../..
colcon build --packages-select textop_ctrl unitree_go unitree_hg CNPY

# Activate the workspace environment
source install/setup.bash
```

Environment Activation: in every time before running
```bash
conda activate textop
cd TextOpDeploy/src/unitree_ros2/ && source setup.sh && cd -
cd TextOpDeploy
source install/setup.bash

export ROS_DOMAIN_ID=0
```

### Run Inference in Sim2Sim

#### Test ONNX Controller without RobotMDAR
```bash
# Open terminal-1, activate the environment
ros2 launch textop_ctrl textop_onnx_controller.launch.py \
    onnx_path:=/path/to/policy.onnx

# Open terminal-2, start unitree_mujoco simulator
./unitree_mujoco -r g1 -n lo -i 0

# Open terminal-3, activate the environment
# This program send an example motion file to the policy as a trigger.
# The example `motion.npz` is corresponding to https://github.com/TeleHuman/PBHC/blob/main/example/motion_data/Horse-stance_pose.pkl
python src/textop_ctrl/scripts/npz_motion_publisher.py --mode single src/textop_ctrl/models/motion.npz

# Joystick: Press start and then press A. The Tracker and RobotMDAR will begin.
```

#### Running Policy with RobotMDAR
```bash
# Open terminal-1, start textop_onnx_controller

# Open terminal-2, start unitree_mujoco

# Open terminal-3, start RobotMDAR: activate the ros workspace and `textop` python environment
# Modify the config of RobotMDAR in `TextOpDeploy/src/textop_ctrl/config/rmdar_config.yaml` to choose the RobotMDAR checkpoint and inference parameters.
python src/textop_ctrl/scripts/rmdar.py 

# (Optional) Open terminal-3, start a motion watcher to visualize the generated reference motion.
python src/textop_ctrl/scripts/motion_watcher.py 

# In your joystick, first press `start` and then press `A`, the Tracker and RobotMDAR should start in the same times. Enter some words to RobotMDAR to instruct the Robot.

# Note: Inference of RobotMDAR is toggled by /dar/toggle channel, 
# The textop_onnx_controller actively manages this. To manually toggle it:
ros2 topic pub /dar/toggle builtin_interfaces/msg/Time "{sec: 0, nanosec: 0}" -t 1 -r 10 
```


### Deployment in Real Robot

This guide assumes you have a host PC with a NVIDIA GPU and a G1-Edu robot connected via Ethernet or a wireless network.

In the following, we will run the Tracker policy on G1's onboard computer (192.168.123.164, called G1 for short) and run the RobotMDAR on PC. The PC sends motion commands to the G1 over the network.

Installation
1. In PC, install everything the same as in Sim2Sim.
2. In G1's onboard computer:
    1. Download `TextOpDeploy`.
    1. Pull `unitree_ros2` and `cnpy` as usual. No need to install `unitree_ros2`.
    2. Install `textop_ctrl`. Compile the controller with the appropriate ONNX Runtime version for `aarch64`.
```bash
# Download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases.
mkdir TextOpDeploy/src/textop_ctrl/thirdparty && cd TextOpDeploy/src/textop_ctrl/thirdparty
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-aarch64-1.22.0.tgz 
tar -xvf onnxruntime-linux-aarch64-1.22.0.tgz 

# Set the environment variable
export LD_LIBRARY_PATH=$(pwd)/src/textop_ctrl/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib:$LD_LIBRARY_PATH


# Build
cd ../../..
colcon build --packages-select textop_ctrl unitree_go unitree_hg CNPY

# Activate the workspace environment
source install/setup.bash
```
3. Ensure that ROS2 is communicating correctly over the network:
```bash
# In both PC and G1: 
export ROS_DOMAIN_ID=0 
ros2 topic list
## You should find a list of G1-related topic
```

Startup Process
1. PAY SPECIAL ATTENTION TO EVERYTHING'S SAFETY. 
    1. Ensure the robot is operating in an open area and is stable before engaging the policy. 
    2. Ensure the Tracker and RobotMDAR policy works well in simulation. 
    3. Ensure you fully understand the operating procedure and code logic. 
    4. We cannot ensure the pretrained models work well in your setting.
2. Start G1. Use unitree's joystick to enter **Debug Mode** by pressing `L2+R2`, `L2+A`, `L2+B`.
3. In G1, start the controller node by this. G1 will enter Zero Torque Mode. Ensure its pose is normal.
```bash
export LD_LIBRARY_PATH=$(pwd)/src/textop_ctrl/thirdparty/onnxruntime-linux-aarch64-1.22.0/lib:$LD_LIBRARY_PATH
source install/setup.bash
ros2 launch textop_ctrl textop_onnx_controller.launch.py \
    onnx_path:=/path/to/policy.onnx
```
4. In PC, start the `rmdar` and `motion_watcher` the same as in Sim2Sim. You can also visualize the state of real robot by:
```bash
python src/textop_ctrl/scripts/show_realrobot.py 
```
5. Press `Start`. G1 will smoothly go to a default pose. Place it on flat ground and it should stand still.
6. Press `A`. The Tracker and RobotMDAR policy will start inference in the same time. G1 will track the reference motion corresponding to default command `stand`. 
    - ATTENTION: In some rare cases the reference motion will step forward even if the text command is `stand`
    - To stop the robot:
        - The system will auto exit if any of the joint angles or joint velocitys exceeds the safety thresholds, defined in `TextOp\TextOpDeploy\src\textop_ctrl\src\textop_onnx_controller.cpp`
        - Press `B` in joystick to stop Tracker policy immediately.
        - Kill the RobotMDAR node, the Tracker will try to track the last reference pose and won't exit.
        - The internal control of G1 will force power off in specific hazardous situations.
7. Enter some words to RobotMDAR's terminal in PC to instruct the robot movement as you like. E.g. `wave hands`, `walk forward`, `punch`.
    - ATTENTION: The motion generation process is stochastic. Entering the same text command multiple times will result in different movements. Please avoid entering unfamiliar words to ensure predictable behavior.



