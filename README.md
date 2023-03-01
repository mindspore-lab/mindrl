# MindSpore Reinforcement

[查看中文](./README_CN.md)

[![Python Version](https://img.shields.io/badge/python-3.7%2F3.8%2F3.9-green)](https://pypi.org/project/mindspore-rl/) [![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/reinforcement/blob/master/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://gitee.com/mindspore/reinforcement/pulls)

<!-- TOC -->

- [MindSpore Reinforcement](#mindspore-reinforcement)
    - [Overview](#overview)
    - [Installation](#installation)
        - [Version dependency](#version-dependency)
        - [Installing from pip command](#installing-from-pip-command)
        - [Installing from source code](#installing-from-source-code)
        - [Verification](#verification)
    - [Quick Start](#quick-start)
    - [Features](#features)
        - [Algorithm](#algorithm)
        - [Environment](#environment)
        - [ReplayBuffer](#replaybuffer)
    - [Future Roadmap](#future-roadmap)
    - [Community](#community)
        - [Governance](#governance)
        - [Communication](#communication)
    - [Contributions](#contributions)
    - [License](#license)

<!-- /TOC -->
## Overview

MindSpore Reinforcement is an open-source reinforcement learning framework that supports the **distributed training** of agents using reinforcement learning algorithms. MindSpore Reinforcement offers a **clean API abstraction** for writing reinforcement learning algorithms, which decouples the algorithm from deployment and execution considerations, including the use of accelerators, the level of parallelism and the distribution of computation across a cluster of workers. MindSpore Reinforcement translates the reinforcement learning algorithm into a series of compiled **computational graphs**, which are then run efficiently by the MindSpore framework on CPUs, GPUs and Ascend AI processors. Its architecture is shown below:

![MindSpore_RL_Architecture](docs/images/mindspore_rl_architecture.png)

## Installation

MindSpore Reinforcement depends on the MindSpore training and inference framework. Therefore, please first install [MindSpore](https://www.mindspore.cn/install/en) following the instruction on the official website, then install MindSpore Reinforcement. You can install from `pip` or source code.

### Version dependency

Due the dependency between MindSpore Reinforcement and MindSpore, please follow the table below and install the corresponding MindSpore verision from [MindSpore download page](https://www.mindspore.cn/versions/en).

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore-Version}/MindSpore/cpu/ubuntu_x86/mindspore-{MindSpore-Version}-cp37-cp37m-linux_x86_64.whl
```

| MindSpore Reinforcement Version |                            Branch                            | MindSpore version |
| :-----------------------------: | :----------------------------------------------------------: | :---------------: |
|              0.6.0              | [r0.6](https://gitee.com/mindspore/reinforcement/tree/r0.6/) |       2.0.0       |
|              0.5.0              | [r0.5](https://gitee.com/mindspore/reinforcement/tree/r0.5/) |       1.8.0       |
|              0.3.0              | [r0.3](https://gitee.com/mindspore/reinforcement/tree/r0.3/) |       1.7.0       |
|              0.2.0              | [r0.2](https://gitee.com/mindspore/reinforcement/tree/r0.2/) |       1.6.0       |
|              0.1.0              | [r0.1](https://gitee.com/mindspore/reinforcement/tree/r0.1/) |       1.5.0       |

### Installing from pip command

If you use the pip command, please download the whl package from [MindSpore Reinforcement](https://www.mindspore.cn/versions/en) page and install it.

```shell
pip install  https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore_version}/Reinforcement/any/mindspore_rl-{Reinforcement_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - Installing whl package will download MindSpore Reinforcement dependencies automatically (detail of dependencies is shown in requirement.txt),  other dependencies should install manually.
> - `{MindSpore_version}` stands for the version of MindSpore. For the version matching relationship between MindSpore and Reinforcement, please refer to [page](https://www.mindspore.cn/versions).
> - `{Reinforcement_version}` stands for the version of Reinforcement. For example, if you would like to download version 0.1.0, you should fill 1.5.0 in `{MindSpore_version}` and fill 0.1.0 in `{Reinforcement_version}`.

### Installing from source code

Download [source code](https://gitee.com/mindspore/reinforcement), then enter the `reinforcement` directory.

```shell
git clone https://gitee.com/mindspore/reinforcement.git
cd reinforcement/
bash build.sh
pip install output/mindspore_rl-{Reinforcement_version}-py3-none-any.whl
```

`build.sh` is the compiling script in `reinforcement` directory. `Reinforcement_version` is the version of MindSpore Reinforcement.

Install dependencies

```shell
cd reinforcement && pip install requirements.txt
```

### Verification

If you can successfully execute following command, then the installation is completed.

```python
import mindspore_rl
```

## Quick Start

The algorithm example of mindcore reinforcement is located under `reinforcement/example/`. A simple algorithm [Deep Q-Learning (DQN)](https://www.mindspore.cn/reinforcement/docs/zh-CN/master/dqn.html) is used to demonstrate how to use MindSpore Reinforcement.

The first way is using script files to run it directly:

```shell
cd reinforcement/example/dqn/scripts
bash run_standalone_train.sh
```

The second way is to use `config.py` and `train.py` to modify the configuration more flexibly:

```shell
cd reinforcement/example/dqn
python train.py --episode 1000 --device_target GPU
```

The first way will generate the logfile `dqn_train_log.txt` in the current directory. The second way prints log information on the screen:

```shell
Episode 0: loss is 0.396, rewards is 42.0
Episode 1: loss is 0.226, rewards is 15.0
Episode 2: loss is 0.202, rewards is 9.0
Episode 3: loss is 0.122, rewards is 15.0
Episode 4: loss is 0.107, rewards is 12.0
Episode 5: loss is 0.078, rewards is 10.0
Episode 6: loss is 0.075, rewards is 8.0
Episode 7: loss is 0.084, rewards is 12.0
Episode 8: loss is 0.069, rewards is 10.0
Episode 9: loss is 0.067, rewards is 10.0
Episode 10: loss is 0.056, rewards is 8.0
-----------------------------------------
Evaluate for episode 10 total rewards is 9.600
-----------------------------------------
```

<center>
<img src=docs/images/cartpole.gif width=400 height=300> <img src=docs/images/episode_rewards_of_dqn.png width=400 height=300>
</center>

For more details about the installation guide, tutorials, and APIs, see [MindSpore Reinforcement API Docs](https://www.mindspore.cn/reinforcement/docs/en/master/index.html).

## Features

### Algorithm

<table align="center">
    <tr>
        <th rowspan="2" align="center">Algorithm</th>
        <th rowspan="2" align="center">RL Version</th>
        <th colspan="2" align="center">Action Space</th>
        <th colspan="3" align="center">Device</th>
        <th rowspan="2" align="center">Example Environment</th>
    </tr>
    <tr>
        <th align="center">Discrete</th><th          align="center">Continuous</th>
        <th align="center">CPU</th><th align="center">GPU</th><th align="center">Ascend</th>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/dqn">DQN</a></td>
        <td align="center">>= 0.1</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/ppo">PPO</a></td>
        <td align="center">>= 0.1</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/ac">AC</a></td>
        <td align="center">>= 0.1</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/a2c">A2C</a></td>
        <td align="center">>= 0.2</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/ddpg">DDPG</a></td>
        <td align="center">>= 0.3</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/qmix">QMIX</a></td>
        <td align="center">>= 0.5</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://github.com/oxwhirl/smac/">SMAC</a>, <a href="https://github.com/openai/multiagent-particle-envs">Simple Spread</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/sac">SAC</a></td>
        <td align="center">>= 0.5</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/td3">TD3</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/c51">C51</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/a3c">A3C</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/cql">CQL</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper">Hopper-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/mappo">MAPPO</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://github.com/openai/multiagent-particle-envs">Simple Spread</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/gail">GAIL</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/mcts">MCTS</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/environment/tic_tac_toe_environment.py">Tic-Tac-Toe</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/awac">AWAC</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/ant">Ant-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/dreamer">Dreamer</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://github.com/deepmind/dm_control">Walker-walk</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/tree/master/example/iql">IQL</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/walker2d/">Walker2d-v2</a></td>
    </tr>
</table>

### Environment

In the field of reinforcement learning, during the interaction between the agent and the environment, the learning strategy maximizes the numerical benefit signal. As a problem to be solved, environment is an important element in reinforcement learning.

At present, there are many kinds of environments used for reinforcement learning:[Mujoco](https://github.com/deepmind/mujoco)、[MPE](https://github.com/openai/multiagent-particle-envs)、[Atari](https://github.com/gsurma/atari)、[PySC2](https://www.github.com/deepmind/pysc2)、[SMAC](https://github/oxwhirl/smac)、[TORCS](https://github.com/ugo-nama-kun/gym_torcs)、[Isaac](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs) etc. At present, MindSpore Reinforcement has access to both `Gym` and `Smac` environments. With the enrichment of algorithms, it will gradually access more environments.

<center>
<img src=docs/images/environment-uml.png width=500 height=350></center>

### ReplayBuffer

In reinforcement learning, ReplayBuffer is a commonly used basic data storage method. It is used to store the data obtained by the interaction between the agent and the environment. ReplayBuffer can solve the following problems:

1. The stored historical experience data can be extracted by sampling or certain priority to break the correlation of the training data and make the sampled data have the characteristics of independent and identical distribution.

2. It can provide temporary storage of data and improve the utilization rate of data.

In general, researchers use native Python data structures or numpy data structures to construct ReplayBuffer, or the general reinforcement learning framework also provides standard API encapsulation. The difference is that MindSpore implements the ReplayBuffer structure on the device. On the one hand, it can reduce the frequent copying of data between the host and the device when using GPU/Ascend hardware. On the other hand, it can express the ReplayBuffer in the form of MindSpore operators, which can build a complete IR graph and enable MindSpore GRAPH_MODE optimization to improve the overall performance.  

<table>
    <tr>
        <th rowspan="2">Type</th>
        <th rowspan="2">Features</th>
        <th colspan="3" align="center">Device</th>
    </tr>
    <tr>
        <th align="center">CPU</th><th align="center">GPU</th><th align="center">Ascend</th>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/core/uniform_replay_buffer.py">UniformReplayBuffer</a></td>
        <td align="left">1 FIFO, fist in fist out. <br>2 Support batch input.</a></td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
        <td align="center">/</td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/core/priority_replay_buffer.py#L25">PriorityReplayBuffer</a></td>
        <td align="left">1 Proportional-based priority strategy. <br>2 Using Sum Tree to improve sample performance.</a></td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
    </tr>
    <tr>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/core/reservoir_replay_buffer.py#L24">ReservoirReplayBuffer</a></td>
        <td align="left">keeps an 'unbiased' sample of previous iterations.</a></td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
    </tr>
</table>

## Future Roadmap

This initial release of MindSpore Reinforcement contains a stable API for implementing reinforcement learning algorithms and executing computation using MindSpore's computational graphs.  Now it supports semi-automatic distributed execution of algorithms and multi-agent, but does not support fully automatic distributed capabilities yet. These features will be included in the subsequent version of MindSpore Reinforcement. Please look forward to it.

## Community

### Governance

[MindSpore Open Governance](https://gitee.com/mindspore/community/blob/master/governance.md)

### Communication

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) developer communication platform
- [MindSpore 论坛](https://bbs.huaweicloud.com/forum/forum-1076-1.html) Welcome to post.
- [Reinforcement issues](https://gitee.com/mindspore/reinforcement/issues) Welcome to submit issues.

## Contributions

Welcome to MindSpore contribution.
MindSpore Reinforcement will be updated every 3 months. If you encounter any problems, please inform us in time. We appreciate all contributions and can submit your questions or modifications in the form of issues or prs.

## License

[Apache License 2.0](https://gitee.com/mindspore/reinforcement/blob/master/LICENSE)
