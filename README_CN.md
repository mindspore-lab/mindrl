# MindSpore Reinforcement

[View English](./README.md)

[![Python Version](https://img.shields.io/badge/python-3.7%2F3.8%2F3.9-green)](https://pypi.org/project/mindspore-rl/) [![LICENSE](https://img.shields.io/github/license/mindspore-ai/mindspore.svg?style=flat-square)](https://github.com/mindspore-ai/reinforcement/blob/master/LICENSE) [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/mindspore-lab/mindrl/pulls)

<!-- TOC -->

- [MindSpore Reinforcement](#mindspore-reinforcement)
    - [概述](#概述)
    - [安装](#安装)
        - [MindSpore版本依赖关系](#mindspore版本依赖关系)
        - [pip安装](#pip安装)
        - [源码编译安装](#源码编译安装)
        - [验证是否成功安装](#验证是否成功安装)
    - [快速入门](#快速入门)
    - [特性](#特性)
        - [算法](#算法)
        - [环境](#环境)
        - [经验回放](#经验回放)
    - [未来路标](#未来路标)
    - [社区](#社区)
        - [治理](#治理)
        - [交流](#交流)
    - [贡献](#贡献)
    - [许可证](#许可证)

<!-- /TOC -->

## 概述

MindSpore Reinforcement是一个开源的强化学习框架，支持使用强化学习算法对agent进行**分布式训练**。MindSpore Reinforcement为编写强化学习算法提供了**干净整洁的API抽象**，它将算法与部署和执行注意事项解耦，包括加速器的使用、并行度和跨worker集群计算的分布。MindSpore Reinforcement将强化学习算法转换为一系列编译后的**计算图**，然后由MindSpore框架在CPU、GPU或Ascend AI处理器上高效运行。MindSpore Reinforcement的架构在如下展示:

![MindSpore_RL_Architecture](docs/images/mindspore_rl_architecture.png)

## 安装

MindSpore Reinforcement依赖MindSpore训练推理框架，安装完[MindSpore](https://gitee.com/mindspore/mindspore#%E5%AE%89%E8%A3%85)，再安装MindSpore Reinforcement。可以采用pip安装或者源码编译安装两种方式。

### MindSpore版本依赖关系

由于MindSpore Reinforcement与MindSpore有依赖关系，请按照根据下表中所指示的对应关系，在[MindSpore下载页面](https://www.mindspore.cn/versions)下载并安装对应的whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore-Version}/MindSpore/cpu/ubuntu_x86/mindspore-{MindSpore-Version}-cp37-cp37m-linux_x86_64.whl
```

| MindSpore Reinforcement |                             分支                             | MindSpore |
| :---------------------: | :----------------------------------------------------------: | :-------: |
|          0.7.0          | [r0.7](https://gitee.com/mindspore/reinforcement/tree/r0.7/) |   2.1.0   |
|          0.6.0          | [r0.6](https://gitee.com/mindspore/reinforcement/tree/r0.6/) |   2.0.0   |
|          0.5.0          | [r0.5](https://gitee.com/mindspore/reinforcement/tree/r0.5/) |   1.8.0   |
|          0.3.0          | [r0.3](https://gitee.com/mindspore/reinforcement/tree/r0.3/) |   1.7.0   |
|          0.2.0          | [r0.2](https://gitee.com/mindspore/reinforcement/tree/r0.2/) |   1.6.0   |
|          0.1.0          | [r0.1](https://gitee.com/mindspore/reinforcement/tree/r0.1/) |   1.5.0   |

### pip安装

使用pip命令安装，请从[MindSpore Reinforcement下载页面](https://www.mindspore.cn/versions)下载并安装whl包。

```shell
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/{MindSpore_version}/Reinforcement/any/mindspore_rl-{Reinforcement_version}-py3-none-any.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> - 在联网状态下，安装whl包时会自动下载MindSpore Reinforcement安装包的依赖项（依赖项详情参见requirement.txt），其余情况需自行安装。
> - `{MindSpore_version}`表示MindSpore版本号，MindSpore和Reinforcement版本配套关系参见[页面](https://www.mindspore.cn/versions)。
> - `{Reinforcement_version}`表示Reinforcement版本号。例如下载0.1.0版本Reinforcement时，`{MindSpore_version}应写为1.5.0，{Reinforcement_version}`应写为0.1.0。

### 源码编译安装

下载[源码](https://github.com/mindspore-lab/mindrl)，下载后进入`mindrl`目录。

```shell
git clone https://github.com/mindspore-lab/mindrl.git
cd mindrl/
bash build.sh
pip install output/mindspore_rl-{Reinforcement_version}-py3-none-any.whl
```

其中，`build.sh`为`mindrl`目录下的编译脚本文件。`{Reinforcement_version}`表示MindSpore Reinforcement版本号。

安装依赖项

```shell
cd mindrl && pip install requirements.txt
```

### 验证是否成功安装

执行以下命令，验证安装结果。导入Python模块不报错即安装成功：

```python
import mindspore_rl
```

## 快速入门

MindSpore Reinforcement的算法示例位于`mindrl/example/`下，以一个简单的算法[Deep Q-Learning (DQN)](https://www.mindspore.cn/reinforcement/docs/zh-CN/master/dqn.html) 示例，演示MindSpore Reinforcement如何使用。

第一种开箱即用方式，使用脚本文件直接运行:

```shell
cd mindrl/example/dqn/scripts
bash run_standalone_train.sh
```

第二种方式，直接使用`config.py`和`train.py`，可以更灵活地修改配置：

```shell
cd mindrl/example/dqn
python train.py --episode 1000 --device_target GPU
```

第一种方式会在当前目录会生成`dqn_train_log.txt`日志文件，第二种在屏幕上打印日志信息：

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

更多有关安装指南、教程和API的详细信息，请参阅[用户文档](https://www.mindspore.cn/reinforcement/docs/zh-CN/master/index.html)。

## 特性

### 算法

<table align="center">
    <tr>
        <th rowspan="2" align="center">算法</th>
        <th rowspan="2" align="center">RL版本</th>
        <th colspan="2" align="center">动作空间</th>
        <th colspan="3" align="center">设备</th>
        <th rowspan="2" align="center">示例环境</th>
    </tr>
    <tr>
        <th align="center">离散</th><th          align="center">连续</th>
        <th align="center">CPU</th><th align="center">GPU</th><th align="center">Ascend</th>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/dqn">DQN</a></td>
        <td align="center">>= 0.1</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/ppo">PPO</a></td>
        <td align="center">>= 0.1</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/ac">AC</a></td>
        <td align="center">>= 0.1</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/a2c">A2C</a></td>
        <td align="center">>= 0.2</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/ddpg">DDPG</a></td>
        <td align="center">>= 0.3</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/qmix">QMIX</a></td>
        <td align="center">>= 0.5</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://github.com/oxwhirl/smac/">SMAC</a>, <a href="https://github.com/openai/multiagent-particle-envs">Simple Spread</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/sac">SAC</a></td>
        <td align="center">>= 0.5</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/td3">TD3</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/c51">C51</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/a3c">A3C</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/cql">CQL</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/hopper">Hopper-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/mappo">MAPPO</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://github.com/openai/multiagent-particle-envs">Simple Spread</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/gail">GAIL</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/half_cheetah/">HalfCheetah-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/mcts">MCTS</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center"><a href="https://gitee.com/mindspore/reinforcement/blob/master/mindspore_rl/environment/tic_tac_toe_environment.py">Tic-Tac-Toe</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/awac">AWAC</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/ant">Ant-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/dreamer">Dreamer</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://github.com/deepmind/dm_control">Walker-walk</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/iql">IQL</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/mujoco/walker2d/">Walker2d-v2</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/maddpg">MADDPG</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://pettingzoo.farama.org/environments/mpe/simple_spread/">simple_spread</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/double_dqn">Double DQN</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/pg">Policy Gradient</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/dueling_dqn">Dueling DQN</a></td>
        <td align="center">>= 0.6</td>
        <td align="center">✔️</td>
        <td align="center">/</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center">✔️</td>
        <td align="center"><a href="https://www.gymlibrary.dev/environments/classic_control/cart_pole/">CartPole-v0</a></td>
    </tr>
</table>

### 环境

强化学习领域中，智能体与环境交互过程中，学习策略来使得数值化的收益信号最大化。“环境”作为待解决的问题，是强化学习领域中重要的要素。当前已支持的环境如下表所示：

<table>
    <tr>
        <th>环境</th>
        <th>版本</th>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/openai/gym">Gym</a></td>
        <td align="center">>= v0.1</td>
    </tr>
    <tr>
        <td align="center"><a href="https://mujoco.org">MuJoCo</a></td>
        <td align="center">>= v0.1</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/openai/multiagent-particle-envs">MPE</a></td>
        <td align="center">>= v0.6</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github/oxwhirl/smac">SMAC</a></td>
        <td align="center">>= v0.5</td>
    </tr>
    <tr>
        <td align="center"><a href="https://www.deepmind.com/open-source/deepmind-control-suite">DMC</a></td>
        <td align="center">>= v0.6</td>
    </tr>
    <tr>
        td align="center"><a href="https://pettingzoo.farama.org/environments/mpe/simple_spread/">PettingZoo-mpe</a></td>
        <td align="center">>= v0.6</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/Farama-Foundation/D4RL">D4RL</a></td>
        <td align="center">>= v0.6</td>
    </tr>
</table>

<center><img src=docs/images/environment-uml.png width=500 height=350></center>

### 经验回放

在强化学习中，ReplayBuffer是一个常用的基本数据存储方式，它的功能在于存放智能体与环境交互得到的数据。 使用ReplayBuffer可以解决以下几个问题：

1. 存储的历史经验数据，可以通过采样或一定优先级的方式抽取，以打破训练数据的相关性，使抽样的数据具有独立同分布的特性。

2. 可以提供数据的临时存储，提高数据的利用率。

一般情况下，算法人员使用原生的Python数据结构或Numpy的数据结构来构造ReplayBuffer, 或者一般的强化学习框架也提供了标准的API封装。不同的是，MindSpore实现了设备端的ReplayBuffer结构，一方面能在使用GPU/Ascend硬件时减少数据在Host和Device之间的频繁拷贝，另一方面，以MindSpore算子的形式表达ReplayBuffer，可以构建完整的IR图，使能MindSpore GRAPH_MODE的各种图优化，提升整体的性能。

<table>
    <tr>
        <th rowspan="2">类别</th>
        <th rowspan="2">特性</th>
        <th colspan="3" align="center">设备</th>
    </tr>
    <tr>
        <th align="center">CPU</th><th align="center">GPU</th><th align="center">Ascend</th>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/core/uniform_replay_buffer.py">UniformReplayBuffer</a></td>
        <td align="left">1 FIFO先进先出 <br>2 支持batch 输入</a></td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
        <td align="center">/</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/core/priority_replay_buffer.py#L25">PriorityReplayBuffer</a></td>
        <td align="left">1 proportional-based优先级策略 <br>2 Sum Tree提升采样效率</a></td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/core/reservoir_replay_buffer.py#L24">ReservoirReplayBuffer</a></td>
        <td align="left">采用无偏采样</a></td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
        <td align="center">✔️ </td>
    </tr>
</table>

### 分布式
MindSpore Reinforcement 将强化学习的算法定义与算法如何并行或分布式执行在硬件上进行了解偶。我们通过一个新的抽象，即数据流片段图（**Fragmented Dataflow Graphs**)来实现这一目标，算法的每一部分都将成为数据流片段，并由MSRL灵活地分发与并行。参考[更多信息](https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/README.md)。

当前已经支持如下分布式策略：

<table>
    <tr>
        <th rowspan="1">策略类别</th>
        <th rowspan="1">策略</th>
        <th rowspan="1">示例</th>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/distribution_policies/multi_actor_single_learner_dp">MultiActorSingleLearnerDP</a></td>
        <td align="left">同步单learner多actor结构分布式策略</a></td>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/ppo/train.py">ppo</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/distribution_policies/async_multi_actor_single_learner_dp">AsyncMultiActorSingleLearnerDP</a></td>
        <td align="left">异步单learner多actor结构分布式策略</a></td>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/a3c/train.py">a3c</td>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/distribution_policies/single_actor_learner_with_multi_env_dp">SingleActorLearnerWithMultEnvDP</a></td>
        <td align="left">单actor learner 多远端环境分布式策略</a></td>
        <td align="center"><a href="https://github.com/mindspore-lab/mindrl/tree/master/example/ppo/train.py">ppo</td>
    </tr>
<table>

<table align="center">
    <tr>
        <td align="center">
            <center>
                <img src=docs/images/multiactorsinglelearnerdp.png width="100%">
                </br><p><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/README_CN.md">MultiActorSingleLearnerDP</a></p>
            </center>
        </td>
        <td align="center">
            <center>
                <img src=docs/images/asyncmultiactorsinglelearnerdp.png width="100%">
                </br><p><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/README_CN.md">AsyncMultiActorSingleLearnerDP</a></p>
            </center>
        </td>
        <td align="center">
            <center>
                <img src=docs/images/singleactorlearnerwithmultienv.png width="100%">
                </br><p><a href="https://github.com/mindspore-lab/mindrl/tree/master/mindspore_rl/distribution/README_CN.md">SingleActorLearnerWithMultEnvDP</a></p>
            </center>
        </td>
    </tr>
<table>

## 未来路标

MindSpore Reinforcement初始版本包含了一个稳定的API， 用于实现强化学习算法和使用MindSpore的计算图执行计算。现已支持算法并行和自动分布式执行能力，支持多智能体，offline-rl，门特卡罗树等多种场景。MindSpore Reinforcement的后续版本将继续完善并提升自动分布式功能以及接入大模型的能力，敬请期待。

## 社区

### 治理

查看MindSpore如何进行[开放治理](https://gitee.com/mindspore/community/blob/master/governance.md)。

### 交流

- [MindSpore Slack](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w) 开发者交流平台。
- [MindSpore 论坛](https://bbs.huaweicloud.com/forum/forum-1076-1.html) 欢迎发帖。
- [Reinforcement issues](https://github.com/mindspore-lab/mindrl/issues) 欢迎提交问题。

## 贡献

欢迎参与贡献。
MindSpore Reinforcement 会按3个月周期更新，如果遇到问题，请及时通知我们。我们感谢所有的贡献，可以通过issue/pr的形式提交您的问题或修改。

## 许可证

[Apache License 2.0](https://gitee.com/mindspore/reinforcement/blob/master/LICENSE)
