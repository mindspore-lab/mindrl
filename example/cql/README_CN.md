# Conservative Q-Learning for Offline Reinforcement Learning (CQL)

## 相关论文

1. Kumar, A., Zhou, A., Tucker, G., & Levine, S.. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
2. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2021). [D4rl: datasets for deep data-driven reinforcement learning](https://arxiv.org/abs/2004.07219)

CQL算法是Offline RL算法中比较经典的一个算法。离线强化学习需要解决的是如何最大化利用离线的历史数据训练智能体(期间不与环境交互），最终应用于实际环境中的问题。通常情况下Offline的方法会有一个致命的缺陷：由于离线数据与实际学习的策略分布不同而引起的Q值过估问题，尤其是当实际应用环境的与训练数据的分布不同时（分布偏移），会导致训练的策略无法正确判断。
本文提出了一种Conservative Q-learning(CQL)算法，它通过学习一个较为保守的Q函数，使得该策略下的Q函数的期望值低于其真实值。

## 使用的游戏

CQL使用了离线强化学习的一个开源数据[D4RL](https://arxiv.org/abs/2004.07219)。它为训练和基准测试算法提供了标准化的环境和数据集。
它包括7个领域的40多个任务，覆盖机器人，导航，自动驾驶等应用领域。

本次默认示例使用[D4RL](https://github.com/Farama-Foundation/D4RL)的`hopper-medium-expert-v0`数据集，控制环境需要[MuJoCo](https://github.com/openai/mujoco-py)作为依赖项。关于更多D4RL的介绍，可参考[网页介绍](https://sites.google.com/view/d4rl/home)。

<img src="../../docs/images/hopper.gif" alt="hopper" style="zoom:80%;" />

## 如何运行CQL

在运行CQL前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

本示例使用[d4rl](https://github.com/Farama-Foundation/d4rl)提供的离线数据，需要安装d4rl:

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

安装成功之后，可以直接通过输入如下指令来运行CQL。

### 训练

以默认配置，训练1e6步。

```shell
> cd example/cql/
> bash scripts/run_standalone_train.sh 1000000
```

你会在`example/cql/cql_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 1100: critic_loss is 55.32, actor_loss is -1.331, per_step_time 15.232 ms
Episode 1200: critic_loss is 59.163, actor_loss is -1.505, per_step_time 14.231 ms
Episode 1300: critic_loss is 59.425, actor_loss is -1.271, per_step_time 15.521 ms
Episode 1400: critic_loss is 54.217, actor_loss is -1.287, per_step_time 16.064 ms
Episode 1500: critic_loss is 53.353, actor_loss is -1.16, per_step_time 15.864 ms
Episode 1600: critic_loss is 53.412, actor_loss is -1.276, per_step_time 15.131 ms
Episode 1700: critic_loss is 56.585, actor_loss is -1.423, per_step_time 14.816 ms
Episode 1800: critic_loss is 63.932, actor_loss is -1.313, per_step_time 15.324 ms
Episode 1900: critic_loss is 58.954, actor_loss is -1.318, per_step_time 15.739 ms
Episode 2000: critic_loss is 58.665, actor_loss is -1.209, per_step_time 16.238 ms
-----------------------------------------
Evaluate for episode 2000 total rewards is 315.292
-----------------------------------------
Episode 2100: critic_loss is 65.397, actor_loss is -1.432, per_step_time 14.230 ms
Episode 2200: critic_loss is 60.845, actor_loss is -1.456, per_step_time 15.209 ms
Episode 2300: critic_loss is 71.774, actor_loss is -1.476, per_step_time 15.423 ms
Episode 2400: critic_loss is 66.351, actor_loss is -1.324, per_step_time 15.768 ms
Episode 2500: critic_loss is 67.588, actor_loss is -1.297, per_step_time 15.826 ms
Episode 2600: critic_loss is 66.918, actor_loss is -1.36, per_step_time 15.903 ms
Episode 2700: critic_loss is 71.617, actor_loss is -1.413, per_step_time 13.343 ms
Episode 2800: critic_loss is 71.606, actor_loss is -1.489, per_step_time 15.221 ms
Episode 2900: critic_loss is 71.235, actor_loss is -1.451, per_step_time 15.815 ms
Episode 3000: critic_loss is 65.778, actor_loss is -1.572, per_step_time 15.559 ms
-----------------------------------------
Evaluate for episode 3000 total rewards is 386.211
```

### 推理

```shell
> cd example/cql/
> bash scripts/run_standalone_eval.sh ./ckpt
```

你会在`example/cql/cql_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file  ./ckpt//policy/policy_1000000.ckpt
Load file  ./ckpt//value_net/value_net_1000000.ckpt
-----------------------------------------
Evaluate result is 3671.575, checkpoint file in ./ckpt/
-----------------------------------------
eval end
```

## 支持平台

CQL算法支持CPU, GPU。
