# Offline Reinforcement Learning with Implicit Q-Learning(IQL)

## 相关论文

1. Ilya Kostrikov, Ashvin Nair, Sergey Levine: ["Offline Reinforcement Learning with Implicit Q-Learning", 2021 ](https://arxiv.org/abs/2110.06169)

2. Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine.[D4rl: datasets for deep data-driven reinforcement learning,2021](https://arxiv.org/abs/2004.07219)

IQL由伯克利的Sergey Levine团队于2021年提出，发表在ICLR2022上，提出了一种离线强化学习的新范式。IQL算法结合了期望分位数回归，聚焦于已经采样到的信息而避免了查询未出现过的动作的价值，实验表明算法可以在D4RL上实现SOTA的效果。

## 使用的游戏

IQL使用了离线强化学习的一个标准benchmark[D4RL](https://arxiv.org/abs/2004.07219)，包括MuJoCo locomotion和Ant Maze tasks等。D4RL为训练和基准测试算法提供了标准化的环境和数据集，它包括7个领域的40多个任务，覆盖机器人，导航，自动驾驶等应用领域。

本次默认示例使用[D4RL](https://github.com/Farama-Foundation/D4RL)的walker2d-medium-v2数据集，控制环境需要[MuJoCo](https://github.com/openai/mujoco-py)作为依赖项。关于更多D4RL的介绍，可参考[网页介绍](https://sites.google.com/view/d4rl/home)。

<img src="../../docs/images/walker2d.gif" alt="ant" style="zoom:80%;" />

## 如何运行IQL

在配置环境时，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- mujoco200
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1
- [D4RL](https://github.com/Farama-Foundation/D4RL)

### 训练

以默认配置，训练100个episode。

```bash
> cd example/iql/
> bash scripts/run_standalone_train.sh 100
```

你会在`example/iql/iql_train_log.txt`中获得和下面内容相似的输出：

```bash
Episode 0: critic_loss is 8.61, actor_loss is -0.135, value_loss is 3.731,mean_std is 0.184,per_step_time 59.416 ms,
Episode 5: critic_loss is 0.559, actor_loss is -0.3, value_loss is 0.079,mean_std is 0.184,per_step_time 16.027 ms,
-----------------------------------------
Evaluate for episode 5 total rewards is 2906.332
-----------------------------------------
Episode 10: critic_loss is 0.593, actor_loss is -0.312, value_loss is 0.064,mean_std is 0.184,per_step_time 16.899 ms,
-----------------------------------------
Evaluate for episode 10 total rewards is 3452.921
-----------------------------------------
Episode 15: critic_loss is 0.561, actor_loss is -0.31, value_loss is 0.077,mean_std is 0.184,per_step_time 15.729 ms,
-----------------------------------------
Evaluate for episode 15 total rewards is 3389.251
-----------------------------------------
Episode 20: critic_loss is 2.208, actor_loss is -0.279, value_loss is 0.086,mean_std is 0.184,per_step_time 15.831 ms,
-----------------------------------------
Evaluate for episode 20 total rewards is 3358.396
```

### 推理

你会在`example/iql/iql_eval_log.txt`中获得和下面内容相似的输出

```bash
Load file  ./ckpt/policy/policy_20.ckpt
Load file  ./ckpt/value_net_1/value_net_1_20.ckpt
Load file  ./ckpt/value_net_2/value_net_2_20.ckpt
Load file  ./ckpt/value_model/value_model_20.ckpt
-----------------------------------------
Evaluate result is 5158.605, checkpoint file in ./ckpt
-----------------------------------------
eval end
```

## 支持的平台

IQL算法支持CPU, GPU 和 Ascend。