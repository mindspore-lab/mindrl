# Accelerating Online Reinforcement Learning with Offline Datasets (AWAC)

## 相关论文

1. Nair, A.,  Dalal, M., Gupta, A., & Levine, S. (2020). [Accelerating online reinforcement learning with offline datasets](https://arxiv.org/abs/2006.09359)
2. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2021). [D4rl: datasets for deep data-driven reinforcement learning](https://arxiv.org/abs/2004.07219)

AWAC算法由UC Berkeley 的 Sergey Levine团队于2020年提出， 利用离线专家数据进行pre-train后进行finetune，属于离线强化学习一类。文章指出，在强化学习的机器人领域，使用在线学习的成本太高，一是数据收集的造价高，而是数以万计的训练步数的时间成本高，因此有必要使用离线数据加速收敛。

## 使用的游戏

AWAC使用了离线强化学习的一个开源数据[D4RL](https://arxiv.org/abs/2004.07219)。它为训练和基准测试算法提供了标准化的环境和数据集。
它包括7个领域的40多个任务，覆盖机器人，导航，自动驾驶等应用领域。

本次默认示例使用[D4RL](https://github.com/Farama-Foundation/D4RL)的`ant-expert-v2`数据集，控制环境需要[MuJoCo](https://github.com/openai/mujoco-py)作为依赖项。关于更多D4RL的介绍，可参考[网页介绍](https://sites.google.com/view/d4rl/home)。

<img src="../../docs/images/ant.gif" alt="ant" style="zoom:80%;" />

## 如何运行AWAC

在运行AWAC前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

本示例使用[d4rl](https://github.com/Farama-Foundation/d4rl)提供的离线数据，需要安装d4rl:

```bash
pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
```

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

安装成功之后，可以直接通过输入如下指令来运行AWAC。

### 训练

以默认配置，训练500个episode。

```shell
> cd example/awac/
> bash scripts/run_standalone_train.sh 500
```

你会在`example/awac/awac_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 0: critic_loss is 0.955, actor_loss is 23433.918, mean_std is 0.063, per_step_time 14.688 ms
Episode 0: critic_loss is 28.976, actor_loss is -4048.753, mean_std is 0.151, per_step_time 11.777 ms
-----------------------------------------
Evaluate for episode 10 total rewards is 669.482
-----------------------------------------
Episode 20: critic_loss is 76.176, actor_loss is -4846.117, mean_std is 0.166, per_step_time 11.579 ms
-----------------------------------------
Evaluate for episode 20 total rewards is 1097.807
-----------------------------------------
Episode 30: critic_loss is 71.176, actor_loss is -4459.788, mean_std is 0.189, per_step_time 11.741 ms
-----------------------------------------
Evaluate for episode 30 total rewards is 1327.440
-----------------------------------------
Episode 40: critic_loss is 86.47, actor_loss is -3203.894, mean_std is 0.187, per_step_time 11.643 ms
-----------------------------------------
Evaluate for episode 40 total rewards is 4652.439
-----------------------------------------
```

### 推理

```shell
> cd example/awac/
> bash scripts/run_standalone_eval.sh ./ckpt
```

你会在`example/awac/awac_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file  ./ckpt//policy/policy_500.ckpt
Load file  ./ckpt//value_net_1/model_1_500.ckpt
Load file  ./ckpt//value_net_2/model_2_500.ckpt
-----------------------------------------
Evaluate result is 5342.521, checkpoint file in ./ckpt/
-----------------------------------------
eval end
```

## 支持平台

AWAC算法支持CPU, GPU。
