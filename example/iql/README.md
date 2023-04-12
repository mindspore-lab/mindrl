# Offline Reinforcement Learning with Implicit Q-Learning(IQL)

## Related Paper

1. Ilya Kostrikov, Ashvin Nair, Sergey Levine: ["Offline Reinforcement Learning with Implicit Q-Learning", 2021 ](https://arxiv.org/abs/2110.06169)

2. Justin Fu, Aviral Kumar, Ofir Nachum, George Tucker, and Sergey Levine.[D4rl: datasets for deep data-driven reinforcement learning,2021](https://arxiv.org/abs/2004.07219)

IQL was proposed by Sergey Levine of Berkeley in 2021, and was published at the ICLR2022, proposing a new paradigm of offline reinforcement learning. The IQL algorithm combines expected quantile regression and focuses on the information that has been sampled to avoid querying the value of  the unseen action. The experiment shows that the algorithm can achieve the effect of SOTA on D4RL.

## Game that this algorithm used

IQL uses an open source data of offline reinforcement learning [D4RL](https://arxiv.org/abs/2004.07219).
It provides a standardized environment and data set for training and benchmarking. The D4RL benchmark includes over 40 tasks across 7 qualitatively distinct domains that cover application areas such as robotic manipulation, navigation, and autonomous driving.

This demo using the default dataset `walker2d-medium-v2` in [D4RL](https://github.com/Farama-Foundation/D4RL), Control environment needs [MuJoCo](https://github.com/openai/mujoco-py) as a dependency. For more details of D4RL, please refer to the [website](https://sites.google.com/view/d4rl/home).

<img src="../../docs/images/walker2d.gif" alt="ant" style="zoom:80%;" />

## How to run IQL

Before running IQL, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- mujoco200
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1
- [D4RL](https://github.com/Farama-Foundation/D4RL)

After the installation is successful, you can directly run IQL by entering the following instructions.

### Training

Train 100 episodes in the default configuration.

```bash
> cd example/iql/
> bash scripts/run_standalone_train.sh 100
```

You will get similar output as following in`example/iql/iql_train_log.txt`

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

### Eval

You will get similar output as following in `example/iql/iql_eval_log.txt`

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

## Supported Platform

IQL algorithm supports GPU CPU and Ascend platform.