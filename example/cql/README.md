# Conservative Q-Learning for Offline Reinforcement Learning (CQL)

## Related Paper

1. Kumar, A., Zhou, A., Tucker, G., & Levine, S.. [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
2. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2021). [D4rl: datasets for deep data-driven reinforcement learning](https://arxiv.org/abs/2004.07219)

CQL is a classical algorithm in Offline RL. Offline reinforcement learning needs to solve the problem of how to maximize the use of offline historical data to train agents (without interaction with the environment during the period), and finally apply it to the actual environment. Generally, the offline method has a fatal flaw: the overestimation of Q value caused by the difference between the offline data and the actual learning policy distribution, especially when the distribution of the actual environment and the training data is different (distribution offset), it will lead to the failure to correctly judge the training policy.
In this paper, a conservative Q-learning (CQL) algorithm is proposed, which aims to address these limitations by learning a conservative Q-function such that the expected value of a policy under this Q-function lower-bounds its true value.

## Game that this algorithm used

CQL uses an open source data of offline reinforcement learning [D4RL](https://arxiv.org/abs/2004.07219).
It provides a standardized environment and data set for training and benchmarking. The D4RL benchmark includes over 40 tasks across 7 qualitatively distinct domains that cover application areas such as robotic manipulation, navigation, and autonomous driving.

This demo using the default dataset `hopper-medium-expert-v0` in [D4RL](https://github.com/Farama-Foundation/D4RL), Control environment needs [MuJoCo](https://github.com/openai/mujoco-py) as a dependency. For more details of D4RL, please refer to the [website](https://sites.google.com/view/d4rl/home).

<img src="../../docs/images/hopper.gif" alt="hopper" style="zoom:80%;" />

## How to run CQL

Before running CQL, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

This example uses the offline data provided by [d4rl](https://github.com/Farama-Foundation/d4rl) , you need to install d4rl:

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

After the installation is successful, you can directly run CQL by entering the following instructions.

### Training

Train 1e6 steps in the default configuration.

```shell
> cd example/cql/
> bash scripts/run_standalone_train.sh 1000000
```

You will get similar output as following in `example/cql/cql_train_log.txt`.

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

### Eval

```shell
> cd example/cql/
> bash scripts/run_standalone_eval.sh ./ckpt
```

You will get similar output as following in  `example/cql/cql_eval_log.txt`.

```shell
Load file  ./ckpt//policy/policy_1000000.ckpt
Load file  ./ckpt//value_net/value_net_1000000.ckpt
-----------------------------------------
Evaluate result is 3671.575, checkpoint file in ./ckpt/
-----------------------------------------
eval end
```

## Supported Platform

CQL algorithm supports GPU and CPU platform.
