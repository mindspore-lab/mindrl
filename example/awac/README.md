# Accelerating Online Reinforcement Learning with Offline Datasets (AWAC)

## Related Paper

1. Nair, A.,  Dalal, M., Gupta, A., & Levine, S. (2020). [Accelerating online reinforcement learning with offline datasets](https://arxiv.org/abs/2006.09359)
2. Fu, J., Kumar, A., Nachum, O., Tucker, G., & Levine, S. (2021). [D4rl: datasets for deep data-driven reinforcement learning](https://arxiv.org/abs/2004.07219)

The AWAC algorithm was proposed by Sergey Levine team of UC Berkeley in 2020. It uses offline expert data to pre train and then fintune, which belongs to the category of offline reinforcement learning. The article points out that in the field of reinforcement learning robots, the cost of using online learning is too high. First, the cost of data collection is high, and the time cost of tens of thousands of training steps is high. Therefore, it is necessary to use offline data to accelerate convergence.

## Game that this algorithm used

AWAC uses an open source data of offline reinforcement learning [D4RL](https://arxiv.org/abs/2004.07219).
It provides a standardized environment and data set for training and benchmarking. The D4RL benchmark includes over 40 tasks across 7 qualitatively distinct domains that cover application areas such as robotic manipulation, navigation, and autonomous driving.

This demo using the default dataset `ant-expert-v0` in [D4RL](https://github.com/Farama-Foundation/D4RL), Control environment needs [MuJoCo](https://github.com/openai/mujoco-py) as a dependency. For more details of D4RL, please refer to the [website](https://sites.google.com/view/d4rl/home).

<img src="../../docs/images/ant.gif" alt="ant" style="zoom:80%;" />

## How to run AWAC

Before running AWAC, you should first install [MindSpore](https://www.mindspore.cn/install) and MindSpore-Reinforcement. Besides, you should also install following dependencies. Please follow the instruction on the official website.

This example uses the offline data provided by [d4rl](https://github.com/Farama-Foundation/d4rl) , you need to install d4rl:

```bash
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
```

- MindSpore >= 1.9.0
- Reinforcement >= 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

After the installation is successful, you can directly run AWAC by entering the following instructions.

### Training

Train 500 episodes in the default configuration.

```shell
> cd example/awac/
> bash scripts/run_standalone_train.sh 500
```

You will get similar output as following in `example/awac/awac_train_log.txt`.

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

### Eval

```shell
> cd example/awac/
> bash scripts/run_standalone_eval.sh ./ckpt
```

You will get similar output as following in  `example/awac/awac_eval_log.txt`.

```shell
Load file  ./ckpt//policy/policy_500.ckpt
Load file  ./ckpt//value_net_1/model_1_500.ckpt
Load file  ./ckpt//value_net_2/model_2_500.ckpt
-----------------------------------------
Evaluate result is 5342.521, checkpoint file in ./ckpt/
-----------------------------------------
eval end
```

## Supported Platform

AWAC algorithm supports GPU and CPU platform.
