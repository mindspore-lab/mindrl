# Generative Adversarial Imitation Learning (GAIL)

## Related Paper

1. Jonathan Ho, Stefano Ermon 2016 ["Generative Adversarial Imitation Learning"](https://arxiv.org/abs/1606.03476)
2. SAC algorithm is applied to generator：Tuomas Haarnoja, et al. ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290)

## Game that this algorithm used

 GAIL has solved the game called [HalfCheetah-v2](https://www.gymlibrary.ml/environments/mujoco/half_cheetah/) in OpenAI Gym. If you would like to run this game, implement and train GAIL algorithm, it is necessary to install an additional library named [MuJoCo](https://github.com/openai/mujoco-py). The interface of this game is shown below (image from https://www.gymlibrary.dev/environments/mujoco/half_cheetah/):

<img src="./img/half_cheetah.gif" alt="half_cheetah" style="zoom:50%;" />

## How to run GAIL

Before running GAIL, you should first install [MindSpore](https://www.mindspore.cn/install/en)(>=2.0.0) and [MindSpore-Reinforcement](https://github.com/mindspore-lab/mindrl/tree/master/README.md#installation). Besides, the following dependencies should be installed. Please follow the installation instructions on their official websites.

- [gym](https://github.com/openai/gym) <= 0.21.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

The Imitation Learning(IL) algorithms require expert demonstrations data. This algorithm tuning on Mujoco environment and the expert data can be download in [here](https://drive.google.com/drive/folders/1cZYLU-Wm11SV76apLZUJHrirk8N4pVyh?usp=sharing).

Then start training:

```python
> cd example/gail
> python train.py --expert_data_path /expert_data_path/mujoco-experts/HalfCheetah/seed-0/exp_trajs_sac_50.pkl
```

## Supported Platform

GAIL algorithm currently supports Ascend, GPU and CPU platform.
