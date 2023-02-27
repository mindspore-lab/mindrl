# Generative Adversarial Imitation Learning (GAIL)

## 相关论文

1. Jonathan Ho, Stefano Ermon 2016 ["Generative Adversarial Imitation Learning"](https://arxiv.org/abs/1606.03476)
2. 算法生成器部分采用SAC算法：Tuomas Haarnoja, et al. ["Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"](https://arxiv.org/abs/1801.01290)

## 使用的游戏

在GAIL算法解决了[HalfCheetah-v2](https://www.gymlibrary.ml/environments/mujoco/half_cheetah/)游戏。如果要运行这个游戏，训练GAIL算法，则必须要安装[MuJoCo](https://github.com/openai/mujoco-py)这个库。游戏界面示意图如下(图源:https://www.gymlibrary.dev/environments/mujoco/half_cheetah/)：

<img src="./img/half_cheetah.gif" alt="half_cheetah" style="zoom:50%;" />

## 如何运行GAIL

在运行GAIL前，首先需要安装[MindSpore](https://www.mindspore.cn/install)(>=2.0.0)和[MindSpore-Reinforcement](https://gitee.com/mindspore/reinforcement/blob/master/README_CN.md#%E5%AE%89%E8%A3%85)。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- [gym](https://github.com/openai/gym) <= 0.21.3
- [mujoco-py](https://github.com/openai/mujoco-py)<2.2,>=2.1

模仿学习算法需要专家展示数据。当前算法对Mujoco环境进行调优，对应的数据可以从[这里](https://drive.google.com/drive/folders/1cZYLU-Wm11SV76apLZUJHrirk8N4pVyh?usp=sharing)下载。

接下来使用如下命令启动训练:

```python
> cd example/gail
> python train.py --expert_data_path /expert_data_path/mujoco-experts/HalfCheetah/seed-0/exp_trajs_sac_50.pkl
```

## 支持平台

GAIL算法支持Ascend, GPU和CPU平台。
