# Advantage Actor-Critic Algorithm (A2C)

## 相关论文

1. Konda, Vijay R., and John N. Tsitsiklis. "[Actor-critic algorithm](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)"

## 使用的游戏

A2C使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在A2C中，解决了倒立摆（[**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## 如何运行A2C

在运行A2C前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3
- tqdm >= 4.46.0

安装成功之后，可以直接通过输入如下指令来运行A2C。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train_and_eval.sh`中修改)。

### 训练

```shell
> cd example/a2c/scripts
> bash run_standalone_train_and_eval.sh [DEVICE_TARGET](可选)
```

你会在`example/a2c/scripts/a2c_log.txt`中获得和下面内容相似的输出

```shell
Solved at episode 353: average reward: 195.74.
Episode 353:  4%|██▏                                       | 353/10000 [08:43<03:38,  1.48s/it, episode_reward=200.0, loss=27.060312, running_reward=196]
training end
```

## 支持平台

A2C算法支持GPU CPU 和 Ascend。
