# Actor-Critic Algorithm (AC)

## 相关论文

1. Konda, Vijay R., and John N. Tsitsiklis. "[Actor-critic algorithm](https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)"

## 使用的游戏

AC使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在AC中，解决了倒立摆（[**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## 如何运行AC

在运行AC前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行AC。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train.sh`中修改)。

### 训练

```shell
> cd example/ac/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET](可选)
```

你会在`example/ac/scripts/log.txt`中获得和下面内容相似的输出

```shell
Episode 0, loss is 386.797, rewards is 20.0
Episode 1, loss is 386.477, rewards is 25.0
Episode 2, loss is 385.673, rewards is 11.0
Episode 3, loss is 386.896, rewards is 17.0
Episode 4, loss is 385.612, rewards is 28.0
Episode 5, loss is 386.764, rewards is 43.0
Episode 6, loss is 386.637, rewards is 32.0
Episode 7, loss is 388.327, rewards is 12.0
Episode 8, loss is 385.753, rewards is 39.0
Episode 9, loss is 386.731, rewards is 17.0
------------------------------------
Evaluate for episode 10 total rewards is 9.600
------------------------------------
```

### 推理

```shell
> cd example/ac/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/ac/scripts/log.txt`中获得和下面内容相似的输出

```shell
-----------------------------------------
Evaluate result is 170.300, checkpoint file in /path/ckpt/actor_net/actor_net_950.ckpt
-----------------------------------------
```

## 支持平台

AC算法支持GPU和CPU。
