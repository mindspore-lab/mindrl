# Policy Grandient Algorithm (PG)

## 相关论文

1. Williams R J. [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://link.springer.com/content/pdf/10.1007/BF00992696.pdf)

## 使用的游戏

PG使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。

在PG中，解决了倒立摆（[**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

## 如何运行PG

在运行PG前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 2.0.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行PG。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train.sh`中修改)。

### 训练

```shell
> cd example/pg/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET] [PRECISION_MODE]
```

你会在`example/pg/scripts/pg_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 520 has 200 steps, cost time: 158.550 ms, per step time: 0.793 ms
-----------------------------------------
Evaluate for episode 520 total rewards is 200.000
-----------------------------------------
Episode 530 has 200 steps, cost time: 156.738 ms, per step time: 0.784 ms
-----------------------------------------
Evaluate for episode 530 total rewards is 200.000
-----------------------------------------
Episode 540 has 200 steps, cost time: 161.918 ms, per step time: 0.810 ms
-----------------------------------------
Evaluate for episode 540 total rewards is 200.000
-----------------------------------------
Episode 550 has 200 steps, cost time: 161.121 ms, per step time: 0.806 ms
-----------------------------------------
Evaluate for episode 550 total rewards is 200.000
-----------------------------------------
Episode 560 has 200 steps, cost time: 159.035 ms, per step time: 0.795 ms
-----------------------------------------
Evaluate for episode 560 total rewards is 200.000
-----------------------------------------
Episode 570 has 200 steps, cost time: 134.095 ms, per step time: 0.670 ms
-----------------------------------------
Evaluate for episode 570 total rewards is 200.000
-----------------------------------------
Episode 580 has 200 steps, cost time: 166.513 ms, per step time: 0.833 ms
-----------------------------------------
Evaluate for episode 580 total rewards is 200.000
-----------------------------------------
Episode 590 has 200 steps, cost time: 131.775 ms, per step time: 0.659 ms
-----------------------------------------
Evaluate for episode 590 total rewards is 200.000
-----------------------------------------
```

### 推理

```shell
> cd example/pg/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/pg/scripts/pg_eval_log.txt`中获得和下面内容相似的输出

```shell
-----------------------------------------
Evaluate result is 200.000, checkpoint file in /path/ckpt/actor_net/actor_net_550.ckpt
-----------------------------------------
```

## 支持平台

PG算法支持GPU, CPU 和 Ascend。
