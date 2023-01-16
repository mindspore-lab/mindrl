# Deep Q-Learning (DQN)

## 相关论文

1. Mnih, Volodymyr, et al. [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

2. Mnih, Volodymyr, *et al.* [Human-level control through deep reinforcement learning. *Nature* **518,** 529–533 (2015).](https://www.nature.com/articles/nature14236)

## 使用的游戏

DQN算法使用了OpenAI开发的一个强化学习环境库[Gym](https://github.com/openai/gym) ，来作为算法的游戏环境。

在DQN算法中，解决了倒立摆([**CartPole-v0**](https://github.com/openai/gym/wiki/CartPole-v0)) 游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://github.com/openai/gym/wiki/CartPole-v0)

## 如何运行DQN

在运行DQN前，首先需要安装[MindSpore](https://www.mindspore.cn/install) 和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore >= 1.6.0

- numpy >= 1.17.0
- matplotlib >=3.1.3
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行DQN。

在CPU综合性能上的考虑，建议统一配置`OMP_NUM_THREADS` 的值为物理核数的1/4，比如`export OMP_NUM_THREADS=32`(需在`run_standalone_train.sh`中修改)。

### 训练

```shell
> cd example/dqn/scripts
> bash run_standalone_train.sh [EPISODE] [DEVICE_TARGET](可选)
```

你会在`example/dqn/scripts/dqn_train_log.txt`中获得和下面内容相似的输出

```shell
Episode 0： loss is 0.223, rewards is 10.0.
Episode 1： loss is 0.186, rewards is 38.0..
Episode 2： loss is 0.152, rewards is 23.0.
Episode 3： loss is 0.118, rewards is 9.0.
Episode 4： loss is 0.1, rewards is 10.0.
Episode 5： loss is 0.146, rewards is 12.0.
Episode 6： loss is 0.062, rewards is 10.0.
Episode 7： loss is 0.144, rewards is 10.0.
Episode 8： loss is 0.086, rewards is 9.0.
Episode 9： loss is 0.125, rewards is 9.0.
Episode 10： loss is 0.143, rewards is 9.0.
-----------------------------------------
Evaluate for episode 10 total rewards is 9.300
-----------------------------------------
```

### 推理

```shell
> cd example/dqn/scripts
> bash run_standalone_eval.sh [CKPT_PATH] [DEVICE_TARGET](可选)
```

你会在`example/dqn/scripts/dqn_eval_log.txt`中获得和下面内容相似的输出

```shell
Load file /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
Evaluate result is 199.300, checkpoint file in /path/ckpt/policy_net/policy_net_600.ckpt
-----------------------------------------
```

## 支持平台

DQN算法支持GPU，CPU和Ascend。

## GPU 运行ONNX推理脚本

首先需要通过前面提到的训练过程获取ckpt文件

```shell
> cd example/dqn/scripts
> bash run_standalone_train.sh [TRAIN_EPISODE](可选) [DEVICE_TARGET](可选)
```

训练得到的ckpt文件位于`example/dqn/scripts/ckpt/`中，
接着需要通过export.py导出onnx文件，导出的onnx文件位于`example/dqn/scripts/onnx/`中。

```python
python export.py
```

运行onnx推理脚本

```shell
> cd example/dqn/scripts
> bash run_infer_onnx.sh [ONNX_PATH](可选)
```

你会在`example/dqn/scripts/dqn_infer_onnx_log.txt`中获得和下面内容相似的输出

```shell
-----------------------------------------
Evaluate result is 200.0, onnx file in ./onnx/policy_net/policy_net_1000.onnx
-----------------------------------------
eval end
```