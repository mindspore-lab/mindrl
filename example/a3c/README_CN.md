# Asynchronous Advantage Actor-critic Algorithm (A3C)

## 相关论文

1. Mnih V,  Badia A P,  Mirza M, et al. [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783?context=cs)

A3C算法是一种支持异步更新的Actor-Critic算法，它通过多个actor和环境交互生成经验数据，并通过这些经验数据计算每个actor网络的梯度。这些梯度将异步地传至唯一的一个learner进行weight的更新，更新后最新的weight将覆盖该actor的网络。由于异步的机制，每个actor在更新时都是全局最新的weight，这样保证了算法在不断的往好的方向发展，使更新效率更高。

![A3C](../../docs/images/a3c_algo_arch.png)

## 使用的游戏

A3C使用了OpenAI开发的一个强化学习环境库叫做[Gym](https://github.com/openai/gym)，来作为算法的游戏环境。相比于A2C的单actor结构，A3C引入了多个actor异步执行的方式来提高采样效率。

在A3C中，解决了倒立摆（[**CartPole-v0**](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)）游戏。“一根杆子通过一个关节连接在一个小车上，这个小车可以沿着没有摩擦的轨道上左右移动。系统会对这个小车施加+1或者-1的力。这个游戏的目标是防止这根杆子从车上倒下。“[1](https://www.gymlibrary.dev/environments/classic_control/cart_pole/)

![A3C](../../docs/images/cartpole.gif)

## 如何运行A3C

在运行A3C前，首先需要安装[MindSpore](https://www.mindspore.cn/install)和MindSpore-Reinforcement。除此之外，还需要安装以下依赖。请根据官网的教程安装。

- MindSpore > 2.0.0
- Reinforcement > 0.6
- numpy >= 1.17.0
- [gym](https://github.com/openai/gym) >= 0.18.3

安装成功之后，可以直接通过输入如下指令来运行A3C。

### 训练

### 单机训练

以单机4卡(1 learner + 3 actors)为例：

```shell
> cd example/a3c/
> bash run_standalone.sh 4
```

### 多机训练

以双机4卡(1 learner + 3 actors)为例，首先在node1 上启动scheduler，用于组网管理。需要设置该scheduler的ip和port。
然后在node1, node2 分别启动2个worker,设置对应scheduler的ip和port。

```shell
> cd example/a3c/
> bash run_distribute.sh scheduler 4 [ip] [port] # on node1
> bash run_distribute.sh worker 4 2 [ip] [port] # on node1
> bash run_distribute.sh worker 4 2 [ip] [port] # on node2
```

你会在`example/a3c/worker_0.txt`中获得和下面内容相似的输出

```shell
Train in actor 0, episode 0, rewards 13, loss 102.990003983
Train in actor 0, episode 1, rewards 8, loss 96.347625657
Train in actor 0, episode 2, rewards 25, loss 34.082636394
Train in actor 0, episode 3, rewards 47, loss 45.092653245
Train in actor 0, episode 4, rewards 168, loss 64.23145143
Train in actor 0, episode 5, rewards 21, loss 32.53667883
Train in actor 0, episode 6, rewards 42, loss 54.13416578
Train in actor 0, episode 7, rewards 29, loss 52.45465789
Train in actor 0, episode 8, rewards 107, loss 32.86542446
Train in actor 0, episode 9, rewards 56, loss 22.34567676
```

## 支持平台

A3C算法支持Ascend 和 GPU。
