# 自动分布式

[View English](./README.md)

## 介绍

MindSpore Reinforcement Learning（MSRL）支持分布式训练，将算法与执行进行了解偶，使RL算法分布在不同的计算资源上执行。
MSRL提出了**Fragmented Dataflow Graphs**的抽象，将Python函数从RL算法中剥离，并映射到并行计算的片段。最终，不同的片段将在不同的设备上执行，通过不同的分布式策略（DP）进行数据汇聚与分发。
与MindSpore已有的分布式方式不同，MSRL的分布式更为复杂且粒度更宏观。MS的模型并行与数据并行将网络模型进行切分并行，而MSRL将RL算法抽象成Actor, Learner, Environment等模块片段，依据不同的分布式策略，自动插入通信算子，汇聚或分发片段间的数据。

<center><img src=../../docs/images/msrl_distribute.png width=600 height=350><br/>MSRL Auto Distribution Architecture</center>

## 分布式策略

MSRL提供了不同的分布式策略，用于将单机RL算法自动分布式地执行在多个设备上，以提高算法的吞吐。
根据的RL算法的不同特点，分布式策略的选择也不同，MSRL目前提供3中不同的分布式策略。

### 单Learner多Actor同步策略

该分布式策略下，会生成多个Actor分布在不同进程，每个Actor上创建有各自的环境。多个Actor将同步收集环境产生的经验数据，并由Learner收集并更新策略网络。同时，Learner的网络参数将同步覆盖每个Actor。

<center><img src=../../docs/images/multiactorsinglelearnerdp_detail.png width=360 height=320><br/>MultiActorSingleLearnerDP</center>

### 单Learner多Actor异步策略

单Learner多Actor异步策略与同步策略的区别在于，多个Actor不再同步收集经验并更新网络，每个Actor将独立异步地发送经验给Learner，随后Learner将该经验应用于策略网络更新，同时覆盖该Actor的策略网络，完成一次网络更新。

<center><img src=../../docs/images/asyncmultiactorsinglelearnerdp_detail.png width=360 height=320><br/>AsyncSingleLearnerMultiActorDP</center>

### 单Learner单Actor多远端环境策略

单Learner单Actor多远端环境策略将Actor和Learner绑定于一个进程，将环境分布于不同的进程。这种分布式策略适合于环境较大或者环境节点为CPU节点的情况。

<center><img src=../../docs/images/multienvdp_detail.png width=360 height=320><br/>SingleLearnerSingleActorWithMultiEnvDP</center>

### 分布式策略与模板

分布式策略([DP](./distribution_policies/distribution_policy.py))用于描述分布式策略的拓扑逻辑，设置通信模式，设置通信内容等，每种不同的分布式结构都有各自的分布式策略。而模板(template.tp)文件为自动生成代码提供基础骨架，配合不同DP使用，可以在模板中预设基础逻辑，如通信算子初始化，通用算子定义，基础循环逻辑等。具体可以参见[已有示例](./distribution_policies/multi_actor_single_learner_dp/template.tp)。

### 自动分布式代码生成

使用分布式策略搭配对应的模板，MSRL可以基于现有标准单机算法实现自动的分布式代码生成。MSRL将分析已有标准算法的流程，使用python ast分析并添加分布式的代码逻辑。最后根据DP的描述，将原算法分割成不同的Fragment片段，并在不同的节点上通过通信算子连接并运行。

<center class="half">
<img src=../../docs/images/codegen.png width=800 height=300>
</center>

### 分布式Fragment

通过上述代码生成逻辑，会在算法执行路径生成对应的Fragment文件，具体以`Fragments`+`pid`的形式，每个进程对应一个Fragment文件。
Fragment文件基于模板生成基础算法执行框架，并通过DP描述将原有算法进行逻辑拆分与通信缝合。
下面为一个Fragment中的`Actor`模块的部分示例，`lerner`部分结构类似。

```python
class Actor(nn.Cell):
    def __init__(self, msrl, rank, duration, episode):
        super(Actor, self).__init__()
        self.msrl = msrl
        self.rank = rank
        self.worker_num = msrl.proc_num
        self.env_per_actor = msrl.collect_environment.num_env_per_worker
        self.allgather = P.AllGather(group=NCCL_WORLD_COMM_GROUP)
        self.broadcast = P.Broadcast(root_rank=0, group=NCCL_WORLD_COMM_GROUP)
        self.assign = P.Assign()
        self.expanddims = P.ExpandDims()
        self.depend = P.Depend()
        self.less = P.Less()

    @mindspore.jit
    def kernel(self):
        action = self.broadcast((self.action_placeholder,))[0]
        action = action[self.action_start:self.action_end]
        new_state, reward, _ = self.msrl.collect_environment.step(action)
        new_state = self.allgather(new_state)
        new_state = self.depend(new_state, action)
        reward = self.allgather(reward)
        reward = self.depend(reward, new_state)
        return reward

    def run(self):
        print('Start actor run ----- episode ', self.episode)
        for i in range(self.episode):
            res = self.gather_state()
            for j in range(self.duration):
                res = self.kernel()
            print('actor episode', i)
```

模板在类`Actor`中，预设了三个方法，分别是`__init__`用于初始化算子和定义变量；`kernel`方法由`jit`装饰，为图模式运行，可以加速性能，该方法中是算法的主逻辑，并添加了通信算子；`run`方法中为算法的循环执行逻辑，并获取算法的输出。

## 使用方式

以[PPO](../../example/ppo/README_CN.md)算法为例, PPO算法默认提供了自动分布式的功能。在PPO对应的[config.py](../algorithm/ppo/config.py)中，预设了[MultiActorSingleLearnerDP](./distribution_policies/multi_actor_single_learner_dp/multi_actor_single_learner_dp.py)。`config.py`文件中增加了`depoly_config`，用于描述分布式所需的基本信息，如分布式策略，节点个数，同步的网络等。

```python
deploy_config = {
    "auto_distribution": True,
    "distribution_policy": MultiActorSingleLearnerDP,
    "worker_num": 2,
    "network": "actor_net",
    "algo_name": "ppo",
    "config": {},
}
```

最后我们可以通过以下命令执行[train.py](../../example/ppo/train.py)文件，实现`MultiActorSingleLearnerDP`描述下的分布式结构。通过替换`config.py`中的`distribution_policy`，可以实现相同算法下的不同分布式策略的切换。

> 目前ppo算法已支持MultiActorSingleLearnerDP 和 SingleLearnerSingleActorWithMultiEnvDP，A3C支持AsyncSingleLearnerMultiActorDP，其余算法正在更新支持中。

```bash
mpirun -n 4 python train.py --enable_distribute True --worker_num 4
```
