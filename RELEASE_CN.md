# MindSpore Reinforcement Release Notes

[View English](./RELEASE.md)

## Reinforcement 0.6.0-rc1 Release Notes

### 主要特性和增强

- [BETA] 支持GAIL(Generative Adversarial Imitation Learning [Jonathan Ho et al..2016](https://proceedings.neurips.cc/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf)) 算法。算法解决了HalfCheetah环境问题，支持CPU，GPU和Ascend后端设备。
- [BETA] 支持C51([Marc G. Bellemare et al..2017](https://arxiv.org/abs/1707.06887)) 算法。算法解决了CartPole环境问题，支持CPU，GPU和Ascend后端设备。
- [BETA] 支持CQL(Conservative Q-Learning [Aviral Kumar et al..2019](https://arxiv.org/pdf/1906.00949)) 算法。算法解决了Hopper环境问题，支持CPU，GPU和Ascend后端设备。
- [BETA] 支持AWAC(Accelerating Online Reinforcement Learning with Offline Datasets [Ashvin Nair et al..2020](https://arxiv.org/abs/2006.09359)) 算法。算法解决了Ant环境问题，支持CPU，GPU和Ascend后端设备。
- [BETA] 支持Dreamer([Danijar Hafner et al..2020](https://arxiv.org/abs/1912.01603)) 算法。算法解决了Walker-walk环境问题，支持GPU后端设备。

### 贡献者

感谢以下人员做出的贡献:

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

欢迎以任何形式对项目提供贡献！

## MindSpore Reinforcement 0.5.0 Release Notes

### 主要特性和增强

- [STABLE] 增加现有接口的中文API文档。
- [STABLE] 增加强化学习多智能体算法QMIX。

### Contributors

感谢以下人员做出的贡献:

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

欢迎以任何形式对项目提供贡献！

## MindSpore Reinforcement 0.3.0 Release Notes

### 主要特性和增强

- [STABLE] 支持DDPG强化学习算法

### 接口变更

#### 后向兼容变更

##### Python接口

- 修改了`Actor`和`Agent`类的接口。它们的方法名被修改成`act(self, phase, params)`和`get_action(self, phase, params)`。除此之外，删除冗余方法(`Actor`类中的`env_setter`, `act_init`, `evaluate`, `reset_collect_actor`, `reset_eval_actor`, `update`, 和`Agent`类中的 `init`, `reset_all`)。修改配置文件中的层级结构，将`actor`目录下的`ReplayBuffer`移出作为`algorithm_config`中的一个单独键值。([Rearrange API PR !29](https://e.gitee.com/mind_spore/repos/mindspore/reinforcement/pulls/29))

- 增加了`Environment`类的虚基类。它提供`step`和`reset`方法以及5个`space`相关的属性(`action_space`, `observation_space`, `reward_space`, `done_space`和`config`)

### Contributors

感谢以下人员作出的贡献：

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

欢迎以任意形式对项目提供贡献!
