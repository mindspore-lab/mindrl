# MindSpore Reinforcement Release Notes

[查看中文](./RELEASE_CN.md)

## Reinforcement 0.6.0-alpha Release Notes

### Major Features and Improvements

- [BETA] Support GAIL(Generative Adversarial Imitation Learning [Jonathan Ho et al..2016](https://proceedings.neurips.cc/paper/2016/file/cc7e2b878868cbae992d1fb743995d8f-Paper.pdf)) Algorithm. The algorithms are tuned on HalfCheetah environment and support GPU backends.
- [BETA] Support C51([Marc G. Bellemare et al..2017](https://arxiv.org/abs/1707.06887)) Algorithm. The algorithms are tuned on CartPole environment and support CPU backends.
- [BETA] Support CQL(Conservative Q-Learning [Aviral Kumar et al..2019](https://arxiv.org/pdf/1906.00949)) Algorithm. The algorithms are tuned on Hopper environment and support CPU and GPU backends.
- [BETA] Support AWAC(Accelerating Online Reinforcement Learning with Offline Datasets [Ashvin Nair et al..2020](https://arxiv.org/abs/2006.09359)) Algorithm. The algorithms are tuned on Ant environment and support CPU and GPU backends.
- [BETA] Support Dreamer([Danijar Hafner et al..2020](https://arxiv.org/abs/1912.01603)) Algorithm. The algorithms are tuned on Walker-walk environment and support GPU backends.

### Contributors

Thanks goes to these wonderful people:

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

## MindSpore Reinforcement 0.5.0 Release Notes

### Major Features and Improvements

- [STABLE] Add Chinese version of all existed API.
- [STABLE] Add reinforcement learning multi-agent algorithm QMIX.

### Contributors

Thanks goes to these wonderful people:

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

## MindSpore Reinforcement 0.3.0 Release Notes

### Major Features and Improvements

- [STABLE] Support DDPG reinforcement learning algorithm.

### API Change

#### Backwards Compatible Change

##### Python API

- Change the API of following classes: `Actor`, `Agent`. Their function names change to `act(self, phase, params)` and `get_action(self, phase, params)`. Moreover, some useless functions are deleted (`env_setter`, `act_init`, `evaluate`, `reset_collect_actor`, `reset_eval_actor, update` in `Actor`class, and `init`, `reset_all` in `Agent` class). Also the hierarchy relationship of configuration file changes. `ReplayBuffer`is moved out from the directory `actor`, and becomes a new key in `algorithm config`. ([Rearrange API PR !29](https://e.gitee.com/mind_spore/repos/mindspore/reinforcement/pulls/29))
- Add the virtual base class of `Environment` class. It has `step`, `reset`functions and 5 `space` properties (`action_space`, `observation_space`, `reward_space`, `done_space` and `config`)

### Contributors

Thanks goes to these wonderful people:

Pro. Peter, Huanzhou Zhu, Bo Zhao, Gang Chen, Weifeng Chen, Liang Shi, Yijie Chen.

Contributions of any kind are welcome!
