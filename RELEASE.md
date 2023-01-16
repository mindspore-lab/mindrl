# MindSpore Reinforcement Release Notes

[查看中文](./RELEASE_CN.md)

## MindSpore Reinforcement 0.5.0 Release Notes

### Major Features and Improvements

- [STABLE] Add Chinese version of all existed API.
- [STABLE] Add reinforcement learning multi-agent algorithm QMIX.

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
