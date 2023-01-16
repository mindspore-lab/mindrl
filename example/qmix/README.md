# QMIX

## Related Paper

1. Rashid et al., 2018 ["QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning"](https://arxiv.org/pdf/1803.11485.pdf)

## Game that this algorithm used

In QMIX, we use [SMAC - StarCraft Multi-Agent Challenge](https://github.com/oxwhirl/smac). SMAC is WhiRL's environment for research in the field of collaborative multi-agent reinforcement learning (MARL) based on Blizzard's StarCraft II RTS game. SMAC makes use of Blizzard's StarCraft II Machine Learning API and DeepMind's PySC2 to provide a convenient interface for autonomous agents to interact with StarCraft II, getting observations and performing actions. Unlike the PySC2, SMAC concentrates on decentralised micromanagement scenarios, where each unit of the game is controlled by an individual RL agent.

## How to run QMIX

QMIX depends on SMAC, thus user needs to install SMAC before running QMIX. For details installation instruction, please have a look at SMAC official Github: [https://github.com/oxwhirl/smac](https://github.com/oxwhirl/smac).

After installation, user can run QMIX by using following command:

```python
python train.py --device_target [YOUR DEVICE]
```

## Supported Platform

QMIX algorithm GPU and CPU.
