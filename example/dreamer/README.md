# Dreamer

## Related Paper

1. Hafner et al., 2019 ["Dream to Control: Learning Behaviors by Latent Imagination"](https://arxiv.org/abs/1912.01603)

## Game that whis algrithm used

In Dreamer, we use Waler walk in DeepMind Control Suite(DMC)(<https://github.com/deepmind/dm_control>). This game simulates a human walking. System controls the torque applied on each node of leg.

<!-- <img src="../../docs/images/mpe_simple_spread.gif" alt="mpe_simple_spread" style="zoom: 67%;" /> -->

## How to run Dreamer

User needs to install this dependency:

```shell
pip install dm_control
```

After installation, user can run Dremer by using following command:

```python
python train.py
```

## Supported Platform

Dreamer algorithm currently supports GPU。
