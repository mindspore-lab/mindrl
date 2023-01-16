# 蒙特卡洛树搜索

## 相关论文

 Coulom, 2006 [Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search](https://link.springer.com/chapter/10.1007/978-3-540-75538-8_7)

## 蒙特卡洛树搜索介绍

蒙特卡洛树搜索（MCTS）是一种决策搜索算法，以MCTS为基础的强化学习算法（如AlphaGo）获得成功之后，MCTS在强化学习算法中的应用越来越广泛。MindSpore Reinforcement v0.5.0提供了通用可扩展的MCTS算法框架。开发者可以使用Python在强化学习算法中直接调用框架内置的MCTS算法，也可以通过扩展来完成自定义逻辑，框架自动将算法编译成MindSpore计算图，实现高效执行。

<img src="../../docs/images/mcts.png" alt="mcts" style="zoom: 50%;" />

## 使用的游戏

在目前实现的蒙特卡洛树搜索中，我们使用井字棋作为游戏。井字棋是一款有名的纸笔游戏[en.wikipedia.org/wiki/Tic-tac-toe](en.wikipedia.org/wiki/Tic-tac-toe)。这个游戏的规则是两个玩家在一个3X3的格子上交互的画O和X。当三个相同的标记在水平，垂直或者对角线连成一条线时，对应的玩家将获得胜利。

## 如何运行蒙特卡洛树搜索

```python
python mcts_demo.py
```

## 支持平台

蒙特卡洛树搜索支持GPU和CPU。

