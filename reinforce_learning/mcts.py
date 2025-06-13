import numpy as np
import math
import random

# 定义一个节点类，表示搜索树中的一个节点
class Node:
    def __init__(self, parent, action):
        self.parent = parent    # 父节点
        self.action = action    # 到达该节点所采取的动作
        self.children = []      # 子节点列表
        self.Q = 0              # 节点的累计奖励
        self.N = 0              # 节点的访问次数

# 定义一个蒙特卡洛树搜索类
class MCTS:
    def __init__(self, root):
        self.root = root        # 根节点

    # 选择一个节点用于扩展
    def selection(self, node):
        if len(node.children) == 0:
            return node

        # 对子节点按照UCB1公式进行排序
        children = sorted(node.children, key=lambda c: c.Q / c.N + math.sqrt(2 * math.log(node.N) / c.N), reverse=True)

        # 递归选择最优子节点
        return self.selection(children[0])

    # 扩展选定的节点
    def expansion(self, node, env):
        actions = env.get_legal_actions()
        random.shuffle(actions)

        # 添加新的子节点
        for action in actions:
            child = Node(node, action)
            node.children.append(child)

        # 返回扩展出的第一个子节点
        return node.children[0]

    # 模拟一次随机走动并返回奖励
    def simulation(self, node, env):
        reward = env.get_reward()

        while not env.is_terminal():
            action = random.choice(env.get_legal_actions())
            env.step(action)
            reward = env.get_reward()

        return reward

    # 更新节点价值和访问次数
    def backpropagation(self, node, reward):
        while node is not None:
            node.N += 1
            node.Q += reward
            reward = -reward
            node = node.parent

    # 返回最优动作
    def get_best_action(self, node):
        # 对子节点按照平均奖励进行排序
        children = sorted(node.children, key=lambda c: c.Q / c.N, reverse=True)
        return children[0].action

    # 运行指定次数的模拟，并返回最优动作
    def run(self, env, n):
        for i in range(n):
            # 选择节点并扩展
            node = self.selection(self.root)
            node = self.expansion(node, env)

            # 模拟并更新节点价值和访问次数
            reward = self.simulation(node, env)
            self.backpropagation(node, reward)

        # 返回最优动作
        return self.get_best_action(self.root)



if __name__=="__main__":
    MCTS()