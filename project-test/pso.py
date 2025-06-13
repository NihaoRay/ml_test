import random
import numpy as np
import matplotlib.pyplot as plt


class PSO:
    def __init__(self, NGEN, possize, low, up):
        """
        particle swarm optimization
        parameter: a list type, like [NGEN, pos_size, var_num_min, var_num_max]
        """
        # 初始化
        # 迭代的代数
        self.NGEN = NGEN
        # 种群大小
        self.pos_size = possize
        # 变量个数
        self.var_num = len(low)
        # 变量的约束范围
        self.bound = []
        self.bound.append(low)
        self.bound.append(up)

        self.pos_x = np.zeros((self.pos_size, self.var_num))    # 所有粒子的位置
        self.pos_v = np.zeros((self.pos_size, self.var_num))    # 所有粒子的速度
        self.p_best = np.zeros((self.pos_size, self.var_num))   # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))   # 全局最优的位置

        # 初始化第0代初始全局最优解
        temp = -1
        for i in range(self.pos_size):
            for j in range(self.var_num):
                self.pos_x[i][j] = random.uniform(self.bound[0][j], self.bound[1][j])
                self.pos_v[i][j] = random.uniform(0, 1)

            self.p_best[i] = self.pos_x[i]      # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, ind_var):
        """
        个体适应值计算
        """
        x1 = ind_var[0]
        x2 = ind_var[1]
        x3 = ind_var[2]
        x4 = ind_var[3]
        y = x1 ** 2 + x2 ** 2 + x3 ** 3 + x4 ** 4
        return y

    def update_operator(self, pos_size):
        """
        更新算子：更新下一时刻的位置和速度
        """
        c1 = 2     # 学习因子，一般为2
        c2 = 2
        w = 0.4    # 自身权重因子
        for i in range(pos_size):
            # 更新速度
            self.pos_v[i] = w * self.pos_v[i] + c1 * random.uniform(0, 1) * (self.p_best[i] - self.pos_x[i]) \
                            + c2 * random.uniform(0, 1) * (self.g_best - self.pos_x[i])

            # 更新位置
            self.pos_x[i] = self.pos_x[i] + self.pos_v[i]

            # 越界保护
            for j in range(self.var_num):
                if self.pos_x[i][j] < self.bound[0][j]:
                    self.pos_x[i][j] = self.bound[0][j]
                if self.pos_x[i][j] > self.bound[1][j]:
                    self.pos_x[i][j] = self.bound[1][j]

            # 更新p_best和g_best
            if self.fitness(self.pos_x[i]) > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pos_x[i]
            if self.fitness(self.pos_x[i]) > self.fitness(self.g_best):
                self.g_best = self.pos_x[i]

    def main(self):
        posobj = []
        self.ng_best = np.zeros((1, self.var_num))[0]
        for gen in range(self.NGEN):

            self.update_operator(self.pos_size)
            posobj.append(self.fitness(self.g_best))
            print('############ Generation {} ############'.format(str(gen + 1)))
            if self.fitness(self.g_best) > self.fitness(self.ng_best):
                self.ng_best = self.g_best.copy()

            print('最好的位置：{}'.format(self.ng_best))
            print('最大的函数值：{}'.format(self.fitness(self.ng_best)))
        print("---- End of (successful) Searching ----")

        plt.figure()
        plt.title("Figure1")
        plt.xlabel("iterators", size=14)
        plt.ylabel("fitness", size=14)
        t = [t for t in range(self.NGEN)]
        plt.plot(t, posobj, color='b', linewidth=2)
        plt.show()


if __name__ == '__main__':
    NGEN = 100
    possize = 100
    low = [1, 1, 1, 1]
    up = [30, 30, 30, 30]

    pso = PSO(NGEN, possize, low, up)
    pso.main()