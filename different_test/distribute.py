# import ray
# ray.init()
#
# @ray.remote
# def f(x):
#     return x * x
#
# futures = [f.remote(i) for i in range(4)]
# print(ray.get(futures)) # [0, 1, 4, 9]


print(3 != 4 < 2)