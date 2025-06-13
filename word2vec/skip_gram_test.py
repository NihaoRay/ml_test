import math
import os
import random
import torch
from d2l import torch as d2l

d2l.DATA_HUB['ptb'] = (d2l.DATA_URL + 'ptb.zip', '319d85e578af0cdc590547f26231e4e31cdf1e42')


def read_ptb():
    """将PTB数据集加载到⽂本⾏的列表中"""
    data_dir = d2l.download_extract('ptb')
    # Readthetrainingset.
    with open(os.path.join(data_dir, 'ptb.train.txt')) as f:
        raw_text = f.read()
    return [line.split() for line in raw_text.split('\n')]


sentences = read_ptb()
#
# print(f'# sentences数: {len(sentences)}')
#
# # 将一个文章数据转换成词汇表
vocab = d2l.Vocab(sentences, min_freq=10)
#
# print(f'vocab size: {len(vocab)}')


# 数据集中的每个词wi将有概率地被丢弃
# 1.删除词频率小于10的词，2.排除未知词元'<unk>'，3.删除频率过大的词。
def subsample(sentences, vocab):
    """下采样高频词"""

    # 排除未知词元'<unk>'与词频率小于10的词（这里我们看不到，这个方法在执行添加词典表函数的时候就已经把小频率词汇去掉了）
    sentences = [[token for token in line if vocab[token] != vocab.unk] for line in sentences]

    # 计算文章预料中的词频率
    counter = d2l.count_corpus(sentences)
    # 计算总频率
    num_tokens = sum(counter.values())

    # 如果在下采样期间保留词元，则返回True，该词的相对比率越高，被丢弃的概率就越大。词频率越大，就是分母越大，越可能比random.uniform(0, 1)小，会被丢弃
    def keep(token):
        return random.uniform(0, 1) < math.sqrt(1e-4 / counter[token] * num_tokens)

    return [[token for token in line if keep(token)] for line in sentences], counter

subsampled, counter = subsample(sentences, vocab)


def compare_counts(token):
    return (
        f'"{token}"的数量：'f'之前={sum([l.count(token) for l in sentences])}, 'f'之后={sum([l.count(token) for l in subsampled])}')

# print(compare_counts('the'))
# print(compare_counts('join'))

# for line in subsampled:
#     print(vocab[line])

# 将词元映射到它们在语料库中的索引，将词转换成ID
corpus = [vocab[line] for line in subsampled]
# print(corpus[:3])



def get_centers_and_contexts(corpus, max_window_size):
    """返回跳元模型中的中⼼词和上下⽂词"""
    centers, contexts = [], []
    for line in corpus:
        # 要形成“中⼼词-上下⽂词”对，每个句⼦⾄少需要有2个词
        if len(line) < 2:
            continue
        centers += line

        for i in range(len(line)): # 上下文窗⼝中间i
            window_size = random.randint(1, max_window_size)
            indices = list(range(max(0, i - window_size),
                                 min(len(line), i + 1 + window_size)))
            # 从上下文词中排除中心词
            indices.remove(i)
            contexts.append([line[idx] for idx in indices])

    return centers, contexts


all_centers, all_contexts = get_centers_and_contexts(corpus, 5)

# tiny_dataset = [list(range(7)), list(range(7, 10))]
# print('数据集', tiny_dataset)
#
# for center, context in zip(*get_centers_and_contexts(tiny_dataset, 2)):
#     print('中心词', center, '的上下文词是', context)


class RandomGenerator:
    """根据n个采样权重在{1,...,n}中随机抽取"""
    def __init__(self, sampling_weights):
        # Exclude
        self.population = list(range(1, len(sampling_weights) + 1))
        self.sampling_weights = sampling_weights
        self.candidates = []
        self.i = 0
    # 产生有分布的随机队列
    def draw(self):
        if self.i == len(self.candidates):
            # 缓存k个随机采样结果 从集群中随机选取k次数据，返回一个列表，可以设置权重
            self.candidates = random.choices(
                self.population, self.sampling_weights, k=10000)
            self.i = 0
        self.i += 1
        return self.candidates[self.i - 1]


generator = RandomGenerator([2, 3, 4])
print([generator.draw() for _ in range(10)])



def get_negatives(all_contexts, vocab, counter, K):
    """返回负采样中的噪声词"""
    # 索引为1、2、...（索引0是词表中排除的未知标记） **0.75表示0.75次幂
    sampling_weights = [counter[vocab.to_tokens(i)] ** 0.75 for i in range(1, len(vocab))]

    all_negatives, generator = [], RandomGenerator(sampling_weights)
    for contexts in all_contexts:
        negatives = []
        while len(negatives) < len(contexts) * K:
            neg = generator.draw()
            # 噪声词不能是上下文词
            if neg not in contexts:
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


all_negatives = get_negatives(all_contexts, vocab, counter, 5)
print(all_negatives)

def batchify(data):
    """返回带有负采样的跳元模型的小批量样本"""
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []

    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [center]
        contexts_negatives += [context + negative + [0] * (max_len - cur_len)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]

    return (torch.tensor(centers).reshape((-1, 1)), torch.tensor(contexts_negatives), torch.tensor(masks), torch.tensor(labels))

x_1 = (1, [2, 2], [3, 3, 3, 3])
x_2 = (1, [2, 2, 2], [3, 3])
batch = batchify((x_1, x_2))















