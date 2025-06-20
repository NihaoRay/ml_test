import collections

symbols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
           'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '[UNK]']

raw_token_freqs = {'fast_': 4, 'faster_': 3, 'tall_': 5, 'taller_': 4}

token_freqs = {}

for token, freq in raw_token_freqs.items():
    token_freqs[' '.join(list(token))] = raw_token_freqs[token]

# print(token_freqs)


def get_max_freq_pair(token_freqs):
    """返回词内最频繁的连续符号对，其中词来自输入词典token_freqs的键"""
    pairs = collections.defaultdict(int)
    for token, freq in token_freqs.items():
        symbols = token.split()
        for i in range(len(symbols) - 1):
            # “pairs”的键是两个连续符号的元组
            pairs[symbols[i], symbols[i + 1]] += freq
    # 重新生成一个pairs的字典，然后更新频率，选择具有最大值的“pairs”键
    ret_max = max(pairs, key=pairs.get)
    print(ret_max, pairs)
    return ret_max

# print(get_max_freq_pair(token_freqs))


def merge_symbols(max_freq_pair, token_freqs, symbols):
    """作为基于连续符号频率的贪心算法，合并最频繁的连续符号对以产生新符号"""
    symbols.append(''.join(max_freq_pair))
    new_token_freqs = dict()
    for token, freq in token_freqs.items():
        new_token = token.replace(' '.join(max_freq_pair),  ''.join(max_freq_pair))
        new_token_freqs[new_token] = token_freqs[token]
    return new_token_freqs


num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(token_freqs)
    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols)
    print(f'合并# {i + 1}:', max_freq_pair)


print(symbols)
print(list(token_freqs.keys()))


def segment_BPE(tokens, symbols):
    outputs = []
    for token in tokens:
        start, end = 0, len(token)
        cur_output = []
        # 具有符号中可能最长子字的词元段
        while start < len(token) and start < end:
            if token[start: end] in symbols:
                cur_output.append(token[start: end])
                start = end
                end = len(token)
            else:
                end -= 1
        if start < len(token):
            cur_output.append('[UNK]')
        outputs.append(' '.join(cur_output))
    return outputs

tokens = ['tallest_', 'fatter_']
print(segment_BPE(tokens, symbols))











