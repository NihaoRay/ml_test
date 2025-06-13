def Levenshtein_Distance_Recursive(str1, str2):
    if len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    elif str1 == str2:
        return 0

    if str1[len(str1) - 1] == str2[len(str2) - 1]:
        d = 0
    else:
        d = 1

    return min(Levenshtein_Distance_Recursive(str1, str2[:-1]) + 1,
               Levenshtein_Distance_Recursive(str1[:-1], str2) + 1,
               Levenshtein_Distance_Recursive(str1[:-1], str2[:-1]) + d)


def Levenshtein_Distance(str1, str2):
    """
    计算字符串 str1 和 str2 的编辑距离
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if (str1[i - 1] == str2[j - 1]):
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]

src_str = 'Simultaneous'
target_str = 'Simultaneous'
print(Levenshtein_Distance(src_str, target_str))


# print(Levenshtein_Distance_Recursive("移动通信基站风光互补系统远程监控设计与实现", "移动通信基站风光互补系统远程设计与实现"))