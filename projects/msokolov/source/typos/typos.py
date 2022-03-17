import numpy as np
import operator

qwerty = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]

GAP = -1
MATCH = 1
ADD = -2
DELETE = -2
MISMATCH = -11


def __pos_fold(pos: [int]):
    for i, p in enumerate(pos):
        if p != -1:
            return i, p

    return None


def distance(xchar: str, ychar: str):
    xpos = __pos_fold(np.char.find(qwerty, xchar.lower()))
    ypos = __pos_fold(np.char.find(qwerty, ychar.lower()))

    if xpos and ypos:
        level_dif = abs(xpos[0] - ypos[0])
        pos_dif = abs(xpos[1] - ypos[1])
        return -(level_dif + pos_dif)
    else:
        return MISMATCH


def nw_score(x: str, y: str):
    curr = np.arange(0, -(len(y) + 1), GAP)
    for xchar in x:
        prev = curr.copy()
        for j, ychar in enumerate(y):
            add, delete, change = prev[j + 1] + ADD, curr[j] + DELETE, prev[j]
            if xchar != ychar:
                change += distance(xchar, ychar)
            else:
                change += MATCH
            curr[j + 1] = max(add, delete, change)

        curr[0] = prev[0] + GAP

    return curr


def hirshberg(x: str, y: str):
    x_len = len(x)
    y_len = len(y)
    if x_len <= 1 or y_len <= 1:
        return nw_score(x, y)[-1]
    else:
        i = x_len // 2
        lf, ls = x[:i], x[i:]
        lu = nw_score(lf, y)  # NW для левого верхнего угла
        rd = nw_score(ls[::-1], y[::-1])  # NW для правого нижнего угла
        sums = map(operator.add, lu, reversed(rd))
        j, score = max(enumerate(sums), key=lambda s: s[1])
        rf, rs = y[:j], y[j:]

        xscore = hirshberg(lf, rf)
        yscore = hirshberg(ls, rs)
        return xscore + yscore


def try_correct(dictionary: [str], text: str):
    hish = enumerate(map(lambda t: hirshberg(t, text), dictionary))
    answer_idx, score = max(hish, key=lambda v: v[1])

    if score < -5:
        return text

    return dictionary[answer_idx]

# def comp_score(xchar: str, ychar: str):
#     if xchar == ychar:
#         return MATCH
#     elif xchar != ' ' and ychar == ' ':
#         return GAP
#     elif xchar == ' ' and ychar != ' ':
#         return DELETE
#     else:
#         return distance(xchar, ychar)
#
#
# def nw_score(x: str, y: str):
#     curr = np.arange(0, -(len(y) + 1), GAP)
#     for xchar in x:
#         prev = curr.copy()
#         for j, ychar in enumerate(y):
#             score = comp_score(xchar, ychar)
#             curr[j + 1] = max(curr[j], prev[j], prev[j + 1]) + score
#         curr[0] = prev[0] + GAP
#
#     return curr
#
#
# def hirshberg(x: str, y: str):
#     x_len = len(x)
#     match x_len:
#         case 0:
#             return '', 0
#         case 1 if x[0] in y:
#             return x[0], MATCH
#         case 1:
#             return '', 0
#         case _:
#             i = x_len // 2
#             lf, ls = x[:i], x[i:]
#             lu = nw_score(lf, y)              # NW для левого верхнего угла
#             rd = nw_score(ls[::-1], y[::-1])  # NW для правого нижнего угла
#             sums = map(operator.add, lu, reversed(rd))
#             j, score = max(enumerate(sums), key=lambda s: s[1])
#             rf, rs = y[:j], y[j:]
#
#             xf, xscore = hirshberg(lf, rf)
#             yf, yscore = hirshberg(ls, rs)
#             return xf + yf, xscore + yscore
#
#
# def try_correct(dictionary: [str], text: str):
#     hish = enumerate(map(lambda t: hirshberg(t, text), dictionary))
#     answer_idx, score = max(hish, key=lambda v: v[1][1])
#
#     if score[1] <= 0:
#         return text
#
#     return dictionary[answer_idx]
