import numpy as np
import operator


def lcs_length(left: str, right: str):
    curr = np.zeros(len(right) + 1, int)
    for l_char in left:
        prev = curr.copy()
        for j, r_char in enumerate(right):
            if l_char == r_char:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])

    return curr


def lcs_hirshberg(left: str, right: str):
    left_len = len(left)
    match left_len:
        case 0:
            return []
        case 1 if left[0] in right:
            return [left[0]]
        case 1:
            return []
        case _:
            i = left_len // 2
            lf, ls = left[:i], left[i:]
            lu = lcs_length(lf, right)              # Левый верхний
            rd = lcs_length(ls[::-1], right[::-1])  # Правый нижний
            sums = map(operator.add, lu, reversed(rd))
            j, _ = max(enumerate(sums), key=lambda s: s[1])
            rf, rs = right[:j], right[j:]
            return lcs_hirshberg(lf, rf) + lcs_hirshberg(ls, rs)
