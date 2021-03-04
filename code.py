import numpy as np


def PS(n):
    """
    Returns powerset of n (excluding null set)
    """

    from itertools import combinations
    P = []
    for sz in range(1, n+1, 1):
        # create all possible combinations of size i
        temp = combinations(range(n), sz)
        for item in temp:
            P.append(item)
    return P


def support_pairs(A, B):
    """
    Returns all possible support pairs
    """

    pl1_strat_no, pl2_strat_no = A.shape
    pl1_strats = PS(pl1_strat_no)
    pl2_strats = PS(pl2_strat_no)

    result = []
    for support1 in pl1_strats:
        for support2 in pl2_strats:
            if len(support1) == len(support2):
                result.append((support1, support2))
    return result


def find_prob_vector(A, row_sup=None, col_sup=None):
    """
    Return probability vector for mixed startegy
    """
    # Ensure differences between pairs of pure strategies are the same
    M = A[np.array(row_sup)]
    M = M - np.roll(A[np.array(row_sup)], 1, axis=0)
    M = M[:-1]

    # Columns that must be played with prob 0
    zero_columns = set(range(A.shape[1]))
    zero_columns = zero_columns - set(col_sup)

    if len(zero_columns):
        X = [[int(i == j) for i, col in enumerate(M.T)] for j in zero_columns]
        M = np.append(M, X, axis=0,)

    # Ensure have probability vector
    M = np.append(M, np.ones((1, M.shape[1])), axis=0)
    b = np.append(np.zeros(len(M) - 1), [1])

    try:
        prob = np.linalg.solve(M, b)
        for ind in prob:
            if(ind < 0):
                return False
        return prob
    except np.linalg.linalg.LinAlgError:
        return False


def obey_support(strategy, support):
    """
    Test if strategy obeys its support
    """

    for index, value in enumerate(strategy):
        if index in support and value <= 0:
            return False
        elif index not in support and value > 0:
            return False
    return True


def calculate_strat(A, B):
    """
    Return possible startegies 
    """
    result = []
    for pair in support_pairs(A, B):

        s1 = find_prob_vector(B.T, *(pair[::-1]))
        s2 = find_prob_vector(A, *pair)

        if s1 is False:
            continue
        elif s2 is False:
            continue
        elif obey_support(s1, pair[0]) and obey_support(s2, pair[1]):
            result.append((s1, s2, pair[0], pair[1]))
    return result


def IS_NE(strategy, support, payoff):
    """
    Test if strategy is NE
    """

    A, B = payoff

    # row player
    u = strategy[1].reshape(strategy[1].size, 1)

    # utilities of row when col player plays strategy
    row_util = np.dot(A, u)

    # utilities for strategies in support
    row_support_util = row_util[np.array(support[0])]

    # check if best response
    fl1 = (row_util.max() == row_support_util.max())

    # col player
    v = strategy[0].reshape(strategy[0].size, 1)

    # utilities of col when row player plays strategy
    col_util = np.dot(B.T, v)

    # utilities for strategies in support
    col_support_util = col_util[np.array(support[1])]

    # check if best response
    fl2 = (col_util.max() == col_support_util.max())

    return (fl1 and fl2)


def support_enumeration(A, B):
    """
    Obtain NE using support enumeration algo
    """

    result = []
    for s1, s2, sup1, sup2 in calculate_strat(A, B):
        if IS_NE((s1, s2), (sup1, sup2), (A, B)):
            result.append((s1, s2))
    return result


if __name__ == "__main__":

    rows = int(input())
    cols = int(input())
    A = np.zeros((rows, cols))
    B = np.zeros((rows, cols))

    nfg_format = input()
    nfg_format = list(nfg_format.split(" "))

    k = 0
    for j in range(cols):
        for i in range(rows):
            A[i][j] = float(nfg_format[k])
            B[i][j] = float(nfg_format[k+1])
            k = k+2

    A = np.array(A)
    B = np.array(B)

    result = support_enumeration(A, B)

    print(len(result))
    for s1, s2 in result:
        for ele in s1:
            print(ele, end=" ")
        print()
        for ele in s2:
            print(ele, end=" ")
        print()
