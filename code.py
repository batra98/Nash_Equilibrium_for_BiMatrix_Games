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


def IS_NE(strategy, support, payoff):
    """
    Test if strategy is NE
    """

    A, B = payoff

    # col player
    v = strategy[0].reshape(strategy[0].size, 1)

    # utilities of col when row player plays strategy
    col_util = np.dot(B.T, v)

    # check if best response
    fl1 = (col_util.max() == (col_util[np.array(support[1])]).max())

    # row player
    u = strategy[1].reshape(strategy[1].size, 1)

    # utilities of row when col player plays strategy
    row_util = np.dot(A, u)

    # check if best response
    fl2 = (row_util.max() == (row_util[np.array(support[0])]).max())

    return (fl1 and fl2)


def find_prob_vector(A, row_sup=None, col_sup=None):
    """
    Return probability vector for mixed startegy
    """

    M = A[np.array(row_sup)]
    M = M - np.roll(M, 1, axis=0)
    M = M[:-1]

    zero_columns = set(range(A.shape[1]))
    zero_columns = zero_columns - set(col_sup)

    if len(zero_columns):
        X = []
        for j in zero_columns:
            temp = []
            for i, col in enumerate(M.T):
                if(i == j):
                    temp.append(1)
                else:
                    temp.append(0)
            X.append(temp)
        M = np.append(M, X, axis=0,)

    I1 = np.ones((1, M.shape[1]))
    M = np.append(M, I1, axis=0)
    I2 = np.zeros(len(M)-1)
    b = np.append(I2, [1])

    try:
        prob = np.linalg.solve(M, b)
        for val in prob:
            if(val < 0):
                return False
        return prob
    except np.linalg.linalg.LinAlgError:
        return False


def calculate_strat(A, B):
    """
    Return possible startegies 
    """
    result = []
    for pair in support_pairs(A, B):

        strat1 = find_prob_vector(B.T, *(pair[::-1]))
        strat2 = find_prob_vector(A, *pair)

        if strat1 is False:
            continue
        elif strat2 is False:
            continue
        elif obey_support(strat1, pair[0]) and obey_support(strat2, pair[1]):
            result.append((strat1, strat2, pair[0], pair[1]))
    return result


def support_enumeration(A, B):
    """
    Obtain NE using support enumeration algo
    """

    result = []
    for strat1, strat2, sup1, sup2 in calculate_strat(A, B):
        if IS_NE((strat1, strat2), (sup1, sup2), (A, B)):
            result.append((strat1, strat2))
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

    if(rows == 1 or cols == 1):

        result = []
        strat1 = np.zeros(rows)
        strat2 = np.zeros(cols)

        if rows == 1:
            strat1[0] = 1.0
            strat2[np.argmax(B)] = 1.0
        else:
            strat1[np.argmax(A)] = 1.0
            strat2[0] = 1.0

        result.append((strat1, strat2))
    else:
        result = support_enumeration(A, B)

    print(len(result))
    for strat1, strat2 in result:
        print_str_strat = ""
        for ind, ele in enumerate(strat1):
            print_str_strat += str(ele) + " "
        print(print_str_strat.strip())
        print_str_strat = ""
        for ind, ele in enumerate(strat2):
            print_str_strat += str(ele) + " "
        print(print_str_strat.strip())
