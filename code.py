import numpy as np


def powerset(n):
    """
    return powerset of n (excluding null set)
    """

    from itertools import combinations
    P = []
    for i in range(1, n+1, 1):
        temp = combinations(range(n), i)
        for j in temp:
            P.append(j)
    return P


def potential_support_pairs(A, B):
    """
    return all possible support pairs
    """

    pl1_strat_no, pl2_strat_no = A.shape
    pl1_strats = powerset(pl1_strat_no)
    pl2_strats = powerset(pl2_strat_no)

    result = []
    for support1 in pl1_strats:
        for support2 in pl2_strats:
            if len(support1) == len(support2):
                result.append((support1, support2))
    return result


def solve_indifference(A, rows=None, columns=None):
    """
    Solve the indifference for a payoff matrix assuming support for the
    strategies given by columns
    Finds vector of probabilities that makes player indifferent between
    rows.  (So finds probability vector for corresponding column player)
    Parameters
    ----------
        A: a 2 dimensional numpy array (A payoff matrix for the row player)
        rows: the support played by the row player
        columns: the support player by the column player
    Returns
    -------
        A numpy array:
        A probability vector for the column player that makes the row
        player indifferent. Will return False if all entries are not >= 0.
    """
    # Ensure differences between pairs of pure strategies are the same
    M = (A[np.array(rows)] - np.roll(A[np.array(rows)], 1, axis=0))[:-1]
    # Columns that must be played with prob 0
    zero_columns = set(range(A.shape[1])) - set(columns)

    if zero_columns != set():
        M = np.append(
            M,
            [[int(i == j) for i, col in enumerate(M.T)] for j in zero_columns],
            axis=0,
        )

    # Ensure have probability vector
    M = np.append(M, np.ones((1, M.shape[1])), axis=0)
    b = np.append(np.zeros(len(M) - 1), [1])

    try:
        prob = np.linalg.solve(M, b)
        if all(prob >= 0):
            return prob
        return False
    except np.linalg.linalg.LinAlgError:
        return False


def obey_support(strategy, support):
    """
    Test if strategy obeys its support
    """

    if strategy is False:
        return False

    if not all((i in support and value > 0) or (i not in support and value <= 0) for i, value in enumerate(strategy)):
        return False
    return True


def indifference_strategies(A, B):
    """
    Return possible startegies 
    """
    result = []
    for pair in potential_support_pairs(A, B):

        s1 = solve_indifference(B.T, *(pair[::-1]))
        s2 = solve_indifference(A, *pair)

        if obey_support(s1, pair[0]) and obey_support(s2, pair[1]):
            result.append((s1, s2, pair[0], pair[1]))
    return result


def is_ne(strategy, support, payoff):
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
    fl1 = (row_util.max() == row_support_util[0])

    # col player
    v = strategy[0].reshape(strategy[0].size, 1)

    # utilities of col when row player plays strategy
    col_util = np.dot(B.T, v)

    # utilities for strategies in support
    col_support_util = col_util[np.array(support[1])]

    # check if best response
    fl2 = (col_util.max() == col_support_util.max())

    return fl1 and fl2


def support_enumeration(A, B):
    """
    Obtain NE using support enumeration algo
    """

    result = []
    for s1, s2, sup1, sup2 in indifference_strategies(A, B):
        if is_ne((s1, s2), (sup1, sup2), (A, B)):
            result.append((s1, s2))
    return result


rows = int(input())
cols = int(input())

nfg_format = input()
nfg_format = list(nfg_format.split(" "))

A = np.zeros((rows,cols))
B = np.zeros((rows,cols))


k = 0
for j in range(cols):
    for i in range(rows):
        # print(nfg_format[k],nfg_format[k+1])
        A[i][j] = float(nfg_format[k])
        B[i][j] = float(nfg_format[k+1])

        k = k+2

A = np.array(A)
B = np.array(B)

# A = np.array([[2, 0], [0, 1]])
# B = np.array([[1, 0], [0, 2]])

# A = np.array([[-2, -1], [-10, -5]])
# B = np.array([[-2, -10], [-1, -5]])
result = support_enumeration(A,B)

print(len(result))

for s1,s2 in result:
    for ele in s1:
        print(ele,end=" ")
    print()
    for ele in s2:
        print(ele,end=" ")
    print()
