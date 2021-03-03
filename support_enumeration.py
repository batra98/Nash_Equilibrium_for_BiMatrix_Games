"""A class for a normal form game"""
import warnings
from itertools import chain, combinations

import numpy as np


def powerset(n):
    """
    A power set of range(n)
    Based on recipe from python itertools documentation:
    https://docs.python.org/2/library/itertools.html#recipes
    """
    return chain.from_iterable(combinations(range(n), r) for r in range(n + 1))


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


def potential_support_pairs(A, B, non_degenerate=False):
    """
    A generator for the potential support pairs
    Returns
    -------
        A generator of all potential support pairs
    """
    p1_num_strategies, p2_num_strategies = A.shape
    for support1 in (s for s in powerset(p1_num_strategies) if len(s) > 0):
        for support2 in (
            s
            for s in powerset(p2_num_strategies)
            if (len(s) > 0 and not non_degenerate) or len(s) == len(support1)
        ):
            yield support1, support2


def indifference_strategies(A, B, non_degenerate=False, tol=10 ** -16):
    """
    A generator for the strategies corresponding to the potential supports
    Returns
    -------
        A generator of all potential strategies that are indifferent on each
        potential support. Return False if they are not valid (not a
        probability vector OR not fully on the given support).
    """
    if non_degenerate:
        tol = min(tol, 0)

    for pair in potential_support_pairs(A, B, non_degenerate=non_degenerate):
        s1 = solve_indifference(B.T, *(pair[::-1]))
        s2 = solve_indifference(A, *pair)

        if obey_support(s1, pair[0], tol=tol) and obey_support(
            s2, pair[1], tol=tol
        ):
            yield s1, s2, pair[0], pair[1]


def obey_support(strategy, support, tol=10 ** -16):
    """
    Test if a strategy obeys its support
    Parameters
    ----------
        strategy: a numpy array
            A given strategy vector
        support: a numpy array
            A strategy support
    Returns
    -------
        A boolean: whether or not that strategy does indeed have the given
        support
    """
    if strategy is False:
        return False
    if not all(
        (i in support and value > tol) or (i not in support and value <= tol)
        for i, value in enumerate(strategy)
    ):
        return False
    return True


def is_ne(strategy_pair, support_pair, payoff_matrices):
    """
    Test if a given strategy pair is a pair of best responses
    Parameters
    ----------
        strategy_pair: a 2-tuple of numpy arrays
        support_pair: a 2-tuple of numpy arrays
    """
    A, B = payoff_matrices
    # Payoff against opponents strategies:
    u = strategy_pair[1].reshape(strategy_pair[1].size, 1)
    row_payoffs = np.dot(A, u)

    v = strategy_pair[0].reshape(strategy_pair[0].size, 1)
    column_payoffs = np.dot(B.T, v)

    # Pure payoffs on current support:
    row_support_payoffs = row_payoffs[np.array(support_pair[0])]
    column_support_payoffs = column_payoffs[np.array(support_pair[1])]

    return (
        row_payoffs.max() == row_support_payoffs.max()
        and column_payoffs.max() == column_support_payoffs.max()
    )


def support_enumeration(A, B, non_degenerate=False, tol=10 ** -16):
    """
    Obtain the Nash equilibria using support enumeration.
    Algorithm implemented here is Algorithm 3.4 of [Nisan2007]_
    1. For each k in 1...min(size of strategy sets)
    2. For each I,J supports of size k
    3. Solve indifference conditions
    4. Check that have Nash Equilibrium.
    Returns
    -------
        equilibria: A generator.
    """
    count = 0
    for s1, s2, sup1, sup2 in indifference_strategies(
        A, B, non_degenerate=non_degenerate, tol=tol
    ):
        if is_ne((s1, s2), (sup1, sup2), (A, B)):
            count += 1
            yield s1, s2
    if count % 2 == 0:
        warning = """
An even number of ({}) equilibria was returned. This
indicates that the game is degenerate. Consider using another algorithm
to investigate.
                  """.format(
            count
        )
        warnings.warn(warning, RuntimeWarning)


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
result = support_enumeration(A,B,True)

for ele in result:
    print(ele)
# print(len(result))

# for s1,s2 in result:
#     for ele in s1:
#         print(ele,end=" ")
#     print()
#     for ele in s2:
#         print(ele,end=" ")
#     print()