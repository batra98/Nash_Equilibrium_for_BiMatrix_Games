import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection
from itertools import product


def build_halfspaces(M):
    number_of_strategies, dimension = M.shape
    b = np.append([-1 for i in range(number_of_strategies)],
                  [0 for i in range(dimension)])
    I = np.eye(dimension)
    M = np.vstack((M, -I))
    halfspaces = np.column_stack((M, b.T))
    return halfspaces


def non_trivial_vertices(halfspaces):
    feasible_point = find_feasible_point(halfspaces)
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    hs.close()
    return ((v, set(np.where(np.isclose(np.dot(halfspaces[:, :-1], v), -halfspaces[:, -1]))[0])) for v in hs.intersections if not np.all(np.isclose(v, 0)) and max(v) < np.inf)


def find_feasible_point(halfspaces):
    """
    Use linear programming to find a point inside the halfspaces (needed to
    define it).
    Code taken from scipy documentation:
    https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.HalfspaceIntersection.html
    Parameters
    ----------
        halfspaces: a matrix representation of halfspaces
    Returns:
    --------
        numpy array
    """
    norm_vector = np.reshape(
        np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1)
    )
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b)
    return res.x[:-1]


def vertex_enumeration(A, B):
    """
    Return NE using vertex enumeration method
    """

    if np.min(A) < 0:
        A = A + abs(np.min(A))

    if np.min(B) < 0:
        B = B + abs(np.min(B))

    number_of_row_strategies, row_dimension = A.shape
    max_label = number_of_row_strategies + row_dimension
    full_labels = set(range(max_label))

    row_halfspaces = build_halfspaces(B.T)
    col_halfspaces = build_halfspaces(A)

    result = []
    for row_v, row_l in non_trivial_vertices(row_halfspaces):
        adjusted_row_l = set((label + number_of_row_strategies) %
                             (max_label) for label in row_l)
        for col_v, col_l in non_trivial_vertices(col_halfspaces):
            if adjusted_row_l.union(col_l) == full_labels:
                result.append((row_v / sum(row_v), col_v / sum(col_v)))
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

    result = vertex_enumeration(A, B)
    tol = 10 ** -10

    print(len(result))
    for s1, s2 in result:
        for ele in s1:
            if ele > tol:
                print(ele, end=" ")
            else:
                print('0.0', end=" ")
        print()
        for ele in s2:
            if ele > tol:
                print(ele, end=" ")
            else:
                print('0.0', end=" ")
        print()
