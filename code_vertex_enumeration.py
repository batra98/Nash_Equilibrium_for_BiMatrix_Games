"""A class for the vertex enumeration algorithm"""
from itertools import product

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


def labels(vertex, halfspaces):
    """
    Return the labels of the facets on which lie a given vertex. This is
    calculated by carrying out the matrix multiplictation.
    Parameters
    ----------
        vertex: a numpy array
        halfspaces: a numpy array
    Returns
    -------
       set
    """
    b = halfspaces[:, -1]
    M = halfspaces[:, :-1]
    return set(np.where(np.isclose(np.dot(M, vertex), -b))[0])


def build_halfspaces(M):
    """
    Build a matrix representation for a halfspace corresponding to:
        Mx <= 1 and x >= 0
    This is of the form:
       [M: -1]
       [-1: 0]
    As specified in
    https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.spatial.HalfspaceIntersection.html
    Parameters
    ----------
        M: a numpy array
    Returns:
    --------
        Numpy array
    """
    number_of_strategies, dimension = M.shape
    b = np.append(-np.ones(number_of_strategies), np.zeros(dimension))
    M = np.append(M, -np.eye(dimension), axis=0)
    halfspaces = np.column_stack((M, b.transpose()))
    return halfspaces

def non_trivial_vertices(halfspaces):
    """
    Returns all vertex, label pairs (ignoring the origin).
    Parameters:
        halfspaces: a numpy array
    Returns:
        generator
    """
    feasible_point = find_feasible_point(halfspaces)
    hs = HalfspaceIntersection(halfspaces, feasible_point)
    hs.close()
    return (
        (v, labels(v, halfspaces))
        for v in hs.intersections
        if not np.all(np.isclose(v, 0)) and max(v) < np.inf
    )

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
    Obtain the Nash equilibria using enumeration of the vertices of the best
    response polytopes.
    Algorithm implemented here is Algorithm 3.5 of [Nisan2007]_
    1. Build best responses polytopes of both players
    2. For each vertex pair of both polytopes
    3. Check if pair is fully labelled
    4. Return the normalised pair
    Returns
    -------
        equilibria: A generator.
    """
    result = []
    if np.min(A) < 0:
        A = A + abs(np.min(A))
    if np.min(B) < 0:
        B = B + abs(np.min(B))

    number_of_row_strategies, row_dimension = A.shape
    max_label = number_of_row_strategies + row_dimension
    full_labels = set(range(max_label))

    row_halfspaces = build_halfspaces(B.transpose())
    col_halfspaces = build_halfspaces(A)

    for row_v, row_l in non_trivial_vertices(row_halfspaces):
        adjusted_row_l = set(
            (label + number_of_row_strategies) % (max_label) for label in row_l
        )

        for col_v, col_l in non_trivial_vertices(col_halfspaces):
            if adjusted_row_l.union(col_l) == full_labels:
                result.append((row_v / sum(row_v), col_v / sum(col_v)))
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
result = vertex_enumeration(A,B)

# for ele in result:
#     print(ele)
tol=10 ** -10

print(len(result))

for s1,s2 in result:
    for ele in s1:
        if ele > tol:
            print(ele,end=" ")
        else:
            print('0.0',end=" ")
    print()
    for ele in s2:
        if ele > tol:
            print(ele,end=" ")
        else:
            print('0.0',end=" ")
    print()
