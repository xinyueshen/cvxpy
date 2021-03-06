__author__ = 'Xinyue'

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.norm import norm
from cvxpy.atoms.affine.hstack import hstack

def col_norm(X, p=2):

    X = Expression.cast_to_const(X)
    vecnorms = [ norm(X[:, i], p) for i in range(X.size[1]) ]
    return hstack(*vecnorms)
