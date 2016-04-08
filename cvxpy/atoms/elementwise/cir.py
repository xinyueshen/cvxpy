import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
import numpy as np
#from cvxpy.atoms.elementwise.power import power
from .power import power
from fractions import Fraction

class cir(Elementwise):
    """Elementwise:  math:`1-sqrt(2x-x^2)`.
    """
    def __init__(self,x):
        super(cir, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        """Returns the value of 1-sqrt(2x-x^2).
        """
        if values[0]<1:
            y = 1-np.sqrt(2*values[0]-values[0]*values[0])
        else:
            y = 0
        return y

    def sign_from_args(self):
        return u.Sign.UNKNOWN

    def func_curvature(self):
        return u.Curvature.CONVEX

    def monotonicity(self):
        return [u.monotonicity.DECREASING]

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        # min 1-sqrt(2z-z^2)
        # s.t. x>=0, z<=1, z = x+s, s<=0
        x = arg_objs[0]
        z = lu.create_var(size)
        s = lu.create_var(size)
        zeros = lu.create_const(np.mat(np.zeros(size)),size)
        ones = lu.create_const(np.mat(np.ones(size)),size)
        z2, constr_square = power.graph_implementation([z],size, (2, (Fraction(1,2), Fraction(1,2))))
        two_z = lu.sum_expr([z,z])
        sub = lu.sub_expr(two_z, z2)
        sq, constr_sqrt = power.graph_implementation([sub],size, (Fraction(1,2), (Fraction(1,2), Fraction(1,2))))
        obj = lu.sub_expr(ones, sq)
        constr = [lu.create_eq(z, lu.sum_expr([x,s]))]+[lu.create_leq(zeros,x)]+[lu.create_leq(z, ones)]+[lu.create_leq(s,zeros)]+constr_square+constr_sqrt
        return (obj, constr)
        
        
