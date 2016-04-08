"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.settings as s
import cvxpy.utilities as u
import cvxpy.interface as intf
from cvxpy.error import SolverError, DCPError
from cvxpy.constraints import EqConstraint, LeqConstraint, PSDConstraint
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.solvers.solver import Solver
from cvxpy.problems.solvers.utilities import SOLVERS
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.expressions.variables.variable import Variable
from cvxpy.atoms.pnorm import pnorm

import warnings
import numpy as np

class Problem(u.Canonical):
    """A convex optimization problem.

    Attributes
    ----------
    objective : Minimize or Maximize
        The expression to minimize or maximize.
    constraints : list
        The constraints on the problem variables.
    """

    # The solve methods available.
    REGISTERED_SOLVE_METHODS = {}

    def __init__(self, objective, constraints=None):
        """

        :rtype : object
        """
        if constraints is None:
            constraints = []
        # Check that objective is Minimize or Maximize.
        if not isinstance(objective, (Minimize, Maximize)):
            raise DCPError("Problem objective must be Minimize or Maximize.")
        # Constraints and objective are immutable.
        self.objective = objective
        self.constraints = constraints
        self._value = None
        self._status = None
        # Cached processed data for each solver.
        self._cached_data = {}
        self._reset_cache()

    def _reset_cache(self):
        """Resets the cached data.
        """
        for solver_name in SOLVERS.keys():
            self._cached_data[solver_name] = ProblemData()

    @property
    def value(self):
        """The value from the last time the problem was solved.

        Returns
        -------
        float or None
        """
        return self._value

    @property
    def status(self):
        """The status from the last time the problem was solved.

        Returns
        -------
        str
        """
        return self._status

    def is_dcp(self):
        """Does the problem satisfy DCP rules?
        """
        return all(exp.is_dcp() for exp in self.constraints + [self.objective])

    def is_dccp(self):
        """Does the problem satisfy DCCP rules?
        """
        flag = True
        for constr in self.constraints + [self.objective]:
            for arg in constr.args:
                if arg.curvature == 'UNKNOWN':
                    flag = False
                    return flag
        return flag

    def iter_DCCP(self,solver,max_iter, tau, miu, tau_max):
        it = 1
        # keep the values from the previous iteration or initialization
        previous_cost = float("inf")
        variable_pres_value = []
        for var in self.variables():
            variable_pres_value.append(var.value)
        # each non-dcp constraint needs a slack variable
        var_slack = []
        for constr in self.constraints:
            if not constr.is_dcp():
                rows, cols = constr.size
                var_slack.append(Variable(rows, cols))

        while it<=max_iter and all(var.value is not None for var in self.variables()):
            constr_new = []
            #cost functions
            if not self.objective.is_dcp():
                temp = self.objective.convexify()
                flag = temp[2]
                flag_var = temp[3]
                while flag == 1:
                    for var in flag_var:
                        var_index = self.variables().index(var)
                        var.value = 0.8*var.value + 0.2* variable_pres_value[var_index]
                    temp = self.objective.convexify()
                    flag = temp[2]
                    flag_var = temp[3]
                cost_new =  temp[0] # new cost function
                for dom in temp[1]: # domain constraints
                    constr_new.append(dom)
            else:
                cost_new = self.objective.args[0]
            #constraints
            count_slack = 0
            for arg in self.constraints:
                if not arg.is_dcp():
                    temp = arg.convexify()
                    flag = temp[2]
                    flag_var = temp[3]
                    while flag == 1:
                        for var in flag_var:
                            var_index = self.variables().index(var)
                            var.value = 0.8*var.value + 0.2* variable_pres_value[var_index]
                        temp = arg.convexify()
                        flag = temp[2]
                        flag_var = temp[3]
                    newcon = temp[0]   #new constraint without slack variable
                    for dom in temp[1]:#domain
                        constr_new.append(dom)
                    right = newcon.args[1] + var_slack[count_slack]
                    constr_new.append(newcon.args[0]<=right)
                    constr_new.append(var_slack[count_slack]>=0)
                    count_slack = count_slack+1
                else:
                    constr_new.append(arg)
            #objective
            if self.objective.NAME == 'minimize':
                for var in var_slack:
                    cost_new += np.ones((var._cols,var._rows))*var*tau
                obj_new = Minimize(cost_new)
            else:
                for var in var_slack:
                    cost_new -= var*tau
                obj_new = Maximize(cost_new)
            #new problem
            prob_new = Problem(obj_new, constr_new)
            variable_pres_value = []
            for var in self.variables():
                variable_pres_value.append(var.value)
            if not var_slack == []:
                if solver is None:
                    print "iteration=",it, "cost value = ", prob_new.solve(), "tau = ", tau
                else:
                    print "iteration=",it, "cost value = ", prob_new.solve(solver = solver), "tau = ", tau
                max_slack = []
                for i in range(len(var_slack)):
                    max_slack.append(np.max(var_slack[i].value))
                max_slack = np.max(max_slack)
                print "max slack = ", max_slack
            else:
                if solver is None:
                    co = prob_new.solve()
                else:
                    co = prob_new.solve(solver = solver)
                print "iteration=",it, "cost value = ", co , "tau = ", tau
            if np.abs(previous_cost - prob_new.value) <= 1e-4: #terminate
                it_real = it
                it = max_iter+1
            else:
                previous_cost = prob_new.value
                it_real = it
                tau = min([tau*miu,tau_max])
                it += 1
        if not var_slack == []:
            return(it_real, tau, max(var_slack[i].value for i in range(len(var_slack))))
        else:
            return(it_real, tau)

    def DCCP_ini(self):
        dom_constr = self.objective.args[0].domain
        for arg in self.constraints:
            for dom in arg.args[0].domain:
                dom_constr.append(dom)
            for dom in arg.args[1].domain:
                dom_constr.append(dom)
        var_store = []
        init_flag = []
        for var in self.variables():
            var_store.append(np.zeros((var._rows,var._cols)))
            init_flag.append(var.value is None)
        times = 3
        for t in range(times):
            ini_cost = 0
            var_ind = 0
            for var in self.variables():
                if init_flag[var_ind]:
                    var.value = np.random.randn(var._rows,var._cols)*10
                ini_cost += pnorm(var-var.value,2)
                var_ind += 1
            ini_obj = Minimize(ini_cost)
            ini_prob = Problem(ini_obj,dom_constr)
            ini_prob.solve()
            var_ind = 0
            for var in self.variables():
                var_store[var_ind] = var_store[var_ind] + var.value/float(times)
                var_ind += 1
        var_ind = 0
        for var in self.variables():
            var.value = var_store[var_ind]
            var_ind += 1


    def DCCP(self, solver,**kwargs):
        if 'max_iter' in kwargs:
            max_iter = kwargs['max_iter']
        else:
            max_iter = 100
        if 'tau' in kwargs:
            tau = kwargs['tau']
        else:
            tau = 0.005
        if 'mu' in kwargs:
            mu = kwargs['mu']
        else:
            mu = 1.2
        if 'tau_max' in kwargs:
            tau_max = kwargs['tau_max']
        else:
            tau_max = 1e8
        # split all the non-affine equality constraints
        constr = []
        for arg in self.constraints:
            if arg.OP_NAME == "==" and not arg.is_dcp():
                sp = arg.split()
                constr.append(sp[0])
                constr.append(sp[1])
            else:
                constr.append(arg)
        self.constraints = constr
        self.DCCP_ini()
        result = self.iter_DCCP(solver,max_iter, tau, mu, tau_max)
        return(result)

    def canonicalize(self):
        """Computes the graph implementation of the problem.

        Returns
        -------
        tuple
            (affine objective,
             constraints dict)
        """
        canon_constr = []
        obj, constr = self.objective.canonical_form
        canon_constr += constr

        for constr in self.constraints:
            canon_constr += constr.canonical_form[1]

        return (obj, canon_constr)

    def variables(self):
        """Returns a list of the variables in the problem.
        """
        vars_ = self.objective.variables()
        for constr in self.constraints:
            vars_ += constr.variables()
        # Remove duplicates.
        return list(set(vars_))

    def parameters(self):
        """Returns a list of the parameters in the problem.
        """
        params = self.objective.parameters()
        for constr in self.constraints:
            params += constr.parameters()
        # Remove duplicates.
        return list(set(params))

    def solve(self, *args, **kwargs):
        """Solves the problem using the specified method.

        Parameters
        ----------
        method : function
            The solve method to use.
        solver : str, optional
            The solver to use.
        verbose : bool, optional
            Overrides the default of hiding solver output.
        solver_specific_opts : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        func_name = kwargs.pop("method", None)
        if func_name is not None:
            func = Problem.REGISTERED_SOLVE_METHODS[func_name]
            return func(self, *args, **kwargs)
        else:
            return self._solve(*args, **kwargs)

    def check_loc_opt(self, Var, point, ep_range):
        constr = self.constraints
        obj = self.objective
        #Var = self.variables()    #but how to know which one is to be check?
        for ind in range(len(Var)-1):         #check the fist variable, others are fixed
            fix_val = Var[ind+1].value
            constr = constr + [ Var[ind+1] == fix_val ]
#       point = Var[0].value
        cost_value = np.zeros((len(ep_range),1))
        var_value = np.zeros((len(ep_range),1))
        for ep_ind in range(len(ep_range)):
            constr_tan, Var_tan = Var[0].tan_sp(ep_range[ep_ind],point)
            constr_opt = constr + constr_tan
            prob_loc_opt = Problem(obj, constr_opt)
            cost_value[ep_ind] = prob_loc_opt.solve()
            var_value[ep_ind] = np.linalg.norm(Var[0].value - point, 'fro')
#        plt.figure(figsize=(8,8))
#        plt.plot(ep_range,cost_value)
#        plt.xlabel('log10 varepsilon', fontsize=16)
#        plt.ylabel('Minimized cost', fontsize=16)
#        plt.figure(figsize=(8,8))
#        plt.plot(np.log10(ep_range),np.log10(var_value))
#        plt.xlabel('log10 varepsilon', fontsize=16)
#        plt.ylabel('distance from the solution', fontsize=16)
        return(cost_value)
        

    @classmethod
    def register_solve(cls, name, func):
        """Adds a solve method to the Problem class.

        Parameters
        ----------
        name : str
            The keyword for the method.
        func : function
            The function that executes the solve method.
        """
        cls.REGISTERED_SOLVE_METHODS[name] = func

    def get_problem_data(self, solver):
        """Returns the problem data used in the call to the solver.

        Parameters
        ----------
        solver : str
            The solver the problem data is for.

        Returns
        -------
        tuple
            arguments to solver
        """
        objective, constraints = self.canonicalize()
        # Raise an error if the solver cannot handle the problem.
        SOLVERS[solver].validate_solver(constraints)
        return SOLVERS[solver].get_problem_data(objective, constraints,
                                                self._cached_data)



    def _solve(self, solver=None, ignore_dcp=False,
               warm_start=False, verbose=False, **kwargs):
        """Solves a DCP compliant optimization problem.

        Saves the values of primal and dual variables in the variable
        and constraint objects, respectively.

        Parameters
        ----------
        solver : str, optional
            The solver to use. Defaults to ECOS.
        ignore_dcp : bool, optional
            Overrides the default of raising an exception if the problem is not
            DCP.
        warm_start : bool, optional
            Should the previous solver result be used to warm start?
        verbose : bool, optional
            Overrides the default of hiding solver output.
        kwargs : dict, optional
            A dict of options that will be passed to the specific solver.
            In general, these options will override any default settings
            imposed by cvxpy.

        Returns
        -------
        float
            The optimal value for the problem, or a string indicating
            why the problem could not be solved.
        """
        if not self.is_dcp() and not ignore_dcp: #if not DCP and not ignore
            #if self.is_dccp():
            #    print("Problem does not follow DCP rules, but follows DCCP rules")
            #    result = self.DCCP(solver,**kwargs)
            #    return(result)
            #else:
            raise DCPError("Problem does not follow DCP rules.")
        else: # is DCP or ignore
            if ignore_dcp and not self.dcp():
                print ("Problem does not follow DCP rules. "
                       "Solving a convex relaxation.")
            objective, constraints = self.canonicalize()
            # Choose a solver/check the chosen solver.
            if solver is None:
                solver_name = Solver.choose_solver(constraints)
                solver = SOLVERS[solver_name]
            elif solver in SOLVERS:
                solver = SOLVERS[solver]
                solver.validate_solver(constraints)
            else:
                raise SolverError("Unknown solver.")

            sym_data = solver.get_sym_data(objective, constraints,
                                       self._cached_data)
            # Presolve couldn't solve the problem.
            if sym_data.presolve_status is None:
                results_dict = solver.solve(objective, constraints,
                                        self._cached_data,
                                        warm_start, verbose, kwargs)
            # Presolve determined problem was unbounded or infeasible.
            else:
                results_dict = {s.STATUS: sym_data.presolve_status}

            self._update_problem_state(results_dict, sym_data, solver)
            return self.value

    def _update_problem_state(self, results_dict, sym_data, solver):
        """Updates the problem state given the solver results.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Parameters
        ----------
        results_dict : dict
            A dictionary containing the solver results.
        sym_data : SymData
            The symbolic data for the problem.
        solver : Solver
            The solver type used to obtain the results.
        """
        if results_dict[s.STATUS] in s.SOLUTION_PRESENT:
            self._save_values(results_dict[s.PRIMAL], self.variables(),
                              sym_data.var_offsets)
            # Not all solvers provide dual variables.
            if s.EQ_DUAL in results_dict:
                self._save_dual_values(results_dict[s.EQ_DUAL],
                                       sym_data.constr_map[s.EQ],
                                       [EqConstraint])
            if s.INEQ_DUAL in results_dict:
                self._save_dual_values(results_dict[s.INEQ_DUAL],
                                       sym_data.constr_map[s.LEQ],
                                       [LeqConstraint, PSDConstraint])
            # Correct optimal value if the objective was Maximize.
            value = results_dict[s.VALUE]
            self._value = self.objective.primal_to_result(value)
        # Infeasible or unbounded.
        elif results_dict[s.STATUS] in s.INF_OR_UNB:
            self._handle_no_solution(results_dict[s.STATUS])
        # Solver failed to solve.
        else:
            raise SolverError(
                "Solver '%s' failed. Try another solver." % solver.name()
            )
        self._status = results_dict[s.STATUS]

    def unpack_results(self, solver_name, results_dict):
        """Parses the output from a solver and updates the problem state.

        Updates problem.status, problem.value and value of
        primal and dual variables.

        Assumes the results are from the given solver.

        Parameters
        ----------
        solver_name : str
            The name of the solver being used.
        results_dict : dict
            The solver output.
        """
        if solver_name not in SOLVERS:
            raise SolverError("Unknown solver.")
        solver = SOLVERS[solver_name]

        objective, constraints = self.canonicalize()
        sym_data = solver.get_sym_data(objective, constraints,
                                       self._cached_data)
        data = {s.DIMS: sym_data.dims, s.OFFSET: 0}
        results_dict = solver.format_results(results_dict, data,
                                             self._cached_data)
        self._update_problem_state(results_dict, sym_data, solver)

    def _handle_no_solution(self, status):
        """Updates value fields when the problem is infeasible or unbounded.

        Parameters
        ----------
        status: str
            The status of the solver.
        """
        # Set all primal and dual variable values to None.
        for var_ in self.variables():
            var_.save_value(None)
        for constr in self.constraints:
            constr.save_value(None)
        # Set the problem value.
        if status in [s.INFEASIBLE, s.INFEASIBLE_INACCURATE]:
            self._value = self.objective.primal_to_result(np.inf)
        elif status in [s.UNBOUNDED, s.UNBOUNDED_INACCURATE]:
            self._value = self.objective.primal_to_result(-np.inf)

    def _save_dual_values(self, result_vec, constraints, constr_types):
        """Saves the values of the dual variables.

        Parameters
        ----------
        result_vec : array_like
            A vector containing the dual variable values.
        constraints : list
            A list of the LinEqConstr/LinLeqConstr in the problem.
        constr_types : type
            A list of constraint types to consider.
        """
        constr_offsets = {}
        offset = 0
        for constr in constraints:
            constr_offsets[constr.constr_id] = offset
            offset += constr.size[0]*constr.size[1]
        active_constraints = []
        for constr in self.constraints:
            # Ignore constraints of the wrong type.
            if type(constr) in constr_types:
                active_constraints.append(constr)
        self._save_values(result_vec, active_constraints, constr_offsets)

    def _save_values(self, result_vec, objects, offset_map):
        """Saves the values of the optimal primal/dual variables.

        Parameters
        ----------
        results_vec : array_like
            A vector containing the variable values.
        objects : list
            The variables or constraints where the values will be stored.
        offset_map : dict
            A map of object id to offset in the results vector.
        """
        if len(result_vec) > 0:
            # Cast to desired matrix type.
            result_vec = intf.DEFAULT_INTF.const_to_matrix(result_vec)
        for obj in objects:
            rows, cols = obj.size
            if obj.id in offset_map:
                offset = offset_map[obj.id]
                # Handle scalars
                if (rows, cols) == (1, 1):
                    value = intf.index(result_vec, (offset, 0))
                else:
                    value = intf.DEFAULT_INTF.zeros(rows, cols)
                    intf.DEFAULT_INTF.block_add(value,
                        result_vec[offset:offset + rows*cols],
                        0, 0, rows, cols)
                offset += rows*cols
            else: # The variable was multiplied by zero.
                value = intf.DEFAULT_INTF.zeros(rows, cols)
            obj.save_value(value)

    def __str__(self):
        if len(self.constraints) == 0:
            return str(self.objective)
        else:
            subject_to = "subject to "
            lines = [str(self.objective),
                     subject_to + str(self.constraints[0])]
            for constr in self.constraints[1:]:
                lines += [len(subject_to)*" " + str(constr)]
            return '\n'.join(lines)

    def __repr__(self):
        return "Problem(%s, %s)" % (repr(self.objective),
                                    repr(self.constraints))

    def __neg__(self):
        return Problem(-self.objective, self.constraints)

    def __add__(self, other):
        if not isinstance(other, Problem):
            return NotImplemented
        return Problem(self.objective + other.objective,
                       list(set(self.constraints + other.constraints)))

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return NotImplemented

    def __sub__(self, other):
        if not isinstance(other, Problem):
            return NotImplemented
        return Problem(self.objective - other.objective,
                       list(set(self.constraints + other.constraints)))

    def __rsub__(self, other):
        if other == 0:
            return -self
        else:
            return NotImplemented

    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Problem(self.objective * other, self.constraints)

    __rmul__ = __mul__

    def __div__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        return Problem(self.objective * (1.0/other), self.constraints)

    __truediv__ = __div__
