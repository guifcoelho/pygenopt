import pygenopt as opt
from pygenopt.xpressapi import Xpress
from pygenopt.highsapi import HiGHS


def main():
    x1 = opt.Variable('x1')
    x2 = opt.Variable('x2')

    constr1 = opt.LinearConstraint(x2 - x1 >= 2, 'constr1')
    constr2 = opt.LinearConstraint(x1 + x2 >= 0, 'constr2')

    objective_function = -(x1 + x2 + 5)

    prob = (
        opt.Problem(name="minimal_problem", solver_api=Xpress)
        .add_vars(x1, x2)
        .add_constrs(constr1, constr2)
        .set_objective(opt.ObjectiveFunction(objective_function, is_minimization=False))
        .solve()
        .fetch_solution()
    )
    print("Solve status:", prob.solve_status)
    print("Objective function value:", prob.get_objectivefunction_value())

    print("x1:", x1.value)
    print("x2:", x2.value)

    prob.to_mps('minimimal.mps')

if __name__ == '__main__':
    main()
