import pygenopt as opt
from pygenopt.solvers import XpressApi, HighsApi


def main():
    x1 = opt.Variable('x1')
    x2 = opt.Variable('x2')

    constr1 = opt.LinearConstraint(-x1 + x2 >= 2, 'constr1')
    constr2 = opt.LinearConstraint(x1 + x2 >= 0, 'constr2')

    objective_function = -(x1 + x2 + 5)

    prob = (
        opt.Problem(name="minimal_problem", solver_api=XpressApi)
        .add_vars(x1, x2)
        .add_constrs(constr1, constr2)
        .set_objective(opt.ObjectiveFunction(objective_function, is_minimization=False))
        .solve()
    )
    x1_val, x2_val = x1.value, x2.value

    prob2 = prob.sync().solve()

    assert prob2.variables[0].value == x1_val and prob2.variables[1].value == x2_val

if __name__ == "__main__":
    main()