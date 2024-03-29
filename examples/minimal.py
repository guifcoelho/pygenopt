import pygenopt as opt
from pygenopt.highsapi import HiGHS


def main():
    x1 = opt.Variable('x1')
    x2 = opt.Variable('x2')

    constr1 = opt.LinearConstraint(x2 - x1 >= 2, 'constr1')
    constr2 = opt.LinearConstraint(x1 + x2 >= 0, 'constr2')

    objective_function = x1 + x2 + 5

    m = (
        opt.Model(name="minimal", solver_api=HiGHS)
        .add_vars(x1, x2)
        .add_constrs(constr1, constr2)
        .set_objective(objective_function, is_minimization=True)
        .set_options({'presolve': 'off'})
        .run()
        .fetch_solution()
    )

    print(x1.value, x2.value)

if __name__ == '__main__':
    main()