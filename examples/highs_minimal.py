import pygenopt as opt
from pygenopt.highsapi import HiGHS


if __name__ == '__main__':
    x1 = opt.Variable('x1')
    x2 = opt.Variable('x2')

    constr1 = opt.LinearConstraint(x2 - x1 >= 2, 'constr1')
    constr2 = opt.LinearConstraint(x1 + x2 >= 0, 'constr2')

    m = (
        opt.Model()
        .add_vars(x1, x2)
        .add_constrs(constr1, constr2)
        .set_objective(x1 + x2)
        .set_options({'presolve': 'off'})
        .run(HiGHS)
        .fetch_solution()
    )

    print(x1.value, x2.value)
