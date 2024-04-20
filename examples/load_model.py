import pygenopt as opt
from parse_solver import get_solver_api
from minimal import get_problem


def main():
    prob, x1, x2, *_ = get_problem()
    prob.solve().to_mps('load_model.mps')
    x1_val, x2_val = x1.value, x2.value

    prob2 = opt.Problem.load_mps('load_model.mps', solver_api=get_solver_api()).solve()

    print("Solve status:", prob2.solve_status)
    print("Objective function value:", prob2.get_objectivefunction_value())

    x1, x2 = prob2.variables

    assert x1.value == x1_val and x2.value == x2_val

if __name__ == '__main__':
    main()
