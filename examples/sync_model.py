from minimal import get_problem


def main():
    prob, x1, x2, *_ = get_problem()
    prob.solve()
    x1_val, x2_val = x1.value, x2.value

    prob2 = prob.pull_from_solver().solve()

    assert prob2.variables[0].value == x1_val and prob2.variables[1].value == x2_val

if __name__ == "__main__":
    main()
