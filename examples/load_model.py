import pygenopt as opt


x1 = opt.Variable('x1')
x2 = opt.Variable('x2')

constr1 = opt.LinearConstraint(x2 - x1 >= 2, 'constr1')
constr2 = opt.LinearConstraint(x1 + x2 >= 0, 'constr2')

objective_function = -(x1 + x2 + 5)

prob = (
    opt.Problem(name="minimal_problem")
    .add_vars(x1, x2)
    .add_constrs(constr1, constr2)
    .set_objective(opt.ObjectiveFunction(objective_function, is_minimization=False))
    .update()
    .to_mps('load_model.mps')
)

prob = opt.Problem.load_mps('load_model.mps').solve()

print("Solve status:", prob.solve_status)
print("Objective function value:", prob.get_objectivefunction_value())

print("x1:", prob.variables[0].value)
print("x2:", prob.variables[1].value)
