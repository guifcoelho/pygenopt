# Example taken from https://www.fico.com/fico-xpress-optimization/docs/latest/examples/python/GUID-D3DC5129-30E7-306D-AEB2-F174CC60FC6F.html

# A company produces two electrical products, A and B. Both require
# two stages of production: wiring and assembly. The production plan
# must meet several goals:
# 1. A profit of $200
# 2. A contractual requirement of 40 units of product B
# 3. To fully utilize the available wiring department hours
# 4. To avoid overtime in the assembly department

import pygenopt as opt
from pygenopt.solvers import HighsApi


# Decision variables for the number of products to make of each type
produceA = opt.Variable(name="produceA", vartype=opt.VarType.INT, lowerbound=0)
produceB = opt.Variable(name="produceB", vartype=opt.VarType.INT, lowerbound=0)

# Deviational variables
# There is a penalty for both under- and over-utilizing each department
surplus_wiring = opt.Variable(name="surplus_wiring", lowerbound=0)
deficit_wiring = opt.Variable(name="deficit_wiring", lowerbound=0)
surplus_assembly = opt.Variable(name="surplus_assembly", lowerbound=0)
deficit_assembly = opt.Variable(name="deficit_assembly", lowerbound=0)
# There is no penalty for surplus in profit or in production of product B, only for deficits
deficit_profit = opt.Variable(name="deficit_profit", lowerbound=0)
deficit_productB = opt.Variable(name="deficit_productB", lowerbound=0)

vars = [
    produceA,
    produceB,
	surplus_wiring,
	deficit_wiring,
	surplus_assembly,
	deficit_assembly,
	deficit_profit,
	deficit_productB
]

# Production constraints
constraints = [
	# Meet or exceed profit goal of $200
	# Profit for products A and B are $7 and $6
	7 * produceA + 6 * produceB + deficit_profit >= 200,

	# Meet or exceed production goal for product B
	produceB + deficit_productB >= 40,

	# Utilize wiring department:
	# Products A and B require 2 and 3 hours of wiring
	# 120 hours are available
	2 * produceA + 3 * produceB - surplus_wiring + deficit_wiring == 120,

	# Utilize assembly department:
	# Products A and B require 6 and 5 hours of assembly
	# 300 hours are available
	6 * produceA + 5 * produceB - surplus_assembly + deficit_assembly == 300
]

prob = (
    opt.Problem(name="multiobjective", solver_api=HighsApi)
	.add_vars(*vars)
	.add_constrs(*constraints)
	.add_objectives(
		deficit_profit,
		deficit_productB,
		surplus_wiring + deficit_wiring,
		surplus_assembly + deficit_assembly
	)
	.solve()
)

if prob.solve_status in [opt.SolveStatus.FEASIBLE, opt.SolveStatus.OPTIMUM]:
	print('\nProduction plan:')
	print(f' - Product A: {int(produceA.value)} units')
	print(f' - Product B: {int(produceB.value)} units')
	print(f' - Profit: ${7 * produceA.value + 6 * produceB.value}')
	if deficit_profit.value > 0:
		print(f' - Profit goal missed by ${deficit_profit.value}')
	if deficit_productB.value > 0:
		print(f' - Contractual goal for product B missed by {int(deficit_productB.value)} units')
	if surplus_wiring.value > 0:
		print(f' - Unused wiring department hours: {surplus_wiring.value}')
	if deficit_wiring.value > 0:
		print(f' - Wiring department overtime: {deficit_wiring.value}')
	if surplus_assembly.value > 0:
		print(f' - Unused assembly department hours: {surplus_assembly.value}')
	if deficit_assembly.value > 0:
		print(f' - Assembly department overtime: {deficit_assembly.value}')
else:
	print('Problem could not be solved')
