from pygenopt import Variable, LinearConstraint
from pygenopt.highsapi import HiGHS


x1 = Variable('x1')
x2 = Variable('x2')

constr1 = LinearConstraint(x2 - x1 >= 2, 'constr1')
constr2 = LinearConstraint(x1 + x2 >= 0, 'constr2')

m = HiGHS()
m.addvars(x1, x2)
m.addconstrs(constr1, constr2)
m.setobj(x1+x2)
m.setoptions({'presolve': 'off'})
m.run()

print(m.getval(x1), m.getval(x2))
