from scipy.optimize import minimize
import numpy as np
e = 1e-10 # 非常接近0的值
fun = lambda x : 2*x[0]**2 + 4*x[1] # f(x,y) = 2*x^2+4*y
cons = (
        {'type': 'ineq', 'fun': lambda x: x[0]+x[1] - 3}, #
        {'type': 'ineq', 'fun': lambda x: 10-2*x[0]-x[1]},
       )
x0 = np.array((2.0,1.0)) # 设置初始值
res = minimize(fun, x0, method='SLSQP', constraints=cons)
print('最大值：',res.fun)
print('最优解：',res.x)
print('迭代终止是否成功：', res.success)
print('迭代终止原因：', res.message)
