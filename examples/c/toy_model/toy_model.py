from casadi import DM, SX, Function, horzcat, jacobian, jtimes, mtimes, vertcat


def sizes():
    return 1, 1

def ode_model(x, u):
    return x**6 + u

MODEL_NAME = 'toy'

nx, nu = sizes()
x, u = SX.sym('x', nx), SX.sym('u', nu)
ode = ode_model(x, u)

Sx = SX.sym('Sx', nx, nx)
Sp = SX.sym('Sp', nx, nu)
vdeX = SX.zeros(nx, nx)
vdeX = vdeX + jtimes(ode, x, Sx)
vdeP = SX.zeros(nx, nu) + jacobian(ode, u)
vdeP = vdeP + jtimes(ode, x, Sp)

vdeFun = Function('vde_' + MODEL_NAME, [x, Sx, Sp, u], [ode, vdeX, vdeP])

jacX = SX.zeros(nx, nx) + jacobian(ode, x)
jacFun = Function('jac_' + MODEL_NAME, [x, u], [ode, jacX])

vdeFun.generate('vde_' + MODEL_NAME)
jacFun.generate('jac_' + MODEL_NAME)

lambdaX = SX.sym('lambdaX', nx)
adj = jtimes(ode, vertcat(x, u), lambdaX, True)

adjFun = Function('adj_' + MODEL_NAME, [x, lambdaX, u], [adj])
adjFun.generate('adj_' + MODEL_NAME)

S_forw = vertcat(horzcat(Sx, Sp), horzcat(DM.zeros(nu, nx), DM.eye(nu)))
hess = mtimes(S_forw.T, jtimes(adj, vertcat(x, u), S_forw))
hess2 = []
for j in range(nx+nu):
    for i in range(j, nx+nu):
        hess2 = vertcat(hess2, hess[i,j])

hessFun = Function('adj_hess_' + MODEL_NAME, [x, Sx, Sp, lambdaX, u], [adj, hess2])
hessFun.generate('adj_hess_' + MODEL_NAME)
