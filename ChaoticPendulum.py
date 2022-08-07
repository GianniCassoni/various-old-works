import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
from mpl_toolkits import mplot3d

#import chaospy as cp
#------------------------------------------------------------
def sim_summary():
    print('--Simulation setup--')
    print('sim time:', t_max)
    print('timestep:', dt)
    print('number of steps:', nt)
def myqr(A,m):
    Q, R = np.linalg.qr(A)
    nr=m
    nc=m
    nn = nr
    Q = Q[:, 0:nc]
    R = R[0:nc, :]
    for ii in range(nn):
        if R[ii, ii] < 0 :

            Q[:, ii] = -Q[:, ii]
            R[ii, ii:m] = -R[ii, ii:m]
    return Q, R
def LyapunovCE(y,J,dJ,nt,m):
    # y : solution of the system 
    # J : jacobian matrix of the system 
    # dJ :  derivative of jacobian matrix of the system 
    # m : order of the system
    # nt : number of time steps
    Qk=np.zeros([m,m])
    Rk=np.zeros([m,m])
    Y0 = np.eye(m);
    Qk, Rk= np.linalg.qr(Y0)
    Yk = Y0
    ll_curr = np.zeros([1, m]);
    lyap=np.zeros([m,nt])
    for H in range(1,nt):
        A=J(y[H,:])
        F= np.linalg.lstsq((dJ()+dt/2.*A),(dJ()-dt/2.*A),rcond=None)[0]
        Qk=np.dot(F,Qk)
        Yk=np.dot(F,Yk)
        Qk, Rk = myqr(Qk,m)
        ll_curr=ll_curr+np.log(Rk.diagonal().T)
        lyap[:,H]=ll_curr/t[H]
    return lyap




class ChaoticPendulum:
    def __init__(self,BCs,M,C,L,W,F,nt):
        # BCs : [ theta, omega ]
        G=9.81 # Gravity [m/s^2]
        origin=(0,0) # Origin
        self.parameters = (M,C,L,G,W,F)
        self.BCs = np.asarray(BCs, dtype='float')
        self.origin = origin
        self.time_elapsed = 0
        self.state = self.BCs * np.pi / 180.
        self.evolution = np.zeros([2,nt])
        self.sim_summary()
    def sim_summary(self):
        (M,C,L,G,W,F)= self.parameters
        print('--System parameters--')
        print('BCs: ', self.BCs) 
        print('K:', G/L)
        print('C:',C/M)
        print('Amplitude of the external force:', F/(M*L))
        print('Frequency of the external force:', W)
    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.state = integrate.odeint(self.model, self.state, [0, dt])[1]
        self.time_elapsed += dt
    def model(self, state, t):
        (M,C,L,G,W,F)= self.parameters
        x1, x2 = state
        f = [x2, F/(M*L)*np.cos(W*t) - (G/L)*np.sin(x1)-(C/M)*x2]
        return f
    def Solve(self,t):
        atol = 1e-6
        rtol = 1e-6
        Solution = odeint(self.model,self.BCs,t,atol=atol,rtol=rtol) 
        self.evolution=Solution
        return Solution
    def JacobianMatrix(self,state):
        (M,C,L,G,W,F)= self.parameters
        return np.array([[0,-1],[-(G/L)*np.cos(state[0]),-C/M]])
    def dJ(self):
        return np.array([[1,0],[0,-1]])
    
    
class ChaoticPendulum_Stochastis:
    def __init__(self,BCs,M,C,L,F,ep,nt):
        # BCs : [ theta, omega ]
        G=9.81 # Gravity [m/s^2]
        origin=(0,0) # Origin
        self.parameters = (M,C,L,G,F,ep)
        self.BCs = np.asarray(BCs, dtype='float')
        self.origin = origin
        self.time_elapsed = 0
        self.state = self.BCs * np.pi / 180.
        self.sim_summary()
    def sim_summary(self):
        (M,C,L,G,F,ep)= self.parameters
        print('--System parameters Stochastic--')
        print('BCs: ', self.BCs) 
        print('K:', G/L)
        print('C:',C/M)
        print('Amplitude of the external force:', F/(M*L))
        print('White noise amplitude:', ep)
    def step(self, dt):
        """execute one time step of length dt and update state"""
        self.state = integrate.odeint(self.model, self.state, [0, dt])[1]
        self.time_elapsed += dt
    def model(self, t,state):
        (M,C,L,G,F,ep)= self.parameters
        x1, x2 = state
        f = np.array([x2, F/(M*L) - (G/L)*np.sin(x1)-(C/M)*x2])
        return f
    def Solve(self,t):
        nt=len(t)
        dt=t[nt-1]/nt
        # Euler-Maruyama numerical discretization
        (M,C,L,G,F,ep)= self.parameters
        x=np.zeros([nt,2])
        x[0,:]=self.BCs
        g=[0,ep]
        for i in range(nt-1):
            Z=np.array([np.random.standard_normal(),np.random.standard_normal()])
            x[i+1,:] = x[i,:]+ self.model(t[i],x[i,:])*dt + g*Z*np.sqrt(dt);
        return x
    def JacobianMatrix(self,state):
        (M,C,L,G,W,F)= self.parameters
        return np.array([[0,-1],[-(G/L)*np.cos(state[0]),-C/M]])
    def dJ(self):
        return np.array([[1,0],[0,-1]])
    def MC_simulation(self,N,t):
#        x_poly=[[] for i in range(N+1)]
        expected=np.zeros([len(t),2])
        A=np.zeros([len(t),2])
        B=np.zeros([len(t),2])
        std=np.zeros([len(t),2])
        x_end=np.zeros([1,N])
        v_end=np.zeros([1,N])
        for i in range(N+1) :
            x = self.Solve(t)
            expected+=x
            A+=x**2
            B+=2*x
            x_end[:,i-1]=x[len(t)-1,0]
            v_end[:,i-1]=x[len(t)-1,0]
#            x_poly[i]=x
        expected=expected/(N+1)
        std=np.sqrt((A-expected*B+(N+1)*expected**2)/(N+1))
#        expected=np.mean(x_poly,axis=0)
#        std=np.std(x_poly,axis=0)
        return expected,std,x_end,v_end

class Lorenz_Sys:
    def __init__(self,BCs,sigma,beta,rho,nt):
        # BCs : [u,v,w]
        origin=(0,0,0) # Origin
        self.parameters = (sigma,beta,rho)
        self.BCs = np.asarray(BCs, dtype='float')
        self.origin = origin
        self.time_elapsed = 0
        self.state = self.BCs 
        self.sim_summary()
    def sim_summary(self):
        (sigma,beta,rho)= self.parameters
        print('--System parameters--')
        print('BCs: ', self.BCs) 
        print('sigma:', sigma)
        print('beta:',beta)
        print('rho:', rho)
    def model(self,state,t):
        (sigma,beta,rho)= self.parameters
        u, v, w = state
        up = -sigma*(u - v)
        vp = rho*u - v - u*w
        wp = -beta*w + u*v
        return up, vp, wp
    def Solve(self,t):
        Solution = odeint(self.model,self.BCs,t) 
        return Solution
    def JacobianMatrix(self,state):
        (sigma,beta,rho)= self.parameters
        x, y, z = [k for k in state]
        return np.array([[sigma, -sigma, 0], [-rho+z, 1, +x], [-y, -x, beta]])
    def dJ(self):
        return np.eye(3)
    
#------------------------------------------------------------
diags_dir='c:/Users/Gianni/Desktop/SuperCod/Result'
#------------------------------------------------------------
print('----------------------Deterministic armonic external force----------------------') 
#------------------------------------------------------------
t_max=20
dt=0.001
nt = int(t_max/dt)
t = np.linspace(0.,t_max, nt)
sim_summary()

theta=0. # [degrees]
omega=0. # [degrees/s]
init_state= [theta, omega]
L=0.1  # length of pendulum 1 in m
C=1  # C/M is the dumping on dot_theta
F=1 # Aplitude of the external force 
M=1 # mass of pendulum 1 in kg
W=2*np.pi # Frequency of the external force
#------------------------------------------------------------
pendulum = ChaoticPendulum(init_state,M,C,L,W,F,nt)
x = pendulum.Solve(t)

m=2  # m : order of the system
lyapunov=LyapunovCE(x,pendulum.JacobianMatrix,pendulum.dJ,nt,m)
#------------------------------------------------------------
fig = plt.figure(num=0,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="x,v [code units]", title='Solution')
ax.plot(t,x[:,0],label='postion',color ='r')
ax.plot(t,x[:,1],label='velocity',color ='b')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Solution.png")
plt.show()

fig = plt.figure(num=1,figsize=[20, 10])
ax = fig.gca(xlabel="x [code units]", ylabel="v [code units]", title='Solution')
ax.plot(x[:,0],x[:,1],label='Deterministic',color ='r')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Phase_space.png")
plt.show()

fig = plt.figure(num=2,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="LCE [code units]", title='LyapunovCE')
ax.plot(t,lyapunov[0,:],label='$\lambda_1$',color ='r')
ax.plot(t,lyapunov[1,:],label='$\lambda_2$',color ='b')
ax.plot(t,lyapunov[1,:]+lyapunov[0,:],label='$\lambda_{tot}$',color ='y')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Solution_Lyap.png")
plt.show()
#------------------------------------------------------------
print('----------------------Lorenz----------------------')
#------------------------------------------------------------
t_max=50
dt=0.005
nt = int(t_max/dt)
t = np.linspace(0.,t_max, nt)
sim_summary()

u=1.
v=1.
w=1.
init_state= [u,v,w]
sigma=10.
beta=8./3.
rho=28.
#------------------------------------------------------------
Lorenz = Lorenz_Sys(init_state,sigma,beta,rho,nt)
x = Lorenz.Solve(t)

m=3  # m : order of the system
lyapunov=LyapunovCE(x,Lorenz.JacobianMatrix,Lorenz.dJ,nt,m)
#------------------------------------------------------------
fig = plt.figure(num=3,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="u,v,w [code units]", title='Solution')
ax.plot(t,x[:,0],label='u',color ='r')
ax.plot(t,x[:,1],label='v',color ='b')
ax.plot(t,x[:,2],label='w',color ='g')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Solution_Lorenz.png")
plt.show()

fig = plt.figure(num=4)
ax = plt.axes(projection ='3d')
ax.plot3D(x[:,0], x[:,1], x[:,2], 'red')
ax.set_title('3D Phase space')
ax.set_xlabel('u [code units]')
ax.set_ylabel('v [code units]')
ax.set_zlabel('w [code units]')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Lorenz_Phasespace.png")
plt.show()

fig = plt.figure(num=5,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="LCE [code units]", title='LyapunovCE')
ax.plot(t,lyapunov[0,:],label='$\lambda_1$',color ='r')
ax.plot(t,lyapunov[1,:],label='$\lambda_2$',color ='b')
ax.plot(t,lyapunov[2,:],label='$\lambda_3$',color ='g')
ax.plot(t,lyapunov[1,:]+lyapunov[0,:]+lyapunov[2,:],label='$\lambda_{tot}$',color ='y')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Solution_Lorenz_Lyap.png")
plt.show()
#------------------------------------------------------------
print('----------------------Stochastic white noise---------------------- ')
#------------------------------------------------------------
# il rumore bianco e sul limite della stabilità, instabilità
#t_max=20
#dt=0.001
#nt = int(t_max/dt)
#t = np.linspace(0.,t_max, nt)
#sim_summary()

#theta=0. # [degrees]
#omega=0. # [degrees/s]
#init_state= [theta, omega]
#L=0.1  # length of pendulum 1 in m
#C=1  # C/M is the dumping on dot_theta
#F=1 # Aplitude of the external force 
#M=1 # mass of pendulum 1 in kg
#ep=4.1 # White noise amplitude 
#------------------------------------------------------------
t_max=5
dt=0.001
nt = int(t_max/dt)
t = np.linspace(0.,t_max, nt)
sim_summary()

theta=0. # [degrees]
omega=0. # [degrees/s]
init_state= [theta, omega]
L=0.1  # length of pendulum 1 in m
C=1  # C/M is the dumping on dot_theta
F=1 # Aplitude of the external force 
M=1 # mass of pendulum 1 in kg
ep=0.02 # White noise amplitude 
#------------------------------------------------------------
pendulum_Chaos = ChaoticPendulum_Stochastis(init_state,M,C,L,F,ep,nt)
x = pendulum_Chaos.Solve(t)

m=2  # m : order of the system
lyapunov=LyapunovCE(x,pendulum_Chaos.JacobianMatrix,pendulum_Chaos.dJ,nt,m)
#------------------------------------------------------------
fig = plt.figure(num=6,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="x,v [code units]", title='Solution')
ax.plot(t,x[:,0],label='postion',color ='r')
ax.plot(t,x[:,1],label='velocity',color ='b')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Solution_Stochastic.png")
plt.show()

fig = plt.figure(num=7,figsize=[20, 10])
ax = fig.gca(xlabel="x [code units]", ylabel="v [code units]", title='Solution')
ax.plot(x[:,0],x[:,1],label='Stochastic',color ='r')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Phase_space_stochastic.png")
plt.show()

fig = plt.figure(num=8,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="LCE [code units]", title='LyapunovCE')
ax.plot(t,lyapunov[0,:],label='$\lambda_1$',color ='r')
ax.plot(t,lyapunov[1,:],label='$\lambda_2$',color ='b')
ax.plot(t,lyapunov[1,:]+lyapunov[0,:],label='$\lambda_{tot}$',color ='y')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Solution_Stochastic_Lyap.png")
plt.show()
#------------------------------------------------------------
N=300 # number of run (1/sqrt(N)) convergence of monte carlo 
expected,std,x_end,v_end=pendulum_Chaos.MC_simulation(N,t)

fig = plt.figure(num=9,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="x,v [code units]", title='Solution')
ax.plot(t,expected[:,0],"-b")
ax.plot(t,expected[:,1],"-r")
ax.fill_between(t, expected[:,0]-std[:,0],expected[:,0]+std[:,0], color='blue',alpha=0.5)
ax.fill_between(t, expected[:,1]-std[:,1],expected[:,1]+std[:,1], color='orange',alpha=0.5)
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/MC_Stochastic.png")
plt.show()

fig = plt.figure(num=10,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="x,v [code units]", title='Solution')
ax.hist(x_end.T,bins=30,align='right', color='purple', edgecolor='black')
#ax.hist(v_end.T,bins=15,align='right', color='green', edgecolor='black')
plt.tight_layout()
plt.show()











#unif_distr =ep*cp.Normal(0, 1)

#K = 4 # quadrature_order
#N = 2 # derees of the polynomials

#nodes, weights = cp.generate_quadrature( K, unif_distr, rule="gaussian")
#polynomial_expansion_unif = cp.generate_expansion(N, unif_distr,normed=True)
#x_poly=[[] for i in range(K+1)]

#for i in range(K+1) :
#    F_new=F+nodes[0,i]*L
#    pendulum = ChaoticPendulum(init_state,M,C,L,0,F_new,nt)
#    x = pendulum.Solve(t)
#    x_poly[i]=x
#f_approx = cp.fit_quadrature(polynomial_expansion_unif, nodes, weights, x_poly)
#expected = cp.E(f_approx, unif_distr)
#std = cp.Std(f_approx, unif_distr)


























