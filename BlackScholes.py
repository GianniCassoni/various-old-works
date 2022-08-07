"""
Created on Wed Sep 15 22:34:31 2021
@author: Gianni
"""
import numpy as np
import matplotlib.pyplot as plt
# Heston's model : 
class HestonModel:
    def __init__(self,BCs,k,theta,sigma_v,rho,mu,nt):
        # BCs : [ theta, omega ]
        origin=(0,0) # Origin
        self.parameters = (k,theta,sigma_v,rho,mu)
        self.BCs = np.asarray(BCs, dtype='float')
        self.origin = origin
        self.time_elapsed = 0
        self.state = self.BCs * np.pi / 180.
        self.evolution = np.zeros([2,nt])
        self.sim_summary()
    def sim_summary(self):
        (k,theta,sigma_v,rho,mu)= self.parameters
        print('--System parameters--')
        print('BCs: ', self.BCs) 
        print('K:', k)
        print('C:',theta)
        print('Amplitude of the external force:', sigma_v)
        print('Frequency of the external force:', rho)
        print('Frequency of the external force:', mu)
    def modelf(self, t,state):
        (k,theta,sigma_v,rho,mu)= self.parameters
        x1, x2 = state
        f = np.array([mu*x1,k*(theta-x2)])
        return f
    def modelg(self, t,state):
        (k,theta,sigma_v,rho,mu)= self.parameters
        x1, x2 = state
        g =np.array([x1*np.sqrt(x2),sigma_v*np.sqrt(x2)])
        return g
    def Solve(self,t):
        nt=len(t)
        dt=t[nt-1]/nt
        # Euler-Maruyama numerical discretization
        (k,theta,sigma_v,rho,mu)= self.parameters
        x=np.zeros([nt,2])
        x[0,:]=self.BCs
        for i in range(nt-1):
            Z1=np.random.standard_normal()
            Z2=np.random.standard_normal()
            rnd =np.array([Z1,rho*Z1+(1-rho**2)**(1/2)*Z2])
            x[i+1,:] = x[i,:]+ self.modelf(t[i],x[i,:])*dt +self.modelg(t[i],x[i,:])*rnd*np.sqrt(dt)
        return x
class HestonWhiteModel:
    def __init__(self,BCs,R,alpha,k,theta,sigma_2,sigma_3,rho,nt):
        # BCs : [ theta, omega ]
        self.parameters = (R,alpha,k,theta,sigma_2,sigma_3,rho)
        self.BCs = np.asarray(BCs, dtype='float')
        self.time_elapsed = 0
        self.state = self.BCs * np.pi / 180.
        self.evolution = np.zeros([3,nt])
        self.sim_summary()
    def sim_summary(self):
        (R,alpha,k,theta,sigma_2,sigma_3,rho)= self.parameters
        print('--System parameters--')
        print('BCs: ', self.BCs) 
        print('K:', k)
        print('C:',theta)
    def modelf(self, t,state):
        (R,alpha,k,theta,sigma_2,sigma_3,rho)= self.parameters
        x1, x2, x3 = state
        f = np.array([x3*x1,k*(theta-x2),alpha*(R-x3)])
        return f
    def modelg(self, t,state):
        (R,alpha,k,theta,sigma_2,sigma_3,rho)= self.parameters
        x1, x2, x3 = state
        g =np.array([[x1*np.sqrt(x2),0,0],
                     [sigma_2*np.sqrt(x2)*rho,sigma_2*np.sqrt(x2)*np.sqrt(1-rho**2),0],
                     [0,0,sigma_3]])
        return g
    def Solve(self,t):
        nt=len(t)
        dt=t[nt-1]/nt
        # Euler-Maruyama numerical discretization
        (R,alpha,k,theta,sigma_2,sigma_3,rho)= self.parameters
        x=np.zeros([nt,3])
        x[0,:]=self.BCs
        for i in range(nt-1):
            Z1=np.random.standard_normal()
            Z2=np.random.standard_normal()
            Z3=np.random.standard_normal()
            rnd =np.array([Z1,Z2,Z3])
            x[i+1,:] = x[i,:]+ self.modelf(t[i],x[i,:])*dt +self.modelg(t[i],x[i,:]).dot(rnd)*np.sqrt(dt)
        return x
#------------------------------------------------------------
print('----------------------Heston model----------------------') 
#------------------------------------------------------------
t_max=10
dt=0.001
nt = int(t_max/dt)
t = np.linspace(0.,t_max, nt)
    
k=1.63
theta=0.0934
sigma_v=0.473
rho=-0.8021
mu=0
v0= 0.0821
S0=2
init_state= [S0, v0]
#------------------------------------------------------------   
solution = HestonModel(init_state,k,theta,sigma_v,rho,mu,nt)
x = solution.Solve(t)

fig = plt.figure(num=0,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="x,v [code units]", title='Solution')
ax.plot(t,x[:,0],label='S',color ='r')
ax.plot(t,x[:,1],label='v',color ='b')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.show()
#------------------------------------------------------------
print('----------------------Heston-Hull-White model----------------------') 
#------------------------------------------------------------
t_max=10
dt=0.001
nt = int(t_max/dt)
t = np.linspace(0.,t_max, nt)
   
R=0.06
alpha=0.1
k=3
theta=0.12
sigma_2=0.04
sigma_3=0.01
rho= 0.6
v0= theta
S0=95
r0=R
init_state= [S0, v0, r0]
#------------------------------------------------------------   
solution = HestonWhiteModel(init_state,R,alpha,k,theta,sigma_2,sigma_3,rho,nt)
x = solution.Solve(t)

fig = plt.figure(num=1,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="x,v [code units]", title='Solution')
ax.plot(t,x[:,0],label='S',color ='r')
ax.plot(t,x[:,1],label='v',color ='b')
ax.plot(t,x[:,2],label='r',color ='g')
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.show()


V=np.zeros([1,1])
for i in range(nt):
    V=+np.exp(-(x[i-1,2]+x[i,2])/2*dt)*(x[i,0]-100)
print(V)