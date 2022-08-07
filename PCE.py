import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint

import chaospy as cp
#------------------------------------------------------------

def sim_summary():
    print('--Simulation setup--')
    print('sim time:', t_max)
    print('timestep:', dt)
    print('number of steps:', nt)
#def model(u, t, k, w, c, f):
#	x1, x2 = u
#	f = [x2, f*np.cos(w*t) - k*np.sin(x1) - c*x2]
#	return f 
def model(u, t, k, w, c, F):
	x1, x2 = u
	f = [x2, F- k*x1 - c*x2]
	return f 

#------------------------------------------------------------
diags_dir='c:/Users/Gianni/Desktop/SuperCod/Result_PCE'
#------------------------------------------------------------
print('---------------------- Polynomial chaos expansion ----------------------') 
#------------------------------------------------------------
t_max=20
dt=0.001
nt = int(t_max/dt)
t = np.linspace(0.,t_max, nt)
sim_summary()

K=5 # Quadrature order
mean=0.2 # centre of the distribution
std=0.1 # standard deviation

nw=1000
xi=np.linspace(mean-6*std,mean+6*std, nw)
rho=1/(std*((2*np.pi)**(1/2)))*np.exp(-1/2*(((xi-mean)/std)**2)) # probability density function
##############################################################################
##                       three terms recursion relation
## Psi_n+1=Psi_n*(Xi-An)-Psi_n-1*Bn
## An=(Xi*Psi_n,Psi_n)/(Psi_n,Psi_n)
## Bn=An=(Psi_n,Psi_n)/(Psi_n-1,Psi_n-1)

# for std different from 1 
## Psi_n+1=Psi_n*(Xi-An)/std-Psi_n-1*Bn
##############################################################################
unif_distr =cp.Normal(mean,std)
nodes, weights = cp.generate_quadrature(K-1, unif_distr, rule="gaussian")
# psi1
A=np.zeros([1,1])
B=np.zeros([1,1])
for i in range(nw):
    A+=(rho[i-1]*(xi[i-1])+rho[i]*(xi[i]))/2*(np.abs(xi[i-1]-xi[i]))
    B+=(rho[i-1]+rho[i])/2*(np.abs(xi[i-1]-xi[i]))
psi_1=(xi-A)/std
# psi2
A1=np.zeros([1,1])
A2=np.zeros([1,1])
B1=np.zeros([1,1])
B2=np.zeros([1,1])
for i in range(nw):
    A1+=(rho[i-1]*(psi_1[0,i-1]**2)*(xi[i-1])+rho[i]*(psi_1[0,i]**2)*xi[i])/2*(np.abs(xi[i-1]-xi[i]))
    A2+=(rho[i-1]*(psi_1[0,i-1]**2)+rho[i]*(psi_1[0,i]**2))/2*(np.abs(xi[i-1]-xi[i]))
    B1+=(rho[i-1]*(psi_1[0,i-1]**2)+rho[i]*(psi_1[0,i]**2))/2*(np.abs(xi[i-1]-xi[i]))
    B2+=(rho[i-1]+rho[i])/2*(np.abs(xi[i-1]-xi[i]))
psi_2=psi_1*(xi-A1/A2)/std-B1/B2
# psi3
A3=np.zeros([1,1])
A4=np.zeros([1,1])
B5=np.zeros([1,1])
B6=np.zeros([1,1])
for i in range(nw):
    A3+=(rho[i-1]*(psi_2[0,i-1]**2)*(xi[i-1])+rho[i]*(psi_2[0,i]**2)*xi[i])/2*(np.abs(xi[i-1]-xi[i]))
    A4+=(rho[i-1]*(psi_2[0,i-1]**2)+rho[i]*(psi_2[0,i]**2))/2*(np.abs(xi[i-1]-xi[i]))
    B5+=(rho[i-1]*(psi_2[0,i-1]**2)+rho[i]*(psi_2[0,i]**2))/2*(np.abs(xi[i-1]-xi[i]))
    B6+=(rho[i-1]*(psi_1[0,i-1]**2)+rho[i]*(psi_1[0,i]**2))/2*(np.abs(xi[i-1]-xi[i]))
psi_3=psi_2*(xi-A3/A4)/std-B5/B6*psi_1

# nodes=np.random.normal(mean, std, size=(1, K))
# For a Noraml disturibution i have to use the Hermite polynomials
weights = np.array([0.01125741, 0.22207592, 0.53333333, 0.22207592, 0.01125741])
nodes=std*((-2*np.log(weights*std*(2*np.pi)**(1/2)))**(1/2))+mean

one=np.ones([1,nw])
fig = plt.figure(num=5,figsize=[20, 10])
ax = fig.gca(xlabel="$\\xi$", ylabel="$H_n(\\xi)$")
ax.plot(xi,one[0,:],"-r", label="$H_0$")
ax.plot(xi,psi_1[0,:],"-b", label="$H_1$")
ax.plot(xi,psi_2[0,:],"-g", label="$H_2$")
ax.plot(xi,psi_3[0,:],"-m", label="$H_3$")
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/HermitePolynomials.png")
plt.show()

unif_distr =cp.Normal(mean,std)
nodes, weights = cp.generate_quadrature(K-1, unif_distr, rule="gaussian")

fig = plt.figure(num=6,figsize=[20, 10])
ax = fig.gca(xlabel="$\\xi$", ylabel="$\\rho(\\xi)$")
ax.scatter(nodes,1/(std*((2*np.pi)**(1/2)))*np.exp(-1/2*(((nodes-mean)/std)**2)),s=40,color='red', label="Weight functions $w_k$")
ax.plot(xi,rho,"-b", label="Probability density function")
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/Weigthfun.png")
plt.show()

    
# Hermite polynomials - da sistemare
N=3 # Polynomial order
# Stieltjes’ three-term recursion relation

psi0=np.ones([1,K])
psi1=(nodes-A)/std
psi2=psi1*(nodes-A1/A2)/std-B1/B2*psi0
psi3=psi2*(nodes-A3/A4)/std-B5/B6*psi1
#------------------------------------------------------------
k=10  # length of pendulum 1 in m
c=1  # C/M is the dumping on dot_theta
f=1 # Aplitude of the external force 
w=2*np.pi # Frequency of the external force

y0  = 0
y1  = 0
init_cond   = y0, y1


model_r = [odeint(model, init_cond, t, args=(k, w,c,nodes)) for nodes in nodes.T]
#------------------------------------------------------------
x0=np.zeros([1,nt])
v0=np.zeros([1,nt])
x1=np.zeros([1,nt])
v1=np.zeros([1,nt])
x2=np.zeros([1,nt])
v2=np.zeros([1,nt])
x3=np.zeros([1,nt])
v3=np.zeros([1,nt])
variancex=np.zeros([1,nt])
variancev=np.zeros([1,nt])

for j in range(K):
    var=model_r[j]
    x0+=var[:,0]*psi0[0,j]*weights[j]
    v0+=var[:,1]*psi0[0,j]*weights[j]
    x1+=var[:,0]*psi1[0,j]*weights[j]
    v1+=var[:,1]*psi1[0,j]*weights[j]
    x2+=var[:,0]*psi2[0,j]*weights[j]
    v2+=var[:,1]*psi2[0,j]*weights[j]
    x3+=var[:,0]*psi3[0,j]*weights[j]
    v3+=var[:,1]*psi3[0,j]*weights[j]
        
variancex=np.sqrt(x1**2+x2**2+x3**2)
variancev=np.sqrt(v1**2+v2**2+v3**2)
    
    
fig = plt.figure(num=1,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="$x,\dot{x}(t)$ [code units]", title='Solution')
ax.plot(t,x0.T,"-r", label="$x(t)$")
ax.plot(t,v0.T,"-b", label="$\dot{x}(t)$")
ax.fill_between(t, x0[0,:]-variancex[0,:],x0[0,:]+variancex[0,:], color='orange',alpha=0.5)
ax.fill_between(t, v0[0,:]-variancev[0,:],v0[0,:]+variancev[0,:], color='blue',alpha=0.5)
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/PCE.png")
plt.show()
#------------------------------------------------------------
N = 3
unif_distr =cp.Normal(mean,std)
# nodes, weights = cp.generate_quadrature( K, unif_distr, rule="gaussian")
polynomial_expansion_unif = cp.generate_expansion(N, unif_distr,normed=True)

f_approx = cp.fit_quadrature(polynomial_expansion_unif, nodes, weights, model_r)
expected = cp.E(f_approx, unif_distr)
std = cp.Std(f_approx, unif_distr)

fig = plt.figure(num=2,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="$x,\dot{x}(t)$ [code units]", title='Solution')
ax.plot(t,expected[:,0],"-r", label="$x(t)$")
ax.plot(t,expected[:,1],"-b", label="$\dot{x}(t)$")
ax.fill_between(t, expected[:,0]-std[:,0],expected[:,0]+std[:,0], color='orange',alpha=0.5)
ax.fill_between(t, expected[:,1]-std[:,1],expected[:,1]+std[:,1], color='blue',alpha=0.5)
ax.legend()
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/PCE_chaospy.png")
plt.show()


fig = plt.figure(num=3,figsize=[20, 10])
ax = fig.gca(xlabel="t [code units]", ylabel="absolute error")
ax.semilogy(t,np.abs(variancex[0,:]-std[:,0]),color='orange',label="error std $x(t)$")
ax.semilogy(t,np.abs(variancev[0,:]-std[:,1]),color='blue',label="error std $\dot{x}(t)$")
ax.semilogy(t,np.abs(x0[0,:]-expected[:,0]),color='green',label="error mean $x(t)$")
ax.semilogy(t,np.abs(v0[0,:]-expected[:,1]),color='red',label="error mean $\dot{x}(t)$")
ax.legend(loc='upper right')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.tight_layout()
plt.savefig(diags_dir+"/error_PCE.png")
plt.show()









