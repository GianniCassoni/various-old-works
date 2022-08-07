"""
Created on Wed Jul 14 18:09:50 2021

@author: Gianni
"""
"""_____________________________________________EXTERNAL_FUNCTION_____________________________________________"""

import numpy as np
from numpy import pi,sqrt
import matplotlib.pyplot as plt
from matplotlib import animation

"""_____________________________________________FUNCTION_____________________________________________"""

def save_1Dquantity(x, y, filename):
    np.savetxt(filename, np.c_[x, y])
def sim_summary():
    print('Simulation setup:')
    print('grid length: ', Lx) 
    print('cell length: ', dx)
    print('sim time:', T)
    print('timestep:', dt)
    print('check Courant condition: dt / dx = ', dt / dx)
    print('number of steps:', nt)
    print('number of grid points:', nx)
def init_EMfield(x, Ex, Ey, Ez, Bx, By, Bz):
    nx = len(x)
    Ex = Ex*np.ones((nx))
    Ey = Ey*np.ones((nx))
    Ez = Ez*np.ones((nx))
    Bx = Bx*np.ones((nx))
    By = By*np.ones((nx))
    Bz = Bz*np.ones((nx))
    return Ex, Ey, Ez, Bx, By, Bz

def gaussian(x, t, a0, k, omega, x_peak, fwhm):
    sigma = fwhm / ( 2. * np.sqrt(np.log(2.)) ) 
    return a0*np.cos(k*x-omega*t)*np.exp(-(k*(x-x_peak)-omega*t)**2/(sigma)**2)
def laser(x, t, Ex, Ey, Ez, Bx, By, Bz,a0, k, omega, x_peak, fwhm):
    Ey = gaussian(x,0.0, a0, k, omega, x_peak, fwhm)
    Bz = np.sign(k)*gaussian(x+0.5*dx,t-0.5*dt, a0, k, omega, x_peak, fwhm)
    return Ex,Ey,Ez,Bx,By,Bz

def advance_B(dx, dt, Ex, Ey, Ez, Bx, By, Bz): 
    # update the B field from time (n-1/2)*dt to time (n+1/2)*dt
    Bx[1:-1] = Bx[1:-1]
    By[1:-1] += dt / dx * (Ez[2:] - Ez[1:-1])
    Bz[1:-1] -= dt / dx * (Ey[2:] - Ey[1:-1])
    return Bx, By, Bz
def advance_E(dx, dt, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz): 
    # update the E field from time n*dt to time (n+1)*dt
    Ex[1:-1] -= dt * Jx[1:-1]
    Ey[1:-1] -= dt/dx * (Bz[1:-1] - Bz[:-2]) + dt * Jy[1:-1]
    Ez[1:-1] += dt/dx * (By[1:-1] - By[:-2]) - dt * Jz[1:-1]
    return Ex, Ey, Ez
def pbc_B(Bx, By, Bz): 
    global nx
    By[0] = By[nx-2]
    By[nx-1] = By[1]
    Bz[0] = Bz[nx-2]
    Bz[nx-1] = Bz[1]
    return Bx, By, Bz
def pbc_E(Ex, Ey, Ez): 
    global nx
    Ey[0] = Ey[nx-2]
    Ey[nx-1] = Ey[1]
    Ez[0] = Ez[nx-2]
    Ez[nx-1] = Ez[1]
    return Ex, Ey, Ez
def compute_emfield_energy(x, Ex, Ey, Ez, Bx, By, Bz): 
    Ex_i = 0.5*(Ex[:-1]+Ex[1:]) 
    By_i = 0.5*(By[:-1]+By[1:]) 
    Bz_i = 0.5*(Bz[:-1]+Bz[1:]) 
    U_Ex = np.trapz(Ex_i[:-1]**2, x[1:-1]) 
    U_Ey = np.trapz(Ey[1:-1]**2, x[1:-1]) 
    U_Ez = np.trapz(Ez[1:-1]**2, x[1:-1]) 
    U_Bx = np.trapz(Bx[1:-1]**2, x[1:-1]) 
    U_By = np.trapz(By_i[:-1]**2, x[1:-1]) 
    U_Bz = np.trapz(Bz_i[:-1]**2, x[1:-1]) 
    U_em = U_Ex + U_Ey + U_Ez + U_Bx + U_By + U_Bz
    return U_Ex, U_Ey, U_Ez, U_Bx, U_By, U_Bz, U_em

def save_1Dquantity(xx, yy, filename, save_dir):
    np.savetxt(save_dir+'/'+filename, np.c_[xx,yy])
def sinusoidal_plasma(x_left, x_right, n0, k=0, dn=0):
    global x 
    if n0-dn>=0: 
        return np.heaviside(x-x_left,1)*np.heaviside(x_right-x,1)*(n0+dn*np.sin(k*x))
    else: 
        raise Exception('negative number density')
def plasma_sheet(x_left, x_right, n0): 
    global x 
    if n0>=0: 
        return n0*(np.heaviside(x-x_left,1)-np.heaviside(x-x_right,1))
    else: 
        raise Exception('negative number density')
def trapezoidal_plasma(x_left, n0, left_ramp=0, plateau =0, right_ramp=0):
    global x
    if n0>=0:
        x_right = x_left + left_ramp + plateau + right_ramp
        if (x_right <= x[-1]) & (x_left >= x[0]): 
            return np.piecewise(x, [ 
                x <= x_left, 
                (x > x_left) & (x <= x_left+left_ramp),
                (x > x_left+left_ramp) & (x<= x_left+left_ramp+plateau),
                (x > x_left+left_ramp+plateau) & (x <= x_right), 
                x > x_right], [ 0., lambda x: n0/left_ramp * (x-x_left), 
                n0, lambda x: n0/right_ramp * (x_right-x), 0. ]) 
        else: 
            raise Exception('trapezoid overflowing the grid')
    else: 
        raise Exception('negative number density') 

class Species: 
    def __init__(self, name , q , m):
        self.name = name
        self.q = q
        self.m = m 
        self.x = np.empty((0))
        self.px = np.empty((0))
        self.py = np.empty((0)) 
        self.pz = np.empty((0))
        self.w = np.empty((0)) 
    def add(self, x0=0., px0=0., py0=0., pz0=0., w0=1.):
        self.x = np.append(self.x, x0)
        self.px = np.append(self.px, px0)
        self.py = np.append(self.py, py0)
        self.pz = np.append(self.pz, pz0)
        self.w = np.append(self.w, w0)
    def init_positions(self, rho, nppc):
        for i in np.arange(nx):
            if(np.abs(rho[i])>0.0):
                pos = np.linspace(x[i]-dx/2., x[i]+dx/2., nppc+2, endpoint=True) 
                for p in np.arange(nppc):
                    self.add(pos[p+1], 0.,0.,0.,np.abs(rho[i]/self.q)/nppc*dx)

    def init_momenta(self, px0, py0, pz0, dpx, dpy, dpz):
        self.px = np.random.uniform(low=px0-dpx, high=px0+dpx, size=np.size(self.x))
        self.py = np.random.uniform(low=py0-dpy, high=py0+dpy, size=np.size(self.x))
        self.pz = np.random.uniform(low=pz0-dpz, high=pz0+dpz, size=np.size(self.x))
    def save_particles(self, step, save_dir): 
        filename = save_dir + '/'+ self.name + '_%04d.txt' % step 
        head = 'charge = ' + str(self.q) \
            + ' mass = ' + str(self.m) \
            + ' number = ' + str(len(self.w))
        if hasattr(self, 'x'): 
            np.savetxt(filename, np.c_[self.x, self.px, self.py, self.pz, self.w], header=head)
        return 0
    def charge_density_deposition(self): 
        rho = np.zeros(np.shape(x))
        Np = np.size(self.x)
        for p in np.arange(Np):
            xx = (self.x[p]-x[0])/dx
            index = int(np.floor(xx+0.5)) # ngp
            xx = (self.x[p]-x[index])/dx # or: xx -= index
            xx2 = xx**2
            rho[index-1] += self.q*self.w[p]/dx*0.5*(0.25 + xx2 - xx) 
            rho[index] += self.q*self.w[p]/dx*(0.75 - xx2)
            rho[index+1] += self.q*self.w[p]/dx*0.5*(0.25 + xx2 + xx)
            # periodic boundary 
        rho[1] += rho[-1]
        rho[-2] += rho[0] 
        rho[0] = rho[-2]
        rho[-1] = rho[1] 
        return rho
    def advance_positions(self):
        global dt
        gamma = sqrt(1.+(self.px**2+self.py**2+self.pz**2)/ self.m**2)
        self.x += dt * self.px / gamma / self.m
    def particles_pbc(self):
        global x, dx, Lx
        Np = np.size(self.x)
        for p in np.arange(Np):
            xx = (self.x[p]-x[0])/dx 
            index = int(np.floor(xx+0.5)) 
            if index==0:  # closest cell is left guard
                self.x[p] += Lx
            elif index==nx-1: # closest cell is right guard
                self.x[p] -= Lx
    def compute_kinetic_energy(self):
        gamma = sqrt(1.+(self.px**2+self.py**2+self.pz**2)/ self.m**2)
        ekin = self.w*self.m*(gamma-1.)
        return np.sum(ekin) 
    def advance_momenta(self, Ex, Ey, Ez, Bx, By, Bz):
        global dt, dx, x 
        Np = np.size(self.x) 
        for p in np.arange(Np):
            xxi = (self.x[p]-x[0])/dx
            xxh = xxi + 0.5 # half-integer grid in shifted to the right 
            i_index = int(np.floor(xxi+0.5)) # integer ngp 
            h_index = int(np.floor(xxh+0.5)) # half-integer ngp 
            xxi -= i_index
            xxi2 = xxi**2
            xxh -= h_index
            xxh2 = xxh**2
            w_i_l = 0.5 * (0.25 + xxi2 - xxi) # weight integer left
            w_i_c = (0.75 - xxi2) # weight integer center
            w_i_r = 0.5 * (0.25 + xxi2 + xxi) # weight integer right
            w_h_l = 0.5 * (0.25 + xxh2 - xxh) # weight half-integer left 
            w_h_c = (0.75 - xxh2) # weight half-integer center
            w_h_r = 0.5 * (0.25 + xxh2 + xxh) # weight half-integer right
            Exp = w_h_l*Ex[h_index-1]+w_h_c*Ex[h_index]+w_h_r*Ex[h_index+1]
            Eyp = w_i_l*Ey[i_index-1]+w_i_c*Ey[i_index]+w_i_r*Ey[i_index+1]
            Ezp = w_i_l*Ez[i_index-1]+w_i_c*Ez[i_index]+w_i_r*Ez[i_index+1]
            Bxp = w_i_l*Bx[i_index-1]+w_i_c*Bx[i_index]+w_i_r*Bx[i_index+1]
            Byp = w_h_l*By[h_index-1]+w_h_c*By[h_index]+w_h_r*By[h_index+1]
            Bzp = w_h_l*Bz[h_index-1]+w_h_c*Bz[h_index]+w_h_r*Bz[h_index+1]
            beta = self.q/self.m * dt/2.
            um = np.array([self.px[p]/self.m + beta*Exp, self.py[p]/self.m + beta*Eyp,self.pz[p]/self.m + beta*Ezp])
            gamma1 = sqrt(1.+um[0]**2+um[1]**2+um[2]**2)
            t = np.array([beta*Bxp/gamma1, beta*Byp/gamma1, beta*Bzp/gamma1])
            t2 = np.linalg.norm(t)**2
            s = 2.*np.asarray(t)/(1.+t2)
            # rotation 
            up = um + np.cross(um + np.cross(um, t), s)
            # half acceleration 
            self.px[p] = self.m*(up[0] + beta*Exp)
            self.py[p] = self.m*(up[1] + beta*Eyp)
            self.pz[p] = self.m*(up[2] + beta*Ezp)
    def advance_J(self,Jx,Jy,Jz):
        global x,dx
#        Jx=Jy=Jz= np.zeros(np.shape(x))
        Np=np.size(self.x)
        gamma=sqrt(1.+(self.px**2+self.py**2+self.pz**2)/self.m**2)
        for p in np.arange(Np):
            xxi = (self.x[p]-x[0])/dx
            xxh = xxi + 0.5 # half-integer grid in shifted to the right 
            i_index = int(np.floor(xxi+0.5)) # integer ngp 
            h_index = int(np.floor(xxh+0.5)) # half-integer ngp
            xxi -= i_index
            xxi2 = xxi**2
            xxh -= h_index
            xxh2 = xxh**2
            w_i_l = self.q*self.w[p]/dx*0.5 * (0.25 + xxi2 - xxi) # weight integer left 
            w_i_c = self.q*self.w[p]/dx*(0.75 - xxi2) # weight integer center
            w_i_r = self.q*self.w[p]/dx*0.5 * (0.25 + xxi2 + xxi) # weight integer right
            w_h_l = self.q*self.w[p]/dx*0.5 * (0.25 + xxh2 - xxh) # weight half-integer left 
            w_h_c = self.q*self.w[p]/dx*(0.75 - xxh2) # weight half-integer center
            w_h_r = self.q*self.w[p]/dx*0.5 * (0.25 + xxh2 + xxh) # weight half-integer right
            Jx[h_index-1] += w_h_l*self.px[p]/self.m/gamma[p]
            Jx[h_index]   += w_h_c*self.px[p]/self.m/gamma[p]
            Jx[h_index+1] += w_h_r*self.px[p]/self.m/gamma[p]
            Jy[i_index-1] +=w_i_l*self.py[p]/self.m/gamma[p]
            Jy[i_index]   +=w_i_c*self.py[p]/self.m/gamma[p]
            Jy[i_index+1] +=w_i_r*self.py[p]/self.m/gamma[p]
            Jz[i_index-1] +=w_i_l*self.pz[p]/self.m/gamma[p]
            Jz[i_index]   +=w_i_c*self.pz[p]/self.m/gamma[p]
            Jz[i_index+1] +=w_i_r*self.pz[p]/self.m/gamma[p]
        Jx[0] = Jx[nx-2]
        Jx[nx-1] = Jx[1]
        Jy[0] = Jy[nx-2]
        Jy[nx-1] = Jy[1]
        Jz[0] = Jz[nx-2]
        Jz[nx-1] = Jz[1]
        return Jx, Jy, Jz
    def pbc_J(self,Jx, Jy, Jz): 
        global nx
        Jx[0] = Jx[nx-2]
        Jx[nx-1] = Jx[1]
        Jy[0] = Jy[nx-2]
        Jy[nx-1] = Jy[1]
        Jz[0] = Jz[nx-2]
        Jz[nx-1] = Jz[1]
        return Jx, Jy, Jz

"""_____________________________________________MAIN_____________________________________________"""

"""Universal_Data"""

diags_dir='c:/Users/Gianni/Desktop/PIC/1DPIC_Laser/Data'

# UNIVERSAL CONSTANTS 
q_e = 4.80320425e-10 # statC
m_e = 9.1093837015e-31*1e3 # g
c = 2.99792458e8*1e2 # cm / s 
# UNITS CGS + GAUSS
wavlen0 = 0.8*1e-4
omega0 = 2.*pi*c/wavlen0 
unit_time = 1./omega0 
unit_length = c/omega0 
unit_em_field = m_e*c*omega0/q_e 
fs = 1e-15/unit_time 
micron = 1e-4/unit_length
# SPACE 
dx = micron/50.
#Lx = 20.*micron 
Lx = 200
x_min = 0.
x_max = Lx 
x = np.arange(x_min-dx, x_max+dx, dx)
nx = len(x)
# TIME
cfl = 0.95
dt = cfl * dx 
T = 40.*fs 
nt = int(T/dt)
t = np.linspace(0.,T, nt)
sim_summary()

"""Particle_Data"""

q_pro = 1.
q_ele = -1.
m_ele = 1.
m_pro = 1836.
# m_pro = 1.


nc=4*m_ele/(4*np.pi*q_ele**2)
print('critical density:',nc)

nppc = 1
n0 = 20
left = 0.4*x_max
right = 0.6*x_max 

#left = 54.2906
#right = 54.2906+dx
rho_ele0 = q_ele*plasma_sheet(right, left, n0)
rho_pro0 = -rho_ele0 / q_pro 

ele = Species('ELE', q=q_ele, m=m_ele)
pro = Species('PRO', q=q_pro, m=m_pro)
all_species = [ele, pro]
n_species = len(all_species)


colors = ("red", "green")


# Create plot
fig = plt.figure()
axs = fig.add_subplot(1, 1, 1)

px0= py0 = pz0 = dpx = dpy = dpz= 0
ele.init_positions(rho_ele0, nppc)
ele.init_momenta(px0 ,py0 ,pz0 ,dpx ,dpy ,dpz)
pro.init_positions(rho_pro0, nppc)
pro.init_momenta(px0 ,py0 ,pz0 ,dpx ,dpy ,dpz)
for s in all_species:
    print('Species summary:')
    print('Name:', s.name, ', charge [e] = ', s.q, \
          ', mass [m_e] = ', s.m, ', number of macro-particles', len(s.w))
    s.save_particles(0, diags_dir)
    
    rho = s.charge_density_deposition() 
#    save_1Dquantity(x, rho, 'RHO_%s_%04d.txt' % (s.name, 0), diags_dir)
for v, color in zip(all_species, colors):
    axs.scatter(x,rho, alpha=0.8, c=color, edgecolors='none', s=30, label=v.name)
plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()


Ex=Ey = Ez = Bx = By = Bz = 0
Ex, Ey, Ez, Bx, By, Bz = init_EMfield(x, Ex, Ey, Ez, Bx, By, Bz)
a0=1
omega=1
k=omega
fwhm =15*fs
x_peak=1.1*fwhm 

Ex, Ey, Ez, Bx, By, Bz = laser(x, 0., Ex, Ey, Ez, Bx, By, Bz,a0, k, omega, x_peak, fwhm)
Jx= np.zeros((nx))
Jy= np.zeros((nx))
Jz= np.zeros((nx))


#fig = plt.figure(figsize=(12, 8))
#axs = fig.gca(xlabel="t [code units]", ylabel="momentum [code units]", title='electron momentum')


every_out=20
# plt.plot(x,Ey)
M1=np.zeros([nx,int(nt/every_out)+1])
M2=np.zeros([nx,int(nt/every_out)+1])
M3=np.zeros([nx,int(nt/every_out)+1])
fr=0
#for index, s in np.ndenumerate(all_species):
#    Jx,Jy,Jz=s.advance_J(Jx,Jy,Jz)
emfield_energy = np.empty((nt,7))
kinetic_energy = np.empty((nt,n_species))
for n in range(nt): 
    if (n % every_out == 0):
        print(n)
        save_1Dquantity(x, np.c_[Ex,Ey,Ez,Bx,By,Bz], 'EM_FIELD_%04d.txt' % n, diags_dir)
        for s in all_species:
            rho = s.charge_density_deposition()
            save_1Dquantity(x, rho, 'RHO_%s_%04d.txt' % (s.name, n), diags_dir)
            s.save_particles(n, diags_dir) 
 #       plt.plot(x,Ey, alpha=0.8)
        M1[:,fr]=Ex
        M2[:,fr]=Ey
        M3[:,fr]=rho
        fr=fr+1
    Bx, By, Bz = advance_B(dx, 0.5*dt, Ex, Ey, Ez, Bx, By, Bz)
    Bx, By, Bz = pbc_B(Bx, By, Bz)
    for index, s in np.ndenumerate(all_species): 
        s.advance_momenta(Ex, Ey, Ez, Bx, By, Bz)
        kinetic_energy[n,index] = s.compute_kinetic_energy()
    Bx, By, Bz = advance_B(dx, 0.5*dt, Ex, Ey, Ez, Bx, By, Bz)
    Bx, By, Bz = pbc_B(Bx, By, Bz)
    for index, s in np.ndenumerate(all_species):
        s.advance_positions() 
        s.particles_pbc() 
        Jx,Jy,Jz=s.advance_J(Jx,Jy,Jz)
        s.advance_positions() 
        s.particles_pbc() 
    Ex, Ey, Ez = advance_E(dx, dt, Ex, Ey, Ez, Bx, By, Bz, Jx, Jy, Jz)
    Jx= np.zeros((nx))
    Jy= np.zeros((nx))
    Jz= np.zeros((nx))
    Ex, Ey, Ez = pbc_E(Ex, Ey, Ez) 
    emfield_energy[n,:] = compute_emfield_energy(x, Ex, Ey, Ez, Bx, By, Bz)
#    for v, color in zip(all_species, colors):
#        axs.scatter(n*dt,v.x, alpha=0.8, c=color, edgecolors='none', s=30)
#        plt.plot(x,rho, alpha=0.8)


save_1Dquantity(t, np.c_[emfield_energy], 'EM_FIELD_ENERGY.txt', diags_dir)
for index, s in np.ndenumerate(all_species):
    save_1Dquantity(t, np.c_[kinetic_energy], 'KINETIC_ENERGY_%s.txt' %s.name, diags_dir)
    
    
    
    

fig = plt.figure()
ydata = []
axis = plt.axes(xlim =(0, x_max), ylim =(-10, 10)) 
line, = axis.plot([], [], lw = 1) 

# data which the line will 
# contain (x, y)
def init(): 
    line.set_data([], [])
    return line,
   
def animate(i):
    """perform animation step"""
    global M3,x, dt    
    line.set_data(x,M3[:,i])
    return line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=int(nt/every_out), interval=100, blit=True)
plt.show()

Ta=10
Ti=18
Time=T/every_out
T1=Time*(Ta-Ti)
T2=Time*Ta
T3=Time*(Ta+Ti)
plt.figure(num=5)
plt.rc("figure", figsize=[20, 10])
plt.subplot(3, 3, 1)
plt.plot(x,M1[:,Ta-Ti],color='blue')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("$E_{x}$ field  [code units]")
plt.title('t = %.2f [code units]' % T1)
plt.subplot(3, 3, 2)
plt.plot(x,M2[:,Ta-Ti],color='violet')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("$E_{y}$ field  [code units]")
plt.title('t = %.2f [code units]' % T1)
plt.subplot(3, 3, 3)
plt.plot(x,M3[:,Ta-Ti],color='green')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("density [code units]")
plt.title('t = %.2f [code units]' % T1)
plt.subplot(3, 3, 4)
plt.plot(x,M1[:,Ta],color='blue')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("$E_{x}$ field  [code units]")
plt.title('t = %.2f [code units]' % T2)
plt.subplot(3, 3, 5)
plt.plot(x,M2[:,Ta],color='violet')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("$E_{y}$ field  [code units]")
plt.title('t = %.2f [code units]' % T2)
plt.subplot(3, 3, 6)
plt.plot(x,M3[:,Ta],color='green')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("density [code units]")
plt.title('t = %.2f [code units]' % T2)
plt.subplot(3, 3, 7)
plt.plot(x,M1[:,Ta+Ti],color='blue')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("$E_{x}$ field  [code units]")
plt.title('t = %.2f [code units]' % T3)
plt.subplot(3, 3, 8)
plt.plot(x,M2[:,Ta+Ti],color='violet')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("$E_{y}$ field  [code units]")
plt.title('t = %.2f [code units]' % T3)
plt.subplot(3, 3, 9)
plt.plot(x,M3[:,Ta+Ti],color='green')
plt.grid(color='black', linestyle='-', linewidth=0.5,alpha=0.2)
plt.xlabel("x [code units]")
plt.ylabel("density [code units]")
plt.title('t = %.2f [code units]' % T3)
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 8))
ax = fig.gca(xlabel="t [code units]", ylabel="field energy [code units]", title='energy contributions')
ax.plot(t,emfield_energy[:,0],label='Ex')
ax.plot(t,emfield_energy[:,1],label='Ey')
ax.plot(t,emfield_energy[:,2],label='Ez')
ax.plot(t,emfield_energy[:,3],label='Bx')
ax.plot(t,emfield_energy[:,4],label='By')
ax.plot(t,emfield_energy[:,5],label='Bz')
ax.plot(t,emfield_energy[:,5],label='Bz')
ax.plot(t,emfield_energy[:,5],label='Bz')
ax.plot(t,kinetic_energy[:,0], label=r'E$_{kin}$ ele')
ax.plot(t,kinetic_energy[:,1], label=r'E$_{kin}$ pro')
ax.legend()
plt.plot(t,emfield_energy[:,0]+emfield_energy[:,1]+emfield_energy[:,2]+emfield_energy[:,3]
         +emfield_energy[:,4]+emfield_energy[:,5]+emfield_energy[:,6]+
         kinetic_energy[:,1]+kinetic_energy[:,0])
    

   

   


   



