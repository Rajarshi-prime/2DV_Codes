import cupy as np
import cupy.fft as fft
from time import time
## This program solves the 2D vorticity equation without forcing using RK4 and FFT
## The 2D vorticity equation is given by 
## d_t \xi  + u.\grad \xi= \nu \laplacian \xi.
## Defining stream funciton and taking fourier transpose we get 
## The dissipation term is (\laplace)^8. 
## In the fourier space, the equation is
## d_t \psi_k = \int_k' (k-k')^2/k^2 [\hat{z} . {(k-k') x k'}] \psi_k' \psi_{k-k'}  - \nu k^2 \psi_k.

## We have a finite number of grid points i.e. we have finite number of ks. 
## We initiate a finite number of wavenumber. Then evolve and fourier the -k^2 \psi_k at every instant. 
## How to evolve the k = 0 mode?

## It is best to define the function which returns the real part of the iifted function as ifft. 
ifft2 = lambda x: np.real(fft.ifft2(x))

## Forming the 2D grid (position space)
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
Nx, Ny = 1024*3,1024*3 #Number of points
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
x,y = np.meshgrid(X,Y)

## Forming the 2D grid (k space)
Kx = 2*np.pi*np.linspace(-(Nx//2) , Nx//2 - 0.5*(1+ (-1)**(Nx%2)),Nx)/Lx
Ky = 2*np.pi*np.linspace(-(Ny//2) , Ny//2 - 0.5*(1+ (-1)**(Ny%2)),Ny)/Ly
Kx = np.append(Kx[Kx>=0], Kx[Kx<0])
Ky = np.append(Ky[Ky>=0], Ky[Ky<0])
kx,ky = np.meshgrid(Kx,Ky)

## Defining the inverese laplacian.
lap = -(kx**2 + ky**2)
lap1 = lap.copy()
lap1[lap1== 0] = np.inf
lapinv = 1.0/lap1


## Ignoring the k = 0 mode as this mode of psi does not contribute in the evolution equation of xi

## Variables with r denotes quantities in position space.
## Underscore denotes derivative.


## Defining the advective term psi = str fn ; xi = vorticity.
def adv(t,xi):
    psi = -lapinv*xi
    ur = ifft2(1j * ky*psi)
    vr = ifft2(-1j * kx*psi)
    xi_xr = ifft2(1j * kx*xi)
    xi_yr = ifft2(1j * ky*xi)
    trm1 = ur*xi_xr
    trm2 = vr*xi_yr
    advterm =   trm1 + trm2
    return -1.0*dealias * fft.fft2( advterm)
    




## The RK4 integration function
def evolve_and_save(f,t,x0):
    h = t[1] - t[0]
    x_old = x0
    etot = 1.0*np.zeros(len(t)//20) +1
    
    for i,ti in enumerate(t[:-1]):
        print(np.round(t[i],2),end= '\r')
        
        k1 = f(ti,x_old)
        k2 = f(ti + h/2, x_old + h/2*k1)
        k3 = f(ti + h/2, x_old + h/2*k2)
        k4 = f(ti + h, x_old + h*k3)
        
        # print(f'x[{i}] = ',x[i])
        # print("k1 = ",k1)
        # print("k2 = ",k2)
        # print("k3 = ",k3)
        # print("k4 = ",k4)
        
        
        xivis = nu *(lap**lp) ## The viscous term 
        x_new = (x_old + h/6.0*(k1 + 2*k2 + 2*k3 + k4))/(1.0 + h*xivis)
        ## Things to do every 1 second (dt = 0.1)
        if i%10 ==0:
            ## Saving the vorticity contour
            np.save(f"Oceans/data/vorticity {np.round(t[i]/t[-1]*100,2)}%",ifft2(x_old))
            
            
            ## The energy density per wavenumber + flux per wavenumber
            psi = -lapinv*x_old
            psi1 = np.conjugate(psi)
            e = np.real(0.5*k**2*psi*psi1)
            g = np.real(k1*psi1)
            # Extracting the energy wavenumbers as and putting them in the bin.
            # a = k1*psi1
            # erate = 0.5*(a + np.conjugate(a) - 0.5*nu*psi*psi1*(lap**lp))
            # e1d,erate1d = e2d_to_e1d(e,erate)        
            e1d,g1d = e2d_to_e1d(e,g)        
            flx = np.cumsum(g1d[::-1])[::-1]
            
            
            np.save(f"Oceans/data/E(k) {np.round(t[i]/t[-1]*100,2)}%",e1d)
            np.save(f"Oceans/data/PI(k) {np.round(t[i]/t[-1]*100,2)}%",flx)
            
            
        x_old = x_new
      
        
        
    ## Saving the last vorticity        
    np.save(f"Oceans/data/vorticity {np.round(t[i+1]/t[-1]*100,2)}%",ifft2(x_old))
    ## The last energy density per wavenumber + flux per wave number
    psi = -lapinv*x_old
    psi1 = np.conjugate(psi)
    e = np.real(0.5*k**2*psi*psi1)
    g = np.real(f(t[i+1],x_old)*psi1)
    # Extracting the energy wavenumbers as and putting them in the bin.
    # a = k1*psi1
    # erate = 0.5*(a + np.conjugate(a) - 0.5*nu*psi*psi1*(lap**lp))
    # e1d,erate1d = e2d_to_e1d(e,erate)        
    e1d,g1d = e2d_to_e1d(e,g)        
    flx = np.cumsum(g1d[::-1])[::-1]
    
    
    np.save(f"Oceans/data/E(k) {np.round(t[i+1]/t[-1]*100,2)}%",e1d)
    np.save(f"Oceans/data/PI(k) {np.round(t[i+1]/t[-1]*100,2)}%",flx)
    


    


## Quantities that will be important later on.
# Power of the laplacian in the hyperviscous term
lp = 8.0

# Co-efficient of kinematic viscosity 
nu = 1e-34
k = (kx**2 + ky**2)**0.5
k1d = np.arange(np.max(np.abs(Kx))+1)




def e2d_to_e1d(e,erate):
    # e = np.ones_like(k)
        
    e1d_f = 1.0*np.zeros_like(k1d)        
    erate1d_f = 1.0*np.zeros_like(k1d)        
    for i in range(len(k1d)):
        cond = (k> k1d[i] -0.5) *(k < k1d[i] +0.5)
        e1d_f[i]  = np.sum(e[cond])
        erate1d_f[i]  = np.sum(erate[cond])
    return e1d_f,erate1d_f
   

dalcutoff = ((2*np.pi*Nx)/Lx)//3,((2*np.pi*Ny)/Ly)//3
dealias = (abs(kx)<dalcutoff[0])*(abs(ky)<dalcutoff[1])
einit = 100 ## Amplitude of psi0s


kinit= 6 ## Intial energy will be distributed among these regions equally in all wave numbers but will have a phase. 
 
 
 
 
 
 
## The initial conditions   
thinit = np.random.uniform(0,2*np.pi,np.shape(k))
psi0  = einit*(k<kinit)*np.exp(1j*thinit)
xi0 = -lap*psi0



            




## Time-step
dt = 0.1
T = 2000
t = np.arange(0,dt + T,dt)
t1 = time()
etot = evolve_and_save(adv,t,xi0)
t2 = time()
print(f"Time taken to evolve for {T} secs in {dt} sec timesteps with gridsize {Nx}x{Nx} is \n {t2-t1} sec")
np.save("Oceans/data/x",X)
np.save("Oceans/data/y",Y)
np.save("Oceans/data/k1d",k1d)
np.save("Oceans/data/t",t[::10])
np.save("Oceans/data/param",np.array([nu,dt,T]))
np.save("Oceans/data/gridsize",f"{Nx}x{Ny}.txt")



# plt.plot(t[::20],etot.get())
# plt.xlabel("time")
# plt.ylabel("Totat \n energy",rotation = 0)
# plt.grid()
# plt.savefig(f"Oceans/plots/etot_v_t.png")

print("Done!")
