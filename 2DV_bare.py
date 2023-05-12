import cupy as np
import cupy.fft as fft
from time import time
# import h5py
import pathlib
curr_path = pathlib.Path(__file__).parent
## --------------- Details of the code ---------------
    # This program solves the 2D vorticity equation without forcing using RK4 and FFT. 
    # The 2D vorticity equation is given by d_t \xi  + u.\grad \xi= \nu \laplacian \xi.
    # Defining stream funciton and taking fourier transpose we get 
    # The dissipation term is (\laplace)^8. 
    # In the fourier space, the equation is
    # d_t \psi_k = \int_k' (k-k')^2/k^2 [\hat{z} . {(k-k') x k'}] \psi_k' \psi_{k-k'}  - \nu k^2 \psi_k.

    # We have a finite number of grid points i.e. we have finite number of ks. 
    # We initiate a finite number of wavenumber. Then evolve and fourier the -k^2 \psi_k at every instant. 


    # Ignoring the k = 0 mode as this mode of psi does not contribute in the evolution equation of xi

    # Variables with r denotes quantities in position space.
    # Underscore denotes derivative.
## ----------------------------------------------------


## ------------ Grid and related operators ------------
## Forming the 2D grid (position space)
Lx, Ly = (2*np.pi),(2*np.pi) #Length of the grid
Nx, Ny = 128*3,128*3 #Number of points
X,Y = np.linspace(0,Lx,Nx,endpoint= False), np.linspace(0,Ly,Ny,endpoint= False)
x,y = np.meshgrid(X,Y,indexing="ij")

## It is best to define the function which returns the real part of the iifted function as ifft. 
ifft2 = lambda x: fft.irfft2(x,(Nx,Ny))

## Forming the 2D grid (k space)
Kx = 2*np.pi*np.linspace(-(Nx//2) , Nx//2 - 0.5*(1+ (-1)**(Nx%2)),Nx)/Lx
Ky = 2*np.pi*np.linspace(-(Ny//2) , Ny//2 - 0.5*(1+ (-1)**(Ny%2)),Ny)/Ly
Kx = np.append(Kx[Kx>=0], Kx[Kx<0])
Ky = np.append(Ky[Ky>=0], -Ky[0])
kx,ky = np.meshgrid(Kx,Ky,indexing="ij")

## Defining the inverese laplacian.
lap = -(kx**2 + ky**2)
# lap1 = lap.copy()
# lap1[lap1== 0] = np.inf
lapinv = 1.0/np.where(lap == 0., np.inf, lap)
## ----------------------------------------------------


## -----------------  Parameters  ----------------------
# Power of the laplacian in the hyperviscous term
lp = 8.0

# Co-efficient of kinematic viscosity 
nu = 1e-32
k = (kx**2 + ky**2)**0.5
xivis =  nu *(lap**lp) ## The viscous term 

dalcutoff = ((2*np.pi*Nx)/Lx)//3,((2*np.pi*Ny)/Ly)//3
dealias = (abs(kx)<dalcutoff[0])*(abs(ky)<dalcutoff[1])
einit = 0.5 ## Amplitude of psi0s


kinit= 100 ## Intial energy will be distributed among these regions equally in all wave numbers but will have a phase. 
## ----------------------------------------------

## ---------------- Time things -----------------
## Time-step
dt = 0.005
T = 10
t = np.arange(0,dt + T,dt)

st = int(1/dt) ## Interval of saving the data
## ----------------------------------------------

## ------------- Dictionary of params ----------
param = dict()
param["N"] = Nx*2/3.
param["T"] = T
param["Re"] = 1/nu
param["dt"] = dt
param["einit"] = einit
param["kinit"] = kinit
## -------------------------------------------------------


# ## ------------Opening the hdf5 files -------------
# file = h5py.File("2dVdata.hdf5","a")
# try: 
#     data = file.create_group(f'Re = {1/nu},dt = {dt}, gridsize = {Nx}x{Ny}, final_time = {T}, einit = {einit}')
#     vorticity  = data.create_dataset("vorticity",(len(t)//st+1,Nx,Ny),dtype= np.float64)
#     velocity_x = data.create_dataset("u",(len(t)//st+1,Nx,Ny),dtype= np.float64)
#     velocity_y = data.create_dataset("v",(len(t)//st+1,Nx,Ny),dtype= np.float64)
#     times = data.create_dataset("time",len(t)//st +1,dtype = np.float64)
# except ValueError: 
#     data = file[f'Re = {1/nu},dt = {dt}, gridsize = {Nx}x{Ny}, final_time = {T}, einit = {einit}']
#     vorticity  = data["vorticity"]
#     velocity_x = data["u"]
#     velocity_y = data["v"]
#     times = data["time"]
# times[:] = t[::st].get()
# ## -------------------------------------------------




## -------------- Initializing the empty arrays -----------------
psi = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
xi = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
x_old = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
x_new = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k1 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k2 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k3 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)
k4 = np.empty((Nx,Ny//2 + 1),dtype = np.complex128)


ur = np.empty((Nx,Ny),dtype = np.float64)
vr = np.empty((Nx,Ny),dtype = np.float64)
xi_xr = np.empty((Nx,Ny),dtype = np.float64)
xi_yr = np.empty((Nx,Ny),dtype = np.float64)
advterm = np.empty((Nx,Ny),dtype = np.float64)

## --------------------------------------------------------------

## ------------------- RHS w/o viscosity --------------------
"""psi = str fn ; xi = vorticity"""
def adv(t,xi,i):
    global psi, ur, vr,xi_xr, xi_yr, advterm,dealias,st
    psi[:] = -lapinv*xi
    ur[:] = ifft2(1j * ky*psi)
    vr[:] = ifft2(-1j * kx*psi) 
    # print(ur.min().get(),end = "\r")
    # velocity_x[i//st,:] = ur.get()
    # velocity_y[i//st,:] = vr.get()
    xi_xr[:] = ifft2(1j * kx*xi)
    xi_yr[:] = ifft2(1j * ky*xi)
    advterm[:] = ur*xi_xr + vr*xi_yr
    return -1.0*dealias * fft.rfft2( advterm)

## ----------------------------------------------------------
    

## -------------The RK4 integration function -----------------
def evolve_and_save(f,t,x0):
    # print()
    h = t[1] - t[0]
    x_old[:] = x0
    etot = 1.0*np.zeros(len(t)//20) +1
    
    for i,ti in enumerate(t[:-1]):
        # print(np.round(t[i],2),ifft2(x_old).min().get(),end= '\r')
        
        
        k1[:] = adv(ti,x_old,i)
        k2[:] = adv(ti + h/2, x_old + h/2*k1,i)
        k3[:] = adv(ti + h/2, x_old + h/2*k2,i)
        k4[:] = adv(ti + h, x_old + h*k3,i)
        
        # print(f'x[{i}] = ',x[i])
        # print("k1 = ",k1)
        # print("k2 = ",k2)
        # print("k3 = ",k3)
        # print("k4 = ",k4)
        
        x_new[:] = (x_old + h/6.0*(k1 + 2*k2 + 2*k3 + k4))/(1.0 + h*xivis)
        ## Things to do every 1 second (dt = 0.1)
        if i%st ==0:
            print(np.round(t[i],2),end= '\r')
            
            ## Saving the vorticity contour
            # np.save(f"data/vorticity {np.round(t[i]/t[-1]*100,2)}%",ifft2(x_old))
            # vorticity[i//st,:] = ifft2(x_old).get()
            savePath = curr_path/f"data/Re_{np.round(1/nu,2)},dt_{dt}/time_{t[i]}"
            savePath.mkdir(parents=True, exist_ok=True)
            np.save(f"data/Re_{np.round(1/nu,2)},dt_{dt}/time_{t[i]}/w", x_old)
            
            
            
        x_old[:] = x_new
      
        
        
    ## Saving the last vorticity        
    # vorticity[i//st+1,:] = ifft2(x_old).get()
    savePath = curr_path/f"data/Re_{np.round(1/nu,2)},dt_{dt}/time_{t[i+1]}"
    savePath.mkdir(parents=True, exist_ok=True)
    np.save(f"data/Re_{np.round(1/nu,2)},dt_{dt}/time_{t[i+1]}/w", x_old)
    return etot
## ---------------------------------------------------------   


    









## ---------------- Initial conditions ----------------
r = np.random.RandomState(309)
thinit = r.uniform(0,2*np.pi,np.shape(k))
psi0  = einit*(k<kinit)*np.exp(1j*thinit)
ur[:] = ifft2(1j * ky*psi)
vr[:] = ifft2(-1j * kx*psi) 
xi0 = -lap*psi0
## ----------------------------------------------------




t1 = time()
etot = evolve_and_save(adv,t,xi0)
t2 = time()
print(f"Time taken to evolve for {T} secs in {dt} sec timesteps with gridsize {Nx}x{Nx} is \n {t2-t1} sec")
np.save(f"data/Re_{np.round(1/nu,2)},dt_{dt}/parameters",param)
# file.close()
