import numpy as np
import matplotlib.pyplot as plt

with open('input.txt', 'r') as file:
    exec(file.read())

Nx = int(L / dx)  
Nt = int(T / dt)  

uss = (a / 5) 
vss = 1 + (uss**2)

def traveling_wave_init(var, var_ss, amplitude, wavelength, speed, t, direction='x', perturbation_region=None):
    x = np.linspace(0, L, var.shape[0])
    y = np.linspace(0, L, var.shape[1])
    xx, yy = np.meshgrid(x, y)
    
    if perturbation_region is not None:
        x_start, x_end, y_start, y_end = perturbation_region
        perturbation_mask = ((xx >= x_start) & (xx <= x_end)) & ((yy >= y_start) & (yy <= y_end))
    else:
        perturbation_mask = np.ones_like(xx, dtype=bool)
    
    if direction == 'x':
        wave = amplitude * np.sin(2 * np.pi * (xx - speed * t) / wavelength)
    elif direction == 'y':
        wave = amplitude * np.sin(2 * np.pi * (yy - speed * t) / wavelength)
    else:
        raise ValueError("Direction must be 'x' or 'y'")
    
    var[perturbation_mask] += wave[perturbation_mask]
    return var

u = np.full((Nx, Nx), uss)
v = np.full((Nx, Nx), vss)

amplitude = 0.1  
wavelength = 50 
speed = 1.0  
t = 0 

x_start, x_end = 195,205
y_start, y_end = 195,205

u = traveling_wave_init(u, uss, amplitude, wavelength, speed, t, direction='x', perturbation_region=(x_start, x_end, y_start, y_end))
v = traveling_wave_init(v, vss, amplitude, wavelength, speed, t, direction='y', perturbation_region=(x_start, x_end, y_start, y_end))

u[:, 0] = u[:, 1]
u[:, -1] = u[:, -2]
u[0, :] = u[1, :]
u[-1, :] = u[-2, :]

v[:, 0] = v[:, 1]
v[:, -1] = v[:, -2]
v[0, :] = v[1, :]
v[-1, :] = v[-2, :]

def laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1, 1:-1] = (u[:-2, 1:-1] + u[2:, 1:-1] + u[1:-1, :-2] + u[1:-1, 2:] - 4 * u[1:-1, 1:-1]) / (dx**2)
    return lap

for t in range(Nt):
    lap_u = laplacian(u, dx)
    lap_v = laplacian(v, dx)
    u += dt * (lap_u + (a - u - ((4 * u * v) / (1 + u**2))))
    v += dt * (d * lap_v + (b * (u - ((u * v) / (1 + u**2)))))
    
    u[:, 0] = u[:, 1]
    u[:, -1] = u[:, -2]
    u[0, :] = u[1, :]
    u[-1, :] = u[-2, :]

    v[:, 0] = v[:, 1]
    v[:, -1] = v[:, -2]
    v[0, :] = v[1, :]
    v[-1, :] = v[-2, :]
    
    if t % save_interval == 0:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(u, cmap='viridis', extent=[0, L, 0, L], origin='lower')
        plt.title(f'Time: {t*dt:.2f}: $u$')
        plt.colorbar(label='Concentration')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.subplot(1, 2, 2)
        plt.imshow(v, cmap='viridis', extent=[0, L, 0, L], origin='lower')
        plt.title(f'Time: {t*dt:.2f}: $v$')
        plt.colorbar(label='Concentration')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"au_{int(t/save_interval):04d}.png")
        plt.close()

final_u=u.copy()
final_v=v.copy()
np.savez('data.npz', final_u=final_u, final_v=final_v)

