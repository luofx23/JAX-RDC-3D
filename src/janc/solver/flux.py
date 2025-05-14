import jax.numpy as jnp
from jax import jit
from ..solver import aux_func
from ..grid import read_grid as gd

p = 2
eps = 1e-6
C1 = 1 / 10
C2 = 3 / 5
C3 = 3 / 10

@jit
def splitFlux_LF(ixy, U, aux):
    rho,u,v,w,Y,p,a = aux_func.U_to_prim(U,aux)
    rhoE = U[4:5]

    zx = (ixy == 1) * gd.dxi_dx + (ixy == 2)*gd.deta_dx + (ixy==3)*gd.ddelta_dx
    zy = (ixy == 1) * gd.dxi_dy + (ixy == 2)*gd.deta_dy + (ixy==3)*gd.ddelta_dy
    zz = (ixy == 1) * gd.dxi_dz + (ixy == 2)*gd.deta_dz + (ixy==3)*gd.ddelta_dz

    F = jnp.concatenate([rho*u,rho*u**2+p,rho*u*v,rho*u*w,u*(rhoE+p),rho*u*Y],axis=0)
    G = jnp.concatenate([rho*v,rho*u*v,rho*v**2+p,rho*v*w,v*(rhoE+p),rho*v*Y],axis=0)
    H = jnp.concatenate([rho*w,rho*u*w,rho*v*w,rho*w**2+p,w*(rhoE+p),rho*w*Y],axis=0)
    vel = zx*u + zy*v + zz*w
    grad_norm = jnp.sqrt(zx**2 + zy**2 + zz**2)
    flux = gd.J*(zx*F + zy*G + zz*H)
    velm = jnp.nanmax(abs(vel) + a*grad_norm)
    
    flux_plus = 0.5 * (flux + velm * gd.J * U)
    flux_minus = 0.5 * (flux - velm * gd.J * U)
    
    return flux_plus, flux_minus

@jit
def WENO_plus_x(f):
    fj = f[:,2:-3,3:-3,3:-3]
    fjp1 = f[:,3:-2,3:-3,3:-3]
    fjp2 = f[:,4:-1,3:-3,3:-3]
    fjm1 = f[:,1:-4,3:-3,3:-3]
    fjm2 = f[:,0:-5,3:-3,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,1:,:,:] - fj_halfp[:,0:-1,:,:]
    return dfj

@jit
def WENO_plus_y(f):

    fj = f[:,3:-3,2:-3,3:-3]
    fjp1 = f[:,3:-3,3:-2,3:-3]
    fjp2 = f[:,3:-3,4:-1,3:-3]
    fjm1 = f[:,3:-3,1:-4,3:-3]
    fjm2 = f[:,3:-3,0:-5,3:-3]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,:,1:,:] - fj_halfp[:,:,0:-1,:]

    return dfj

@jit
def WENO_plus_z(f):

    fj = f[:,3:-3,3:-3,2:-3]
    fjp1 = f[:,3:-3,3:-3,3:-2]
    fjp2 = f[:,3:-3,3:-3,4:-1]
    fjm1 = f[:,3:-3,3:-3,1:-4]
    fjm2 = f[:,3:-3,3:-3,0:-5]

    IS1 = 1 / 4 * jnp.power((fjm2 - 4 * fjm1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjm2 - 2 * fjm1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjm1 - fjp1), 2) + 13 / 12 * jnp.power((fjm1 - 2 * fj + fjp1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjp1 + fjp2), 2) + 13 / 12 * jnp.power((fj - 2 * fjp1 + fjp2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfp1 = 1 / 3 * fjm2 - 7 / 6 * fjm1 + 11 / 6 * fj
    fj_halfp2 = -1 / 6 * fjm1 + 5 / 6 * fj + 1 / 3 * fjp1
    fj_halfp3 = 1 / 3 * fj + 5 / 6 * fjp1 - 1 / 6 * fjp2

    fj_halfp = w1 * fj_halfp1 + w2 * fj_halfp2 + w3 * fj_halfp3
    dfj = fj_halfp[:,:,:,1:] - fj_halfp[:,:,:,0:-1]

    return dfj



@jit
def WENO_minus_x(f):

    fj = f[:,3:-2,3:-3,3:-3]
    fjp1 = f[:,4:-1,3:-3,3:-3]
    fjp2 = f[:,5:,3:-3,3:-3]
    fjm1 = f[:,2:-3,3:-3,3:-3]
    fjm2 = f[:,1:-4,3:-3,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,1:,:,:] - fj_halfm[:,0:-1,:,:])

    return dfj

@jit
def WENO_minus_y(f):

    fj = f[:,3:-3,3:-2,3:-3]
    fjp1 = f[:,3:-3,4:-1,3:-3]
    fjp2 = f[:,3:-3,5:,3:-3]
    fjm1 = f[:,3:-3,2:-3,3:-3]
    fjm2 = f[:,3:-3,1:-4,3:-3]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,:,1:,:] - fj_halfm[:,:,0:-1,:])

    return dfj

@jit
def WENO_minus_z(f):

    fj = f[:,3:-3,3:-3,3:-2]
    fjp1 = f[:,3:-3,3:-3,4:-1]
    fjp2 = f[:,3:-3,3:-3,5:,]
    fjm1 = f[:,3:-3,3:-3,2:-3]
    fjm2 = f[:,3:-3,3:-3,1:-4]

    IS1 = 1 / 4 * jnp.power((fjp2 - 4 * fjp1 + 3 * fj), 2) + 13 / 12 * jnp.power((fjp2 - 2 * fjp1 + fj), 2)
    IS2 = 1 / 4 * jnp.power((fjp1 - fjm1), 2) + 13 / 12 * jnp.power((fjp1 - 2 * fj + fjm1), 2)
    IS3 = 1 / 4 * jnp.power((3 * fj - 4 * fjm1 + fjm2), 2) + 13 / 12 * jnp.power((fj - 2 * fjm1 + fjm2), 2)

    alpha1 = C1 / jnp.power((eps + IS1), p)
    alpha2 = C2 / jnp.power((eps + IS2), p)
    alpha3 = C3 / jnp.power((eps + IS3), p)


    w1 = alpha1 / (alpha1 + alpha2 + alpha3)
    w2 = alpha2 / (alpha1 + alpha2 + alpha3)
    w3 = alpha3 / (alpha1 + alpha2 + alpha3)

    fj_halfm1 = 1 / 3 * fjp2 - 7 / 6 * fjp1 + 11 / 6 * fj
    fj_halfm2 = -1 / 6 * fjp1 + 5 / 6 * fj + 1 / 3 * fjm1
    fj_halfm3 = 1 / 3 * fj + 5 / 6 * fjm1 - 1 / 6 * fjm2

    fj_halfm = w1 * fj_halfm1 + w2 * fj_halfm2 + w3 * fj_halfm3
    dfj = (fj_halfm[:,:,:,1:] - fj_halfm[:,:,:,0:-1])

    return dfj

@jit
def weno5(U,aux,dx,dy,dz):
    Fplus, Fminus = splitFlux_LF(1, U, aux)
    Gplus, Gminus = splitFlux_LF(2, U, aux)
    Hplus, Hminus = splitFlux_LF(3, U, aux)

    dFp = WENO_plus_x(Fplus)
    dFm = WENO_minus_x(Fminus)

    dGp = WENO_plus_y(Gplus)
    dGm = WENO_minus_y(Gminus)
    
    dHp = WENO_plus_z(Hplus)
    dHm = WENO_minus_z(Hminus)

    dF = dFp + dFm
    dG = dGp + dGm
    dH = dHp + dHm

    netflux = dF/dx + dG/dy + dH/dz

    return -netflux/gd.J[:,3:-3,3:-3,3:-3]
