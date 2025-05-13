import jax.numpy as jnp
from jax import jit,vmap,pmap
from ..solver import aux_func
from .flux import weno5
from ..thermodynamics import thermo
from ..thermodynamics import chemical
from ..boundary import boundary
#from ..grid import read_grid as gd
from functools import partial



def CFL(field,dx,dy,dz,cfl=0.20):
    U, aux = field[0:-2],field[-2:]
    _,u,v,w,_,_,a = aux_func.U_to_prim(U,aux)
    cx = jnp.max(abs(u) + a)
    cy = jnp.max(abs(v) + a)
    cz = jnp.max(abs(w) + a)
    dt = jnp.minimum(cfl*dx/cx,cfl*dy/cy,cfl*dz/cz)
    return dt

def set_solver(thermo_set, boundary_set, source_set = None, nondim_set = None):
    thermo.set_thermo(thermo_set,nondim_set)
    boundary.set_boundary(boundary_set)
    aux_func.set_source_terms(source_set)
    
    def rhs(U,aux,dx,dy,dz,theta=None):
        aux = aux_func.update_aux(U, aux)
        U_with_ghost,aux_with_ghost = boundary.boundary_conditions(U,aux,theta)
        physical_rhs = weno5(U_with_ghost,aux_with_ghost,dx,dy,dz) + aux_func.source_terms(U, aux, theta)
        return physical_rhs
    

    def advance_flux(field,dt,dx,dy,dz,theta=None):
        
        U, aux = field[0:-2],field[-2:]
        U1 = U + dt * rhs(U,aux,dx,dy,dz,theta)
        U2 = 3/4*U + 1/4 * (U1 + dt * rhs(U1,aux,dx,dy,dz,theta))
        U3 = 1/3*U + 2/3 * (U2 + dt * rhs(U2,aux,dx,dy,dz,theta))
        field = jnp.concatenate([U3,aux],axis=0)
        
        return field
    
    def advance_source_term(field,dt):
        U, aux = field[0:-2],field[-2:]
        aux = aux_func.update_aux(U, aux)
        _,T = aux_func.aux_to_thermo(U,aux)
        rho = U[0:1]
        Y = U[5:]/rho
        drhoY = chemical.solve_implicit_rate(T,rho,Y,dt)

        p1 = U[0:5]
        p2 = U[5:] + drhoY
        U_new = jnp.concatenate([p1,p2],axis=0)
        return jnp.concatenate([U_new,aux],axis=0)


    @jit    
    def advance_one_step(field,dt,dx,dy,dz,theta=None):
        field_adv = advance_flux(field,dt,dx,dy,dz,theta)
        field = advance_source_term(field_adv,dt)
        return field

    
    print('solver is initialized successfully!')
    
    return advance_one_step,rhs
        

    


