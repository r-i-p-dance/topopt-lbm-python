import numpy as np
from lbm.src.core.lattice import BaseLattice
from topopt.src.core.objective import dJ_df
from lbm.src.core.kernels import stream_kernel
    

class AdjointLattice(BaseLattice):
    """Solves the discrete adjoint of BrinkmanForced (or any forward lattice).

    Holds a reference to the forward solver — reads its ux, uy, omega_eff,
    f_eq — and computes the adjoint distribution f_hat that satisfies
    the adjoint equation.
    """

    def __init__(self, forward, **kwargs):
        super().__init__(nx=forward.nx, ny=forward.ny,
                     tau_lbm=forward.tau_lbm, **kwargs)
        self.fwd = forward
        self.obstacle = forward.obstacle 
        self.periodic_x = forward.periodic_x
        self.opposite = forward.opposite
        self.f = np.zeros_like(forward.f)
        self.f_new = np.zeros_like(forward.f)
        self.g = np.zeros_like(forward.f)
        self.it = 0
        self.source = dJ_df(
            forward.rho_bar, forward.ux, forward.uy, forward.rho,
            forward.cx, forward.cy,
            forward.alpha_max, forward.q
        )
        self.neg_cx = -forward.cx
        self.neg_cy = -forward.cy

    def adjoint_equilibrium(self):
        """f_adj_eq = A + B_x*(c_ix - u_x) + B_y*(c_iy - u_y)
    
        where the moments carry the derivative structure of the forward equilibrium:
            A   = sum_j w_j * E_j    * f_hat_j
            B_x = sum_j w_j * D_jx  * f_hat_j
            B_y = sum_j w_j * D_jy  * f_hat_j
        with
            E_j  = 1 + 3(c_j·u) + 9/2*(c_j·u)^2 - 3/2*|u|^2
            D_jx = 3*c_jx + 9*(c_j·u)*c_jx - 3*u_x
            D_jy = 3*c_jy + 9*(c_j·u)*c_jy - 3*u_y
        """
        f = self.fwd
        w, cx, cy = f.w, f.cx, f.cy
        ux, uy = f.ux, f.uy

        # Compute c_j · u for each direction j at every cell
        cu = cx[:, None, None] * ux[None, :, :] + cy[:, None, None] * uy[None, :, :]
        usq = ux**2 + uy**2

        # E_j and D_j at every cell (shape: 9, nx, ny)
        E = 1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * usq[None, :, :]
        D_x = 3.0 * cx[:, None, None] + 9.0 * cu * cx[:, None, None] - 3.0 * ux[None, :, :]
        D_y = 3.0 * cy[:, None, None] + 9.0 * cu * cy[:, None, None] - 3.0 * uy[None, :, :]

        # Moments A, B_x, B_y — contract f_hat against these weightings
        A   = np.sum(w[:, None, None] * E   * self.f, axis=0)
        B_x = np.sum(w[:, None, None] * D_x * self.f, axis=0)
        B_y = np.sum(w[:, None, None] * D_y * self.f, axis=0)

        # Build f_adj_eq: A + B_x*(c_ix - u_x) + B_y*(c_iy - u_y)
        f_adj_eq = np.zeros_like(self.f)
        for i in range(9):
            f_adj_eq[i] = A + B_x * (cx[i] - ux) + B_y * (cy[i] - uy)

        return f_adj_eq

    def collision(self):
        self.g = self.f.copy()        # state before C^T, for the sensitivity
        adjoint_collide_kernel(self.f, self.fwd.ux, self.fwd.uy,
                               self.fwd.omega_eff, self.source,
                               self.w, self.cx, self.cy, self.obstacle)

    def stream(self):
        stream_kernel(self.f, self.f_new, self.neg_cx, self.neg_cy, self.fwd.periodic_x)
        self.f, self.f_new = self.f_new, self.f

    def apply_boundary_conditions(self):
        pass

    def step(self):
        self.apply_boundary_conditions()
        self.stream()                       
        self.bounce_back_obstacle()
        self.collision()                   
        