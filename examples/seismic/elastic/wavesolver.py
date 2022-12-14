from devito.tools import memoized_meth
from devito import Function, VectorTimeFunction, TensorTimeFunction
from examples.seismic import PointSource

from examples.seismic.elastic.operators import (
    ForwardOperator, AdjointOperator, GradientOperator, BornOperator
)


class ElasticWaveSolver(object):
    """
    Solver object that provides operators for seismic inversion problems
    and encapsulates the time and space discretization for a given problem
    setup.

    Parameters
    ----------
    model : Model
        Physical model with domain parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Order of the spatial stencil discretisation. Defaults to 4.
    """
    def __init__(self, model, geometry, space_order=4, **kwargs):
        self.model = model
        self.model._initialize_bcs(bcs="mask")
        self.geometry = geometry

        self.space_order = space_order
        # Cache compiler options
        self._kwargs = kwargs

    @property
    def dt(self):
        return self.model.critical_dt

    @memoized_meth
    def op_fwd(self, save=None):
        """Cached operator for forward runs with buffered wavefield"""
        return ForwardOperator(self.model, save=save, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_adj(self):
        """Cached operator for adjoint runs"""
        return AdjointOperator(self.model, save=None, geometry=self.geometry,
                               space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_grad(self, save=True):
        """Cached operator for gradient runs"""
        return GradientOperator(self.model, save=save, geometry=self.geometry,
                                space_order=self.space_order, **self._kwargs)

    @memoized_meth
    def op_born(self):
        """Cached operator for born runs"""
        return BornOperator(self.model, save=None, geometry=self.geometry,
                            space_order=self.space_order, **self._kwargs)

    def forward(self, src=None, rec_tau=None, rec_vx=None, rec_vz=None, rec_vy=None,
                v=None, tau=None, model=None, save=None, **kwargs):
        """
        Forward modelling function that creates the necessary
        data objects for running a forward modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec1 : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the pressure (tzz).
        rec2 : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the particle velocities.
        rec3 : SparseTimeFunction or array_like, optional
            The interpolated receiver data of the particle velocities.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        tau : TensorTimeFunction, optional
            The computed symmetric stress tensor.
        model : Model, optional
            Object containing the physical parameters.
        lam : Function, optional
            The time-constant first Lame parameter `rho * (vp**2 - 2 * vs **2)`.
        mu : Function, optional
            The Shear modulus `(rho * vs*2)`.
        b : Function, optional
            The time-constant inverse density (b=1 for water).
        save : bool, optional
            Whether or not to save the entire (unrolled) wavefield.

        Returns
        -------
        Rec1(tzz), Rec2(div(v)), particle velocities v, stress tensor tau and
        performance summary.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec_vx = rec_vx or self.geometry.new_rec(name='rec_vx')
        rec_vz = rec_vz or self.geometry.new_rec(name='rec_vz')
        if self.model.grid.dim == 3:
            rec_vy = rec_vy or self.geometry.new_rec(name='rec_vy')
            kwargs.update({'rec_vy': rec_vy})
        rec_tau = rec_tau or self.geometry.new_rec(name='rec_tau')

        # Create all the fields vx, vz, tau_xx, tau_zz, tau_xz
        save_t = src.nt if save else None
        v = v or VectorTimeFunction(name='v', grid=self.model.grid, save=save_t,
                                    space_order=self.space_order, time_order=1)
        tau = tau or TensorTimeFunction(name='tau', grid=self.model.grid, save=save_t,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in tau})

        model = model or self.model
        # Pick Lame parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_fwd(save).apply(src=src, rec_tau=rec_tau, rec_vx=rec_vx,
                                          rec_vz=rec_vz, dt=kwargs.pop('dt', self.dt),
                                          **kwargs)
        if self.model.grid.dim == 2:
            return rec_tau, rec_vx, rec_vz, v, tau, summary
        return rec_tau, rec_vx, rec_vz, rec_vy, v, tau, summary

    def adjoint(self, rec, srca=None, u=None, sig=None, model=None, **kwargs):
        """
        Adjoint modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        rec : SparseTimeFunction or array-like
            The receiver data. Please note that
            these act as the source term in the adjoint run.
        srca : SparseTimeFunction or array-like
            The resulting data for the interpolated at the
            original source location.
        u : VectorTimeFunction, optional
            The computed particle velocity.
        sig : TensorTimeFunction, optional
            The computed symmetric stress tensor.
        model : Model, optional
            Object containing the physical parameters.
        lam : Function, optional
            The time-constant first Lame parameter `rho * (vp**2 - 2 * vs **2)`.
        mu : Function, optional
            The Shear modulus `(rho * vs*2)`.
        b : Function, optional
            The time-constant inverse density (b=1 for water).

        Returns
        -------
        Adjoint source, wavefield and performance summary.
        """
        # Create a new adjoint source and receiver symbol
        srca = srca or PointSource(name='srca', grid=self.model.grid,
                                   time_range=self.geometry.time_axis,
                                   coordinates=self.geometry.src_positions)

        u = u or VectorTimeFunction(name="u", grid=self.model.grid,
                                    time_order=1, space_order=self.space_order)
        sig = sig or TensorTimeFunction(name='sig', grid=self.model.grid,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in u})
        kwargs.update({k.name: k for k in sig})
        kwargs['time_m'] = 0

        model = model or self.model
        # Pick vp and physical parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_adj().apply(src=srca, rec=rec, dt=kwargs.pop('dt', self.dt),
                                      **kwargs)
        return srca, u, sig, summary

    def jacobian_adjoint(self, rec_vx, rec_vz, v, u=None, sig=None, rec_vy=None,
                         grad_lam=None, grad_mu=None, grad_rho=None, model=None,
                         checkpointing=False, **kwargs):
        """
        Gradient modelling function for computing the adjoint of the
        Linearized Born modelling function, ie. the action of the
        Jacobian adjoint on an input data.

        Parameters
        ----------
        rec : SparseTimeFunction
            Receiver data.
        p : TimeFunction
            Full wavefield `p` (created with save=True).
        pa : TimeFunction, optional
            Stores the computed wavefield.
        grad : Function, optional
            Stores the gradient field.
        r : TimeFunction, optional
            The computed attenuation memory variable.
        va : VectorTimeFunction, optional
            The computed particle velocity.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        qp : Function, optional
            The P-wave quality factor.
        b : Function, optional
            The time-constant inverse density.

        Returns
        -------
        Gradient field and performance summary.
        """
        # Gradient symbol
        grad_lam = grad_lam or Function(name='grad_lam', grid=self.model.grid)
        grad_mu = grad_mu or Function(name='grad_mu', grid=self.model.grid)
        grad_rho = grad_rho or Function(name='grad_rho', grid=self.model.grid)

        u = u or VectorTimeFunction(name="u", grid=self.model.grid,
                                    time_order=1, space_order=self.space_order)
        sig = sig or TensorTimeFunction(name='sig', grid=self.model.grid,
                                        space_order=self.space_order, time_order=1)
        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in u})
        kwargs.update({k.name: k for k in sig})
        if self.model.grid.dim == 3:
            kwargs.update({'rec_vy': rec_vy})
        kwargs['time_m'] = 0

        model = model or self.model
        # Pick vp and physical parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        summary = self.op_grad().apply(rec_vx=rec_vx, rec_vz=rec_vz, grad_lam=grad_lam,
                                       grad_mu=grad_mu, grad_rho=grad_rho,
                                       dt=kwargs.pop('dt', self.dt), **kwargs)

        return grad_lam, grad_mu, grad_rho, summary

    def jacobian(self, dlam=None, drho=None, dmu=None, src=None, rec_tau=None,
                 rec_vx=None, rec_vz=None, rec_vy=None, v=None, dv=None, tau=None,
                 dtau=None, model=None, **kwargs):
        """
        Linearized Born modelling function that creates the necessary
        data objects for running an adjoint modelling operator.

        Parameters
        ----------
        src : SparseTimeFunction or array_like, optional
            Time series data for the injected source term.
        rec : SparseTimeFunction or array_like, optional
            The interpolated receiver data.
        p : TimeFunction, optional
            The forward wavefield.
        P : TimeFunction, optional
            The linearized wavefield.
        rp : TimeFunction, optional
            The computed attenuation memory variable.
        rP : TimeFunction, optional
            The computed attenuation memory variable.
        v : VectorTimeFunction, optional
            The computed particle velocity.
        dv : VectorTimeFunction, optional
            The computed particle velocity.
        model : Model, optional
            Object containing the physical parameters.
        vp : Function or float, optional
            The time-constant velocity.
        qp : Function, optional
            The P-wave quality factor.
        b : Function, optional
            The time-constant inverse density.
        """
        # Source term is read-only, so re-use the default
        src = src or self.geometry.src
        # Create a new receiver object to store the result
        rec_vx = rec_vx or self.geometry.new_rec(name='rec_vx')
        rec_vz = rec_vz or self.geometry.new_rec(name='rec_vz')
        if self.model.grid.dim == 3:
            rec_vy = rec_vy or self.geometry.new_rec(name='rec_vy')
            kwargs.update({'rec_vy': rec_vy})
        rec_tau = rec_tau or self.geometry.new_rec(name='rec_tau')

        dlam = dlam or Function(name='dlam', grid=self.model.grid, space_order=0)
        drho = drho or Function(name='drho', grid=self.model.grid, space_order=0)
        dmu = dmu or Function(name='dmu', grid=self.model.grid, space_order=0)

        # Create the forward wavefields u and U if not provided
        v = v or VectorTimeFunction(name='v', grid=self.model.grid,
                                    space_order=self.space_order, time_order=1)
        tau = tau or TensorTimeFunction(name='tau', grid=self.model.grid,
                                        space_order=self.space_order, time_order=1)
        dv = dv or VectorTimeFunction(name='dv', grid=self.model.grid,
                                      space_order=self.space_order, time_order=1)
        dtau = dtau or TensorTimeFunction(name='dtau', grid=self.model.grid,
                                          space_order=self.space_order, time_order=1)

        kwargs.update({k.name: k for k in v})
        kwargs.update({k.name: k for k in tau})
        kwargs.update({k.name: k for k in dv})
        kwargs.update({k.name: k for k in dtau})

        model = model or self.model
        # Pick vp and physical parameters from model unless explicitly provided
        kwargs.update(model.physical_params(**kwargs))

        # Execute operator and return wavefield and receiver data
        summary = self.op_born().apply(drho=drho, dlam=dlam, dmu=dmu, src=src,
                                       rec_tau=rec_tau, rec_vx=rec_vx, rec_vz=rec_vz,
                                       dt=kwargs.pop('dt', self.dt), **kwargs)

        if self.model.grid.dim == 2:
            return rec_tau, rec_vx, rec_vz, v, tau, dv, dtau, summary
        return rec_tau, rec_vx, rec_vz, rec_vy, v, tau, dv, dtau, summary
