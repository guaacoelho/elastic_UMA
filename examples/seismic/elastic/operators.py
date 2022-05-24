from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag, solve
from examples.seismic import PointSource, Receiver


def ForwardOperator(model, geometry, space_order=4, save=False, **kwargs):
    """
    Construct method for the forward modelling operator in an elastic media.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    save : int or Buffer
        Saving flag, True saves all time steps, False saves three buffered
        indices (last three time steps). Defaults to False.
    """

    v = VectorTimeFunction(name='v', grid=model.grid,
                           save=geometry.nt if save else None,
                           space_order=space_order, time_order=1)
    tau = TensorTimeFunction(name='tau', grid=model.grid,
                             save=geometry.nt if save else None,
                             space_order=space_order, time_order=1)

    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    rec3 = Receiver(name='rec3', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)

    lam, mu, b, damp = model.lam, model.mu, model.b, model.damp

    s = model.grid.stepping_dim.spacing

    pde_v = v.dt - b * div(tau)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    pde_tau = tau.dt - lam * diag(div(v.forward)) - mu * \
        (grad(v.forward) + grad(v.forward).T)
    u_tau = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

    # The source injection term
    src_xx = src.inject(field=tau[0, 0].forward, expr=s*src)
    src_yy = src.inject(field=tau[-1, -1].forward, expr=s*src)
    src_expr = src_xx + src_yy
    if model.grid.dim == 3:
        src_expr += src.inject(field=tau[1, 1].forward, expr=s*src)

    # Create interpolation expression for receivers
    rec_term1 = rec1.interpolate(expr=tau[1, 1])
    rec_term2 = rec2.interpolate(expr=div(v))
    expr = tau[0, 0] + tau[-1, -1]
    if model.grid.dim == 3:
        expr += tau[1, 1]
    rec_term3 = rec3.interpolate(expr=expr)
    rec_expr = rec_term1 + rec_term2 + rec_term3

    op = Operator([u_v, u_tau] + src_expr + rec_expr, subs=model.spacing_map,
                  name="ForwardElastic", **kwargs)
    # Substitute spacing terms to reduce flops
    return op


def AdjointOperator(model, geometry, space_order=4, **kwargs):
    """
    Construct an adjoint modelling operator in a viscoacoustic medium.

    Parameters
    ----------
    model : Model
        Object containing the physical parameters.
    geometry : AcquisitionGeometry
        Geometry object that contains the source (SparseTimeFunction) and
        receivers (SparseTimeFunction) and their position.
    space_order : int, optional
        Space discretization order.
    """

    u = VectorTimeFunction(name='u', grid=model.grid, space_order=space_order,
                           time_order=1)

    sig = TensorTimeFunction(name='sig', grid=model.grid, space_order=space_order,
                             time_order=1)

    srca = PointSource(name='srca', grid=model.grid, time_range=geometry.time_axis,
                       npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)

    lam, mu, b, damp = model.lam, model.mu, model.b, model.damp

    lpmu = lam + 2. * mu

    rho = 1. / b

    s = model.grid.stepping_dim.spacing

    pde_ux = rho * u[0].dt.T - (lpmu * sig[0, 0]).dx - (lam * sig[1, 1]).dx - \
        (mu * sig[0, 1]).dy
    pde_uy = rho * u[1].dt.T - (lpmu * sig[1, 1]).dy - (lam * sig[0, 0]).dy - \
        (mu * sig[0, 1]).dx
    pde_sigxx = sig[0, 0].dt.T - u[0].backward.dx
    pde_sigyy = sig[1, 1].dt.T - u[1].backward.dy
    pde_sigxy = sig[0, 1].dt.T - u[0].backward.dy - u[1].backward.dx

    if model.grid.dim == 3:
        pde_uz = rho * u[2].dt.T - (lpmu * sig[2, 2]).dz - (lam * sig[0, 0]).dz - \
            (lam * sig[1, 1]).dz - (mu * sig[0, 2]).dx - (mu * sig[1, 2]).dy
        u_uz = Eq(u[2].backward, damp * solve(pde_uz, u[2].backward))

        pde_sigzz = sig[2, 2].dt.T - u[2].backward.dz
        u_sigzz = Eq(sig[2, 2].backward, damp * solve(pde_sigzz, sig[2, 2].backward))

        pde_ux += -(lam * sig[2, 2]).dx - (mu * sig[0, 2]).dz
        pde_uy += -(lam * sig[2, 2]).dy - (mu * sig[1, 2]).dz

        pde_sigxz = sig[0, 2].dt.T - u[0].backward.dz - u[2].backward.dx
        u_sigxz = Eq(sig[0, 2].backward, damp * solve(pde_sigxz, sig[0, 2].backward))

        pde_sigyz = sig[1, 2].dt.T - u[1].backward.dz - u[2].backward.dy
        u_sigyz = Eq(sig[1, 2].backward, damp * solve(pde_sigyz, sig[1, 2].backward))

    u_ux = Eq(u[0].backward, damp * solve(pde_ux, u[0].backward))
    u_uy = Eq(u[1].backward, damp * solve(pde_uy, u[1].backward))
    u_sigxx = Eq(sig[0, 0].backward, damp * solve(pde_sigxx, sig[0, 0].backward))
    u_sigyy = Eq(sig[1, 1].backward, damp * solve(pde_sigyy, sig[1, 1].backward))
    u_sigxy = Eq(sig[0, 1].backward, damp * solve(pde_sigxy, sig[0, 1].backward))

    stencil_kernel = [u_ux, u_uy, u_sigxx, u_sigyy, u_sigxy]

    rec_xx = rec.inject(field=sig[0, 0].backward, expr=rec*s)
    rec_yy = rec.inject(field=sig[-1, -1].backward, expr=rec*s)
    rec_expr = rec_xx + rec_yy
    src_expr = sig[0, 0] + sig[-1, -1]
    if model.grid.dim == 3:
        stencil_kernel += [u_uz, u_sigzz, u_sigxz, u_sigyz]
        rec_expr += rec.inject(field=sig[1, 1].backward, expr=rec*s)
        src_expr += sig[1, 1]
    source_a = srca.interpolate(expr=src_expr)

    # Substitute spacing terms to reduce flops
    return Operator(stencil_kernel + rec_expr + source_a,
                    subs=model.spacing_map, name='AdjointElastic', **kwargs)
