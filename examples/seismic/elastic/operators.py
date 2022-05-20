from devito import Eq, Operator, VectorTimeFunction, TensorTimeFunction
from devito import div, grad, diag, solve
from examples.seismic import PointSource, Receiver


def src_rec(v, tau, model, geometry, **kwargs):
    """
    Source injection and receiver interpolation
    """
    s = model.grid.time_dim.spacing
    # Source symbol with input wavelet
    src = PointSource(name='src', grid=model.grid, time_range=geometry.time_axis,
                      npoint=geometry.nsrc)
    rec = Receiver(name='rec', grid=model.grid, time_range=geometry.time_axis,
                   npoint=geometry.nrec)
    rec1 = Receiver(name='rec1', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)
    rec2 = Receiver(name='rec2', grid=model.grid, time_range=geometry.time_axis,
                    npoint=geometry.nrec)

    forward = kwargs.get('forward', True)

    if forward:
        # The source injection term
        src_xx = src.inject(field=tau[0, 0].forward, expr=src * s)
        src_zz = src.inject(field=tau[-1, -1].forward, expr=src * s)
        src_expr = src_xx + src_zz
        if model.grid.dim == 3:
            src_yy = src.inject(field=tau[1, 1].forward, expr=src * s)
            src_expr += src_yy

        # Create interpolation expression for receivers
        rec_term1 = rec1.interpolate(expr=tau[-1, -1])
        rec_term2 = rec2.interpolate(expr=div(v))
        rec_term3 = rec.interpolate(expr=tau[0, 0] + tau[1, 1])
        if model.grid.dim == 3:
            rec_term3 += rec.interpolate(expr=tau[2, 2])
        rec_expr = rec_term1 + rec_term2 + rec_term3

    else:
        # Construct expression to inject receiver values
        rec_term1 = rec.inject(field=tau[0, 0].backward, expr=rec*s)
        rec_term2 = rec.inject(field=tau[1, 1].backward, expr=rec*s)
        rec_expr = rec_term1 + rec_term2
        if model.grid.dim == 3:
            rec_term3 = rec.inject(field=tau[2, 2].backward, expr=rec*s)
            rec_expr += rec_term3

        # Create interpolation expression for the adjoint-source
        src_expr = src.interpolate(expr=tau[0, 0] + tau[1, 1])
        if model.grid.dim == 3:
            src_expr += src.interpolate(expr=tau[2, 2])

    return src_expr + rec_expr


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

    lam, mu, b, damp = model.lam, model.mu, model.b, model.damp

    pde_v = v.dt - b * div(tau)
    u_v = Eq(v.forward, damp * solve(pde_v, v.forward))

    pde_tau = tau.dt - lam * diag(div(v.forward)) - mu * \
        (grad(v.forward) + grad(v.forward).T)
    u_tau = Eq(tau.forward, damp * solve(pde_tau, tau.forward))

    srcrec = src_rec(v, tau, model, geometry)
    op = Operator([u_v, u_tau] + srcrec, subs=model.spacing_map, name="ForwardElastic",
                  **kwargs)
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

    lam, mu, b, damp = model.lam, model.mu, model.b, model.damp

    lpmu = lam + 2. * mu

    rho = 1. / b

    pde_ux = -rho * u[0].dt.T - (lpmu * sig[0, 0]).dx - (lam * sig[1, 1]).dx - \
        (mu * sig[0, 1]).dy
    u_ux = Eq(u[0].backward, damp * solve(pde_ux, u[0].backward))

    pde_uy = -rho * u[1].dt.T - (lpmu * sig[1, 1]).dy - (lam * sig[0, 0]).dy - \
        (mu * sig[0, 1]).dx
    u_uy = Eq(u[1].backward, damp * solve(pde_uy, u[1].backward))

    pde_sigxx = -sig[0, 0].dt.T - u[0].backward.dx
    u_sigxx = Eq(sig[0, 0].backward, damp * solve(pde_sigxx, sig[0, 0].backward))

    pde_sigyy = -sig[1, 1].dt.T - u[1].backward.dy
    u_sigyy = Eq(sig[1, 1].backward, damp * solve(pde_sigyy, sig[1, 1].backward))

    pde_sigxy = -sig[0, 1].dt.T - u[0].backward.dy - u[1].backward.dx
    u_sigxy = Eq(sig[0, 1].backward, damp * solve(pde_sigxy, sig[0, 1].backward))

    srcrec = src_rec(u, sig, model, geometry, forward=False)

    # Substitute spacing terms to reduce flops
    return Operator([u_ux, u_uy, u_sigxx, u_sigyy, u_sigxy] + srcrec,
                    subs=model.spacing_map, name='AdjointElastic', **kwargs)
