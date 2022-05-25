import jax.numpy as np
from jax import jit, grad, vmap
from jax.tree_util import tree_map

from neural_tangents._src.utils.kernel import Kernel
from neural_tangents._src.utils import utils
from neural_tangents._src.stax.requirements import layer, get_req, _has_req, _set_req, requires, _fuse_requirements, _DEFAULT_INPUT_REQ, _set_shapes, _cov, _cov_diag_batch


def _not_implemented(*args, **kwargs):
  raise NotImplementedError


def layer_extension(layer_fn):
  name = layer_fn.__name__

  @utils.wraps(layer_fn)
  def new_layer_fns(*args, **kwargs):
    _, _, kernel_fn = layer_fn(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn_extension(kernel_fn)
    kernel_fn.__name__ = name
    return kernel_fn

  return _not_implemented, _not_implemented, new_layer_fns


def _preprocess_kernel_fn_extension(kernel_fn):
  init_fn = apply_fn = _not_implemented

  # Set empty requirements if none specified.
  if not _has_req(kernel_fn):
    kernel_fn = requires()(kernel_fn)

  def kernel_fn_kernel(kernel, **kwargs):
    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  def kernel_fn_x1(x1, x2, get, **kwargs):
    # Get input requirements requested by network layers, user, or defaults.
    kernel_fn_reqs = get_req(kernel_fn)
    reqs = _fuse_requirements(kernel_fn_reqs, _DEFAULT_INPUT_REQ, **kwargs)
    compute_ntk = (get is None) or ('ntk' in get)

    x = x_i = x_b = which = None
    if x2 is None:
      if type(x1) is tuple and len(x1)==2:
        x_i, x_b = x1
        x_i, x_b = x_i.astype(np.float64), x_b.astype(np.float64)
        x1 = np.concatenate((x_i, x_b))
        which='kdd'
      elif isinstance(x1, np.ndarray):
        x = x1
        which='ktt'
      x2 = tree_map(lambda x: None, x1)
    elif isinstance(x1, np.ndarray) and type(x2) is tuple and len(x2)==2:
      x = x1
      x_i, x_b = x2
      x2 = np.concatenate((x_i, x_b))
      which='ktd'
    else:
      raise ValueError('invalid inputs for kernel_fn.')
    kernel = _inputs_to_kernel_extension(x1, x2, compute_ntk=compute_ntk, **reqs)
    out_kernel = kernel_fn(kernel, x=x, x_i=x_i, x_b=x_b, which=which, **kwargs)
    print(out_kernel)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  @utils.get_namedtuple('AnalyticKernel')
  def kernel_fn_any(x1_or_kernel, x2=None, get=None, *, pattern=None, mask_constant=None, diagonal_batch=None, diagonal_spatial=None, **kwargs):
    
    if utils.is_nt_tree_of(x1_or_kernel, Kernel):
      return kernel_fn_kernel(x1_or_kernel, pattern=pattern, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, **kwargs)

    return kernel_fn_x1(x1_or_kernel, x2, get, pattern=pattern, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, mask_constant=mask_constant, **kwargs)

  _set_req(kernel_fn_any, get_req(kernel_fn))
  return kernel_fn_any

@utils.nt_tree_fn(2)
def _inputs_to_kernel_extension(x1, x2, *, diagonal_batch, diagonal_spatial, compute_ntk, batch_axis, channel_axis, mask_constant, eps=1e-12, method=None, c2=None, **kwargs):
  if not (isinstance(x1, np.ndarray) and (x2 is None or isinstance(x2, np.ndarray))):
    raise TypeError(f'Wrong input types given. Found `x1` of type {type(x1)} and `x2` of type {type(x2)}, need both to be `np.ndarray`s (`x2` can be `None`).')

  batch_axis %= x1.ndim
  diagonal_spatial = bool(diagonal_spatial)

  assert batch_axis == 0

  if channel_axis is None:
    def flatten(x):
      if x is None:
        return x
      return np.moveaxis(x, batch_axis, 0).reshape((x.shape[batch_axis], -1))

    x1, x2 = flatten(x1), flatten(x2)
    batch_axis, channel_axis = 0, 1
    diagonal_spatial = False

  else:
    channel_axis %= x1.ndim

  def get_x_cov_mask(x):
    if x is None:
      return None, None, None

    if x.ndim < 2:
      raise ValueError(f'Inputs must be at least 2D (a batch dimension and a channel/feature dimension), got {x.ndim}.')

    x = utils.get_masked_array(x, mask_constant)
    x, mask = x.masked_value, x.mask

    if diagonal_batch:
      cov = _cov_diag_batch(x, diagonal_spatial, batch_axis, channel_axis)
    else:
      cov = _cov(x, x, diagonal_spatial, batch_axis, channel_axis)

    return x, cov, mask

  x1, cov1, mask1 = get_x_cov_mask(x1)
  x2, cov2, mask2 = get_x_cov_mask(x2)
  print('method', method, kwargs)
  if method is not None:
    if method=='fem':
      assert x1.shape[channel_axis]==1
      nngp = _fem(x1, x2)
    elif c2 is None:
      nngp = _rbf(x1, x2, channel_axis, method)
    else:
      nngp = _rbf(x1, x2, channel_axis, method, c2=c2)
    ntk = None
  else:
    nngp = _cov(x1, x2, diagonal_spatial, batch_axis, channel_axis)
    ntk = np.zeros((), nngp.dtype) if compute_ntk else None
  is_gaussian = False
  is_reversed = False
  x1_is_x2 = utils.x1_is_x2(x1, x2, eps=eps)
  is_input = False

  return Kernel(cov1=cov1, cov2=cov2, nngp=nngp, ntk=ntk, x1_is_x2=x1_is_x2, is_gaussian=is_gaussian, is_reversed=is_reversed, is_input=is_input, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, shape1=x1.shape, shape2=x1.shape if x2 is None else x2.shape, batch_axis=batch_axis, channel_axis=channel_axis, mask1=mask1, mask2=mask2)


def _rbf(x1, x2, channel_axis, method, c2=1):
  assert channel_axis==1 or len(x1.shape)+channel_axis==1
  x2 = x1 if x2 is None else x2
  r2 = np.sum((x1[:,None]-x2[None])**2, axis=2)
  if method=='gaussian':
    return np.exp(-r2/c2)
  elif method=='imq':
    return 1/np.sqrt(1+r2/c2)
  else:
    raise NotImplementedError


def _fem(x1, x2):
  if x2 is None: # k_dd
    h = np.diff(np.sort(x1, axis=None))
    L = np.diag(1/h[:-1]+1/h[1:], 0)-np.diag(1/h[1:-1], -1)-np.diag(1/h[1:-1], 1)
    return L
  else: # k_td
    h = np.diff(np.sort(x2, axis=None))
    d = x1-x2.reshape(1, -1)
    v1 = d[:,:-1]*(d[:,:-1]>0)*(d[:,:-1]<=h)/h
    v1 = np.concatenate((np.zeros((v1.shape[0], 1)), v1), axis=1)
    v2 = -d[:,1:]*(d[:,1:]<0)*(d[:,1:]>-h)/h
    v2 = np.concatenate((v2, np.zeros((v2.shape[0], 1))), axis=1)
    v3 = -d[:,1:]*(d[:,1:]<0)*(d[:,1:]==-h)*(x1==0)/h
    v3 = np.concatenate((v3, np.zeros((v3.shape[0], 1))), axis=1)
    return (v1+v2+v3)[:,1:-1]


@layer_extension
def Solve(kdd, ktd, ktt):
  def new_kernel_fn(k, x=None, x_i=None, x_b=None, which=None, **kwargs):
    ntk = k.ntk
    if ntk is None:
      get = 'nngp'
    else:
      get = None
    if which=='kdd':
      x1 = np.concatenate((x_i, x_b))
      x2 = None
      nngp = kdd(x1, x2, x_i=x_i, x_b=x_b, get=get)
    elif which=='ktd':
      x1 = x
      x2 = np.concatenate((x_i, x_b))
      nngp = ktd(x1, x2, x=x, x_i=x_i, x_b=x_b, get=get)
    elif which=='ktt':
      x1 = x
      x2 = None
      nngp = ktt(x1, x2, x=x, get=get)
    if ntk is not None:
      nngp, ntk = nngp.nngp, nngp.ntk
    return k.replace(nngp=nngp, ntk=ntk, is_gaussian=True, is_input=False)
  return _not_implemented, _not_implemented, new_kernel_fn


@layer
def Hcombine(layer_fn_0, layer_fn_1):
    _, _, kernel_fn_0 = layer_fn_0
    _, _, kernel_fn_1 = layer_fn_1

    def new_kernel_fn(k, x=None, x_i=None, x_b=None, **kwargs):
        ntk = k.ntk
        if ntk is None:
            get = 'nngp'
        else:
            get = None

        if x is None:
            x = np.concatenate((x_i, x_b))
            nngp_0 = kernel_fn_0(x, x_i, x=x_i, x_i=x_i, x_b=x_b, get=get)
            nngp_1 = kernel_fn_1(x, x_b, x=x_b, x_i=x_i, x_b=x_b, get=get)
            if ntk is not None:
                nngp_0, ntk_0 = nngp_0.nngp, nngp_0.ntk
                nngp_1, ntk_1 = nngp_1.nngp, nngp_1.ntk
                ntk = np.concatenate((ntk_0, ntk_1), axis=1)
            nngp = np.concatenate((nngp_0, nngp_1), axis=1)                
        else:
            nngp_0 = kernel_fn_0(x, x_i, _x1=x, _x2=x_i, get=get)
            nngp_1 = kernel_fn_1(x, x_b, _x1=x, _x2=x_b, get=get)
            if ntk is not None:
                nngp_0, ntk_0 = nngp_0.nngp, nngp_0.ntk
                nngp_1, ntk_1 = nngp_1.nngp, nngp_1.ntk
                ntk = np.concatenate((ntk_0, ntk_1), axis=1)
            nngp = np.concatenate((nngp_0, nngp_1), axis=1)  
        return k.replace(nngp=nngp, ntk=ntk, is_gaussian=True, is_input=False)
    return _not_implemented, _not_implemented, new_kernel_fn


@layer
def Vcombine(layer_fn_0, layer_fn_1):
    _, _, kernel_fn_0 = layer_fn_0
    _, _, kernel_fn_1 = layer_fn_1

    def new_kernel_fn(k, x=None, x_i=None, x_b=None, **kwargs):
        ntk = k.ntk
        if ntk is None:
            get = 'nngp'
        else:
            get = None

        if x is None:
            x = np.concatenate((x_i, x_b))
            nngp_0 = kernel_fn_0(x_i, x, x=x_i, x_i=x_i, x_b=x_b, get=get)
            nngp_1 = kernel_fn_1(x_b, x, x=x_b, x_i=x_i, x_b=x_b, get=get)
            if ntk is not None:
                nngp_0, ntk_0 = nngp_0.nngp, nngp_0.ntk
                nngp_1, ntk_1 = nngp_1.nngp, nngp_1.ntk
                ntk = np.concatenate((ntk_0, ntk_1), axis=0)
            nngp = np.concatenate((nngp_0, nngp_1), axis=0)                
        else:
            nngp_0 = kernel_fn_0(x_i, x, _x1=x_i, _x2=x, get=get)
            nngp_1 = kernel_fn_1(x_b, x, _x1=x_b, _x2=x, get=get)
            if ntk is not None:
                nngp_0, ntk_0 = nngp_0.nngp, nngp_0.ntk
                nngp_1, ntk_1 = nngp_1.nngp, nngp_1.ntk
                ntk = np.concatenate((ntk_0, ntk_1), axis=0)
            nngp = np.concatenate((nngp_0, nngp_1), axis=0)  
        return k.replace(nngp=nngp, ntk=ntk, is_gaussian=True, is_input=False)
    return _not_implemented, _not_implemented, new_kernel_fn


@layer
def Deriv(serial, order_wrt_x1, order_wrt_x2):
    """Layer constructor function for calculating the mixed derivative of the kernel function of a model."""
    _, _, kernel_fn = serial

    def new_kernel_fn(k, *, _x1, _x2, **kwargs):
        """Compute the transformed kernels after a `Deriv` layer."""
        # `x1` and `x2` are used to calculate the output kernel instead of `k`
        x1 = _x1
        x2 = _x1 if _x2 is None else _x2
        x1, x2 = x1.reshape(-1,1,1), x2.reshape(-1,1,1)
        def deriv(x1, x2, order_wrt_x1, order_wrt_x2, get):
            if order_wrt_x1==0:
                if order_wrt_x2==0:
                    return kernel_fn(x1, x2, get).squeeze()
                return grad(deriv, argnums=1)(x1, x2, order_wrt_x1, order_wrt_x2-1, get).squeeze()
            return grad(deriv, argnums=0)(x1, x2, order_wrt_x1-1, order_wrt_x2, get).squeeze()
        nngp = vmap(vmap(deriv, in_axes=(None, 0, None, None, None)), in_axes=(0, None, None, None, None))(x1, x2, order_wrt_x1, order_wrt_x2, 'nngp')
        ntk = k.ntk
        if ntk is not None:
            ntk = vmap(vmap(deriv, in_axes=(None, 0, None, None, None)), in_axes=(0, None, None, None, None))(x1, x2, order_wrt_x1, order_wrt_x2, 'ntk')
        return k.replace(nngp=nngp, ntk=ntk, is_gaussian=True, is_input=False)
    return _not_implemented, _not_implemented, new_kernel_fn


def Poisson(model, d_eq, d_sl):
    equation = Deriv(model, d_eq, d_eq)
    sl_eq = Deriv(model, d_sl, d_eq)
    solution = Deriv(model, d_sl, d_sl)
    eq_sl = Deriv(model, d_eq, d_sl)
    _, _, kernel_dd = Vcombine(Hcombine(equation, eq_sl), Hcombine(sl_eq, solution))
    _, _, kernel_td = Hcombine(sl_eq, solution)
    _, _, kernel_tt = solution
    return Solve(kernel_dd, kernel_td, kernel_tt)