from jaxlib.xla_extension import DeviceArray
from jax.tree_util import tree_map
from neural_tangents._src.utils.kernel import Kernel
from neural_tangents._src.utils import utils
from neural_tangents._src.stax.requirements import layer, get_req, _has_req, _set_req, requires, _fuse_requirements, _DEFAULT_INPUT_REQ, _inputs_to_kernel, _set_shapes


def layer_extension(layer_fn):
  name = layer_fn.__name__

  @utils.wraps(layer_fn)
  def new_layer_fns(*args, **kwargs):
    kernel_fn = layer_fn(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn_extension(kernel_fn)
    kernel_fn.__name__ = name
    return kernel_fn

  return new_layer_fns


def _not_implemented(*args, **kwargs):
  raise NotImplementedError


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
      elif type(x1) is DeviceArray:
        x = x1
        which='ktt'
      x2 = tree_map(lambda x: None, x1)
    elif type(x1) is DeviceArray and type(x2) is tuple and len(x2)==2:
      x = x1
      x_i, x_b = x2
      x2 = np.concatenate((x_i, x_b))
      which='ktd'
    else:
      raise ValueError('invalid inputs for kernel_fn.')
    kernel = _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **reqs)
    out_kernel = kernel_fn(kernel, x=x, x_i=x_i, x_b=x_b, which=which, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  @utils.get_namedtuple('AnalyticKernel')
  def kernel_fn_any(x1_or_kernel, x2=None, get=None, *, pattern=None, mask_constant=None, diagonal_batch=None, diagonal_spatial=None, **kwargs):
    
    if utils.is_nt_tree_of(x1_or_kernel, Kernel):
      return kernel_fn_kernel(x1_or_kernel, pattern=pattern, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, **kwargs)

    return kernel_fn_x1(x1_or_kernel, x2, get, pattern=pattern, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, mask_constant=mask_constant, **kwargs)

  _set_req(kernel_fn_any, get_req(kernel_fn))
  return kernel_fn_any


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
  return new_kernel_fn


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

    def new_kernel_fn(k, _x1, _x2, **kwargs):
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