import jax.numpy as np
from jax import jit, grad, vmap
from jax.tree_util import tree_map

from neural_tangents._src.utils.kernel import Kernel
from neural_tangents._src.utils import utils
from neural_tangents._src.stax.requirements import layer, get_req, _has_req, _set_req, requires, _fuse_requirements, _DEFAULT_INPUT_REQ, _inputs_to_kernel, _set_shapes, _cov, _cov_diag_batch


### Examples ###
from neural_tangents import stax, predict


# Example codes for reproducing experiment results in our paper


class Infinite_time_inference:
  """
  Attributes:
    rmse:
      Contains the root-mean-square error of the approximated solution for each call to the `run` method.
    cd_a:
      Contains the Amotized condition number of the kernel matrix for each call to the `run` method. Must set compute_cond=True when calling `run`.
    cd_e:
      Contains the Effective condition number of the kernel matrix for each call to the `run` method. Must set compute_cond=True when calling `run`.
  """
  def __init__(self, model, equation, **model_kwargs):
    """
    Args:
      model:
        The underlying model to generate the approximated solution. One of the `nn`, `gaussian`, `imq`, `fem` defined in this module or a custom method.
      equation:
        An object that describes the differential equation and its closed-form solution. For the Poisson's equation, must implement the `source` and `solution` methods.
    """
    self.rmse = []
    self.cd_a = []
    self.cd_e = []
    self.model = model
    self.equation = equation
    if model_kwargs or model is fem:
      self._kernels(**model_kwargs)
  def _kernels(self, **model_kwargs):
    self.kernel = self.model(**model_kwargs)
  def run(self, test_x, n_sample, diag_reg=0., compute_cond=False, precision=np.float32, **model_kwargs):
    """
    Args:
      test_x:
        Array of test inputs.
      n_sample:
        Specify the number of sample nodes used for training. The `node density` in paper equals n_sample-1 in this case.
      diag_reg:
        Parameter for the Tikhonov regularization.
      compute_cond:
        Set to `True` in order to acquire the Effective condition number and the Amotized condition number from `self.cd_e` and `self.cd_a` respectively.
      precision:
        Precision used during inference. Can be `np.float32` or `np.float64`.
    """
    source = self.equation.source
    solution = self.equation.solution
    test_y = solution(test_x)
    interior = np.linspace(0, 1, n_sample)[1:-1].reshape(-1,1).astype(precision)
    left_bound = np.zeros((1, 1)).astype(precision)
    right_bound = np.ones((1, 1)).astype(precision)
    if self.model is fem:
      train_x = np.concatenate((left_bound, interior, right_bound))
      h = np.diff(np.sort(train_x, axis=None))
      M = np.diag(h[:-1]/3+h[1:]/3, 0)+np.diag(h[1:-1]/6, -1)+np.diag(h[1:-1]/6, 1)
      train_y = M@source(interior)
    else:
      train_x = (interior, np.concatenate((left_bound, right_bound)))
      train_y = np.concatenate((source(interior), solution(left_bound), solution(right_bound)))

    if model_kwargs:
      self._kernels(**model_kwargs)
    cond = [-1, -1] if compute_cond else None
    predict_fn = predict.gradient_descent_mse_ensemble(self.kernel, train_x, train_y, diag_reg=diag_reg, cond=cond, _b=train_y)        
    sl = predict_fn(x_test=test_x, get='ntk')
    mean_out = sl.reshape(-1)
    if self.model is fem:
      mean_out = -mean_out+solution(left_bound)+(solution(right_bound)-solution(left_bound))*test_x.reshape(-1)
    self.rmse.append(np.sqrt(np.square(mean_out-test_y.reshape(-1)).mean()).item())
    if compute_cond:
      self.cd_a.append(cond[0])
      self.cd_e.append(cond[1])


class Finite_time_inference:
  """
  Attributes:
    rmse:
      Contains the root-mean-square error of the approximated solution for each time stamp in the input array `t` for the `run` method.
    loss:
      Contains the training loss for each time stamp in `t`.
    one_sigma:
      Contains the 1-Sigma deviation of the rmse for each time stamp in `t`. Must set compute_cov=True when calling `run`.
  """
  def __init__(self, model, equation, **model_kwargs):
    """
    Args:
      model:
        The underlying model to generate the approximated solution. One of the `nn`, `gaussian`, `imq`, `fem` defined in this module or a custom method.
      equation:
        An object that describes the differential equation and its closed-form solution. For the Poisson's equation, must implement the `source` and `solution` methods.
    """
    self.model = model
    self.equation = equation
    if model_kwargs:
      self._kernels(**model_kwargs)
  def _kernels(self, **model_kwargs):
    self.kernel = self.model(**model_kwargs)
  def run(self, test_x, n_sample, t=None, diag_reg=0., compute_cov=False, precision=np.float32, **model_kwargs):
    """
    Args:
      test_x:
        Array of test inputs.
      n_sample:
        Specify the number of sample nodes used for training. The `node density` in paper equals n_sample-1 in this case.
      t:
        Array of effective training time. Equals to `# training iteration` * `learning rate`.
      diag_reg:
        Parameter for the Tikhonov regularization.
      compute_cov:
        Set to `True` in order to acquire the uncertainty of the rmse from `self.one_sigma`.
      precision:
        Precision used during inference. Can be `np.float32` or `np.float64`.
    """
    source = self.equation.source
    solution = self.equation.solution
    test_y = solution(test_x)
    interior = np.linspace(0, 1, n_sample)[1:-1].reshape(-1,1).astype(precision)
    left_bound = np.zeros((1, 1)).astype(precision)
    right_bound = np.ones((1, 1)).astype(precision)
    train_x = (interior, np.concatenate((left_bound, right_bound)))
    train_y = np.concatenate((source(interior), solution(left_bound), solution(right_bound)))
    if model_kwargs:
      self._kernels(**model_kwargs)
    predict_fn = predict.gradient_descent_mse_ensemble(self.kernel, train_x, train_y, diag_reg=diag_reg)
    sl = predict_fn(x_test=test_x, t=t, get='ntk', compute_cov=compute_cov)
    if compute_cov:
      mean_out, cov_out = sl
      self.one_sigma = [i.item() for i in np.sqrt(np.mean(np.diagonal(cov_out, axis1=1, axis2=2), axis=1))]
    else:
      mean_out = sl
    self.rmse = [i.item() for i in np.sqrt(np.mean((mean_out.reshape(-1, len(test_x))-test_y.reshape((1, -1)))**2, axis=1))]
    loss_out = predict_fn(x_test=None, t=t, get='ntk')
    self.loss = [i.item() for i in 1/2*np.mean((loss_out.reshape(-1, n_sample)-train_y.reshape((1, -1)))**2, axis=1)]


# Example models for solving 1D Poission's Equation


def nn(std):
  """Example of a neural network model. See https://github.com/google/neural-tangents for more details"""
  model = stax.serial(
    stax.Dense(512, W_std=std, b_std=std),
    stax.Erf(),
    stax.Dense(  1, W_std=std, b_std=std),
  )
  return Poisson(model)


def gaussian(c):
  """RBF interpolation with Gaussian function"""
  model = Gaussian_model(c=c)
  return Poisson(model)


def imq(c, d_eq=2, d_sl=0):
  """RBF interpolation with Inverse Multi-quadratic function"""
  model = IMQ_model(c=c)
  return Poisson(model)


def fem(d_eq=2, d_sl=0):
  """A naive implementation of the FEM"""
  assert d_eq==2
  assert d_sl==0
  model = FEM_model()
  _, _, kernel_fn = model
  return kernel_fn


# Example definitions of differential equations 


def Poisson(model):
  equation = Deriv(model, 2, 2)
  sl_eq = Deriv(model, 0, 2)
  solution = Deriv(model, 0, 0)
  eq_sl = Deriv(model, 2, 0)
  _, _, kernel_dd = Vcombine(Hcombine(equation, eq_sl), Hcombine(sl_eq, solution))
  _, _, kernel_td = Hcombine(sl_eq, solution)
  _, _, kernel_tt = solution
  return Solve(kernel_dd, kernel_td, kernel_tt)


### Scripts ###


@layer
def Hcombine(layer_fn_0, layer_fn_1):
  """Layer constructor function for combining two sub kernel matrix horizontally. Can be nested."""
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
  """Layer constructor function for combining two sub kernel matrix vertically. Can be nested."""
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


def Gaussian_model(c):
  kernel_fn = _traditional_kernel('gaussian', c**2)
  print(kernel_fn)
  return _not_implemented, _not_implemented, kernel_fn


def IMQ_model(c):
  kernel_fn = _traditional_kernel('imq', c**2)
  print(kernel_fn)
  return _not_implemented, _not_implemented, kernel_fn


def FEM_model():
  kernel_fn = _traditional_kernel('fem')
  print(kernel_fn)
  return _not_implemented, _not_implemented, kernel_fn


def layer_extension(layer_fn):
  """Rewrites the `layer` function in the Neural Tangents library to support solving differential equations."""
  name = layer_fn.__name__

  @utils.wraps(layer_fn)
  def new_layer_fns(*args, **kwargs):
    kernel_fn = layer_fn(*args, **kwargs)
    kernel_fn = _preprocess_kernel_fn_extension(kernel_fn)
    kernel_fn.__name__ = name
    return kernel_fn

  return new_layer_fns


@layer_extension
def Solve(kdd, ktd, ktt):
  """Unite the kernel_train_train (kdd), kernel_test_train (ktd), and kernel_test_test (ktt) into a single kernel function."""
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
      nngp = ktt(x1, x2, _x1=x1, _x2=x2, get=get)
    if ntk is not None:
      nngp, ntk = nngp.nngp, nngp.ntk
    return k.replace(nngp=nngp, ntk=ntk, is_gaussian=True, is_input=False)
  return new_kernel_fn


def _not_implemented(*args, **kwargs):
  raise NotImplementedError


def calculate_condition_numbers(A, b):
  """
  Returns:
    A tuple of (Amotized condition number, Effective condition number).
  """
  A = np.array(A, dtype=np.float64)
  b = np.array(b, dtype=np.float64)
  w, v = np.linalg.eigh(A)
  beta = np.dot(v.transpose(), b).reshape(-1)
  b_norm = np.sqrt(np.sum(beta**2))
  x_norm = np.sqrt(np.sum((beta/w)**2))
  eff_sigma = np.sqrt(np.square(1/w).mean())
  #print(list(w))
  return (b_norm/x_norm*eff_sigma).item(), (b_norm/x_norm/np.abs(w[0])).item()


def _preprocess_kernel_fn_extension(kernel_fn):
  init_fn = apply_fn = _not_implemented

  # Set empty requirements if none specified.
  if not _has_req(kernel_fn):
    kernel_fn = requires()(kernel_fn)

  def kernel_fn_kernel(kernel, **kwargs):
    out_kernel = kernel_fn(kernel, **kwargs)
    return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)

  def kernel_fn_x1(x1, x2, get, cond=None, _b=None, **kwargs):
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
    kernel = _inputs_to_kernel(x1, x2, compute_ntk=compute_ntk, **reqs)
    out_kernel = kernel_fn(kernel, x=x, x_i=x_i, x_b=x_b, which=which, **kwargs)
    #
    kernel = _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)
    if which=='kdd' and cond is not None:
      cond[:] = calculate_condition_numbers(kernel.ntk, _b)

    #return _set_shapes(init_fn, apply_fn, kernel, out_kernel, **kwargs)
    return kernel

  @utils.get_namedtuple('AnalyticKernel')
  def kernel_fn_any(x1_or_kernel, x2=None, get=None, *, pattern=None, mask_constant=None, diagonal_batch=None, diagonal_spatial=None, **kwargs):
    
    if utils.is_nt_tree_of(x1_or_kernel, Kernel):
      return kernel_fn_kernel(x1_or_kernel, pattern=pattern, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, **kwargs)

    return kernel_fn_x1(x1_or_kernel, x2, get, pattern=pattern, diagonal_batch=diagonal_batch, diagonal_spatial=diagonal_spatial, mask_constant=mask_constant, **kwargs)

  _set_req(kernel_fn_any, get_req(kernel_fn))
  return kernel_fn_any


@utils.nt_tree_fn(2)
def _traditional_kernel(method, c2=None):
  def kernel_fn(x1, x2, get, **kwargs):
    if method=='fem':
      nngp = _fem(x1, x2)
    else:
      nngp = _rbf(x1, x2, method, c2=c2)
    ntk = None
    is_gaussian = False
    is_reversed = False
    x1_is_x2 = utils.x1_is_x2(x1, x2, eps=1e-12)
    is_input = False
    cov1 = _cov_diag_batch(x1, False, 0, 1)
    kernel = Kernel(cov1=cov1, cov2=None, nngp=nngp, ntk=ntk, x1_is_x2=x1_is_x2, is_gaussian=is_gaussian, is_reversed=is_reversed, is_input=is_input, diagonal_batch=True, diagonal_spatial=False, shape1=x1.shape, shape2=x1.shape if x2 is None else x2.shape, batch_axis=0, channel_axis=1, mask1=None, mask2=None)
    #print(type(kernel), type(kernel.nngp))
    return kernel.nngp
  return kernel_fn


def _rbf(x1, x2, method, c2):
  #assert channel_axis==1 or len(x1.shape)+channel_axis==1
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