# neural-tangents_extension


\section{Overview}


This package is an extension of the Neural Tangents library (URL: https://github.com/google/neural-tangents). We added new features to support the kernel computation in the context of solving differential equations. For the complexity of the cost function, a set of methods are implemented to aid the construction of the partitioned kernel matrix. See the example notebooks for more information.


\section{Symbols}


The Neural Tangent Kernel (NTK) approximate the outputs of a wide neural network as a Gaussian process with mean.
$$
  E\,[\,\hat Y(x)\,] = \Theta(x,\,\mathcal X)\,\Theta(\mathcal X,\,\mathcal X)^{-1}(I-e^{-\eta\,\Theta(\mathcal X,\,\mathcal X)\,t})\,\mathcal Y
$$
The matrix $\Theta(\mathcal X,\,\mathcal X)$ is referred to as `kernel_train_train` or `kdd`, and the matrix $\Theta(x,\,\mathcal X)$ is referred to as `kernel_test_train` or `ktd` following the naming convention of Neural Tangents.


In the context of neural network solver for differential equations, the matrices take different forms. When solving a differential equation with linear operator $\mathcal L$, the `kdd` becomes
\[
  \left[\begin{array}{rr}
        \mathcal L\mathcal L_{(2)}\Theta_{i,i} & \mathcal L\,\Theta_{i,b} \\
        \mathcal L_{(2)}\Theta_{b,i} & \Theta_{b,b}
    \end{array}\right]
\]
and the `ktd` becomes
$$
  \left[\,\mathcal L_{(2)}\Theta(x,\,\mathcal X_{i})\,,\,\Theta(x,\,\mathcal X_{b})\,\right]
$$
Here, $\Theta_{i,i} = \Theta(\mathcal X_{i},\,\mathcal X_{i})\,$, $\Theta_{b,i} = \Theta(\mathcal X_{b},\,\mathcal X_{i})\,$ and so on; $\mathcal L$ and $\mathcal L_{(2)}$ are the same differential operator but act on the first and the second variable of the covariance function $\Theta(x,\,x')$ respectively.


The generating function for `ktd` and `kdd` are different for differential equation solvers, in contrast to the original NTK. Traditional numerical methods including RBF interpolation collocation methods and a naive implementation of the Finite Element method are also included in this extension.