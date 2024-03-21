# Bayesian-Opt-Information-theory

![Example GIF](https://github.com/Sara-Munafo/Bayesian-Opt-Information-theory/blob/main/simple_optimization_ei.gif)

This repository contains the implementation of the Bayesian Optimization method for hyperparameters tuning, a project for the course of Information theory and Inference of the Master's degree in Physics of Data at University of padua.


useful links:

- http://www.sfu.ca/~ssurjano/optimization.html
- https://medium.com/@okanyenigun/step-by-step-guide-to-bayesian-optimization-a-python-based-approach-3558985c6818
- https://towardsdatascience.com/bayesian-optimization-concept-explained-in-layman-terms-1d2bcdeaf12f

Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) method: 
- https://en.wikipedia.org/wiki/Quasi-Newton_method
- https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm
- https://github.com/trsav/bfgs/blob/master/BFGS.py

tutorial bayesian optimization:

- https://arxiv.org/pdf/1807.02811.pdf


Bayesian MC (?):

- https://www.sciencedirect.com/science/article/pii/S0951832010000992


Sequential MC:
- https://jblevins.org/notes/smc-intro
- https://diglib.eg.org/bitstream/handle/10.2312/SCA.SCA10.103-112/103-112.pdf?sequence=1&isAllowed=y

Vihola algorithm
- https://www.slideshare.net/xianblog/vihola
  
Source code likelihood optimization 
- https://github.com/scikit-learn/scikit-learn/blob/f07e0138bfee41cd2c0a5d0251dc3fe03e6e1084/sklearn/gaussian_process/_gpr.py
-             # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite."
                    )
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial, bounds)
                    )
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.kernel_._check_bounds_params()

            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = self.log_marginal_likelihood(
                self.kernel_.theta, clone_kernel=False
            )

        # Precompute quantities required for predictions which are independent
        # of actual query points
        # Alg. 2.1, page 19, line 2 -> L = cholesky(K + sigma^2 I)
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        try:
            self.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
        except np.linalg.LinAlgError as exc:
            exc.args = (
                (
                    f"The kernel, {self.kernel_}, is not returning a positive "
                    "definite matrix. Try gradually increasing the 'alpha' "
                    "parameter of your GaussianProcessRegressor estimator."
                ),
            ) + exc.args
            raise
        # Alg 2.1, page 19, line 3 -> alpha = L^T \ (L \ y)
        self.alpha_ = cho_solve(
            (self.L_, GPR_CHOLESKY_LOWER),
            self.y_train_,
            check_finite=False,
        )
        return self
