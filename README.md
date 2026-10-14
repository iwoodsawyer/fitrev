# fitrev
Extreme Value (EV) regression for fitting generalized extreme value models. The function `fitrev` fits a Gaussian process model for **generalized extreme value** (**GEV** or **Gumbel**) regression. This function delegates fitting to `RegressionEV` class. This class provides a fitrgp-style API for extreme value Gaussian process regression. It is intended to resemble MATLAB's `RegressionGP` and `fitrgp` interface and conventions, but implements regression via approximate inference using Laplace with GEV/Gumbel distribution.

Run `demo_fitrev` in MATLAB for a reproducible Gumbel-regression example with prediction, residual, return-level, and probability-integral-transform plots.

Run `gumbel_regression_examples` for synthetic wind, flood, and temperature
examples. These estimate the observation scale (`ConstantSigma=false`) and
use fixed kernel parameters to avoid confounding kernel tuning with the
demonstrations. The Laplace/EP comparison uses identical fitted parameters.

`predict` returns the **response mean**, not the GEV location. For Gumbel,
the location is `yhat - 0.5772156649015329*model.Sigma`. Gumbel exceedance
probabilities conditional on this location are
`-expm1(-exp(-(threshold-location)/model.Sigma))`. Return levels and conditional
probabilities computed this way do not integrate over latent GP uncertainty;
the returned intervals are symmetric normal approximations, not exact GEV
quantile intervals.

EP uses deterministic sweeps and mode-centered Gauss-Hermite quadrature.
Its defaults are 100 sweeps, damping 0.5, and 40 quadrature points; these can
be overridden with `InferenceOptions`. Convergence checks use relative site
parameter changes. Automatic basis initialization is centered on the estimated
location when fitting parameters, but `FitMethod='none'` preserves supplied
parameters and the zero default basis coefficients.

AIC/BIC variants use the plug-in GEV likelihood at the posterior location,
counting free kernel, basis, scale, shape, and jitter parameters. They do not
account for GP latent-function flexibility and are not a substitute for
held-out predictive assessment. `loss(...,'LossFun','nloglikelihood')` remains
a Gaussian predictive negative log score; the existing WAIC option is also
a Gaussian-score proxy, not GEV WAIC.
