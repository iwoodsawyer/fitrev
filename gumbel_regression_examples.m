%% Gumbel Regression Examples from Literature
% This script demonstrates practical applications of Gumbel regression
% using the RegressionEV class with Shape=0 (Gumbel limit)

warning('off','RegressionEV:MaxIterReached')

%% Example 1: WIND SPEED EXTREMES (Coles 2001)
% Modeling annual maximum wind speeds
% Reference: Coles, S. (2001). An introduction to statistical modeling of extreme values

fprintf('\n========================================\n');
fprintf('Example 1: Annual Maximum Wind Speeds\n');
fprintf('========================================\n\n');

rng(123);
n = 150;

year = (1:n)' / n * 2 - 1;
latitude = 30 + 20 * rand(n, 1);
latitude_norm = (latitude - mean(latitude)) / std(latitude);

X_wind = [year, latitude_norm];
f_wind = 15 + 2*year - 1.5*latitude_norm;

sigma_wind = 2.0;
u = -log(-log(rand(n, 1)));
Y_wind = f_wind + sigma_wind * u;

fprintf('Fitting Gumbel regression model to wind speed data...\n');
mdl_wind = fitrev(X_wind, Y_wind, ...
    'Inference', 'Laplace', ...
    'ConstantShape', true, ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Number of observations: %d\n', mdl_wind.NumObservations);
fprintf('  Estimated Sigma (scale): %.4f (true = %.4f)\n', mdl_wind.Sigma, sigma_wind);
fprintf('  Shape parameter: %.4f (fixed at Gumbel)\n', mdl_wind.Shape);
fprintf('  Log-Likelihood: %.4f\n', mdl_wind.LogLikelihood);
fprintf('  Basis coefficients (Beta): '); fprintf('%.4f ', mdl_wind.Beta); fprintf('\n\n');

X_new = [0, 0];
[yhat, ysd, ci] = mdl_wind.predict(X_new);
fprintf('Prediction at (year=mean, latitude=mean):\n');
fprintf('  Predicted location: %.4f\n', yhat);
fprintf('  Predicted std dev:  %.4f\n', ysd);
fprintf('  95%% CI: [%.4f, %.4f]\n\n', ci(1), ci(2));

%% Example 2: FLOOD MAGNITUDE ANALYSIS (Katz et al. 2002)
% Modeling annual maximum river discharge

fprintf('========================================\n');
fprintf('Example 2: Annual Maximum River Discharge (Flood Analysis)\n');
fprintf('========================================\n\n');

rng(456);
n_flood = 80;

year_flood = (1:n_flood)' / n_flood * 3 - 1.5;
season_effect = sin(2*pi*(1:n_flood)'/n_flood);

X_flood = [year_flood, season_effect];
f_flood = 8 + 0.5*year_flood + 0.3*season_effect;

sigma_flood = 1.2;
u_flood = -log(-log(rand(n_flood, 1)));
Y_flood = f_flood + sigma_flood * u_flood;

fprintf('Fitting Gumbel model to flood discharge data...\n');
mdl_flood = fitrev(X_flood, Y_flood, ...
    'Inference', 'Laplace', ...
    'ConstantShape', true, ...
    'ConstantSigma', false, ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Estimated Sigma: %.4f (true = %.4f)\n', mdl_flood.Sigma, sigma_flood);
fprintf('  Shape: %.4f (Gumbel)\n', mdl_flood.Shape);
fprintf('  Log-Likelihood: %.4f\n', mdl_flood.LogLikelihood);
fprintf('  Basis coefficients: '); fprintf('%.4f ', mdl_flood.Beta); fprintf('\n\n');

threshold = 10;
[yhat_pred, ysd_pred] = mdl_flood.predict(X_flood(1:5, :));
fprintf('First 5 predictions:\n');
for i = 1:5
    z = (threshold - yhat_pred(i)) / mdl_flood.Sigma;
    prob_exceed = exp(-exp(-z));
    fprintf('  Time %d: E[Y]=%.3f, P(Y>%.1f)=%.4f\n', i, yhat_pred(i), threshold, prob_exceed);
end
fprintf('\n');

%% Example 3: TEMPERATURE EXTREMES (Energy Demand)

fprintf('========================================\n');
fprintf('Example 3: Daily Maximum Temperature\n');
fprintf('========================================\n\n');

rng(789);
n_temp = 365;

day_of_year = (1:n_temp)';
day_norm = (day_of_year - 182.5) / 182.5;

region_id = randi([1, 3], n_temp, 1);
region_effect = region_id * 0.5;
region_effect = (region_effect - mean(region_effect)) / std(region_effect);

X_temp = [day_norm, region_effect];
f_temp = 20 + 10*sin(2*pi*day_norm/2) + 2*region_effect;

sigma_temp = 1.5;
u_temp = -log(-log(rand(n_temp, 1)));
Y_temp = f_temp + sigma_temp * u_temp;

fprintf('Fitting Gumbel model to daily max temperature...\n');
mdl_temp = fitrev(X_temp, Y_temp, ...
    'Inference', 'Laplace', ...
    'ConstantShape', true, ...
    'ConstantSigma', false, ...
    'BasisFunction', 'linear', ...
    'Standardize', true, ...
    'Verbose', 0);

fprintf('Model Results:\n');
fprintf('  Estimated Sigma: %.4f\n', mdl_temp.Sigma);
fprintf('  Linear basis coefficients: '); fprintf('%.4f ', mdl_temp.Beta); fprintf('\n');
fprintf('  Log-Likelihood: %.4f\n\n', mdl_temp.LogLikelihood);

loss_mse = mdl_temp.loss(X_temp, Y_temp, 'LossFun', 'mse');
loss_mae = mdl_temp.loss(X_temp, Y_temp, 'LossFun', 'mae');
loss_nll = mdl_temp.loss(X_temp, Y_temp, 'LossFun', 'nloglikelihood');

fprintf('Loss Metrics (on training data):\n');
fprintf('  MSE: %.4f\n', loss_mse);
fprintf('  MAE: %.4f\n', loss_mae);
fprintf('  Neg. Log-Likelihood: %.4f\n\n', loss_nll);

%% Example 4: LAPLACE vs EXPECTATION PROPAGATION

fprintf('========================================\n');
fprintf('Example 4: Laplace vs Expectation Propagation\n');
fprintf('========================================\n\n');

rng(999);
n_comp = 100;
X_comp = [-1 + 2*rand(n_comp, 1), -1 + 2*rand(n_comp, 1)];
f_comp = 10 + 2*X_comp(:, 1) - 1.5*X_comp(:, 2);
Y_comp = f_comp + 1.0 * (-log(-log(rand(n_comp, 1))));

fprintf('Fitting with Laplace approximation...\n');
mdl_lap = fitrev(X_comp, Y_comp, ...
    'Inference', 'Laplace', ...
    'ConstantShape', true, ...
    'Verbose', 0);

fprintf('Fitting with Expectation Propagation...\n');
mdl_ep = fitrev(X_comp, Y_comp, ...
    'Inference', 'EP', ...
    'ConstantShape', true, ...
    'Verbose', 0);

X_test = [-1 + 2*rand(20, 1), -1 + 2*rand(20, 1)];
[yhat_lap, ysd_lap] = mdl_lap.predict(X_test);
[yhat_ep, ysd_ep] = mdl_ep.predict(X_test);

fprintf('Comparison on test set (first 5 predictions):\n');
fprintf('Idx | Laplace Mean | Laplace SD | EP Mean | EP SD | Mean Diff\n');
fprintf('----+----------------------------------------------\n');
for i = 1:5
    diff = abs(yhat_lap(i) - yhat_ep(i));
    fprintf('%3d | %12.4f | %10.4f | %7.4f | %6.4f | %8.4f\n', ...
        i, yhat_lap(i), ysd_lap(i), yhat_ep(i), ysd_ep(i), diff);
end
fprintf('\nMean absolute difference in predictions: %.6f\n\n', ...
    mean(abs(yhat_lap - yhat_ep)));

%% Example 5: MODEL SELECTION

fprintf('========================================\n');
fprintf('Example 5: Model Selection via Information Criteria\n');
fprintf('========================================\n\n');

rng(111);
n_select = 120;
X_select = [-1 + 2*rand(n_select, 1), -1 + 2*rand(n_select, 1)];
f_select = 12 + X_select(:, 1) + 0.5*X_select(:, 2);
Y_select = f_select + 1.5 * (-log(-log(rand(n_select, 1))));

fprintf('Fitting models with different basis functions...\n\n');

basis_options = {'constant', 'linear', 'purequadratic'};
results = [];

for i = 1:numel(basis_options)
    basis = basis_options{i};
    mdl = fitrev(X_select, Y_select, ...
        'BasisFunction', basis, ...
        'ConstantShape', true, ...
        'Standardize', true, ...
        'Verbose', 0);
    
    aic = mdl.criterion('CriterionFun', 'AIC');
    bic = mdl.criterion('CriterionFun', 'BIC');
    ll = mdl.LogLikelihood;
    nbeta = numel(mdl.Beta);
    
    fprintf('Basis: %-15s | AIC: %10.4f | BIC: %10.4f | LL: %10.4f | Beta: %d\n', ...
        basis, aic, bic, ll, nbeta);
    
    results = [results; aic, bic, ll, nbeta];
end

fprintf('\nModel Selection Summary:\n');
[~, idx_aic] = min(results(:,1));
[~, idx_bic] = min(results(:,2));
fprintf('  Best AIC: %s\n', basis_options{idx_aic});
fprintf('  Best BIC: %s\n', basis_options{idx_bic});

%% Summary
fprintf('\n========================================\n');
fprintf('Summary: Gumbel Regression Examples\n');
fprintf('========================================\n\n');
fprintf('Applications:\n');
fprintf('  1. Wind speed extremes: climate/meteorology\n');
fprintf('  2. Flood magnitudes: hydrology/civil engineering\n');
fprintf('  3. Temperature extremes: energy/climate\n');
fprintf('  4. Inference comparison: Laplace vs EP\n');
fprintf('  5. Model selection: AIC/BIC criteria\n\n');
fprintf('Key Features:\n');
fprintf('  - Gaussian Process prior on latent function f(x)\n');
fprintf('  - Gumbel observation model (Shape = 0)\n');
fprintf('  - Laplace or EP approximate inference\n');
fprintf('  - Multiple information criteria\n');
fprintf('  - Standardization and active set selection\n');