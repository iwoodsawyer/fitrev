%% FITREV DEMO: Smoothly changing extreme events
% Reproducible synthetic annual maxima with a seasonal, nonlinear location.
% Run this script from the repository folder.

rng(17);

%% Generate Gumbel observations
n = 90;
xTrain = sort(10 * rand(n, 1));
location = @(x) 12 + 0.55*x + 2.1*sin(2*pi*x/6) + 0.7*sin(2*pi*x/2.8);
sigma = 0.9;
u = -log(-log(rand(n, 1)));
yTrain = location(xTrain) + sigma*u;

%% Fit a Gumbel GP model
mdl = fitrev(xTrain, yTrain, ...
    'Inference', 'Laplace', ...
    'BasisFunction', 'linear', ...
    'Shape', 0, ...
    'ConstantShape', true, ...
    'Sigma', sigma, ...
    'ConstantSigma', true, ...
    'Standardize', true, ...
    'Optimizer', 'fminsearch', ...
    'OptimizerOptions', optimset('Display', 'off', 'MaxIter', 250, 'MaxFunEvals', 700), ...
    'Verbose', 0);

xGrid = linspace(0, 10, 250)';
[yMean, ~, yInterval] = predict(mdl, xGrid);
[yMeanTrain, ~, trainInterval] = predict(mdl, xTrain);
locationFit = yMean - 0.5772156649015329 * mdl.Sigma;
q50 = locationFit - mdl.Sigma * log(-log(0.50));
q90 = locationFit - mdl.Sigma * log(-log(0.90));
q99 = locationFit - mdl.Sigma * log(-log(0.99));

fprintf('RegressionEV Gumbel demo\n');
fprintf('  Observations: %d\n', mdl.NumObservations);
fprintf('  Fitted scale: %.3f (generating scale %.3f)\n', mdl.Sigma, sigma);
fprintf('  Fitted shape: %.3f\n', mdl.Shape);
fprintf('  Approximate 95%% interval coverage on training data: %.1f%%\n', ...
    100 * mean(yTrain >= trainInterval(:,1) & yTrain <= trainInterval(:,2)));

%% Visual diagnostics
figure('Name', 'fitrev: smoothly changing extreme events', 'Color', 'w');
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

nexttile;
hold on;
fill([xGrid; flipud(xGrid)], ...
    [yInterval(:,1); flipud(yInterval(:,2))], ...
    [0.84, 0.90, 0.98], 'EdgeColor', 'none', 'DisplayName', 'Approx. 95% predictive interval');
plot(xTrain, yTrain, '.', 'Color', [0.25, 0.25, 0.25], ...
    'DisplayName', 'Observed maxima');
plot(xGrid, location(xGrid), '--', 'Color', [0.1, 0.55, 0.35], ...
    'LineWidth', 1.5, 'DisplayName', 'True location');
plot(xGrid, yMean, 'Color', [0.05, 0.25, 0.65], ...
    'LineWidth', 2, 'DisplayName', 'Predicted response mean');
plot(xGrid, q90, ':', 'Color', [0.75, 0.2, 0.15], ...
    'LineWidth', 1.8, 'DisplayName', 'Predicted 90th percentile');
title('Gumbel GP fit and extreme-value risk');
xlabel('Time'); ylabel('Annual maximum');
legend('Location', 'northwest');
grid on; box on;

nexttile;
residual = yTrain - yMeanTrain;
scatter(xTrain, residual, 28, [0.15, 0.45, 0.75], 'filled');
yline(0, 'k-');
yline(mdl.Sigma, 'k:', 'Scale');
yline(-mdl.Sigma, 'k:');
title('Residuals around the predicted response mean');
xlabel('Time'); ylabel('Observed - predicted mean');
grid on; box on;

nexttile;
hold on;
plot(xGrid, q50, 'LineWidth', 1.7, 'DisplayName', 'Median');
plot(xGrid, q90, 'LineWidth', 1.7, 'DisplayName', '90th percentile');
plot(xGrid, q99, 'LineWidth', 1.7, 'DisplayName', '99th percentile');
scatter(xTrain, yTrain, 12, [0.25, 0.25, 0.25], 'filled', ...
    'MarkerFaceAlpha', 0.35, 'DisplayName', 'Observed maxima');
title('Conditional Gumbel return levels');
xlabel('Time'); ylabel('Annual maximum level');
legend('Location', 'northwest');
grid on; box on;

nexttile;
locationAtData = yMeanTrain - 0.5772156649015329 * mdl.Sigma;
pit = exp(-exp(-(yTrain - locationAtData) ./ mdl.Sigma));
pit = sort(pit);
plot([0; pit; 1], [0; (1:n)'/n; 1], 'o-', ...
    'Color', [0.2, 0.45, 0.7], 'MarkerSize', 3);
hold on;
plot([0, 1], [0, 1], 'k--', 'DisplayName', 'Uniform reference');
title('Conditional probability integral transform');
xlabel('Uniform quantile'); ylabel('Empirical PIT');
axis square; xlim([0, 1]); ylim([0, 1]);
grid on; box on;
