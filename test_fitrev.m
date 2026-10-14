classdef test_fitrev < matlab.unittest.TestCase
    %TEST_FITREV Consolidated Qualification Tests for RegressionEV (fitrev)
    % Comprehensive test suite covering GEV regression model fitting, prediction,
    % loss computation, criterion evaluation, and hyperparameter optimization.

    properties
        X_data
        Y_gumbel
        Y_gev
        Tbl
        X_1d
        Y_1d
    end

    methods (TestMethodSetup)
        function createData(testCase)
            rng(42);
            
            n = 100;
            x1 = randn(n, 1);
            x2 = randn(n, 1);
            x3 = randn(n, 1);
            testCase.X_data = [x1, x2, x3];

            f = 2 + 0.5*x1 - 0.3*x2 + 0.4*x3;
            
            u = -log(-log(rand(n, 1)));
            testCase.Y_gumbel = f + u;
            
            shape = 0.1;
            sigma = 1.0;
            z = (1 + shape * (u / sigma));
            z = max(z, eps);
            gev = f + sigma * ((z.^(-1/shape)) - 1) / shape;
            testCase.Y_gev = gev;

            testCase.Tbl = table(x1, x2, x3, testCase.Y_gumbel, ...
                'VariableNames', {'x1', 'x2', 'x3', 'Response'});
                
            testCase.X_1d = (1:20)' + 0.1 * randn(20, 1);
            testCase.Y_1d = 5 + 0.3 * testCase.X_1d + exprnd(2, 20, 1);
        end
    end

    methods (Test)

        function testFitBasicGumbel(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Inference', 'Laplace', 'FitMethod', 'exact', 'Verbose', 0);
            testCase.verifyClass(mdl, 'RegressionEV');
            testCase.verifyEqual(mdl.NumObservations, 100);
            testCase.verifyEqual(abs(mdl.Shape), 0, 'AbsTol', 1e-8);
        end

        function testFitGEVWithShapeOptimization(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gev, ...
                'Inference', 'Laplace', 'ConstantShape', false, ...
                'FitMethod', 'exact', 'Verbose', 0);
            testCase.verifyClass(mdl, 'RegressionEV');
            testCase.verifyTrue(mdl.Shape > -0.95 && mdl.Shape < 0.95);
            testCase.verifyTrue(isfinite(mdl.Shape));
        end

        function testFitWithSigmaOptimization(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gev, ...
                'Inference', 'Laplace', 'ConstantSigma', false, ...
                'FitMethod', 'exact', 'Verbose', 0);
            testCase.verifyClass(mdl, 'RegressionEV');
            testCase.verifyTrue(mdl.Sigma > 0);
            testCase.verifyTrue(isfinite(mdl.Sigma));
        end

        function testFitWithTableInput(testCase)
            mdl = fitrev(testCase.Tbl, 'Response', 'Verbose', 0);
            testCase.verifyClass(mdl, 'RegressionEV');
            testCase.verifyEqual(string(mdl.ResponseName), "Response");
            testCase.verifyEqual(numel(mdl.PredictorNames), 3);
        end

        function testInferenceLaplace(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Inference', 'Laplace', 'Verbose', 0);
            testCase.verifyEqual(lower(mdl.Inference), 'laplace');
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testInferenceEP(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Inference', 'EP', 'Verbose', 0);
            testCase.verifyEqual(lower(mdl.Inference), 'ep');
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testInferenceLaplaceVsEPDiffer(testCase)
            mdl_lap = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Inference', 'Laplace', 'FitMethod', 'none', 'Verbose', 0);
            mdl_ep = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Inference', 'EP', 'FitMethod', 'none', 'Verbose', 0);
            
            Xte = testCase.X_data(1:10, :);
            [y_lap, ~] = mdl_lap.predict(Xte);
            [y_ep, ~] = mdl_ep.predict(Xte);
            
            testCase.verifyGreaterThan(max(abs(y_lap - y_ep)), 1e-8);
        end

        function testPredictOutputShape(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            Xte = testCase.X_data(1:15, :);
            
            [yhat, ysd, ci] = mdl.predict(Xte);
            
            testCase.verifySize(yhat, [15 1]);
            testCase.verifySize(ysd, [15 1]);
            testCase.verifySize(ci, [15 2]);
        end

        function testPredictOutputValidity(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            Xte = testCase.X_data(1:20, :);
            
            [yhat, ysd, ci] = mdl.predict(Xte);
            
            testCase.verifyTrue(all(isfinite(yhat)));
            testCase.verifyTrue(all(isfinite(ysd) | isnan(ysd)));
            testCase.verifyTrue(all(ci(:, 1) <= yhat + 1e-10));
            testCase.verifyTrue(all(ci(:, 2) >= yhat - 1e-10));
        end

        function testPredictConfidenceInterval(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            Xte = testCase.X_data(1:10, :);
            
            [~, ysd, ci] = mdl.predict(Xte, 'Alpha', 0.05);
            
            testCase.verifyTrue(all(ci(:, 1) < ci(:, 2)));
            testCase.verifyTrue(all(isfinite(ysd) | isnan(ysd)));
        end

        function testPredictSizeMismatch(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            Xbad = testCase.X_data(:, 1:2);
            
            testCase.verifyError(@() mdl.predict(Xbad), ...
                'RegressionEV:SizeMismatch');
        end

        function testLossMSE(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'mse');
            testCase.verifyTrue(isfinite(L));
            testCase.verifyGreaterThan(L, 0);
        end

        function testLossRMSE(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L_rmse = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'rmse');
            L_mse = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'mse');
            
            testCase.verifyEqual(L_rmse^2, L_mse, 'RelTol', 1e-10);
        end

        function testLossMAE(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'mae');
            testCase.verifyTrue(isfinite(L));
            testCase.verifyGreaterThan(L, 0);
        end

        function testLossNLogLikelihood(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'nloglikelihood');
            testCase.verifyTrue(isfinite(L));
        end

        function testLossSMSE(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'smse');
            testCase.verifyTrue(isfinite(L));
        end

        function testLossMSLL(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'msll');
            testCase.verifyTrue(isfinite(L));
        end

        function testLossWithWeights(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            w = 0.5 + rand(100, 1);
            L = mdl.loss(testCase.X_data, testCase.Y_gumbel, ...
                'LossFun', 'mse', 'Weights', w);
            testCase.verifyTrue(isfinite(L));
        end

        function testResubPredictMatchesPredict(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            
            [yhat_resub, ysd_resub] = mdl.resubPredict();
            [yhat_pred, ysd_pred] = mdl.predict(testCase.X_data);
            
            testCase.verifyLessThan(max(abs(yhat_resub - yhat_pred)), 1e-10);
            testCase.verifyLessThan(max(abs(ysd_resub - ysd_pred)), 1e-10);
        end

        function testResubLoss(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            L_resub = mdl.resubLoss('LossFun', 'mse');
            L_direct = mdl.loss(testCase.X_data, testCase.Y_gumbel, 'LossFun', 'mse');
            
            testCase.verifyEqual(L_resub, L_direct, 'RelTol', 1e-10);
        end

        function testCriterionAIC(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'AIC');
            testCase.verifyTrue(isfinite(C));
        end

        function testCriterionAICc(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'AICc');
            testCase.verifyTrue(isfinite(C));
        end

        function testCriterionBIC(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'BIC');
            testCase.verifyTrue(isfinite(C));
        end

        function testCriterionCAIC(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'CAIC');
            testCase.verifyTrue(isfinite(C));
        end

        function testCriterionGCV(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'GCV');
            testCase.verifyTrue(isfinite(C));
            testCase.verifyGreaterThan(C, 0);
        end

        function testCriterionLOOCV(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'LOOCV');
            testCase.verifyTrue(isfinite(C));
        end

        function testCriterionWAIC(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            C = mdl.criterion('CriterionFun', 'WAIC');
            testCase.verifyTrue(isfinite(C));
        end

        function testKernelSquaredExponential(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'squaredexponential', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testKernelExponential(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'exponential', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testKernelMatern32(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'matern32', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testKernelMatern52(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'matern52', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testKernelRationalQuadratic(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'rationalquadratic', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testKernelARDSquaredExponential(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'ardsquaredexponential', 'Verbose', 0);
            n_params = numel(mdl.KernelInformation.KernelParameters);
            testCase.verifyEqual(n_params, 4);
        end

        function testKernelARDMatern52(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'KernelFunction', 'ardmatern52', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testBasisNone(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'BasisFunction', 'none', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 0);
        end

        function testBasisConstant(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'BasisFunction', 'constant', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 1);
        end

        function testBasisLinear(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'BasisFunction', 'linear', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 4);
        end

        function testBasisPureQuadratic(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'BasisFunction', 'purequadratic', 'Verbose', 0);
            testCase.verifyEqual(numel(mdl.Beta), 7);
        end

        function testLambdaZeroAllowed(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Lambda', 0, 'ConstantLambda', true, 'Verbose', 0);
            testCase.verifyEqual(mdl.Lambda, 0);
        end

        function testLambdaNonZero(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Lambda', 1e-4, 'ConstantLambda', true, 'Verbose', 0);
            testCase.verifyEqual(mdl.Lambda, 1e-4);
        end

        function testStandardizeFalse(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Standardize', false, 'FitMethod', 'none', 'Verbose', 0);
            testCase.verifyEqual(mdl.PredictorLocation, zeros(1, 3));
            testCase.verifyEqual(mdl.PredictorScale, ones(1, 3));
        end

        function testStandardizeTrue(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'Standardize', true, 'FitMethod', 'none', 'Verbose', 0);
            testCase.verifyNotEqual(mdl.PredictorLocation, zeros(1, 3));
            testCase.verifyNotEqual(mdl.PredictorScale, ones(1, 3));
        end

        function testActiveSetRandom(testCase)
            m = 40;
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ActiveSetMethod', 'random', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
        end

        function testActiveSetFirst(testCase)
            m = 40;
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ActiveSetMethod', 'first', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
            testCase.verifyTrue(all(find(mdl.IsActiveSetVector) == (1:m)'));
        end

        function testActiveSetLast(testCase)
            m = 40;
            n = size(testCase.X_data, 1);
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ActiveSetMethod', 'last', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
            testCase.verifyTrue(all(find(mdl.IsActiveSetVector) == (n-m+1:n)'));
        end

        function testActiveSetQR(testCase)
            m = 40;
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ActiveSetMethod', 'qr', 'ActiveSetSize', m, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), m);
        end

        function testExplicitActiveSetIndices(testCase)
            idx = 1:30;
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ActiveSet', idx, 'Verbose', 0);
            testCase.verifyEqual(sum(mdl.IsActiveSetVector), numel(idx));
        end

        function testCompactRemovesData(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            mdl_compact = mdl.compact();
            
            testCase.verifyEmpty(mdl_compact.X);
            testCase.verifyEmpty(mdl_compact.Y);
            testCase.verifyEmpty(mdl_compact.W);
            testCase.verifyEqual(mdl_compact.NumObservations, 0);
        end

        function testCompactPreservesPrediction(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            Xte = testCase.X_data(1:10, :);
            
            [yhat_full, ysd_full] = mdl.predict(Xte);
            [yhat_compact, ysd_compact] = mdl.compact().predict(Xte);
            
            testCase.verifyLessThan(max(abs(yhat_full - yhat_compact)), 1e-10);
            testCase.verifyLessThan(max(abs(ysd_full - ysd_compact)), 1e-10);
        end

        function testCompactErrorsOnResub(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, 'Verbose', 0);
            mdl_compact = mdl.compact();
            
            testCase.verifyError(@() mdl_compact.resubLoss(), ...
                'RegressionEV:Compact');
            testCase.verifyError(@() mdl_compact.resubPredict(), ...
                'RegressionEV:Compact');
            testCase.verifyError(@() mdl_compact.criterion(), ...
                'RegressionEV:Compact');
        end

        function testFitMethodNone(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'FitMethod', 'none', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testFitMethodExact(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'FitMethod', 'exact', 'Verbose', 0);
            testCase.verifyTrue(isfinite(mdl.LogLikelihood));
        end

        function testEmptyXError(testCase)
            testCase.verifyError(@() fitrev([], testCase.Y_gumbel), ...
                'RegressionEV:EmptyX');
        end

        function testEmptyYError(testCase)
            testCase.verifyError(@() fitrev(testCase.X_data, []), ...
                'RegressionEV:EmptyY');
        end

        function testSizeMismatchError(testCase)
            testCase.verifyError( ...
                @() fitrev(testCase.X_data, testCase.Y_gumbel(1:50)), ...
                'RegressionEV:SizeMismatch');
        end

        function testNaNInXError(testCase)
            Xbad = testCase.X_data;
            Xbad(1, 1) = NaN;
            testCase.verifyError(@() fitrev(Xbad, testCase.Y_gumbel), ...
                'RegressionEV:InvalidX');
        end

        function testNaNInYError(testCase)
            Ybad = testCase.Y_gumbel;
            Ybad(1) = NaN;
            testCase.verifyError(@() fitrev(testCase.X_data, Ybad), ...
                'RegressionEV:BadY');
        end

        function testPredictorNames(testCase)
            names = {'feat1', 'feat2', 'feat3'};
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'PredictorNames', names, 'Verbose', 0);
            testCase.verifyEqual(mdl.PredictorNames, string(names));
        end

        function testResponseName(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ResponseName', 'maximum_value', 'Verbose', 0);
            testCase.verifyEqual(string(mdl.ResponseName), "maximum_value");
        end

        function testShapeParameterBounded(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gev, ...
                'ConstantShape', false, 'Verbose', 0);
            testCase.verifyTrue(mdl.Shape > -0.95);
            testCase.verifyTrue(mdl.Shape < 0.95);
        end

        function testSigmaParameterPositive(testCase)
            mdl = fitrev(testCase.X_data, testCase.Y_gumbel, ...
                'ConstantSigma', false, 'Verbose', 0);
            testCase.verifyGreaterThan(mdl.Sigma, 0);
        end

        function testGumbelPredictiveMomentsAtUncorrelatedPoint(testCase)
            x = (0:5)';
            y = 2 + 0.4*x;
            sigma = 0.8;
            mdl = fitrev(x, y, ...
                'FitMethod', 'none', ...
                'BasisFunction', 'constant', ...
                'Beta', 2, ...
                'KernelParameters', [1; 0.1], ...
                'Sigma', sigma, ...
                'Shape', 0, ...
                'Standardize', false, ...
                'Verbose', 0);

            [yhat, ysd] = predict(mdl, 1e4);
            expectedMean = 2 + 0.5772156649015329*sigma;
            expectedVariance = 0.1^2 + (pi^2/6)*sigma^2;

            testCase.verifyEqual(yhat, expectedMean, 'AbsTol', 1e-10);
            testCase.verifyEqual(ysd^2, expectedVariance, 'AbsTol', 1e-10);
        end

        function testEPPredictionIsFinite(testCase)
            rng(7);
            mdl = fitrev(testCase.X_data(1:30,1), testCase.Y_gumbel(1:30), ...
                'Inference', 'EP', ...
                'FitMethod', 'none', ...
                'InferenceOptions', struct('MaxIter', 30, 'Tol', 1e-6, ...
                    'Damping', 0.6, 'NumGHPoints', 30), ...
                'Verbose', 0);

            [yhat, ysd, ci] = predict(mdl, testCase.X_data(31:35,1));
            testCase.verifyTrue(all(isfinite(yhat)));
            testCase.verifyTrue(all(isfinite(ysd) & ysd >= 0));
            testCase.verifyTrue(all(isfinite(ci(:))));
            testCase.verifyTrue(all(ci(:,1) <= yhat & yhat <= ci(:,2)));
        end

        function testEPEvidenceMatchesSingleObservationIntegral(testCase)
            sigma = 0.8;
            priorMean = 1.2;
            priorSD = 0.5;
            y = 1.4;
            options = struct('MaxIter', 100, 'Tol', 1e-9, ...
                'Damping', 0.5, 'NumGHPoints', 100);
            mdl = fitrev(0, y, 'FitMethod', 'none', 'Inference', 'EP', ...
                'Sigma', sigma, 'Beta', priorMean, ...
                'KernelParameters', [1; priorSD], 'InferenceOptions', options);
            density = @(f) exp(-log(sigma) - (y-f)/sigma - exp(-(y-f)/sigma)) .* ...
                exp(-0.5*((f-priorMean)/priorSD).^2)/(sqrt(2*pi)*priorSD);
            expected = log(integral(density, -Inf, Inf, 'AbsTol', 1e-12));
            testCase.verifyEqual(mdl.LogLikelihood, expected, 'AbsTol', 1e-7);
            testCase.verifyLessThanOrEqual(mdl.LogLikelihood, -log(sigma)-1);
        end

        function testEPWeightsExponentiateLikelihood(testCase)
            sigma = 0.8;
            priorMean = 1.2;
            priorSD = 0.5;
            y = 1.4;
            mdl = fitrev(0, y, 'FitMethod', 'none', 'Inference', 'EP', ...
                'Sigma', sigma, 'Beta', priorMean, 'Weights', 2, ...
                'KernelParameters', [1; priorSD], ...
                'InferenceOptions', struct('MaxIter', 100, 'Tol', 1e-9, ...
                'Damping', 0.5, 'NumGHPoints', 100));
            density = @(f) exp(2*(-log(sigma) - (y-f)/sigma - exp(-(y-f)/sigma))) .* ...
                exp(-0.5*((f-priorMean)/priorSD).^2)/(sqrt(2*pi)*priorSD);
            expected = log(integral(density, -Inf, Inf, 'AbsTol', 1e-12));
            testCase.verifyEqual(mdl.LogLikelihood, expected, 'AbsTol', 1e-7);
        end

        function testEPQuadratureResolvesDistantNarrowLikelihood(testCase)
            sigma = 0.1;
            y = 6;
            mdl = fitrev(0, y, 'FitMethod', 'none', 'Inference', 'EP', ...
                'Sigma', sigma, 'Beta', 0, 'KernelParameters', [1; 1], ...
                'InferenceOptions', struct('MaxIter', 100, 'Tol', 1e-9, ...
                'Damping', 0.5, 'NumGHPoints', 100));
            density = @(f) exp(-log(sigma) - (y-f)/sigma - exp(-(y-f)/sigma)) .* ...
                exp(-0.5*f.^2)/sqrt(2*pi);
            normalizer = integral(density, -5, 15, 'AbsTol', 1e-16, 'RelTol', 1e-10);
            meanLocation = integral(@(f) f.*density(f), -5, 15, ...
                'AbsTol', 1e-16, 'RelTol', 1e-10) / normalizer;
            varianceLocation = integral(@(f) (f-meanLocation).^2.*density(f), -5, 15, ...
                'AbsTol', 1e-16, 'RelTol', 1e-10) / normalizer;
            [responseMean, responseSD] = predict(mdl, 0);
            testCase.verifyEqual(mdl.LogLikelihood, log(normalizer), 'AbsTol', 1e-5);
            testCase.verifyEqual(responseMean, meanLocation + 0.5772156649015329*sigma, ...
                'AbsTol', 1e-5);
            testCase.verifyEqual(responseSD^2, varianceLocation + (pi^2/6)*sigma^2, ...
                'AbsTol', 1e-5);
        end

        function testEPDeterministicEvidence(testCase)
            x = (1:8)';
            y = 2 + sin(x);
            args = {'Inference', 'EP', 'FitMethod', 'none', 'Beta', 2, ...
                'Sigma', 1, 'KernelParameters', [1; 0.5]};
            rng(1);
            first = fitrev(x, y, args{:});
            rng(2);
            second = fitrev(x, y, args{:});
            testCase.verifyEqual(first.LogLikelihood, second.LogLikelihood);
            testCase.verifyEqual(predict(first, x), predict(second, x));
        end

        function testInformationCriteriaUseGevLikelihood(testCase)
            x = (1:10)';
            y = 2 + sin(x);
            mdl = fitrev(x, y, 'Beta', 2, 'Sigma', 1, ...
                'KernelParameters', [1; 0.5], ...
                'ConstantKernelParameters', true(2,1), 'Verbose', 0);
            mu = predict(mdl, x) - 0.5772156649015329*mdl.Sigma;
            z = (y-mu)/mdl.Sigma;
            ll = sum(-log(mdl.Sigma)-z-exp(-z));
            k = numel(mdl.Beta);
            testCase.verifyEqual(mdl.criterion('CriterionFun', 'AIC'), ...
                2*k-2*ll, 'AbsTol', 1e-10);
            testCase.verifyEqual(mdl.criterion('CriterionFun', 'BIC'), ...
                k*log(numel(y))-2*ll, 'AbsTol', 1e-10);
        end

    end
end