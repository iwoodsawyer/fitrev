classdef RegressionEV
    %REGRESSIONEV Gaussian Process Generalized Extreme Value (GEV) regression model.
    %
    %   RegressionEV is a Gaussian process model for regression with a
    %   generalized extreme value (GEV) response distribution. This model can
    %   predict response given new data. This model also stores data used for
    %   training and can compute resubstitution predictions.
    %
    %   An object of this class cannot be created by calling the constructor.
    %   Use FITREV to create a RegressionEV object by fitting a GP extreme
    %   value regression model to training data.
    %
    %   This class provides a fitrgp-style API for Gaussian process regression
    %   with non-Gaussian responses following a generalized extreme value
    %   distribution:
    %
    %       Y | f(X) ~ GEV(location = f(X), scale = Sigma, shape = Shape)
    %
    %   where the latent function f(X) follows a Gaussian process prior and
    %   inference is carried out approximately using either Laplace
    %   approximation or Expectation Propagation (EP). The Gumbel model is
    %   recovered as the special case Shape = 0.
    %
    %   RegressionEV properties:
    %       NumObservations        - Number of observations.
    %       X                      - Matrix of predictors used to train this model.
    %       Y                      - Observed response used to train this model.
    %       W                      - Weights of observations used to train this model.
    %       ModelParameters        - RegressionEV parameters.
    %       PredictorNames         - Names of predictors used for this model.
    %       ExpandedPredictorNames - Names of expanded predictors.
    %       ResponseName           - Name of the response variable.
    %       ResponseTransform      - Transformation applied to predicted response.
    %       KernelFunction         - Kernel function used in this model.
    %       KernelInformation      - Information about parameters of this kernel function.
    %       BasisFunction          - Basis function used in this model.
    %       Beta                   - Estimated value of basis function coefficients.
    %       Sigma                  - Scale parameter of the GEV response distribution.
    %       Shape                  - Shape parameter of the GEV response distribution.
    %       Lambda                 - Jitter to force positive definiteness of kernel matrix.
    %       PredictorLocation      - A vector of predictor means if standardization is used.
    %       PredictorScale         - A vector of predictor standard deviations if standardization is used.
    %       Alpha                  - Vector of weights for computing predictions.
    %       ActiveSetVectors       - Subset of the training data needed to make predictions.
    %       Inference              - Approximate inference method used ('Laplace' or 'EP').
    %       FitMethod              - Method used to estimate parameters.
    %       PredictMethod          - Method used to make predictions.
    %       ActiveSetMethod        - Method used to select the active set.
    %       ActiveSetSize          - Size of the active set.
    %       IsActiveSetVector      - Logical vector marking the active set.
    %       LogLikelihood          - Maximized approximate marginal log likelihood.
    %       ActiveSetHistory       - History of active set selection for sparse methods.
    %       RowsUsed               - Logical index for rows used in fit.
    %       HyperparameterOptimizationResults - Results of optional hyperparameter optimization.
    %
    %   RegressionEV methods:
    %       compact                - Compact this model.
    %       criterion              - Criterion of this model for comparison.
    %       loss                   - Regression loss.
    %       predict                - Predicted response of this model.
    %       resubLoss              - Resubstitution regression loss.
    %       resubPredict           - Resubstitution predictions.
    %
    %   ALGORITHMS:
    %   Laplace Approximation:
    %     Rasmussen & Williams (2006), "Gaussian Processes for Machine Learning"
    %     Section 3.4: Laplace Approximation
    %
    %   Expectation Propagation:
    %     Minka (2001), "Expectation Propagation for approximate Bayesian inference"
    %     Rasmussen & Williams (2006), Section 3.6
    %
    %   Extreme Value Likelihood:
    %     The observed response is modeled with a generalized extreme value
    %     distribution whose location parameter is the latent GP value and whose
    %     scale and shape parameters are Sigma and Shape, respectively.
    %
    %   Kernel Functions:
    %     Standard stationary kernels including squared exponential (RBF),
    %     Matérn family (3/2, 5/2), and rational quadratic.
    %
    %   REFERENCES:
    %   [1] Rasmussen, C. E., & Williams, C. K. (2006). Gaussian processes
    %       for machine learning. MIT press.
    %   [2] Minka, T. P. (2001). Expectation propagation for approximate
    %       Bayesian inference. UAI.
    %   [3] Coles, S. (2001). An introduction to statistical modeling of
    %       extreme values. Springer.
    %
    %   See also: fitrev

    properties (Constant, Access=private)
        DEFAULT_ACTIVE_SET_SIZE = 2000
        MAX_FLOAT = 1e10
        MIN_JITTER = 1e-10
        MAX_JITTER = 1e-6
        DEFAULT_MAX_ITER = 50
        DEFAULT_TOL = 1e-6
        DEFAULT_DAMPING = 0.8
        DEFAULT_GH_POINTS = 20
        GEV_ZERO_TOL = 1e-8
        SHAPE_BOUND = 0.95
        EULER_MASCHERONI = 0.5772156649015329
    end

    properties
        % ---- Data / bookkeeping (fitrgp-like public properties) ----
        NumObservations
        X
        Y
        W
        ModelParameters
        PredictorNames
        ExpandedPredictorNames
        ResponseName
        ResponseTransform

        % ---- Data / bookkeeping (fitrgp-like public properties) ----
        KernelFunction
        KernelInformation
        BasisFunction
        Beta
        Sigma
        Shape
        Lambda

        % ---- Standardization stats (fitrgp-like public properties) ----
        PredictorLocation
        PredictorScale

        % ---- Posterior representation hooks (fitrgp-like public properties) ----
        Alpha
        ActiveSetVectors

        % ---- Training/inference options and results (fitrgp-like public properties) ----
        Inference                 % 'Laplace' or 'EP' (internal name), derived from user's 'Inference'
        FitMethod
        PredictMethod
        ActiveSetSize
        ActiveSetMethod
        IsActiveSetVector
        LogLikelihood
        ActiveSetHistory
        RowsUsed
        HyperparameterOptimizationResults
    end

    properties (Access=private)
        % ---- Internal caches not meant to be user-facing ----
        Posterior_
        InferenceOptions_
        Standardize_ logical
        DistanceMethod_ char
    end

    methods (Static)
        function [this, hyperOptResults] = fit(X, Y, varargin)
            %FIT Fit a Gaussian Process Extreme Value (EV) Regression model.
            %
            % This static constructor-style method matches `RegressionEV.fit`
            % conventions (called by `fitrev`).

            RegressionEV.validateInputs(X, Y);
            args = RegressionEV.parseFitArgs(X, Y, varargin{:});

            % Convert table inputs, extract predictor names, normalize response, etc.
            [Xraw, y, w, meta] = RegressionEV.prepareData(args);

            % Create the model object and fill public, fitrgp-like properties.
            this = RegressionEV();
            this.NumObservations = size(Xraw,1);
            this.X = Xraw;
            this.Y = y;
            this.W = w;
            this.PredictorNames = meta.PredictorNames;
            this.ExpandedPredictorNames = meta.PredictorNames;
            this.ResponseName = meta.ResponseName;
            this.ResponseTransform = args.ResponseTransform;
            this.RowsUsed = true(size(Xraw,1),1);

            this.KernelFunction = args.KernelFunction;
            this.KernelInformation = struct('Name', args.KernelFunction, ...
                'KernelParameters', args.KernelParameters);
            this.BasisFunction = args.BasisFunction;
            this.Beta = [];
            this.Sigma = args.Sigma;
            this.Shape = args.Shape;
            this.Lambda = args.Lambda;

            % Derived inference selections:
            %   args.InferenceMethod is 'Laplace' or 'EP'
            this.Inference = args.InferenceMethod;
            this.FitMethod = args.FitMethod;
            this.PredictMethod = args.PredictMethod;
            this.ActiveSetSize = args.ActiveSetSize;
            this.ActiveSetMethod = args.ActiveSetMethod;

            this.InferenceOptions_ = args.InferenceOptions;
            this.Standardize_ = args.Standardize;
            this.DistanceMethod_ = char(string(args.DistanceMethod));

            hyperOptResults = [];

            % ---- Bayesopt Hyperparameter Optimization ----
            % ---- Optional: bayesopt outer loop ----
            % If enabled, bayesopt will select some hyperparameters (e.g., Lambda,
            % KernelFunction, Standardize...) and then we refit at the optimum.
            if ~strcmpi(string(args.OptimizeHyperparameters), "none")
                [bestArgs, hyperOptResults] = RegressionEV.runBayesopt(Xraw, y, w, meta, args);
                args = bestArgs;

                % Update key model-visible selections to the optimized choices.
                this.HyperparameterOptimizationResults = hyperOptResults;
                this.KernelFunction = args.KernelFunction;
                this.BasisFunction = args.BasisFunction;
                this.Standardize_ = args.Standardize;
                this.Inference = args.InferenceMethod;
            end

            % ---- Standardization ----
            % Keep this.X raw for user parity; do math on standardized copy.
            if this.Standardize_
                [Xstd, mu, sig] = RegressionEV.standardizeX(Xraw);
                this.PredictorLocation = mu;
                this.PredictorScale = sig;
            else
                Xstd = Xraw;
                this.PredictorLocation = zeros(1,size(Xraw,2));
                this.PredictorScale = ones(1,size(Xraw,2));
            end

            % ---- Active set selection ----
            % This provides a compute-limiting subset of training points used for
            % posterior inference and prediction (similar spirit to sparse methods).
            [XaStd, ya, wa, isActive] = RegressionEV.selectActiveSet(Xstd, y, w, args);
            this.IsActiveSetVector = isActive;
            this.ActiveSetVectors = Xstd;

            % ---- Basis function setup ----
            % Explicit basis mean: m(x)=H(x)*Beta. This matches fitrgp style.
            Ha = RegressionEV.basisMatrix(XaStd, args.BasisFunction);
            beta0 = RegressionEV.ensureBetaSize(args.Beta, size(Ha,2));
            this.Beta = beta0;

            % ---- FitMethod='none' => no hyperparameter estimation ----
            % We only run inference with the provided initial parameters.
            if strcmpi(string(args.FitMethod), "none")
                [post, ll] = RegressionEV.runInferenceFromArgs(args, XaStd, ya, wa, Ha, beta0);
                this.Posterior_ = post;
                this.LogLikelihood = ll;
                this.Alpha = post.alpha;
                this.ModelParameters = RegressionEV.buildModelParameters(args, this);
            else
                % ---- Local evidence maximization ----
                % Optimize kernel parameters / Beta / Lambda by maximizing the approximate
                % marginal log likelihood (Laplace/EP evidence approximation).
                args = RegressionEV.optimizeLocally(args, XaStd, ya, wa, Ha);

                % Copy optimized parameters back to public properties.
                this.Lambda = args.Lambda;
                this.Sigma = args.Sigma;
                this.Shape = args.Shape;
                this.KernelFunction = args.KernelFunction;
                this.BasisFunction = args.BasisFunction;
                this.Beta = args.Beta(:);
                this.KernelInformation.KernelParameters = args.KernelParameters;

                % ---- Final inference with optimized parameters ----
                [post, ll] = RegressionEV.runInferenceFromArgs(args, XaStd, ya, wa, Ha, this.Beta);
                this.Posterior_ = post;
                this.LogLikelihood = ll;
                this.Alpha = post.alpha;

                this.ModelParameters = RegressionEV.buildModelParameters(args, this);
            end
        end
    end

    methods
        function compacted = compact(this)
            %COMPACT Create a compact version of this model.
            %
            % Similar in spirit to `compact(RegressionGP)`:
            % removes training data arrays to reduce memory footprint while
            % keeping enough information to predict.
            compacted = this;
            compacted.X = [];
            compacted.Y = [];
            compacted.W = [];
            compacted.NumObservations = 0;
            compacted.RowsUsed = [];
            compacted.HyperparameterOptimizationResults = [];
        end

        function [yhat, ysd, ci] = predict(this, Xnew, varargin)
            %PREDICT Predict GEV response mean and standard deviation.
            %
            %   [YHAT, YSD] = predict(MODEL, XNEW)
            %   returns:
            %     YHAT : N-by-1 predicted response means
            %     YSD  : N-by-1 predicted response standard deviations
            %
            %   [YHAT, YSD, CI] also returns approximate confidence intervals.
            %   CI is N-by-2 with columns [lower, upper].
            %
            %   If the GEV mean or variance is undefined (Shape >= 1 or
            %   Shape >= 0.5 respectively), YHAT returns the latent location
            %   mean and YSD returns NaN.
            %
            % Name-value:
            %   'Alpha' : significance level in (0,1). Default 0.05 -> 95% CI.

            if size(Xnew,2) ~= size(this.ActiveSetVectors,2)
                error('RegressionEV:SizeMismatch', ...
                    'Number of columns in Xnew (%d) must equal columns of X (%d).', ...
                    size(Xnew,2), size(this.ActiveSetVectors,2));
            end

            ip = inputParser;
            ip.addParameter('Alpha', 0.05, @(a) isnumeric(a) && isscalar(a) && a > 0 && a < 1);
            ip.parse(varargin{:});
            alphaLevel = ip.Results.Alpha;

            % Accept either numeric matrix or table; for table align columns by PredictorNames.
            Xq = RegressionEV.toMatrixPredictors(Xnew, this.PredictorNames);

            % Apply training standardization parameters
            XqStd = (Xq - this.PredictorLocation) ./ this.PredictorScale;

            % Get active set vectors used during training
            XaStd = this.ActiveSetVectors(this.IsActiveSetVector,:);

            % Cross-covariances K(X_active, X_new) and diagonal of K(X_new, X_new)
            args = this.ModelParameters;
            Kxs = RegressionEV.kernelMatrix(XaStd, XqStd, args);
            Kss = RegressionEV.kernelDiag(XqStd, args);

            % Mean function m(x)=H(x)*Beta
            Hq = RegressionEV.basisMatrix(XqStd, this.BasisFunction);
            mq = Hq * this.Beta;

            switch lower(this.Inference)
                case 'ep'
                    [muPred, varLatent] = RegressionEV.predictLatentEP(this.Posterior_, Kxs, Kss, mq);
                otherwise
                    [muPred, varLatent] = RegressionEV.predictLatentLaplace(this.Posterior_, Kxs, Kss, mq);
            end

            meanShift = RegressionEV.gevMeanShift(this.Sigma, this.Shape);
            noiseVar = RegressionEV.gevVariance(this.Sigma, this.Shape);

            if isnan(meanShift)
                warning('RegressionEV:UndefinedMean', ...
                    'GEV mean is undefined for Shape >= 1. Returning latent location mean.');
                yhat = muPred;
            else
                yhat = muPred + meanShift;
            end

            if isnan(noiseVar)
                ysd = NaN(size(muPred));
            else
                ysd = sqrt(max(varLatent + noiseVar, 0));
            end

            % Optional user transformation of probability outputs
            if ~isempty(this.ResponseTransform)
                yhat = this.ResponseTransform(yhat);
            end

            % Optional confidence interval (approximate)
            ci = [];
            if nargout > 2
                if any(isnan(ysd))
                    ci = NaN(numel(yhat), 2);
                else
                    z = RegressionEV.norminvSafe(1 - alphaLevel/2);
                    ci = [yhat - z*ysd, yhat + z*ysd];
                end
            end
        end

        function L = loss(this, X, Y, varargin)
            %LOSS Compute regression loss on (X,Y).
            %
            % Name-value:
            %   'LossFun':
            %       - 'mse'            (default) Mean Squared Error
            %       - 'rmse'           Root Mean Squared Error
            %       - 'mae'            Mean Absolute Error
            %       - 'nloglikelihood' Negative Gaussian log likelihood
            %       - 'smse'           Standardized Mean Squared Error
            %       - 'msll'           Mean Standardized Log Loss
            %       - function handle: @(Ytrue, Ypred, Ysd) -> scalar
            %   'Weights': vector of weights (default ones(size(Y,1),1))

            RegressionEV.validateInputs(X, Y);

            ip = inputParser;
            ip.addParameter('LossFun', 'mse');
            ip.addParameter('Weights', ones(size(Y,1),1));
            ip.parse(varargin{:});
            lossFun = ip.Results.LossFun;
            w = ip.Results.Weights(:);

            % Validate Y
            yTrue = RegressionEV.normalizeY(Y);
            [yPred, ysd] = this.predict(X);

            if isa(lossFun, 'function_handle')
                L = lossFun(yTrue, yPred, ysd);
                return;
            end

            switch lower(string(lossFun))
                case "mse"
                    % Mean Squared Error
                    L = sum(w .* (yPred - yTrue).^2) / sum(w);
                case "rmse"
                    % Root Mean Squared Error
                    L = sqrt(sum(w .* (yPred - yTrue).^2) / sum(w));
                case "mae"
                    % Mean Absolute Error
                    L = sum(w .* abs(yPred - yTrue)) / sum(w);
                case "nloglikelihood"
                    % Negative Log Likelihood
                    sigma2 = ysd.^2;
                    sigma2 = max(sigma2, eps);
                    sigma2(~isfinite(sigma2)) = NaN;
                    L = 0.5 * sum(w .* (log(2*pi*sigma2) + ((yTrue - yPred).^2) ./ sigma2), 'omitnan');
                case "smse"
                    % Standardized Mean Squared Error
                    mu0 = sum(w .* yTrue) / sum(w);
                    denom = sum(w .* (yTrue - mu0).^2);
                    L = sum(w .* (yPred - yTrue).^2) / max(eps, denom);
                case "msll"
                    % Mean Standardized Log Loss
                    baseMu = sum(w .* yTrue) / sum(w);
                    baseVar = sum(w .* (yTrue - baseMu).^2) / sum(w);
                    sigma2 = ysd.^2;
                    sigma2 = max(sigma2, eps);
                    sigma2(~isfinite(sigma2)) = NaN;
                    nlpp  = 0.5 * (log(2*pi*sigma2) + ((yTrue - yPred).^2) ./ sigma2);
                    nlpp0 = 0.5 * (log(2*pi*baseVar) + ((yTrue - baseMu).^2) ./ baseVar);
                    L = sum(w .* (nlpp - nlpp0), 'omitnan') / sum(w);
                otherwise
                    error('RegressionEV:BadLossFun', 'Unsupported LossFun: %s', string(lossFun));
            end
        end

        function L = resubLoss(this, varargin)
            %RESUBLOSS Resubstitution loss (evaluate on training data).
            if ~isempty(this.X) && ~isempty(this.Y)
                L = this.loss(this.X, this.Y, 'Weights', this.W, varargin{:});
            else
                error('RegressionEV:Compact', 'Unsupported for compacted model');
            end
        end

        function [yhat, ysd, ci] = resubPredict(this, varargin)
            %RESUBPREDICT Resubstitution predictions (predict on training data).
            if ~isempty(this.X)
                [yhat, ysd, ci] = this.predict(this.X, varargin{:});
            else
                error('RegressionEV:Compact', 'Unsupported for compacted model');
            end
        end

        function C = criterion(this, varargin)
            %CRITERION Compute model selection criterion (evaluated on training data).
            %
            % Name-value:
            %   'CriterionFun': 'GCV' (default) | 'AIC' | 'AICc' | 'BIC' |
            %                   'CAIC' | 'LOOCV' | 'WAIC'

            if isempty(this.X) || isempty(this.Y)
                error('RegressionEV:Compact', 'Unsupported for compacted model');
            end

            ip = inputParser;
            ip.addParameter('CriterionFun', 'GCV');
            ip.parse(varargin{:});
            criterionFun = ip.Results.CriterionFun;

            k_hyper = numel(this.KernelInformation.KernelParameters) + numel(this.Beta) + 2;
            n_obs = this.NumObservations;
            n_act = sum(this.IsActiveSetVector);
            post = this.Posterior_;

            % Predict posterior probability
            yTrue = this.Y(:);
            w = this.W(:);
            [yPred, ysd] = this.predict(this.X);
            sigma2 = max(ysd.^2, eps);
            sigma2(~isfinite(sigma2)) = NaN;

            % Log Likelihood
            L = -0.5 * (log(2*pi*sigma2) + ((yTrue - yPred).^2) ./ sigma2);

            switch lower(string(criterionFun))
                case "aic"
                    % Akaike Information Criterion
                    C = 2*k_hyper - 2*sum(w .* L, 'omitnan');
                case "aicc"
                    % Corrected Akaike Information Criterion
                    C = 2*k_hyper - 2*sum(w .* L, 'omitnan');
                    C = C + (2*k_hyper*(k_hyper + 1)) / max(1, n_obs - k_hyper - 1);
                case "bic"
                    % Bayesian Information Criterion
                    C = k_hyper*log(n_obs) - 2*sum(w .* L, 'omitnan');
                case "caic"
                    % Consistent Akaike Information Criterion
                    C = k_hyper*(log(n_obs) + 1) - 2*sum(w .* L, 'omitnan');
                case "gcv"
                    % Generalized Cross-Validation
                    % Compute Effective Degrees of Freedom
                    if strcmpi(this.Inference, 'ep')
                        effDoF = max(0, sum(diag(post.Sigma) .* post.tau(:)));
                    else
                        effDoF = max(0, n_act - sum(sum((post.L \ speye(n_act)).^2)));
                    end
                    rss = sum(w .* (yPred - yTrue).^2);

                    % Calculate Generalized Cross-Validation
                    C = rss / (1 - effDoF / n_obs)^2;
                case "loocv"
                    % Leave-One-Out Cross-Validation
                    h = zeros(n_obs, 1);

                    % Compute leverage (diagonal of hat matrix)
                    if strcmpi(this.Inference, 'ep')
                        % For EP, leverage is diag(Sigma) * tau
                        h(this.IsActiveSetVector) = max(0, diag(post.Sigma) .* post.tau(:));
                    else
                        % For Laplace, leverage is derived from the diagonal of inv(B)
                        % where B = post.L * post.L'
                        h(this.IsActiveSetVector) = max(0, 1 - sum((post.L \ speye(n_act)).^2, 1)');
                    end
                    resid = (yPred - yTrue) ./ max(sqrt(eps), 1 - h);

                    % Calculate LOOCV error functionally via the pseudo-response inverse scalar
                    C = sum(w .* resid.^2) / sum(w);
                case "waic"
                    % Widely Applicable Information Criterion

                    % approx Gaussian quadrature
                    gw = [0.125 0.750 0.125];

                    % Log-likelihood at each support point
                    LL = [L - 0.5, L, L - 0.5];

                    % Weighted mean log-likelihood
                    maxLL = max(LL, [], 2);
                    LPPD = maxLL + log(sum(gw .* exp(LL - maxLL), 2));

                    % Weighted variance
                    PWAIC = sum(gw .* (LL - sum(gw .* LL, 2)).^2, 2);

                    % WAIC
                    C = -2 * sum(w .* (LPPD - PWAIC), 'omitnan');
                otherwise
                    error('RegressionEV:BadCriterionFun', 'Unsupported CriterionFun: %s', string(criterionFun));
            end
        end
    end

    methods (Static, Access=private)
        function validateInputs(X, Y)
            % Validate input data for GP extreme value regression.

            % Check X
            if istable(X)
                Xtbl = X;
                if (ischar(Y) || isstring(Y)) && isscalar(string(Y)) && ...
                        ismember(string(Y), string(X.Properties.VariableNames))
                    Xtbl.(char(string(Y))) = [];
                    Y = X.(char(string(Y)));
                end
                numVars = varfun(@(v) isnumeric(v) || islogical(v), Xtbl, 'OutputFormat', 'uniform');
                X_numeric = table2array(Xtbl(:, numVars));
            else
                X_numeric = X;
            end

            if isempty(X_numeric)
                error('RegressionEV:EmptyX', 'Input X cannot be empty.');
            end

            if ~isnumeric(X_numeric) && ~islogical(X_numeric)
                error('RegressionEV:InvalidX', 'X must be numeric or logical.');
            end

            if any(isnan(X_numeric(:))) || any(isinf(X_numeric(:)))
                error('RegressionEV:InvalidX', 'X contains NaN or Inf values.');
            end

            % Check Y
            if isempty(Y)
                error('RegressionEV:EmptyY', 'Response Y cannot be empty.');
            end

            % Check size compatibility
            if size(X_numeric, 1) ~= size(Y,1)
                error('RegressionEV:SizeMismatch', ...
                    'Number of rows in X (%d) must equal rows of Y (%d).', ...
                    size(X_numeric, 1), size(Y,1));
            end
        end

        function validateKernelParameters(theta, kernelFunction, d)
            % Validate that kernel parameters have the correct count and are positive.

            kf = lower(string(kernelFunction));

            % Expected parameter counts
            if startsWith(kf, "ard")
                if contains(kf, "rationalquadratic")
                    expected = d + 2; % [ell1,...,elld, sf, alpha]
                else
                    expected = d + 1; % [ell1,...,elld, sf]
                end
            else
                if contains(kf, "rationalquadratic")
                    expected = 3; % [ell, sf, alpha]
                else
                    expected = 2; % [ell, sf]
                end
            end

            if numel(theta) ~= expected
                % Warning only, as some custom calls might differ
                warning('RegressionEV:KernelParamMismatch', ...
                    'Kernel ''%s'' with d=%d typically requires %d parameters, got %d.', ...
                    kf, d, expected, numel(theta));
            end

            % All parameters must be positive
            if any(theta <= 0)
                error('RegressionEV:BadKernelParams', ...
                    'All kernel parameters must be positive.');
            end

            % Warn about extreme values
            if any(theta < 1e-6) || any(theta > 1e6)
                warning('RegressionEV:ExtremeKernelParams', ...
                    'Kernel parameters contain extreme values (< 1e-6 or > 1e6). This may cause numerical issues.');
            end
        end

        function args = parseFitArgs(X, Y, varargin)
            % Parse name/value pairs and create fitrgp-like defaults.

            ip = inputParser;
            ip.FunctionName = 'fitrev';

            % ----- Model structure -----
            ip.addParameter('KernelFunction', 'squaredexponential');
            ip.addParameter('KernelParameters', [], @(v) isempty(v) || isnumeric(v));
            ip.addParameter('ConstantKernelParameters', [], @(v) isempty(v) || islogical(v));
            ip.addParameter('DistanceMethod', 'fast', @(s) any(strcmpi(string(s), ["fast","accurate"])));
            ip.addParameter('BasisFunction', 'constant');
            ip.addParameter('Beta', [], @(v) isempty(v) || isnumeric(v));

            ip.addParameter('Sigma', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v) && v > 0));
            ip.addParameter('ConstantSigma', true, @(b) islogical(b) && isscalar(b));

            ip.addParameter('Shape', 0, @(v) isempty(v) || (isnumeric(v) && isscalar(v)));
            ip.addParameter('ConstantShape', true, @(b) islogical(b) && isscalar(b));

            ip.addParameter('Lambda', 0, @(v) isempty(v) || (isnumeric(v) && isscalar(v) && v >= 0));
            ip.addParameter('ConstantLambda', true, @(b) islogical(b) && isscalar(b));

            % ----- Inference choice (user-facing) -----
            ip.addParameter('Inference', 'Laplace', @(s) ischar(s) || isstring(s));
            ip.addParameter('InferenceOptions', struct(), @(s) isstruct(s));

            % ----- Approximation / compute knobs -----
            ip.addParameter('FitMethod', [], @(s) isempty(s) || ischar(s) || isstring(s));
            ip.addParameter('PredictMethod', [], @(s) isempty(s) || ischar(s) || isstring(s));
            ip.addParameter('ActiveSet', [], @(v) isempty(v) || islogical(v) || isnumeric(v));
            ip.addParameter('ActiveSetSize', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v) && v >= 1));
            ip.addParameter('ActiveSetMethod', [], @(s) isempty(s) || ischar(s) || isstring(s));

            % Standardize predictors (centering/scaling)
            ip.addParameter('Standardize', false, @(b) islogical(b) || isscalar(b));

            % Diagnostics
            ip.addParameter('Verbose', 0, @(v) isnumeric(v) && isscalar(v));

            % ----- Local optimizer (fitrgp-like) -----
            ip.addParameter('Optimizer', 'quasinewton', @(s) ischar(s) || isstring(s));
            ip.addParameter('OptimizerOptions', [], @(o) isempty(o) || isstruct(o) || isobject(o));

            % ----- Names / transforms -----
            ip.addParameter('PredictorNames', [], @(v) isempty(v) || iscellstr(v) || isstring(v));
            ip.addParameter('ResponseName', 'Y', @(s) ischar(s) || isstring(s));
            ip.addParameter('ResponseTransform', [], @(f) isempty(f) || isa(f, 'function_handle'));

            % ----- bayesopt hyperparameter optimization -----
            ip.addParameter('OptimizeHyperparameters', 'none');
            ip.addParameter('HyperparameterOptimizationOptions', struct(), @(s) isstruct(s));

            % Cross validation options for hyperparameter optimization
            ip.addParameter('Holdout', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v) && v>0 && v<1));
            ip.addParameter('KFold', [], @(v) isempty(v) || (isnumeric(v) && isscalar(v) && mod(v,1)==0));
            ip.addParameter('Leaveout', [], @(v) isempty(v) || (islogical(v) && isscalar(v)) || (isnumeric(v) && isscalar(v) && (v==0 || v==1)));

            % Optional weights
            ip.addParameter('Weights', [], @(v) isempty(v) || (isnumeric(v) && isvector(v)));

            ip.parse(varargin{:});
            args = ip.Results;
            args.X = X;
            args.Y = Y;

            % Check CV options
            cvOpts = [~isempty(args.Holdout), ~isempty(args.KFold), ~isempty(args.Leaveout)];
            if sum(cvOpts) > 1
                error('RegressionEV:TooManyCVOptions', ...
                    'You can only specify one of ''Holdout'', ''KFold'', or ''Leaveout''.');
            end

            % Convert standardize to logical
            args.Standardize = logical(args.Standardize);

            % Default FitMethod/PredictMethod heuristic similar to fitrgp
            n = size(RegressionEV.peekN(X), 1);
            if isempty(args.FitMethod)
                args.FitMethod = 'exact';
            end
            if isempty(args.PredictMethod)
                args.PredictMethod = 'exact';
            end

            % Default active set size/method
            if isempty(args.ActiveSetSize)
                args.ActiveSetSize = min(RegressionEV.DEFAULT_ACTIVE_SET_SIZE, n);
            end
            if isempty(args.ActiveSetMethod)
                args.ActiveSetMethod = 'random';
            end

            % Inference mapping requested in prompt
            infStr = lower(string(args.Inference));
            if infStr == "ep"
                args.InferenceMethod = 'EP';
            else
                args.InferenceMethod = 'Laplace';
            end

            % Default Lambda
            if isempty(args.Lambda)
                args.Lambda = 0;
            end
            if isempty(args.Shape)
                args.Shape = 0;
            end
            if isempty(args.Sigma)
                y0 = Y;
                if istable(X) && (ischar(Y) || isstring(Y)) && ...
                        ismember(string(Y), string(X.Properties.VariableNames))
                    y0 = X.(char(string(Y)));
                end
                y0 = RegressionEV.normalizeY(y0);
                s0 = std(y0, 0, 1, 'omitnan');
                args.Sigma = max(0.1*s0, 1e-3);
            end

            % Inference iteration defaults
            opt = args.InferenceOptions;
            if ~isfield(opt, 'MaxIter')
                opt.MaxIter = RegressionEV.DEFAULT_MAX_ITER;
            end
            if ~isfield(opt, 'Tol')
                opt.Tol = RegressionEV.DEFAULT_TOL;
            end
            if ~isfield(opt, 'Damping')
                opt.Damping = RegressionEV.DEFAULT_DAMPING;
            end
            if ~isfield(opt, 'NumGHPoints')
                opt.NumGHPoints = RegressionEV.DEFAULT_GH_POINTS;
            end
            args.InferenceOptions = opt;
        end

        function [Xmat, y, w, meta] = prepareData(args)
            % Convert X to numeric matrix, determine names, normalize Y.
            X = args.X;
            Y = args.Y;

            meta = struct();

            if istable(X)
                % If Y is a variable name, pull it from the table.
                if (ischar(Y) || isstring(Y)) && isscalar(string(Y))
                    yName = string(Y);
                    meta.ResponseName = yName;
                    yVec = X.(char(yName));
                    Xtbl = X;
                    Xtbl.(char(yName)) = [];
                else
                    yVec = Y;
                    Xtbl = X;
                    meta.ResponseName = string(args.ResponseName);
                end

                if isempty(args.PredictorNames)
                    predictorNames = string(Xtbl.Properties.VariableNames);
                else
                    predictorNames = string(args.PredictorNames);
                    if ~all(ismember(predictorNames, string(Xtbl.Properties.VariableNames)))
                        error('RegressionEV:BadPredictorNames', ...
                            'PredictorNames must be a subset of the table variable names.');
                    end
                    Xtbl = Xtbl(:, cellstr(predictorNames));
                end

                Xmat = table2array(Xtbl);
                meta.PredictorNames = predictorNames;
                y = RegressionEV.normalizeY(yVec);
            else
                % Matrix input
                Xmat = X;
                meta.ResponseName = string(args.ResponseName);
                if isempty(args.PredictorNames)
                    p = size(Xmat,2);
                    meta.PredictorNames = "x" + (1:p);
                else
                    meta.PredictorNames = string(args.PredictorNames);
                end
                y = RegressionEV.normalizeY(Y);
            end

            if isempty(args.Weights)
                w = ones(size(Xmat,1),1);
            else
                w = args.Weights(:);
            end
        end

        function y = normalizeY(Y)
            % Validate and convert Y to a column vector.
            if istable(Y)
                error('RegressionEV:BadY', 'Y must be numeric.');
            end
            if ~isnumeric(Y) || ~isvector(Y)
                error('RegressionEV:BadY', 'Y must be a numeric vector.');
            end
            y = Y(:);
            if any(isnan(y)) || any(isinf(y))
                error('RegressionEV:BadY', 'Y contains NaN or Inf values.');
            end
        end

        function Xq = toMatrixPredictors(Xnew, predictorNames)
            % Convert predictors to matrix. For tables, respect PredictorNames order.
            if istable(Xnew)
                Xq = table2array(Xnew(:, cellstr(predictorNames)));
            else
                Xq = Xnew;
            end
        end

        function [Xs, mu, sig] = standardizeX(X)
            % Standardize each predictor column to zero mean and unit std.
            mu = mean(X, 1, 'omitnan');
            sig = std(X, 0, 1, 'omitnan');
            sig(sig == 0) = 1;
            Xs = (X - mu) ./ sig;
        end

        function beta = ensureBetaSize(beta, p)
            % Ensure Beta is the correct length given basis matrix size.
            if isempty(beta)
                beta = zeros(p,1);
            else
                beta = beta(:);
                if numel(beta) ~= p
                    error('RegressionEV:BadBeta', 'Beta must be %d-by-1.', p);
                end
            end
        end

        function H = basisMatrix(X, basis)
            % Construct explicit basis matrix H given BasisFunction.
            if isa(basis, 'function_handle')
                H = basis(X);
                return;
            end
            b = lower(string(basis));
            n = size(X,1);
            switch b
                case "none"
                    H = zeros(n,0);
                case "constant"
                    H = ones(n,1);
                case "linear"
                    H = [ones(n,1), X];
                case "purequadratic"
                    H = [ones(n,1), X, X.^2];
                otherwise
                    error('RegressionEV:BadBasis', 'Unsupported BasisFunction: %s', b);
            end
        end

        function [Xa, ya, wa, isActive] = selectActiveSet(X, y, w, args)
            % Choose active set indices for compute control.
            %
            % User can supply `ActiveSet` explicitly. Otherwise we choose M points
            % according to `ActiveSetMethod`.

            n = size(X,1);

            if ~isempty(args.ActiveSet)
                as = args.ActiveSet;

                if islogical(as) || all(ismember(as(:), [0 1]))
                    isActive = logical(as(:));
                    if numel(isActive) ~= n
                        error('RegressionEV:ActiveSetSizeMismatch', ...
                            'Number of elements in ActiveSet (%d) must equal rows of X (%d).', ...
                            numel(isActive), n);
                    end
                    if ~any(isActive)
                        error('RegressionEV:ActiveSetEmpty', ...
                            'ActiveSet must contain at least one selected observation.');
                    end
                else
                    as = round(as(:));
                    if numel(as) > n
                        error('RegressionEV:ActiveSetSizeMismatch', ...
                            'Number of ActiveSet indices (%d) must be less than or equal to rows of X (%d).', ...
                            numel(as), n);
                    end
                    if numel(as) ~= numel(unique(as))
                        error('RegressionEV:ActiveSetNotUnique', ...
                            'ActiveSet must contain unique numbers.');
                    end
                    if any(as > n)
                        error('RegressionEV:ActiveSetBadIdxs', ...
                            'ActiveSet indices must be lower than or equal to rows of X (%d).', n);
                    end
                    if any(as <= 0)
                        error('RegressionEV:ActiveSetPosInts', ...
                            'ActiveSet indices must be positive integers.');
                    end
                    isActive = false(n,1);
                    isActive(as) = true;
                end
            else
                m = min(args.ActiveSetSize, n);
                method = lower(string(args.ActiveSetMethod));
                switch method
                    case "first"
                        idx = (1:m)';
                    case "last"
                        idx = (n-m+1:n)';
                    case "random"
                        idx = randperm(n, m)';
                    case "qr"
                        % QR selection
                        Xc = X - mean(X,1);
                        [~,~,p] = qr(Xc', 'econ', 'vector');
                        idx = p(1:m);
                    otherwise
                        error('RegressionEV:BadActiveSetMethod', 'Unsupported ActiveSetMethod: %s', method);
                end
                isActive = false(n,1);
                isActive(idx) = true;
            end

            Xa = X(isActive,:);
            ya = y(isActive,:);
            wa = w(isActive,:);
        end

        function argsOut = optimizeLocally(argsIn, XaStd, ya, wa, Ha)
            % Maximize approximate marginal log likelihood over free hyperparameters.

            argsOut = argsIn;

            [x0, packInfo] = RegressionEV.packFreeParameters(argsIn, Ha);
            if isempty(x0)
                return;
            end

            obj = @(x) RegressionEV.negLogEvidenceFromFree(x, packInfo, argsIn, XaStd, ya, wa, Ha);
            optName = lower(string(argsIn.Optimizer));
            [xBest, fBest] = RegressionEV.runLocalOptimizer(obj, x0, optName, argsIn.OptimizerOptions, argsIn.Verbose);
            argsOut = RegressionEV.unpackFreeParameters(xBest, packInfo, argsIn, Ha);

            % If optimization looks failed, keep initial args
            if ~isfinite(fBest)
                argsOut = argsIn;
            end
        end

        function [x0, info] = packFreeParameters(args, H)
            % Pack free model parameters into an unconstrained vector.
            %   - Positive parameters (kernel, Sigma, Lambda) -> log space.
            %   - Shape -> atanh-bounded space via packShape.
            %   - Beta  -> linear space.

            theta = args.KernelParameters;
            if isempty(theta)
                theta = RegressionEV.defaultKernelParameters(args.KernelFunction, size(args.X,2));
            end
            theta = theta(:);

            ck = args.ConstantKernelParameters;
            if isempty(ck)
                ck = false(numel(theta),1);
            else
                ck = ck(:);
                if numel(ck) ~= numel(theta)
                    error('RegressionEV:BadConstantKernelParameters', ...
                        'ConstantKernelParameters must match length of KernelParameters.');
                end
            end

            freeThetaIdx = find(~ck);
            xTheta0 = log(theta(freeThetaIdx));

            beta0 = RegressionEV.ensureBetaSize(args.Beta, size(H,2));
            xBeta0 = beta0(:);

            if args.ConstantSigma
                xSigma0 = [];
            else
                xSigma0 = log(args.Sigma);
            end

            if args.ConstantShape
                xShape0 = [];
            else
                xShape0 = RegressionEV.packShape(args.Shape);
            end

            if args.ConstantLambda
                xLambda0 = [];
            else
                xLambda0 = log(max(args.Lambda, RegressionEV.MIN_JITTER));
            end

            x0 = [xTheta0; xBeta0; xSigma0; xShape0; xLambda0];

            info = struct();
            info.theta0 = theta;
            info.freeThetaIdx = freeThetaIdx;
            info.betaSize = size(H,2);
            info.hasSigma = ~args.ConstantSigma;
            info.hasShape = ~args.ConstantShape;
            info.hasLambda = ~args.ConstantLambda;
        end

        function args = unpackFreeParameters(x, info, args, H)
            % Unpack unconstrained optimization vector back into structured args.
            theta = info.theta0;
            k = numel(info.freeThetaIdx);
            if k > 0
                theta(info.freeThetaIdx) = exp(x(1:k));
            end

            betaStart = k + 1;
            betaEnd = betaStart + info.betaSize - 1;
            beta = x(betaStart:betaEnd);

            pos = betaEnd;
            if info.hasSigma
                pos = pos + 1;
                sigma = exp(x(pos));
            else
                sigma = args.Sigma;
            end

            if info.hasShape
                pos = pos + 1;
                shape = RegressionEV.unpackShape(x(pos));
            else
                shape = args.Shape;
            end

            if info.hasLambda
                pos = pos + 1;
                lambda = exp(x(pos));
            else
                lambda = args.Lambda;
            end

            args.KernelParameters = theta;
            args.Beta = beta(:);
            args.Sigma = sigma;
            args.Shape = shape;
            args.Lambda = lambda;

            RegressionEV.ensureBetaSize(args.Beta, size(H,2));
        end

        function x = packShape(shape)
            % Map shape in (-SHAPE_BOUND, SHAPE_BOUND) to unconstrained real via atanh.
            b = RegressionEV.SHAPE_BOUND;
            s = min(max(shape, -0.999*b), 0.999*b);
            x = atanh(s / b);
        end

        function shape = unpackShape(x)
            % Inverse of packShape: map unconstrained real back to bounded shape.
            b = RegressionEV.SHAPE_BOUND;
            shape = b * tanh(x);
        end

        function nle = negLogEvidenceFromFree(x, info, args0, XaStd, ya, wa, Ha)
            % Negative approximate log evidence for use by local optimizer.
            args = RegressionEV.unpackFreeParameters(x, info, args0, Ha);
            beta = args.Beta(:);

            [~, ll] = RegressionEV.runInferenceFromArgs(args, XaStd, ya, wa, Ha, beta);
            if ~isfinite(ll)
                nle = RegressionEV.MAX_FLOAT;
            else
                nle = -ll;
            end
        end

        function [xBest, fBest] = runLocalOptimizer(obj, x0, optName, optOptions, verbose)
            % Run a chosen local optimizer.

            switch verbose
                case 0
                    dispOpt = 'off';
                case 1
                    dispOpt = 'iter';
                otherwise
                    dispOpt = 'iter-detailed';
            end

            switch optName
                case "fminsearch"
                    if isempty(optOptions)
                        optOptions = optimset('Display', dispOpt);
                    end
                    [xBest, fBest] = fminsearch(obj, x0, optOptions);

                case {"quasinewton","lbfgs","fminunc"}
                    if exist('fminunc', 'file') ~= 2
                        error('RegressionEV:NoFminunc', 'Optimizer fminunc requires Optimization Toolbox.');
                    end
                    if isempty(optOptions)
                        optOptions = optimoptions('fminunc', 'Algorithm', 'quasi-newton', 'Display', dispOpt);
                    else
                        try
                            if ~isa(optOptions, 'optim.options.Fminunc') && isstruct(optOptions)
                                optOptions = optimoptions('fminunc', optOptions);
                            end
                        catch
                        end
                    end
                    [xBest, fBest] = fminunc(obj, x0, optOptions);

                case "fmincon"
                    if exist('fmincon', 'file') ~= 2
                        error('RegressionEV:NoFmincon', 'Optimizer ''fmincon'' requires Optimization Toolbox.');
                    end
                    if isempty(optOptions)
                        optOptions = optimoptions('fmincon', 'Display', dispOpt);
                    else
                        try
                            if ~isa(optOptions, 'optim.options.Fmincon') && isstruct(optOptions)
                                optOptions = optimoptions('fmincon', optOptions);
                            end
                        catch
                        end
                    end
                    [xBest, fBest] = fmincon(obj, x0, [], [], [], [], [], [], [], optOptions);

                otherwise
                    error('RegressionEV:BadOptimizer', 'Unsupported Optimizer: %s', optName);
            end
        end

        function [post, ll] = runInferenceFromArgs(args, XaStd, ya, wa, Ha, beta)
            % Build kernel matrix and prior mean, then dispatch to Laplace or EP.
            m = Ha * beta;
            K = RegressionEV.kernelMatrix(XaStd, XaStd, args) + abs(args.Lambda) .* speye(size(XaStd,1));

            switch lower(args.InferenceMethod)
                case 'laplace'
                    [post, ll] = RegressionEV.inferLaplace(K, ya, wa, m, args.Sigma, args.Shape, args.InferenceOptions);
                case 'ep'
                    [post, ll] = RegressionEV.inferEP(K, ya, wa, m, args.Sigma, args.Shape, args.InferenceOptions);
                otherwise
                    error('RegressionEV:BadInference', 'InferenceMethod must be Laplace or EP.');
            end
        end

        function mp = buildModelParameters(args, this)
            % Build a struct summarizing the fitted model configuration.
            mp = struct( ...
                'KernelFunction', args.KernelFunction, ...
                'KernelParameters', args.KernelParameters, ...
                'BasisFunction', args.BasisFunction, ...
                'Beta', this.Beta, ...
                'Sigma', this.Sigma, ...
                'Shape', this.Shape, ...
                'Lambda', this.Lambda, ...
                'Inference', this.Inference, ...
                'Standardize', this.Standardize_, ...
                'FitMethod', this.FitMethod, ...
                'PredictMethod', this.PredictMethod, ...
                'ActiveSetSize', this.ActiveSetSize, ...
                'ActiveSetMethod', this.ActiveSetMethod, ...
                'DistanceMethod', this.DistanceMethod_, ...
                'Optimizer', args.Optimizer, ...
                'OptimizerOptions', args.OptimizerOptions, ...
                'ConstantSigma', args.ConstantSigma, ...
                'ConstantShape', args.ConstantShape, ...
                'ConstantLambda', args.ConstantLambda, ...
                'ConstantKernelParameters', args.ConstantKernelParameters);
        end

        function [bestArgs, results] = runBayesopt(Xraw, y, w, ~, args0)
            % Run bayesopt outer loop to select discrete/continuous hyperparameters.
            % Objective is negative approximate log evidence evaluated via CV.

            if exist('bayesopt','file') ~= 2
                error('RegressionEV:NoBayesopt', 'OptimizeHyperparameters requires bayesopt.');
            end

            hyp = string(args0.OptimizeHyperparameters);

            if strcmpi(hyp, "auto")
                hypList = ["Sigma","Shape"];
            elseif strcmpi(hyp, "all")
                hypList = ["BasisFunction","KernelFunction","KernelScale","Standardize","Sigma","Shape"];
            elseif isa(args0.OptimizeHyperparameters, 'optimizableVariable')
                vars = args0.OptimizeHyperparameters;
                hypList = string({vars.Name});
            else
                hypList = string(args0.OptimizeHyperparameters);
                if isscalar(hypList) && strcmpi(hypList, "none")
                    bestArgs = args0;
                    results = [];
                    return;
                end
            end

            if ~exist('vars','var')
                vars = RegressionEV.defaultOptimizableVariables(hypList);
            end

            bo = args0.HyperparameterOptimizationOptions;
            if ~isfield(bo,'MaxObjectiveEvaluations')
                bo.MaxObjectiveEvaluations = 30;
            end
            if ~isfield(bo,'Verbose')
                bo.Verbose = 0;
            end
            if ~isfield(bo,'IsObjectiveDeterministic')
                bo.IsObjectiveDeterministic = true;
            end
            if ~isfield(bo,'AcquisitionFunctionName')
                bo.AcquisitionFunctionName = 'expected-improvement-plus';
            end

            % Create CV partition
            n = size(Xraw, 1);
            if ~isempty(args0.Holdout)
                cvp = cvpartition(n, 'Holdout', args0.Holdout);
            elseif ~isempty(args0.Leaveout) && args0.Leaveout
                cvp = cvpartition(n, 'LeaveOut');
            elseif ~isempty(args0.KFold) && args0.KFold > 1
                cvp = cvpartition(n, 'KFold', args0.KFold);
            else
                cvp = cvpartition(n, 'KFold', 10);
            end

            % Create objective function
            objFcn = @(T) RegressionEV.bayesObjective(T, Xraw, y, w, args0, cvp);

            % Perform optimization
            resultsBO = bayesopt(objFcn, vars, ...
                'MaxObjectiveEvaluations', bo.MaxObjectiveEvaluations, ...
                'Verbose', bo.Verbose, ...
                'IsObjectiveDeterministic', bo.IsObjectiveDeterministic, ...
                'AcquisitionFunctionName', bo.AcquisitionFunctionName);

            bestT = resultsBO.XAtMinObjective;
            bestArgs = RegressionEV.applyBayesVarsToArgs(bestT, args0);

            results = struct();
            results.OptimizationResults = resultsBO;
            results.XAtMinObjective = bestT;
            results.MinObjective = resultsBO.MinObjective;
        end

        function vars = defaultOptimizableVariables(hypList)
            % Default hyperparameter search space for bayesopt.
            hypList = string(hypList);
            vars = optimizableVariable.empty(0,1);

            if any(hypList == "Standardize")
                vars(end+1) = optimizableVariable('Standardize', [0 1], 'Type', 'integer');
            end
            if any(hypList == "Lambda")
                vars(end+1) = optimizableVariable('Lambda', [RegressionEV.MIN_JITTER RegressionEV.MAX_JITTER], 'Transform', 'log');
            end
            if any(hypList == "Sigma")
                % Search range from 1e-3 up to a reasonable scale
                vars(end+1) = optimizableVariable('Sigma', [1e-3 1e2], 'Transform', 'log');
            end
            if any(hypList == "Shape")
                % Search range within typical GEV bounds [-0.5, 0.5]
                vars(end+1) = optimizableVariable('Shape', [-0.5 0.5]);
            end
            if any(hypList == "KernelFunction")
                vars(end+1) = optimizableVariable('KernelFunction', ...
                    ["squaredexponential","exponential","matern32","matern52","rationalquadratic"], ...
                    'Type', 'categorical');
            end
            if any(hypList == "BasisFunction")
                vars(end+1) = optimizableVariable('BasisFunction', ...
                    ["none","constant","linear","purequadratic"], ...
                    'Type', 'categorical');
            end
            if any(hypList == "KernelScale")
                vars(end+1) = optimizableVariable('KernelScale', [1e-3 1e3], 'Transform', 'log');
            end
        end

        function obj = bayesObjective(T, Xraw, y, w, args0, cvp)
            % Bayesopt objective: returns total negative log evidence across CV folds.

            args = RegressionEV.applyBayesVarsToArgs(T, args0);

            if ~isempty(cvp)
                nFolds = cvp.NumTestSets;
            else
                nFolds = 1;
            end
            valLoss = zeros(nFolds, 1);

            for i = 1:nFolds
                if ~isempty(cvp)
                    trIdx = cvp.training(i);
                    teIdx = cvp.test(i);
                    Xtr = Xraw(trIdx, :);
                    ytr = y(trIdx);
                    wtr = w(trIdx);
                else
                    Xtr = Xraw;
                    ytr = y;
                    wtr = w;
                end

                % Standardize
                if args.Standardize
                    [XtrStd, mu, sig] = RegressionEV.standardizeX(Xtr);
                else
                    XtrStd = Xtr;
                end

                % Select active data set
                [XaStd, ytr_a, wa] = RegressionEV.selectActiveSet(XtrStd, ytr, wtr, args);

                % Create basis functions
                Ha = RegressionEV.basisMatrix(XaStd, args.BasisFunction);
                argsTrain = args;
                argsTrain.Beta = RegressionEV.ensureBetaSize(argsTrain.Beta, size(Ha,2));

                % Local optimization inside each bayesopt trial.
                if ~strcmpi(string(argsTrain.FitMethod), "none")
                    argsTrain = RegressionEV.optimizeLocally(argsTrain, XaStd, ytr_a, wa, Ha);
                end
                beta = argsTrain.Beta(:);
                [~, ll] = RegressionEV.runInferenceFromArgs(argsTrain, XaStd, ytr_a, wa, Ha, beta);

                if ~isempty(cvp) && (numel(teIdx) > 0)
                    Xte = Xraw(teIdx, :);
                    yte = y(teIdx);
                    wte = w(teIdx);
                    if args.Standardize
                        XteStd = (Xte - mu) ./ sig;
                    else
                        XteStd = Xte;
                    end

                    % Predict on training set
                    Kxs = RegressionEV.kernelMatrix(XaStd, XteStd, argsTrain);
                    Kss = RegressionEV.kernelDiag(XteStd, argsTrain);
                    Hq  = RegressionEV.basisMatrix(XteStd, argsTrain.BasisFunction);
                    mq = Hq*beta;

                    switch lower(argsTrain.InferenceMethod)
                        case 'ep'
                            [muPred, varLatent] = RegressionEV.predictLatentEP( ...
                                RegressionEV.runInferenceFromArgs(argsTrain, XaStd, ytr_a, wa, Ha, beta), ...
                                Kxs, Kss, mq);
                        otherwise
                            [muPred, varLatent] = RegressionEV.predictLatentLaplace( ...
                                RegressionEV.runInferenceFromArgs(argsTrain, XaStd, ytr_a, wa, Ha, beta), ...
                                Kxs, Kss, mq);
                    end

                    sigma2 = max(varLatent + RegressionEV.gevVariance(argsTrain.Sigma, argsTrain.Shape), eps);
                    meanShift = RegressionEV.gevMeanShift(argsTrain.Sigma, argsTrain.Shape);
                    if isnan(meanShift)
                        yhatTe = muPred;
                    else
                        yhatTe = muPred + meanShift;
                    end
                    ll = -0.5 * sum(wte .* (log(2*pi*sigma2) + ((yte - yhatTe).^2) ./ sigma2));
                end

                % Cross validation loss
                if ~isfinite(ll)
                    valLoss(i) = RegressionEV.MAX_FLOAT;
                else
                    valLoss(i) = -ll;
                end
            end

            obj = sum(valLoss);
            if isnan(obj) || ~isfinite(obj)
                obj = RegressionEV.MAX_FLOAT;
            end
        end

        function args = applyBayesVarsToArgs(T, args)
            % Apply bayesopt-selected variables into args struct.
            varNames = T.Properties.VariableNames;

            if ismember('Standardize', varNames)
                args.Standardize = logical(T.Standardize);
            end
            if ismember('Lambda', varNames)
                args.Lambda = T.Lambda;
                args.ConstantLambda = true; % treat as fixed during local optimization
            end
            if ismember('Sigma', varNames)
                args.Sigma = T.Sigma;
                args.ConstantSigma = true; % treat as fixed during local optimization
            end
            if ismember('Shape', varNames)
                args.Shape = T.Shape;
                args.ConstantShape = true; % treat as fixed during local optimization
            end
            if ismember('KernelFunction', varNames)
                args.KernelFunction = char(string(T.KernelFunction));
                args.KernelParameters = [];
                args.ConstantKernelParameters = [];
            end
            if ismember('BasisFunction', varNames)
                args.BasisFunction = char(string(T.BasisFunction));
                args.Beta = [];
            end
            if ismember('KernelScale', varNames)
                % Interpret as a shared length scale for non-ARD kernels.
                kf = lower(string(args.KernelFunction));
                if startsWith(kf, "ard")
                    error('RegressionEV:KernelScaleNotForARD', ...
                        'KernelScale cannot be optimized for ARD kernels.');
                end

                ell = T.KernelScale;
                theta = args.KernelParameters;
                if isempty(theta)
                    theta = RegressionEV.defaultKernelParameters(args.KernelFunction, size(args.X,2));
                end
                theta = theta(:);
                theta(1) = ell;
                args.KernelParameters = theta;

                ck = args.ConstantKernelParameters;
                if isempty(ck)
                    ck = false(numel(theta),1);
                else
                    ck = ck(:);
                end
                ck(1) = true;
                args.ConstantKernelParameters = ck;
            end
        end

        function theta = defaultKernelParameters(kernelFunction, d)
            % Default kernel parameters when user does not supply initial values.
            kf = lower(string(kernelFunction));
            if startsWith(kf, "ard")
                theta = [ones(d,1); 1];
            else
                theta = [1; 1];
            end
            if contains(kf, "rationalquadratic")
                theta = [theta; 1];
            end
        end

        function K = kernelMatrix(X1, X2, args)
            % Evaluate covariance matrix K(X1,X2) for supported kernels.

            kf = args.KernelFunction;
            theta = args.KernelParameters;

            if isempty(theta)
                theta = RegressionEV.defaultKernelParameters(kf, size(X1,2));
            end
            theta = theta(:);
            RegressionEV.validateKernelParameters(theta, kf, size(X1,2));

            if isa(kf, 'function_handle')
                K = kf(X1, X2, theta);
                return;
            end

            name = lower(string(kf));
            dist = lower(string(args.DistanceMethod));

            % Optimized ARD/Iso selection
            if startsWith(name, "ard")
                d = size(X1,2);
                ell = theta(1:d)';
                sf = theta(d+1);
                base = erase(name, "ard");
                % Use pdist2 with weights (1./ell)
                if strcmpi(dist, "accurate")
                    R = pdist2(X1 ./ ell, X2 ./ ell);
                else
                    R = pdist2(X1 ./ ell, X2 ./ ell, 'fasteuclidean');
                end
            else
                ell = theta(1);
                sf = theta(2);
                % Use pdist2 with weights (1./ell)
                if strcmpi(dist, "accurate")
                    R = pdist2(X1 ./ ell, X2 ./ ell);
                else
                    R = pdist2(X1 ./ ell, X2 ./ ell, 'fasteuclidean');
                end
                base = name;
            end
            K = RegressionEV.kernelFromR(base, R, sf, theta);
        end

        function kdiag = kernelDiag(X, args)
            % Return diag(K(X,X)) efficiently for stationary kernels.
            kf = args.KernelFunction;
            theta = args.KernelParameters;

            if isempty(theta)
                theta = RegressionEV.defaultKernelParameters(kf, size(X,2));
            end
            theta = theta(:);

            if isa(kf, 'function_handle')
                K = kf(X, X, theta);
                kdiag = diag(K);
                return;
            end

            name = lower(string(kf));
            d = size(X,2);
            if startsWith(name, "ard")
                sf = theta(d+1);
            else
                sf = theta(2);
            end
            kdiag = (sf^2) * ones(size(X,1),1);
        end

        function K = kernelFromR(name, R, sf, theta)
            % Build K given precomputed distance matrix R.
            sf2 = sf^2;
            switch lower(string(name))
                case "squaredexponential"
                    K = sf2 * exp(-0.5 * R.^2);
                case "exponential"
                    K = sf2 * exp(-R);
                case "matern32"
                    a = sqrt(3) * R;
                    K = sf2 * (1 + a) .* exp(-a);
                case "matern52"
                    a = sqrt(5) * R;
                    K = sf2 * (1 + a + (a.^2)/3) .* exp(-a);
                case "rationalquadratic"
                    alpha = theta(end);
                    K = sf2 * (1 + (R.^2) ./ (2*alpha)).^(-alpha);
                otherwise
                    error('RegressionEV:BadKernel', 'Unsupported kernel: %s', name);
            end
        end

        function [post, logZ] = inferLaplace(K, y, w, m, sigma, shape, opt)
            % inferLaplace - Laplace approximation for GEV GP regression.
            %
            %   Implements Newton-Raphson with backtracking line search to find
            %   the posterior mode, then computes the Laplace evidence approximation.
            %
            %   [post, logZ] = inferLaplace(K, y, w, m, sigma, shape, opt)
            %   where logZ is the approximate marginal log likelihood.

            n = size(K,1);
            y = y(:);
            w = w(:);
            if isempty(m)
                m = zeros(n,1);
            end

            % Initialize at prior mean
            f = m;

            % Newton Raphson with Line Search
            obj_old = -inf;
            alpha = zeros(n,1);

            for it = 1:opt.MaxIter
                % Compute Likelihood derivatives
                [logL, grad, W] = RegressionEV.gevLikelihoodMoments(f, y, w, sigma, shape);

                % Objective
                obj = sum(logL) - 0.5 * (alpha' * (f - m));

                % Convergence check on objective (ascent check)
                if abs(obj - obj_old) < opt.Tol * max(1, abs(obj))
                    break;
                end
                obj_old = obj;

                % Compute Newton Direction
                sW = sqrt(W);
                B = (sW .* (K .* sW')) + speye(n);
                L = RegressionEV.cholSafe(B);

                % Newton Step
                b = W .* (f - m) + grad;

                % a_dir is the target alpha for the full Newton step
                a_dir = b - sW .* (L' \ (L \ (sW .* (K * b))));

                % Direction d_alpha = a_dir - alpha
                d_alpha = a_dir - alpha;
                d_f = K * d_alpha;

                % Backtracking Line Search
                step = 1.0;
                while step > eps
                    % Propose new state
                    alpha_new = alpha + step * d_alpha;
                    f_new = f + step * d_f;

                    % Evaluate Objective
                    logL_new = RegressionEV.gevLikelihoodMoments(f_new, y, w, sigma, shape);
                    obj_new = sum(logL_new) - 0.5 * (alpha_new' * (f_new - m));

                    % Simple ascent check
                    if obj_new > obj
                        f = f_new;
                        alpha = alpha_new;
                        break;
                    end
                    step = 0.5 * step;
                end
            end

            if (it==opt.MaxIter)
                warning('RegressionEV:MaxIterReached', ...
                    'Maximum iterations reached without convergence');
            end

            [logL, ~, W] = RegressionEV.gevLikelihoodMoments(f, y, w, sigma, shape);

            % Final quantities
            sW = sqrt(W);
            B = (sW .* (K .* sW')) + speye(n);
            L = RegressionEV.cholSafe(B);

            % Approximate log evidence
            quad = alpha' * (f - m);
            logdetB = 2 * sum(log(diag(L)));
            logZ = sum(logL) - 0.5 * quad - 0.5 * logdetB;

            post = struct();
            post.f_hat = f;
            post.W = W;
            post.L = L;
            post.sW = sW;
            post.alpha = alpha;
            post.K = K;
            post.m = m;
            post.sigma = sigma;
            post.shape = shape;
            post.inference = 'laplace';
        end

        function [post, logZ] = inferEP(K, y, w, m, sigma, shape, opt)
            % inferEP - Expectation Propagation for GEV GP regression.
            %
            %   Sequential EP with rank-1 posterior updates and Gauss-Hermite
            %   quadrature for computing tilted distribution moments.
            %
            %   [post, logZ] = inferEP(K, y, w, m, sigma, shape, opt)

            n = size(K,1);
            y = y(:);
            w = w(:);
            if isempty(m)
                m = zeros(n,1);
            end

            % Initialize Sites (Gaussian approximations to likelihood)
            tau = zeros(n,1); % Site precisions
            nu  = zeros(n,1); % Site precision-mean products

            % Initialize Posterior (Starts as Prior)
            Sigma = K;
            mu = m;

            [ghx, ghw] = RegressionEV.gaussHermite(opt.NumGHPoints);

            % Iteration Settings
            sqrt_eps = sqrt(eps);
            for it = 1:opt.MaxIter
                tau_old_sweep = tau;
                nu_old_sweep = nu;

                % 1. Randomizing order improves stability and prevents local oscillations
                perm = randperm(n); 
    
                for i = perm
                    sig_i = Sigma(i,i);

                    % 2. Cavity Distribution (Divide Posterior by Site i)
                    % precision_cav = precision_post - site_precision
                    denom = max(1 - sig_i*tau(i), sqrt_eps);
                    var_cav = max(sig_i / denom, sqrt_eps);
                    mu_cav = (mu(i) - sig_i*nu(i)) / denom;
   
                    % 3. Moment Matching
                    [mu_hat, var_hat, logZi] = RegressionEV.gevTiltedMoments(mu_cav, var_cav, y(i), sigma, shape, ghx, ghw);
                    if ~isfinite(logZi) || ~isfinite(var_hat) || ~isfinite(mu_hat)
                        continue;
                    end

                    % 4. Update Site Parameters with CLAMPING and DAMPING
                    % Fix: Corrected variable name from nu_nu to nu
                    delta_tau = w(i)*((1/var_hat) - (1/var_cav)) - tau(i);
                    delta_nu  = w(i)*((mu_hat/var_hat) - (mu_cav/var_cav)) - nu(i);

                    % Damping helps convergence in non-log-concave regions
                    tau_new = max(tau(i) + delta_tau * opt.Damping, 0); 
                    nu_new  = nu(i) + delta_nu * opt.Damping;
                    
                    % 5. Sequential Rank-1 Update of Posterior
                    dt = tau_new - tau(i);
                    dn = nu_new - nu(i);
                    up_denom = 1 + dt*sig_i;

                    % If update is too singular, skip it
                    if abs(up_denom) > sqrt_eps
                        K_fact = dt / up_denom;
                        mu_fact = (dn - dt*mu(i)) / up_denom;
    
                        si = Sigma(:,i);
                        Sigma = Sigma - (K_fact * si) * si';
                        mu = mu + mu_fact*si;
                        
                        tau(i) = tau_new;
                        nu(i)  = nu_new;
                    end
                end

                % 6. Recompute globally to purge numerical drift from rank-1 updates.
                sW = sqrt(max(tau,0));
                B = (sW.*(K.*sW')) + speye(n);
                L = RegressionEV.cholSafe(B);
                
                % Scale ROWS of K (sW.*K) ensures symmetric Sigma_new
                V = L \ (sW .* K); 
                Sigma = K - (V' * V);
    
                % Recompute mu using the corrected stable identity
                mu = Sigma * nu + m - Sigma * (tau .* m);
                if ~all(isfinite(mu(:)))
                    warning('RegressionEV:EPMuNonFinite', 'EP mu became non-finite at iter %d', it);
                    break;
                end

                % Check Convergence
                if max(abs(tau - tau_old_sweep)) < opt.Tol && max(abs(nu - nu_old_sweep)) < opt.Tol
                    break; 
                end
            end

            if (it==opt.MaxIter)
                warning('RegressionEV:MaxIterReached', ...
                    'Maximum iterations reached without convergence');
            end

            % 7. Recompute cavity one last time using final stable posterior
            diag_S = diag(Sigma);
            var_cav_f = max(diag_S ./ (1 - diag_S .* tau), sqrt_eps);
            mu_cav_f  = (mu - diag_S .* nu) ./ (1 - diag_S .* tau);


            % Compute Moment Matching Normalization (Z_hat)
            logP_sites = zeros(n,1);
            for i = 1:n
                [~, ~, logP_sites(i)] = RegressionEV.gevTiltedMoments( ...
                    mu_cav_f(i), var_cav_f(i), y(i), sigma, shape, ghx, ghw);
            end

            % Log-determinant of (I + S^1/2 K S^1/2)
            logdetB = 2*sum(log(diag(L)));

            % Full Evidence (accounts for prior mean m)
            % Terms: Site log-integrals - Complexity - Cavity/Prior Energy
            logZ = sum(logP_sites) - 0.5*logdetB + 0.5*nu'*(Sigma*nu - 2*Sigma*(tau.*m)) ...
                - 0.5*sum( (tau.*(mu_cav_f - m).^2) ./ (1 + tau.*var_cav_f) );

            % Alpha used for predictions: alpha = K^-1 * (mu - m)
            alpha = nu - tau .* mu;

            post = struct('tau',tau, 'nu',nu, 'Sigma',Sigma, 'mu',mu, 'alpha',alpha, 'K',K, 'm',m, 'L',L, 'sigma',sigma, 'shape',shape, 'inference','ep');
        end

        function [muPred, varLatent] = predictLatentLaplace(post, Kxs, Kss, mq)
            % Predictive latent mean and variance using Laplace posterior.
            muPred    = mq(:) + (Kxs' * post.alpha);
            v         = post.L \ (post.sW .* Kxs);
            varLatent = max(Kss(:) - sum(v.^2, 1)', 0);
        end

        function [muPred, varLatent] = predictLatentEP(post, Kxs, Kss, mq)
            % Predictive latent mean and variance using EP posterior.
            muPred    = mq(:) + (Kxs' * post.alpha);
            sW        = sqrt(max(post.tau, 0));
            v         = post.L \ (sW .* Kxs);
            varLatent = max(Kss(:) - sum(v.^2, 1)', 0);
        end

        function [logp, grad, W] = gevLikelihoodMoments(f, y, w, sigma, shape)
            % Improved Robust GEV Likelihood with Support Guidance and Overflow Protection
            %
            % Returns:
            %   logp : weighted log-likelihood at each observation
            %   grad : gradient d(logp)/df
            %   W    : negative Hessian diagonal (used as Fisher information)

            f = f(:); y = y(:); w = w(:);
            sigma = max(sigma, 1e-9);
            z = (y - f) ./ sigma;
            LIMIT = 500; % Prevents double-precision overflow
            EXP_LIMIT = exp(LIMIT);

            % --- Case A: Gumbel Limit (Shape -> 0) ---
            if abs(shape) < RegressionEV.GEV_ZERO_TOL
                penalty = exp(min(-z, LIMIT));
                logp = w .* (-log(sigma) - z - penalty);

                if nargout > 1
                    grad = w .* (1 - penalty) ./ sigma;
                end
                if nargout > 2
                    hess = -w .* penalty ./ (sigma^2);
                    W = max(-hess, 1e-12);
                end
                return;
            end

            % --- Case B: Generalized Extreme Value ---
            sz = shape .* z;
            t = 1 + sz;
            valid = (t > 1e-12);
            invalid = ~valid;

            logp = zeros(size(f));
            grad = zeros(size(f));
            W    = zeros(size(f));

            % 1. Inside Support Region
            if any(valid)
                tv = t(valid);
                szv = sz(valid);
                log_tv = log1p(szv); % Stable calculation of log(1 + shape*z)

                % p_term = tv.^(-1/shape) = exp(-log(tv)/shape)
                p_term = exp(min(-log_tv ./ shape, LIMIT));
                logp(valid) = -log(sigma) - (1 + 1/shape) .* log_tv - p_term;

                if nargout > 1
                    % dL/df = (1/sigma) * [(1+shape)/tv - tv^(-1/shape - 1)]
                    term2 = min(p_term ./ tv, EXP_LIMIT);
                    grad(valid) = ((1 + shape) ./ tv - term2) ./ sigma;
                end

                if nargout > 2
                    % d2L/df2 = ((1+shape)/sigma^2) * [shape/tv^2 - tv^(-1/shape - 2)]
                    term3 = min(term2 ./ tv, EXP_LIMIT);
                    h_core = (shape ./ (tv.^2) - term3);
                    hess_v = ((1 + shape) ./ (sigma^2)) .* h_core;
                    W(valid) = max(-hess_v, 1e-12);
                end
            end

            % 2. Outside Support (Guidance Mode)
            if any(invalid)
                dist = abs(t(invalid));
                logp(invalid) = -1e12 - 1e6 * dist;
                if nargout > 1
                    % Gradient now points toward the boundary
                    grad(invalid) = -sign(shape) * 1e6 / sigma;
                end
                if nargout > 2
                    % High artificial precision to force f back into valid region
                    W(invalid) = 1e6 / (sigma^2);
                end
            end

            % Apply weights
            logp = w .* logp;
            if nargout > 1, grad = w .* grad; end
            if nargout > 2, W    = w .* W;    end
        end

        function [muHat, varHat, logZ] = gevTiltedMoments(mu, varf, y, sigma, shape, ghx, ghw)
            % Compute tilted distribution moments via Gauss-Hermite quadrature.
            % Uses robust likelihood and stabilized weight normalization.

            % Standard GH transformation: f = mu + sqrt(2*varf)*x
            s = sqrt(max(2 * varf, eps));
            f = mu + s * ghx(:);

            % Use the improved likelihood function
            loglik = RegressionEV.gevLogLikelihood(f, y, sigma, shape);

            % logw_i = log(prior_weight_i) + log(likelihood_i)
            % Note: 1/sqrt(pi) is the GH normalization constant
            logw = log(max(ghw(:), eps)) - 0.5 * log(pi) + loglik;

            % Stabilized Log-Sum-Exp
            maxLogW = max(logw);

            % Handle edge case where all weights are -Inf (no overlap)
            if maxLogW < -1e11
                muHat = mu;
                varHat = varf;
                logZ = -1e12; % Representing zero probability
                return;
            end

            ww = exp(logw - maxLogW);
            Z  = sum(ww);

            % Probability weights for moments
            p = ww / Z;

            % Mean and Variance of the tilted distribution
            Ef  = sum(p .* f);
            Ef2 = sum(p .* (f.^2));

            muHat  = Ef;
            % Ensure variance remains positive and stable
            varHat = max(Ef2 - Ef^2, 1e-12 * varf);
            logZ   = maxLogW + log(Z);
        end

        function loglik = gevLogLikelihood(f, y, sigma, shape)
            % Pointwise GEV log-likelihood with stability and support guidance.

            f = f(:);
            sigma = max(sigma, 1e-9);
            z = (y - f) ./ sigma;
            LIMIT = 500; % Prevents double-precision overflow

            % --- Case A: Gumbel Limit (Shape -> 0) ---
            if abs(shape) < RegressionEV.GEV_ZERO_TOL
                % Use min(-z, LIMIT) to prevent exp() overflow
                loglik = -log(sigma) - z - exp(min(-z, LIMIT));
                return;
            end

            % --- Case B: Generalized Extreme Value ---
            sz = shape .* z;
            t = 1 + sz;
            loglik = zeros(size(f));
            valid = (t > 1e-12);
            invalid = ~valid;

            % 1. Inside Support
            if any(valid)
                szv = sz(valid);
                log_tv = log1p(szv); % Stable calculation of log(1 + shape*z)

                % p_term = tv.^(-1/shape) = exp(-log(tv)/shape)
                p_term = exp(min(-log_tv ./ shape, LIMIT));
                loglik(valid) = -log(sigma) - (1 + 1/shape) .* log_tv - p_term;
            end

            % 2. Outside Support (Guidance Penalty)
            % Corrected: Use a large NEGATIVE value, not -log(sqrt(eps))
            if any(invalid)
                dist = abs(t(invalid));
                loglik(invalid) = -1e12 - 1e6 * dist;
            end
        end

        function m = gevMeanShift(sigma, shape)
            % Expected shift E[Y - f] = E[GEV(0,sigma,shape)] for the response mean.
            if abs(shape) < RegressionEV.GEV_ZERO_TOL
                m = sigma * RegressionEV.EULER_MASCHERONI;
            elseif shape < 1
                m = sigma * (gamma(1 - shape) - 1) / shape;
            else
                m = NaN;
            end
        end

        function v = gevVariance(sigma, shape)
            % Marginal variance of GEV(0, sigma, shape). NaN when undefined (shape >= 0.5).
            if abs(shape) < RegressionEV.GEV_ZERO_TOL
                v = (pi^2 / 6) * sigma^2;
            elseif shape < 0.5
                g1 = gamma(1 - shape);
                g2 = gamma(1 - 2*shape);
                v  = sigma^2 * (g2 - g1^2) / (shape^2);
            else
                v = NaN;
            end
        end

        function [x, w] = gaussHermite(n)
            % Gauss-Hermite quadrature nodes and weights of order n.
            i  = (1:n-1)';
            a  = sqrt(i/2);
            CM = diag(a,1) + diag(a,-1);
            [V,D] = eig(CM);
            x = diag(D);
            [x, idx] = sort(x);
            V = V(:,idx);
            w = sqrt(pi) * (V(1,:)'.^2);
        end

        function L = cholSafe(X)
            % Numerically stable Cholesky decomposition with adaptive jitter.
            [L,p] = chol(X, 'Lower');

            % Force positive definite
            if p
                % Force symmetry
                I = speye(size(X,1),size(X,2));
                X = 0.5.*(X+X') + 1e-12.*I;

                % Compute the symmetric polar factor of X.
                %[U,Sigma,V] = svd(X);
                %X = 0.5.*(U*Sigma*U'+V*Sigma*V');

                % Cholesky decomposition
                [L,p] = chol(X,'Lower');

                % Add adaptive jitter for final step
                max_jitter     = RegressionEV.MAX_JITTER;
                current_jitter = RegressionEV.MIN_JITTER;
                while p && (current_jitter <= max_jitter)
                    [L,p] = chol(X + current_jitter .* I, 'lower');
                    current_jitter = 10 * max(current_jitter, 1e-10);
                end
            end

            % Return NaN when stil failing
            if p
                warning('RegressionEV:CholeskyFailed', 'Matrix is ill-conditioned.');
                L = NaN.*I;
            end
        end

        function x = norminvSafe(p)
            % Normal inverse CDF via erfcinv, avoiding Statistics Toolbox dependency.
            p = min(max(p, eps), 1-eps);
            x = -sqrt(2) * erfcinv(2*p);
            if ~isfinite(x)
                if p < 0.5, x = -8.5; else, x = 8.5; end
            end
        end

        function X = peekN(Xin)
            % Extract numeric matrix for size inference, accepting tables or arrays.
            if istable(Xin)
                numVars = varfun(@(v) isnumeric(v) || islogical(v), Xin, 'OutputFormat', 'uniform');
                X = table2array(Xin(:, numVars));
            else
                X = Xin;
            end
        end
    end
end