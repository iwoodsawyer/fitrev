function [this, varargout] = fitrev(X, Y, varargin)
%FITREV Fit a Gaussian Process Extreme Value (EV) Regression model.
%   MODEL=FITREV(TBL,Y) returns a GPC model MODEL for predictors in table
%   TBL and response Y. Y must be a numeric vector of continuous responses.
%
%   MODEL is an object of class `RegressionEV`. This function follows
%   the fitrgp pattern and delegates fitting to `RegressionEV.fit`.
%
%   MODEL=FITREV(X,Y) is an alternative syntax that accepts X as an N-by-P
%   matrix of predictors with one row per observation and one column per
%   predictor. Y is the response vector. 
%
%   [MODEL, HYPEROPTR] = FITREV(...) returns bayesopt results (struct) when
%   'OptimizeHyperparameters' is not 'none'. Otherwise HYPEROPTR is [].
%
% See also: RegressionEV, fitcgp

[this, hyperOptResults] = RegressionEV.fit(X, Y, varargin{:});

if nargout > 1
    varargout{1} = hyperOptResults;
end
end