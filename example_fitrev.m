load fisheriris
X = meas(:,1:2);
y = meas(:,1);

mdlL = fitrev(X, y, 'Inference', 'Laplace', 'Standardize', true);
[yhatL, ysdL, ciL] = predict(mdlL, X);

mdlE = fitrev(X, y, 'Inference', 'EP', 'Standardize', true);
[yhatE, ysdE, ciE] = predict(mdlE, X);