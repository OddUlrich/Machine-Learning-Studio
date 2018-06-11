functiton [cost, grad] = costFunction(theta, X, y)

% Useful values
m = size(X, 1);
n = size(X, 2);

% Compute cost and gradient
y_hat = X * theta;
cost = -1/(2*m) * (y'*log(y_hat) + (1 - y')*log(1 - y_hat));
grad = 1/m * X'*(y_hat - y)

end