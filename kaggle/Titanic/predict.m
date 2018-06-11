function p = predict(theta, X)

% Useful values
m = size(X, 1);
n = size(X, 2);

% initialize
p = zeros(m, 1);

z = X * theta;

for iter = 1:m
    if z(iter) >= 0.5
        p(iter) = 1;
    else
        p(iter) = 0;
    end % if
end % for

end % function

        


