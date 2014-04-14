function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Compute J
% for i=1:m
%    H = sigmoid(X(i,:) * theta);
%    J = J + -y(i,:) * log(H) - (1 - y(i,:)) * log(1 - H);
% end
% J = J/m;

% Compute J (Vectorized)
H = sigmoid(X*theta);
J = (-y' * log(H) - (1 - y') * log(1 - H))/m;

% Compute regularized term
% regular = 0;
% for j=2:n
%     regular = regular + theta(j)^2;
% end
% regular = regular * lambda / (2 * m);

% Apply regularization (Vectorized)
regular = (sum(theta.^2) - theta(1)) * lambda / (2 * m);
J = J + regular;

% Compute grad
for j_iter = 1:length(theta)
    dJ = 0;
    for i=1:m
        H = sigmoid(X(i,:) * theta);
        dJ = dJ + (H - y(i,:)) * X(i,j_iter);
    end
    dJ = dJ / m;

    if (j_iter < 2)
      grad(j_iter) = dJ;
    else
      grad(j_iter) = dJ + (lambda * theta(j_iter)/ m);
    end
end

% for j_iter = 1:length(theta)
%     H = sigmoid(X * theta);
%     dJ = sum((H - y) .* X(:,j_iter))/m;
% 
%     if (j_iter < 2)
%         grad(j_iter) = dJ;
%     else
%         grad(j_iter) = dJ + (lambda * theta(j_iter)/ m); 
%     end
% end

% Compute grad (Vectorized)
% H = sigmoid(X * theta);
% grad = sum(X' * (H - y))/m;

% Apply regularization (Vectorized)
% regular = lambda * sum(theta)-theta(1) / m;
% grad = grad + regular;

% =============================================================

end
