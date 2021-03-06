function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

% Compute J
for i=1:m
    H = sigmoid(theta' * X(i,:)');
    J = J + -y(i,:) * log(H) - (1 - y(i,:)) * log(1 - H);
end
J = J/m;

% Compute grad
for j_iter = 1:length(theta)
  dJ = 0;
  for i=1:m
      H = sigmoid(theta' * X(i,:)');
      dJ = dJ + (H - y(i,:)) * X(i,j_iter);
  end 
  dJ = dJ / m;

  grad(j_iter) = dJ;
end 

% =============================================================

end
