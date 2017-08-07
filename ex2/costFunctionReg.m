function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

regularized_theta = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

hypothesis = X * theta;
hypothesis = sigmoid(hypothesis);

pos_part = (transpose(y) * log(hypothesis)) .* -1;
neg_part = transpose(1 .- y) * log( 1 .- hypothesis);

result = pos_part - neg_part;

cost = result ./ m;

%not regularizing the first theta parameter

regularized_theta = (theta .^ 2) .* (lambda / (2*m) );
regularized_theta(1) = 0;
regularized_result = cost + sum(regularized_theta);

J = regularized_result;

%not regularizing the first theta parameter

regularized_theta = theta .* (lambda / m );
regularized_theta(1) = 0;

partial_derivative = transpose(X) * (hypothesis - y);

grad = grad - ((partial_derivative ./ m) + regularized_theta) .* -1;

% =============================================================

end
