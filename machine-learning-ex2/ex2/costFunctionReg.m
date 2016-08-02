function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(X(1,:)); % number of features.
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

J_1 = 0;
J_2 = 0;
for i = 1:m,
	J_1 = J_1 + (-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
end;
J_1 = J_1/m;

for j=2:n,
	J_2 = J_2 + theta(j)^2;
end;
J_2 = (lambda/(2*m))*J_2;

J = J_1+J_2;

for i=1:m,
	grad(1) = grad(1) + (sigmoid(X(i,:)*theta)-y(i))*X(i,1);
end;
grad(1) = grad(1)/m;

for j=2:size(theta),
	for i=1:m,
		grad(j) = grad(j) + (sigmoid(X(i,:)*theta)-y(i))*X(i,j);
	end;
	grad(j) = grad(j)/m + (lambda/m)*theta(j);
end;


% =============================================================

end
