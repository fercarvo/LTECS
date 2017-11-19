%test_JPP%

X = [1,2,3;4,5,6;7,8,9];
R = [1,1,1;2,2,2;3,3,3];
k = 3;
alpha = 10000000;
lambda = 0.05;
epsilon = 0.01;
maxiter = 100;
verbose = false;
[W, H, M, ObjHistory] = JPP(X, R, k, alpha, lambda, epsilon, maxiter, verbose);

fprintf('***************\n')
W
H
M
ObjHistory
fprintf('***************\n')