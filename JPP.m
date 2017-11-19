%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright (c) 2014 Yahoo! Inc.
%Copyrights licensed under the MIT License. See the accompanying LICENSE file for terms.
% Author : Amin Mantrach - amantrac at yahoo - inc dot com - http://iridia.ulb.ac.be/~amantrac/
% Implements the method presented in: 
%Carmen Vaca, Amin Mantrach, Alejandro Jaimes and Marco Saerens. 
%A Time-based Collective Factorization for Topic Discovery and Monitoring in News. 
%Proc. of 23rd International World Wide Web Conference (WWW 2014), p527-538, April 2014, Seoul, Korea 
function [W, H, M, ObjHistory] = JPP(X, R, k, alpha, lambda, epsilon, maxiter, verbose)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% X is document x term matrix
% R is the preivous H matrix of previous time step  (init first time with normal NMF H output)
% 
% Optimizes the formulation:
% ||X - W*H||^2 + ||X - W*M*R||^2  + alpha*||M-I||^2 + lambda*[l1 norm Regularization of W and H]
%
% with multiplicative rules.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fix seed for reproducable experiments
rand('seed', 14111981); %inicia semilla random

% initilasation
n = size(X, 1); %filas X
v1 = size(X, 2); %columnas X

% randomly initialize W, Hu, Hs.
%W  = abs(rand(n, k));
W = [0.1,0.2,0.3;0.1,0.2,0.3;0.1,0.2,0.3];
%H = abs(rand(k, v1));
H = [0.01,0.02,0.03;0.01,0.02,0.03;0.01,0.02,0.03];
%M = abs(rand(k,k));
M = [0.001,0.002,0.003;0.001,0.002,0.003;0.001,0.002,0.003];

I =speye(k,k); %Matriz identidad (k filas)x(k columnas)
Ilambda = I*lambda; %multiplica la identidad I por lam

% constants
trXX = tr(X, X); %multiplicacion termino por termino y despues suma filas y despues columnas, queda escalar



% iteration counters
itNum = 1;
Obj = 10000000;

prevObj = 2*Obj;

%this improves sparsity, not mandatory.

while((abs(prevObj-Obj) > epsilon) && (itNum <= maxiter)),

     J= M*R;
     W =  W .* ( X*(H'+J')  ./ max(W*((J*J')+(H*H')+ lambda),eps) ); % eps = 2^(-52)
     WtW =W'*W;
     WtX = W'*X;     
     M = M .* ( ((WtX*R') + (alpha*I)) ./ max( (WtW*M*R*R') + ( (alpha)*M)+lambda,eps) );    %%%%%%%%%%%%%%%
     H = H .* (WtX./max(WtW*H+lambda,eps));
     prevObj = Obj;
     Obj = computeLoss(X,W,H,M,R,lambda,alpha, trXX, I);%%%%%%%
     delta = abs(prevObj-Obj);
 	 ObjHistory(itNum) = Obj; %%%%%%% esto no se usa
 	 if verbose,
            fprintf('It: %d \t Obj: %f \t Delta: %f  \n', itNum, Obj, delta); 
     end
  	 itNum = itNum + 1;
end

function [trAB] = tr(A, B) %ya implementado
	trAB = sum(sum(A.*B));
end
    
function Obj = computeLoss(X,W,H,M,R,reg_norm,reg_temp, trXX, I)
    WtW = W' * W;
    MR = M*R;
    WH = W * H;
    WMR = W * MR;    
    tr1 = trXX - 2*tr(X,WH) + tr(WH,WH);
    tr2 = trXX - 2*tr(X,WMR) + tr(WMR,WMR);
    tr3 = reg_temp*(tr(M,M) - 2*trace(M)+ trace(I)); %trace = suma de la diagonal
    tr4 = reg_norm*(sum(sum(H)) + sum(sum(W)) + sum(sum(M)) ); %reg_temp
    Obj = tr1+ tr2 + tr3+ tr4;    
end



end
