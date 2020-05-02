function [w,b,sv] = svm(phi_x, label, C, tol)
K = phi_x*phi_x';
[a,b] = smo(K,label',C,tol);
index = find(a>0);
w = phi_x'*(a'.*label);
sv = phi_x(index,:);
