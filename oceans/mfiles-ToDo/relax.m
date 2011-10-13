function [C, N, conv] = relax(A, B, b, sp, delta, omega)

% This function takes an input matrix, A, a boundary condition matrix, B, 
% an index of relavent locations in the boundary condition matrix, b, and 
% the precision desired, delta, and the relaxation parameter, omega.
% the spacing in the x- and y-directions sp = [delta_x,delta_y]. The function
% then determines the steady state solution of $\nabla^2 A = 0$, returns the 
% solution, C, and the number of iterations, N.
%
% To call the function enter:
%
% [C, N, conv] = relax(A, B, b, sp, delta, omega)
%
% It is assumed that a "buffer region" has been created around the matrix, 
% so solutions are only seeked in the area of (A(2:end-1,2:end-1)). Also it
% size(A) needs to be the same as size(B). b can give an index or a
% two-dimensional coordinate.

%                                                  Ramzi Mirshak |:| 09Oct02
%                                                  mirshak@phys.ocean.dal.ca

% define new matrices

C = zeros(size(A));

C(b) = 1;
a = find(C==0);
C(b) = 0;

D = A;

N = 0;

dx = sp(1);
dy = sp(2);

a_ = dy * dy / 2 / (dx * dx + dy * dy);
b_ = a_;
c_ = dx * dx /  2 / (dx * dx + dy * dy);
d_ = c_;
e_ = - 1;

while(sqrt(mean((C(a)-D(a)).^2)) > delta) 
  C = D;
%===============
% This works!!!
%===============
  D(2:end-1,2:end-1) = C(2:end-1,2:end-1) * (1 - omega) + omega * ... 
     (C(1:end-2,2:end-1) + C(3:end,2:end-1) + ...
     C(2:end-1,1:end-2) + C(2:end-1,3:end)) / 4;

%=========================================================
% This is the numerical recipes version, but takes longer
% (algebraicly it is the same but I guess it works 
% with under more scenarios.
%=========================================================

%  xi = a_ * C(3:end,2:end-1) + b_ * C(1:end-2,2:end-1) + c_ * ...
%            C(2:end-1,3:end) + d_ * C(2:end-1,1:end-2) + e_ * ...
%            C(2:end-1,2:end-1);
%
%  D(2:end-1,2:end-1) = C(2:end-1,2:end-1) - omega * xi / e_;

  D(end,:) = D(end-1,:);
  D(b) = B(b);
  N = N + 1;
  conv(N) = sqrt(mean((C(a)-D(a)).^2))/delta;
end






