function itg=integral(f,dx,smooth);
% INTEGRAL  ANS=INTEGRAL(F,DX)
%           This function computes the integral of F(X)DX where the integrand
%           is specified at discrete points F spaced DX apart (F is a vector,
%           DX is a scalar).  Simpsons Rule is used, so that the error
%           is O(dx^5*F4). (F4 is the 4th derivative of F).
%
%           If F is a matrix, then the integration is done for each column.
%
%           If F is really spiky, then INTEGRAL(F,DX,'smooth') may
%           provide a better looking result (the result is smoothed
%           with a 3 point triangular filter).
%

% Author: RP (WHOI) 15/Aug/92


[N,M]=size(f);

if (N==1 | M==1),
   N=max(size(f));
   itg=zeros(size(f));
   itg(1)=0;                                    % first element
   itg(2)=(5*f(1)+8*f(2)-f(3))*dx/12;           % Parabolic approx to second
   itg(3:N)=(f(1:N-2)+4*f(2:N-1)+f(3:N))*dx/3;  % Simpsons rule for 2-segment
                                                % intervals
   itg(1:2:N)=cumsum(itg(1:2:N));    % Sum up 2-seg integrals
   itg(2:2:N)=cumsum(itg(2:2:N));
   if (nargin>2),                    % ... apply smoothing
      itg(2:N-1)=(itg(1:N-2)+2*itg(2:N-1)+itg(3:N))/4;
      itg(N)= (itg(N-1)+itg(N))/2;
   end
else
   itg=zeros(size(f));
   itg(1,:)=zeros(1,M);
   itg(2,:)=(5*f(1,:)+8*f(2,:)-f(3,:))*dx/12;
   itg(3:N,:)=(f(1:N-2,:)+4*f(2:N-1,:)+f(3:N,:))*dx/3;

   itg(1:2:N,:)=cumsum(itg(1:2:N,:));    % Sum up 2-seg integrals
   itg(2:2:N,:)=cumsum(itg(2:2:N,:));

   if (nargin>2),                    % ... apply smoothing
         itg(2:N-1,:)=(itg(1:N-2,:)+2*itg(2:N-1,:)+itg(3:N,:))/4;
         itg(N,:)= (itg(N-1,:)+itg(N,:))/2;
   end
end