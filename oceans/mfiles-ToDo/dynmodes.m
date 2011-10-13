function [wmodes,pmodes,ce]=dynmodes(Nsq,p,nmodes)
% DYNMODES calculates ocean dynamic vertical modes
%  taking a column vector of Brunt-Vaisala values (Nsq) at
%  different pressures (p) and calculating some number of 
%  dynamic modes (nmodes). 
%  Note: The input pressures need not be uniformly spaced, 
%    and the deepest pressure is assumed to be the bottom.
%
%  USAGE: [wmodes,pmodes,ce]=dynmodes(Nsq,p,nmodes);
%                               or
%                            dynmodes;  % to show demo 
%
%     Inputs: 	Nsq = column vector of Brunt-Vaisala buoyancy frequency (s^-2)
%		    	  p = column vector of pressure (decibars)
%           nmodes = number of vertical modes to calculate 
%  
%       Outputs:   wmodes = vertical velocity structure
%                  pmodes = horizontal velocity structure
%                      ce = modal speed (m/s)
%  developed by J. Klinck. July, 1999
%  send comments and corrections to klinck@ccpo.odu.edu

if nargin<1
   help(mfilename);
   nplot=3;
%    test problems
%      problem 1
%    solution is h = ho sin(z /ce) where ce = 1 / n pi
%     ce = 0.3183 / n = [ 0.3183 0.1591 0.1061]
%p=0:.05:1;
%z=-p;
%n=length(p);
%Nsq(1:n)=1;
%
%      problem 2
%    solution is h = ho sin(No z /ce) where ce = No H / n pi
%    for No=1.e-3 and H = 400, the test values are 
%     ce = 0.127324 / n = [ 0.127324, 0.063662, 0.042441]
%
	p=0:10:400;
	z=-p;
	n=length(p);
	Nsq(1:n)=1.e-6;

	nmodes=3;

	[wmodes,pmodes,ce]=dynmodes(Nsq,p,nmodes);

	figure(1)
	plot(Nsq,z);
	title('Buoyancy Frequency Squared (s^{-2})')

	figure(2)
	plot(ce(1:nplot),'r:o');
	title(' Modal Speed (m/s)')

	figure(3)
	plot(wmodes(:,1:nplot),z);
	title('Vertical Velocity Structure')

	figure(4)
	plot(pmodes(:,1:nplot),z);
	title('Horizontal Velocity Structure')

        figure(gcf)
        return
end
  
rho0=1028;

%    convert to column vector if necessary
[m,n] = size(p);
if n == 1
   p=p';
end
[m,n] = size(Nsq);
if n == 1
   Nsq=Nsq';
   n=m;
end

%                 check for surface value
if p(1) > 0
%             add surface pressure with top Nsq value
    z(1)=0;
    z(2:n+1)=-p(1:n);
    N2(1)=Nsq(1);
    N2(2:n+1)=Nsq(1:n);
    nz=n+1;
else
    z=-p;
    N2=Nsq;
    nz=n;
end

%          calculate depths and spacing
%        spacing
dz(1:nz-1)=z(1:nz-1)-z(2:nz);
%        midpoint depth
zm(1:nz-1)=z(1:nz-1)-.5*dz(1:nz-1)'';
%        midpoint spacing
dzm=zeros(1,nz);
dzm(2:nz-1)=zm(1:nz-2)-zm(2:nz-1);
dzm(1)=dzm(2);
dzm(nz)=dzm(nz-1);

%        get dynamic modes
A = zeros(nz,nz);
B = zeros(nz,nz);
%             create matrices   
for i=2:nz-1
  A(i,i) = 1/(dz(i-1)*dzm(i))  + 1/(dz(i)*dzm(i));
  A(i,i-1) = -1/(dz(i-1)*dzm(i));
  A(i,i+1) = -1/(dz(i)*dzm(i));
end
for i=1:nz
  B(i,i)=N2(i);
end
%             set boundary conditions
A(1,1)=-1.;
A(nz,1)=-1.;

[wmodes,e] = eig(A,B);

%          extract eigenvalues
e=diag(e);
%
ind=find(imag(e)==0);
e=e(ind);
wmodes=wmodes(:,ind);
%
ind=find(e>=1.e-10);
e=e(ind);
wmodes=wmodes(:,ind);
%
[e,ind]=sort(e);
wmodes=wmodes(:,ind);

nm=length(e);
ce=1./sqrt(e);
%                   create pressure structure
pmodes=zeros(size(wmodes));

for i=1:nm
%           calculate first deriv of vertical modes
  pr=diff(wmodes(:,i));   
  pr(1:nz-1)= pr(1:nz-1)./dz(1:nz-1)';
  pr=pr*rho0*ce(i)*ce(i);
%       linearly interpolate back to original depths
  pmodes(2:nz-1,i)=.5*(pr(2:nz-1)+pr(1:nz-2));
  pmodes(1,i)=pr(1);
  pmodes(nz,i)=pr(nz-1);
end
