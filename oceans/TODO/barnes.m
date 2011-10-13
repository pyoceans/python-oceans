function zo = barnes(x,y,z,xo,yo,xr,yr,npassmax)
%
% USAGE
%
% Barnes gridding:
% 
% ZO = barnes(X,Y,Z,XO,YO,XR,YR) 
%
%  -OR-
%
% ZO = barnes(X,Y,Z,XO,YO,XR,YR,NPASS) 
%
% (to set the number of Barnes iterations, NPASS) 
%
% DESCRIPTION
%
% Barnes objective analysis:  a successive corrections scheme
% which is an easy-to-use cousin of kriging, classic objective
% analysis, or optimal interpolation. 
%
% Returns matrix ZO containing elements corresponding to the elements
% of matrices XO and YO and determined by Barnes objective analysis.
% X and Y are vectors containing the input data coordinates, and Z
% is a vector with the input data.
%
% XR and YR are the Barnes smoothing length scales in the x and
% y senses.  These remain fixed throughout the iterations (as
% recommended in the later Barnes papers).  Optional NPASS sets the 
% number of Barnes iterations (defaults to 3, which is usually
% sufficient).
%
% HISTORY
%
% Bug fix, 8-Sep-10, Carlos Carrillo 
% Original, 10-Aug-09, S. Pierce
%
% REFERENCES
% 
% Barnes, S. L. (1994) Applications of the Barnes objective analysis
% scheme.  Part I:  effects of undersampling, wave position, and station
% randomness.  J. of Atmos. and Oceanic Tech., 11, 1433-1448.
%
% Barnes, S. L. (1994) Applications of the Barnes objective analysis
% scheme.  Part II:  Improving derivative estimates.  J. of Atmos. and 
% Oceanic Tech., 11, 1449-1458.
%
% Barnes, S. L. (1994) Applications of the Barnes objective analysis
% scheme.  Part III:  Tuning for minimum error.  J. of Atmos. and Oceanic 
% Tech., 11, 1459-1479.
%
% Daley, R. (1991) Atmospheric data analysis, Cambridge Press, New York.
% Section 3.6.

if ( exist('xyzchk') )
	[msg,x,y,z,xo,yo]=xyzchk(x,y,z,xo,yo);
	if length(msg)>0, error(msg); end
end

if ( nargin == 7 )
	npassmax = 3;	% default # of passes if not selected
end

in=find(~isnan(x)); x=x(in); y=y(in); z=z(in);

x = row2col(x); y = row2col(y); z = row2col(z);
% 
[m,n] = size(xo); 

for i=1:m
	for j=1:n
		xp((i-1)*n+j) = xo(i,j);
		yp((i-1)*n+j) = yo(i,j);
		zp((i-1)*n+j,1) = 0.;
	end
end

xr2=xr^2; yr2=yr^2;

% first analysis at output pts zp
for np=1:m*n
	dx=abs(x-xp(np)); dy=abs(y-yp(np));
	w=exp (-dx.^2/xr2 - dy.^2/yr2 );
    	zp(np)=sum(z.*w)/sum(w);
end


zp = row2col ( zp );
zpp = zp;
fprintf(1,'     std of initial solution:  %f\n',std(zpp));

zpb = row2col(zeros(size(z)));


if (npassmax > 1 ) 

for npass=2:npassmax

	for ni=1:size(x)
		dx=abs(x-x(ni)); dy=abs(y-y(ni));
		w=exp (-dx.^2/xr2 - dy.^2/yr2 );
		if ( mod(npass,2) == 0 )
			zpa(ni)=zpb(ni)+sum((z-zpb).*w)/sum(w);
		else
			zpb(ni)=zpa(ni)+sum((z-zpa).*w)/sum(w);
		end
	end
	zpa = row2col(zpa); zpb = row2col(zpb);

	for np=1:m*n 
		dx=abs(x-xp(np)); dy=abs(y-yp(np));
		w=exp (-dx.^2/xr2 - dy.^2/yr2 );
		if ( mod(npass,2) == 0 )
			zp(np)=zp(np)+sum((z-zpa).*w)/sum(w);
		else
			zp(np)=zp(np)+sum((z-zpb).*w)/sum(w);
		end
	end
	zp = row2col(zp);
	if ( mod(npass,2) == 0 )
		zpn = zp;
	else
		zpp = zp;
	end
	fprintf(1,'rms z adjustment after pass %d:  %f\n',npass,std(zpn-zpp));

end

end

for i=1:m
	for j=1:n
		zo(i,j) = zp((i-1)*n+j,1) ;
	end
end


function[col]=row2col(row)

if size(row,2)>size(row,1)
	if ~ischar(row)
     		col=conj(row');
	else 	
		col=row';
	end
else
	col=row;
end

