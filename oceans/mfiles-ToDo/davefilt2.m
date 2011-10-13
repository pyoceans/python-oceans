function y = davefilt2(A,x,y,type)

% function applies x by y 2-d Median filter to array
% Routine filters out NaN values, unlike the native [and correct]
% medfilt2 command in the Image Processing Toolbox
%
% Apri 2000 DGF

% set size for search radius
xs = round(x/2);
ys = round(y/2);

% get size of array
s=size(A);
ny=s(1);
nx=s(2);

fA(1:ny,1:nx)=nan;

for i = xs+1:nx-xs,
  for j = ys+1:ny-ys,
    tmp(1:2*ys+1,1:2*xs+1)=A(j-ys:j+ys,i-xs:i+xs);
    ind=find(~isnan(tmp));
    if length(ind)>0
      if type == 'median'
        fA(j,i)=median(tmp(ind));
      elseif type == 'mean'
        fA(j,i) = mean(tmp(ind));
      elseif type == 'max'
        fA(j,i) = max(tmp(ind));
      elseif type == 'min'
        fA(j,i) = min(tmp(ind));
      else 
        fA(j,i) = median(tmp(ind));
      end
    end
  end
end

y = fA;


      


