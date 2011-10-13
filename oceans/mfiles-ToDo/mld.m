function h=mld(t,depth,toff);

[len,num]=size(t);
h=nan*ones(len,1);
for i=find(isfinite(t(:,1)))'
   T=t(i,:)'-(t(i,1)-toff);
   ii=find(~isnan(T));
   T=T(ii);
   Z=depth(ii);
   pp=spline(Z,T);
   i0=min(find(T < 0));
   [breaks,coefs,l,k]=unmkpp(pp);
   r=roots(coefs(i0-1,:))+breaks(i0-1);
   h(i)=max(r(find(r < Z(i0) & r > Z(i0-1) & imag(r) == 0)));
end
