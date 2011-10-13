clear all, clc

load u.dat
u = u;
[line column] = size(u); % line="time" and column="stations"

u = detrend(u','linear');

u = u./repmat(std(u,0,2),1,column);

C = u*u'./(line-1); % covariance matrix of data

[evec,eval] = eig(C,'nobalance'); %calc eigvalues and vectors
eval2 = eval;
eval = diag(eval,0);                                  %eigenvalues
[new_eval,ord]=sort(eval);                             %arrange eigenvalues
new_evec=evec(:,[ord]);
if mean(new_evec(:,1)-evec(:,1))~=0;
    eval=new_eval;
    evec=new_evec;
    disp('vectors sorted');
end
if abs(eval(1))<abs(eval(length(eval)))
    eval=eval(length(eval):-1:1);
    evec=evec(:,length(eval):-1:1);
end

for i=1:6
  evec_new(:,i) = evec(:,i)./max(abs(evec(:,i)));
end

amp = (evec'*data)'

figure(1)
  subplot(311), hold on
    plot(-amp(:,1),'k','linewidth',2.5)
    plot([0 8],[0 0],'k')
    axis([0 8 -3 3])
  subplot(312), hold on
    plot(amp(:,2),'k','linewidth',2.5)
    plot([0 8],[0 0],'k')
    axis([0 8 -3 3])
  subplot(313), hold on
    plot(amp(:,3),'k','linewidth',2.5)
    plot([0 8],[0 0],'k')
    axis([0 8 -3 3])

    packrows(3,1)