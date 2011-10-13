clear all, clc
time   =  [1:8];                                 % book data
site1u =  [-.3,-.1,-.1,.2,.3,.5,.2,-.5];         % book data
site1v =  [0.0,.3,-.4,.6,-.1,0.0,.2,-.9];        % book data
site2u =  [.4,.4,0.0,0.0,-.6,.9,-.1,0.0];        % book data
site2v = -[.4,.3,.5,.6,.3,.6,.7,.6];            % book data
site3u =  [-.8,-1.1,0.0,-.7,0.0,.6,1.2,0.0];     % book data
site3v =  [-1.4,0.0,-2.5,.4,-.3,.3,-2.8,-1.8];   % book data

site1uP = detrend(site1u-mean(site1u),'linear');    % demean & detrend
site1vP = detrend(site1v-mean(site1v),'linear');    % demean & detrend
site2uP = detrend(site2u-mean(site2u),'linear');    % demean & detrend
site2vP = detrend(site2v-mean(site2v),'linear');    % demean & detrend
site3uP = detrend(site3u-mean(site3u),'linear');    % demean & detrend
site3vP = detrend(site3v-mean(site3v),'linear');    % demean & detrend

site1uP_stdev = std(site1uP);                   % calc std deviation
site1vP_stdev = std(site1vP);                   % calc std deviation
site2uP_stdev = std(site2uP);                   % calc std deviation
site2vP_stdev = std(site2vP);                   % calc std deviation
site3uP_stdev = std(site3uP);                   % calc std deviation
site3vP_stdev = std(site3vP);                   % calc std deviation

site1u_norm = site1uP/site1uP_stdev;            %normalize the data
site1v_norm = site1vP/site1vP_stdev;            %normalize the data
site2u_norm = site2uP/site2uP_stdev;            %normalize the data
site2v_norm = site2vP/site2vP_stdev;            %normalize the data
site3u_norm = site3uP/site3uP_stdev;            %normalize the data
site3v_norm = site3vP/site3vP_stdev;            %normalize the data

data = [site1u_norm;...                         %create data marix
        site1v_norm;...
        site2u_norm;...
        site2v_norm;...
        site3u_norm;...
        site3v_norm];

data2= [site1u;...                         %create data marix
        site1v;...
        site2u;...
        site2v;...
        site3u;...
        site3v];

C    = data*transpose(data)./(8-1);              %covariance matrix of data

[evec,eval] = eig(C,'nobalance');                   %calc eigvalues and vectors
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