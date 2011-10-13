% Here is a simple script to test caleof.m from PCATOOL package.
% 

% gmaze@mit.edu
% 2006/07/31
%

clear

field = 1; % The best one

% Generate the data
x = -10 : 1 : 10 ;
y = x;
[X,Y] = meshgrid(x,y);

switch field
 case 1   % H. Bjornson and S.A. Venegas
   HH = 12*( 1.2 - 0.35*sqrt( (X-0).^2 + (Y-0).^2 ) );
   HH = HH - mean(mean(HH));
   H(1,:,:) = 1012 + HH;
   H(2,:,:) = 1012 + (1012-squeeze(H(1,:,:)));;
   H(3,:,:) = 1022.8 - 3.6*Y;
   H(4,:,:) = 1012 + (1012-squeeze(H(3,:,:)));
   H(5,:,:) = 1001.2 + 3.6*X;
   H(6,:,:) = 1012 + (1012-squeeze(H(5,:,:)));
  cont = [980:5:1050];
  N = 2;
  
 case 11   % H. Bjornson and S.A. Venegas modified
   % Here it must be only 2 non-zero EOF (45degre oriented ) of similar variance 50%
   H = ones(4,length(x),length(y));
   H(1,:,:) =  Y;
   H(2,:,:) = -Y;
   H(3,:,:) = X;
   H(4,:,:) = -X;
   N = 2;

 case 2 % Hartmann eg:  analysis in place
  a = [2 4 -6 8 ; 1 2 -3 4];
  H(1,:,:) = a(1,:)';
  H(2,:,:) = a(2,:)';
  
  % Get EOFs:
  N = 4;
  for method = 2 : -1 : 1
    [E,pc,expvar] = caleof(H,N,method);
    %E'
  end

  return
  
 case 3  % My signal
  np = 4; % Numer of random signal (...and eof)
  nt = 10; % Number of timestep
  H = zeros(nt,length(x),length(y));
  for ip = 1 : np
    xc(ip) = abs(fix(rand(1)*10));
    yc(ip) = abs(fix(rand(1)*10));
    if ip>=fix(np/2)
      xc(ip) = -xc(ip);
      yc(ip) = -yc(ip);
    end
    dd(ip) = fix(rand(1)*10);
  end %for ip
  f = 1/nt;
  for it = 1 : nt 
    H2 = zeros(length(x),length(y));
    for ip = 1 : ip
        if it==1,[xc(ip) yc(ip) dd(ip)],end
        HH = 12*( 1.2 - 0.35*sqrt(  ((X-xc(ip)).^2 + (Y-yc(ip)).^2 )/dd(ip) ));
        H2 = HH - mean(mean(HH));
	H(it,:,:) = squeeze(H(it,:,:)) + H2.*cos(pi*it/dd(ip)) ;
    end %for ip
  end %for it
  H = 1012 + H;
  cont = [980:5:1050];
  N = 3;
  
 case 4 % My signal 2
  x = -pi : pi/6 : pi ;
  y = x;
  [X,Y] = meshgrid(x,y);
  HH  = cos(X) + cos(Y);
  HH2 = cos(Y);
  nt = 12;
  for it = 1 : nt
%     H(it,:,:) = cos(pi*it/nt)*HH; cont=[-2 2];
     H(it,:,:) = 2*cos(pi*it/nt)*HH + 3*cos(pi*it/nt/2)*HH2; cont=[-10 10];
     xtrm(squeeze(H(it,:,:)));
  end
  cont=[-2 2];
  N = 3;
  
end % switch field

%return

% Plot field time serie:
if 1
figure;iw=2;jw=size(H,1)/iw;
set(gcf,'position',[11 533 560 420]);

for i=1:iw*jw
  C = squeeze(H(i,:,:));
  
  subplot(iw,jw,i);
  pcolor(X,Y,C);
  title(num2str(i));
  if i==1,cx=caxis;end
  axis square
  caxis(cx);

end %for i
end %fi
%return

% Get EOFs:
G = map2mat(ones(size(H,2),size(H,3)),H);

for method = 1 : 4
  [E,pc,expvar] = caleof(G,N,method);  
  eof = mat2map(ones(size(H,2),size(H,3)),E);
  

figure;iw=1;jw=N+1;
set(gcf,'MenuBar','none');
posi = [576 0 710 205];
set(gcf,'position',[posi(1) (method-1)*250+50 posi(3) posi(4)]);

for i=1:iw*jw
  if i<= iw*jw-1
    
  C = squeeze(eof(i,:,:));
  subplot(iw,jw,i);
  cont = 12;
  [cs,h] = contourf(X,Y,C,cont);
  clabel(cs,h); 
  title(strcat('EOF:',num2str(i),'/',num2str(expvar(i)),'%'));
  axis square;
  %caxis([cont(1) cont(end)]);
  
  else
  subplot(iw,jw,iw*jw);
  plot(pc');
  grid on
  xlabel('time')
  title('PC')
  legend(num2str([1:N]'),2);
  box on
  

  end %if
  
end %for i
suptitle(strcat('METHODE:',num2str(method)));

end %for method


