%ps3 command gets rotation matrix for adcp
%from manual, page 10

function [ve, vn, vup, verr] = rdi_coordTransform(adcp,cfg)

%to convert beam to instrument coords.
switch ( cfg.beam_pattern)
    case ('concave')
        c = -1; % for concave
    case ('convex')
        c = 1; % for convex
end

% Get values from RDI Coord Transformation Manual
% see also equation(2) of turbulent_slope.pdf

a = 1 / (2 * sin (cfg.beam_angle * pi / 180));
b = 1 / (4 * cos (cfg.beam_angle * pi / 180));

%not using "d" anymore
%d = a / sqrt(2); 

x = c*a * (adcp.v1 - adcp.v2); 
% (above) this appears to be a left-handed system, see 
% figure 1, equation 2 of ~/Documents/Texts/Thesis/adcp/turbulent_slope.pdf
% (below) try right-handed system:
x = -x;
y = c*a * (adcp.v4 - adcp.v3);

z = - b * (adcp.v1 + adcp.v2 + adcp.v3 + adcp.v4);
%err = d * (adcp.v1 + adcp.v2 - adcp.v3 - adcp.v4);
err = b * (adcp.v1 + adcp.v2 - adcp.v3 - adcp.v4);

M = zeros(length(adcp.heading),3,3);
U = zeros(length(adcp.heading),3,length(x(:,1)));
disp(size(M))

%to convert to ship or earth coords
for ii = 1:length(adcp.heading)

  % Heading angle, transform from map to math coordinates
  %H = (- adcp.heading(ii) + 90) * pi / 180; % convert ro radians
  % negative not needed because we are "unrotating"
  % 90 degree shift is unnecessary b/c we don't want to align
  % "north" with the x-axis
  
  H = (adcp.heading(ii)) * pi / 180; % convert ro radians
  
  % pitch and roll need to be negative

  %Undo pitch angle					    
  P = - adcp.pitch(ii)*pi/180; % convert to radians

  %Undo roll angle
  R = - adcp.roll(ii)*pi/180;  % convert to radians

  CH = cos(H); SH = sin(H);
  CP = cos(P); SP = sin(P);
  CR = cos(R); SR = sin(R);
    
  M(ii,:,:) = [ CH SH 0; ...   % generate rotation matrix
	       -SH CH 0; ...
	         0 0  1] * [1 0  0; ...
 		            0 CP SP; ...
		            0 -SP CP] * [CR 0 -SR; ...
		                         0  1  0; ...
		                         SR 0 CR];
  
  for jj = 1:length(x(:,1));
    u = [x(jj,ii); y(jj,ii); z(jj,ii)];
    U(ii,:,jj) = (squeeze(M(ii,:,:)) * u)';
  end
end

ve = squeeze(U(:,1,:))';
vn = squeeze(U(:,2,:))';
vup = squeeze(U(:,3,:))';
verr = err;



