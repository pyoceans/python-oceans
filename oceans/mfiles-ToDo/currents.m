function [varargout] = currents(varargin)

% This function takes the wind vector field and SSH field as inputs
% and returns Ekman and Geostrophic currents. This implements safe
% equator-crossing calculations and safe f-plane calculations using a
% hybrid of the models of Pond and Pickard and Lagerloef et al.
%
% CALLING THE FUNCTION
% ====================
% 
% Possible inputs are:
%  a) [LONG,LAT,SSH,GEO] = CURRENTS('SSH.DAT',RESOLUTION)
%  b) [LONG,LAT,WIND,EKMAN] = CURRENTS('WIND_EW.DAT','WIND_NS.DAT', ...
%		RESOLUTION)
%  c) [LONG,LAT,WIND,SSH,TOTAL,EKMAN,GEO] = ...
%		CURRENTS('WIND_EW.DAT','WIND_NS.DAT','SSH.DAT', ...
%		RESOLUTION)
%
% a) will return the Geostrophic Current
% b) will return the Ekman Current
% c) will return the Total Current, as well as the two components
%    individually.
%
% FORMAT OF THE INPUT FILES
% =========================
%
% The files 'SSH.DAT', 'WIND_EW.DAT' and 'WIND_NS.DAT' should all be in 
% the following format with *NO* header. The files need to give the data
% in column format as:
%
% lon(1)   lat(1)   value
% lon(2)   lat(1)   value
% lon(3)   lat(1)   value
%  ...      ...      ...
% lon(1)   lat(2)   value
% lon(2)   lat(2)   value
% lon(3)   lat(2)   value
%  ...      ...      ...
%  etc.
% where lon(1) is the westernmost longitude and lat(1) is the
% northernmost latitude.
%  
% Note that longitudes and latitudes should be in degrees and that the
% value for the wind should be in m/s (SI) and the value for SSH should be
% in in cm (CGI).
%
% It is assumed that the lon and lat points are equally spaced with
% a separation distance of RESOLUTION degrees.
%
% Also see comments at lines 139 and 263 of the code.
%
% FORMAT OF THE OUTPUT
% ====================
%
% LAT and LONG are returned as row and column vectors, respectively.
% GEO, EKMAN and TOTAL are returned as vectors of size [LAT LONG].

%							rsm 31Jul01 ||
%					     ramzi.mirshak@noaa.gov ||

% UPDATE: Limits dependency removed, and bug fixes reported by
% S. Joseph of INCOIS addressed.       rsm 12July08 || ramzi@dal.ca ||

% Load in the data

if(nargin < 2);
   error('Not enough inputs called to function CURRENTS');
elseif (nargin == 2);
   if(nargout ~= 4)
      error(['Must have 4 outputs for the inputs given to CURRENTS. See help comments for details.']);
   end
   disp('Function CURRENTS calculating geostrophic currents only.');
   flag = 0;
   fprintf('\nLoading data from files...');
     eval(['load ',varargin{1},' -ascii;']);
     sshname = getfilename(varargin{1});
     eval(['ssh = ',sshname,';']);
     ssh(:,[1 2]) = ssh(:,[2 1]);
     eval(['clear ',sshname]);
   fprintf('done!');
elseif (nargin == 3);
   if(nargout ~= 4)
      error(['Must have 4 outputs for the inputs given to CURRENTS. See help comments for details.']);
   end
   disp('Function CURRENTS calculating Ekman currents only.');
   flag = 2;

   fprintf('\nLoading data from files...');
     eval(['load ',varargin{2},' -ascii;']);
     eval(['load ',varargin{1},' -ascii;']);
     tauz_name = getfilename(varargin{1});
     taum_name = getfilename(varargin{2});
     eval(['u = ',tauz_name,';']);
     eval(['u(:,4) = ',taum_name,'(:,3);']);
     eval(['clear ', taum_name ,' ', tauz_name]);
     u(:,[1 2]) = u(:,[2 1]);
   fprintf('done!');

elseif (nargin == 4);
   if(nargout ~= 7)
      error(['Must have 7 outputs for the inputs given to CURRENTS. See help comments for details.']);
   end
   disp('Function CURRENTS calculating geostrophic and Ekman currents.');
   flag = 1;

   fprintf('\nLoading data from files...');
     eval(['load ',varargin{2},' -ascii;']);
     eval(['load ',varargin{1},' -ascii;']);
     uz_name = getfilename(varargin{1});
     um_name = getfilename(varargin{2});
     eval(['u = ',uz_name,';']);
     eval(['u(:,4) = ',um_name,'(:,3);']);
     eval(['clear ', um_name ,' ', uz_name]);
     u(:,[1 2]) = u(:,[2 1]);

     eval(['load ',varargin{3},' -ascii;']);
     sshname = getfilename(varargin{3});
     eval(['ssh = ',sshname,';']);
     ssh(:,[1 2]) = ssh(:,[2 1]);
     eval(['clear ',sshname]);
   fprintf('done!');

else
   error('Too many input files called to function CURRENTS');
end

resolution = varargin{end};

if(flag >= 1) % Determine Ekman currents
   fprintf('\nDetermining Ekman currents:\n\tFiltering Wind Speed...');

   latEk = [min(u(:,1)):resolution:max(u(:,1))];
   lonEk = [min(u(:,2)):resolution:max(u(:,2))];

   ux10 = reshape(squeeze(u(:,3)),length(lonEk),length(latEk))';
   uy10 = reshape(squeeze(u(:,4)),length(lonEk),length(latEk))';
   Y = reshape(squeeze(u(:,2)),length(lonEk),length(latEk))';

% << NOTE >> IF your plots are coming out upside down, uncomment
% these lines
%
%   ux10(1:end,:) = ux10(end:-1:1,:);
%   uy10(1:end,:) = uy10(end:-1:1,:);
%   Y(1:end,:) = Y(end:-1:1,:);

   fux10=davefilt2(ux10,5,5,'median');
   fuy10=davefilt2(uy10,5,5,'median');

   fprintf('done!\n\tDetermining wind stress...');

   umod10=sqrt(fux10.*fux10 + fuy10.*fuy10);

   % density of surface ocean water
   rho = 1025; % kg/m^3
   rho_air = 1.22; % density of air

   % angular frequency of earth's rotation
   om = 7.29e-5;

   % load drag coefficients for wind stress computation.
   load drag_coeff_smith1988.dat

   cdref=1e-3*drag_coeff_smith1988(:,2);
   u10ref=drag_coeff_smith1988(:,1);

   npts=size(fux10);
   nx=npts(2);
   ny=npts(1);

   cd(1:ny,1:nx) = NaN;

   for i= 1:ny,
      for j = 1:nx,
         if(~isnan(umod10(i,j)))
           cd(i,j) = linterp(u10ref,cdref,umod10(i,j));
         end
      end
   end

   taux = rho_air * cd .* umod10 .* fux10;
   tauy = rho_air * cd .* umod10 .* fuy10;

   fprintf('done!\n\tCalculating Ekman currents...');

   %======================
   % POND AND PICKARD PART
   %======================

   %===============================================================
   % now do coriolis parameter (take 1S to 1N out of consideration)

   fspin(1:ny,1:nx)=nan;
   ind=find(Y<=-1 | Y>=1);
   fspin(ind)=2*om*sin(2*pi*Y(ind)/360);

   %=================================================================
   % now get Ekman Depth (take out 1S to 1N again)
   % method uses empirical relationship outlined by Pond and Pickard

   eld(1:ny,1:nx)=nan;
   ind=find(Y<=-1 | Y>=1);
   eld(ind)=4.3*umod10(ind)./sqrt(sin(abs(2*pi*Y(ind)/360)));

   %=================================================================
   % now calculate Ekman components.
   % note: need density, rho, in kg m^-3.

   ue(1:ny,1:nx)=nan;
   ve(1:ny,1:nx)=nan;

   u_p = tauy./(eld.*fspin*rho);
   v_p = -taux./(eld.*fspin*rho);

   %======================
   % LAGERLOEF ET AL. PART
   %======================

   % fricton term from Lagerloef et al. 
   r = 2.15e-4; %m/s

   % depth term from Lagerloef et al.
   h = 32.5; %m
   f = 2 * om * sin(deg2rad(latEk)); 		% f [1/s]

   i = sqrt(-1); % reinitialize value   

   a2 = (r + i * f * h) ./ (r^2 + h^2 .* f.^2);

   tau = taux + i * tauy;
   Ut = (a2' * ones(size(lonEk))) .* tau / rho;

   u_l = real(Ut);
   v_l = imag(Ut);

   %=============================================================
   % COMBINE THE TWO PARTS OUTSIDE FOR LATITUDES GREATER THAN 5
   % DEGREES FROM THE EQUATOR
   %=============================================================
   % use the lengthscale of 2.2 degrees given by lagerloef et al.
   %-------------------------------------------------------------

   u_e(1:ny,1:nx) = nan;
   v_e(1:ny,1:nx) = nan;

   eq = find(u(:,1) <= 5);
   u_e(eq) = u_l(eq); v_e(eq) = v_l(eq);

   midlat = find(u(:,1) > 5);
   u_e(midlat) = exp(-((abs(u(midlat,1))-5)/2.2).^2) .* u_l(midlat) ...
		+ (1 - exp(-((abs(u(midlat,1))-5)/2.2).^2)) .* u_p(midlat);
   v_e(midlat) = exp(-((abs(u(midlat,1))-5)/2.2).^2) .* v_l(midlat) ...
		+ (1 - exp(-((abs(u(midlat,1))-5)/2.2).^2)) .* v_p(midlat);
   fprintf('done!');
end


if(flag <= 1) % Determine Geostrophic currents
   fprintf('\nDetermining geostrophic currents:\n\tFiltering data...');
   latG = [min(ssh(:,1)):resolution:max(ssh(:,1))];
   lonG = [min(ssh(:,2)):resolution:max(ssh(:,2))];
   SSH = reshape(squeeze(ssh(:,3)),length(lonG),length(latG))';

   % <<NOTE>> IF your plots are coming out upside down, uncomment
   % this line
   %   SSH(1:end,:) = SSH(end:-1:1,:);
   a = find(isnan(SSH));
   SSH2 = davefilt2(SSH,5,5,'median');
   SSH2(a) = NaN;

   fprintf('done!\n\tCalculating SSH gradient and determining currents...');

   %===========================
   % Determind the SSH gradient

   i = sqrt(-1);
   [FX, FY] = gradient(SSH2);
   Z = FX + i * FY;

   % convert dimensions of Z from cm/deg to m/m
   % Z = Z * (1 deg / 111.1175 km) * (1 m / 100 cm)
   Z = Z / 111177.5 / 100; 

   %==========================================================
   %Use SSH gradient (Z) to determine the geostrophic currents

   om = 7.29e-5; % angular frequency of earth's rotation [1/s]
   g = 9.8; 	 % gravitational acceleration [m/s^2]

   f = 2 * om * sin(deg2rad(latG)); 		% f [1/s]
   beta = 2 * om / 111000 * cos(deg2rad(latG)); % beta [1/m/s]

   % coefficients a1 and b1 for f-plane and beta plane components of 
   % flow.
   warning off % There is a divide by zero at the equator for f and at the poles for beta
   a1 = (1 - exp(-(latG/2.2).^2)) .* (g ./ f);
   a1(find(latG==0))=0;

   b1 = (exp(-(latG/2.2).^2)) .* (i * g ./ beta);
   b1(find(beta==0)) = 0;
   warning on % The divide by zero is finished.

   % Geostrophic Current :: f-plane and beta-plane

   Uf = i * (a1' * ones(size(lonG))) .* Z; 
   Ub = (b1(2:end-1)' * ones(size(lonG))) .* (Z(3:end,:)-Z(1:end-2,:)) ...
		./ (2 * 111177.5);
   Ub(2:end+1,:) = Ub;
   Ub(end+1,:) = 0;	

   u_g = real(Uf) + real(Ub);
   v_g = imag(Uf) + imag(Ub);

   fprintf('done!');
end

i = sqrt(-1);

if(flag == 0), %geostrophic only
   varargout{1} = lonG;
   varargout{2} = latG;
   varargout{3} = SSH2;
   varargout{4} = u_g + i * v_g;
elseif(flag == 2), %Ekman only
   varargout{1} = lonEk;
   varargout{2} = latEk;
   varargout{3} = ux10 + i * uy10;
   varargout{4} = u_e + i * v_e;
else, % both geostrophic and Ekman
   if(min(lonEk) == min(lonG) & max(lonEk) == max(lonG) & ...
	min(latEk) == min(latG) & max(latEk) == max(latG)),
      varargout{1} = lonEk;
      varargout{2} = latEk;
      varargout{3} = ux10 + i * uy10;
      varargout{4} = SSH2;
      varargout{6} = u_e + i * v_e;
      varargout{7} = u_g + i * v_g;
      varargout{5} = varargout{6} + varargout{7};
      minlon = min(lonEk);
      maxlon = max(lonEk);
      minlat = min(latEk);
      maxlat = max(latEk);
   else
      if(min(lonEk) < min(lonG))
         minlon = min(lonEk);
         difference = min(lonG) - min(lonEk);
         SSH2(:,difference*resolution+1:difference*resolution+end) = SSH2;
         u_g(:,difference*resolution+1:difference*resolution+end) = u_g;
         v_g(:,difference*resolution+1:difference*resolution+end) = v_g;
          
         u_g(:,1:difference*resolution) = NaN;
         SSH2(:,1:difference*resolution) = NaN;
         v_g(:,1:difference*resolution) = NaN;
      elseif(min(lonEk) > min(lonG))
         minlon = min(lonEg);
         difference = min(lonEk) - min(lonEg);
         u_e(:,difference*resolution+1:difference*resolution+end) = u_e;
         v_e(:,difference*resolution+1:difference*resolution+end) = v_e;
          
         u_e(:,1:difference*resolution) = NaN;
         v_e(:,1:difference*resolution) = NaN;
      else
         minlon = min(lonEk);
      end

      if(max(lonEk) > max(lonG));
         maxlon = max(lonEk);
         difference = max(lonEk) - max(lonG);
         SSH2(:,end+1:end+difference*resolution) = NaN;
         u_g(:,end+1:end+difference*resolution) = NaN;
         v_g(:,end+1:end+difference*resolution) = NaN;
      elseif(max(lonEk) < max(lonG));
         maxlon = max(lonG);
         difference = max(lonG) - max(lonEk);
         u_e(:,end+1:end+difference*resolution) = NaN;
         v_e(:,end+1:end+difference*resolution) = NaN;
      else
         maxlon = max(lonEk);
      end

      if(min(latEk) < min(latG))
         minlat = min(latEk);
         difference = min(latG) - min(latEk);
         SSH2(difference*resolution+1:difference*resolution+end,:) = SSH2;
         u_g(difference*resolution+1:difference*resolution+end,:) = u_g;
         v_g(difference*resolution+1:difference*resolution+end,:) = v_g;
          
         SSH2(1:difference*resolution,:) = NaN;
         u_g(1:difference*resolution,:) = NaN;
         v_g(1:difference*resolution,:) = NaN;
      elseif(min(latEk) > min(latG))
         minlat = min(latG)
         difference = min(latEk) - min(latG);
         u_e(difference*resolution+1:difference*resolution+end,:) = u_e;
         v_e(difference*resolution+1:difference*resolution+end,:) = v_e;
          
         u_e(1:difference*resolution,:) = NaN;
         v_e(1:difference*resolution,:) = NaN;
      else
         minlat = min(latEk);
      end

      if(max(latEk) > max(latG));
         maxlat = max(latEk);
         difference = max(latEk) - max(latG);
         SSH2(end+1:end+difference*resolution,:) = NaN;
         u_g(end+1:end+difference*resolution,:) = NaN;
         v_g(end+1:end+difference*resolution,:) = NaN;
      elseif(max(latEk) < max(latG));
         maxlat = max(latG);
         difference = max(latG) - max(latEk);
         u_e(end+1:end+difference*resolution,:) = NaN;
         v_e(end+1:end+difference*resolution,:) = NaN;
      else
         maxlat = max(latEk);
      end
   end
   i = sqrt(-1);
   varargout{1} = [minlon:resolution:maxlon];
   varargout{2} = [minlat:resolution:maxlat];
   varargout{3} = ux10 + i * uy10;
   varargout{4} = SSH2;
   varargout{6} = u_e + i * v_e;
   varargout{7} = u_g + i * v_g;
   varargout{5} = varargout{6} + varargout{7};
end

fprintf('\n')

