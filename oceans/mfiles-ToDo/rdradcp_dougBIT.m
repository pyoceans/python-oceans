function [adcp,cfg,ens]=rdradcp(name,varargin);
% RDRADCP  Read (raw binary) RDI ADCP files, 
%  ADCP=RDRADCP(NAME) reads the raw binary RDI BB/Workhorse ADCP file NAME and
%  puts all the relevant configuration and measured data into a data structure 
%  ADCP (which is self-explanatory). This program is designed for handling data
%  recorded by moored instruments (primarily Workhorse-type but can also read
%  Broadband) and then downloaded post-deployment. For vessel-mount data I
%  usually make p-files (which integrate nav info and do coordinate transformations)
%  and then use RDPADCP. 
%
%  This current version does have some handling of VMDAS and WINRIVER output
%  files, but it is still 'beta'.
%
%  [ADCP,CFG]=RDRADCP(...) returns configuration data in a
%  separate data structure.
%
%  Various options can be specified on input:
%  [..]=RDRADCP(NAME,NUMAV) averages NUMAV ensembles together in the result.
%  [..]=RDRADCP(NAME,NUMAV,NENS) reads only NENS ensembles (-1 for all).
%  [..]=RDRADCP(NAME,NUMAV,[NFIRST NEND]) reads only the specified range
%   of ensembles. This is useful if you want to get rid of bad data before/after
%   the deployment period.
%
%  Note - sometimes the ends of files are filled with garbage. In this case you may
%         have to rerun things explicitly specifying how many records to read (or the
%         last record to read). I don't handle bad data very well.
%
%       - I don't read in absolutely every parameter stored in the binaries;
%         just the ones that are 'most' useful.
%
%  String parameter/option pairs can be added after these initial parameters:
%
%  'baseyear'    : Base century for BB/v8WH firmware (default to 2000).
%
%  'despike'    : [ 'no' | 'yes' | 3-element vector ]
%                 Controls ensemble averaging. With 'no' a simple mean is used 
%                 (default). With 'yes' a mean is applied to all values that fall 
%                 within a window around the median (giving some outlier rejection). 
%                 This is useful for noisy data. Window sizes are [.3 .3 .3] m/s 
%                 for [ horiz_vel vert_vel error_vel ] values. If you want to 
%                 change these values, set 'despike' to the 3-element vector.
%
% R. Pawlowicz (rich@ocgy.ubc.ca) - 17/09/99

% R. Pawlowicz - 17/Oct/99 
%          5/july/00 - handled byte offsets (and mysterious 'extra" bytes) slightly better, Y2K
%          5/Oct/00 - bug fix - size of ens stayed 2 when NUMAV==1 due to initialization,
%                     hopefully this is now fixed.
%          10/Mar/02 - #bytes per record changes mysteriously,
%                      tried a more robust workaround. Guess that we have an extra
%                      2 bytes if the record length is even?
%          28/Mar/02 - added more firmware-dependent changes to format; hopefully this
%                      works for everything now (put previous changes on firmer footing?)
%          30/Mar/02 - made cfg output more intuitive by decoding things.
%                    - An early version of WAVESMON and PARSE which split out this
%                      data from a wave recorder inserted an extra two bytes per record.
%                      I have removed the code to handle this but if you need it see line 509
%         29/Nov/02  - A change in the bottom-track block for version 4.05 (very old!).
%         29/Jan/03  - Status block in v4.25 150khzBB two bytes short?
%         14/Oct/03  - Added code to at least 'ignore' WinRiver GPS blocks.
%         11/Nov/03  - VMDAS navigation block, added hooks to output
%                      navigation data.

num_av=5;   % Block filtering and decimation parameter (# ensembles to block together).
nens=-1;   % Read all ensembles.
century=2000;  % ADCP clock does not have century prior to firmware 16.05.
vels='no';   % Default to simple averaging

lv=length(varargin);
if lv>=1 & ~isstr(varargin{1}),
  num_av=varargin{1}; % Block filtering and decimation parameter (# ensembles to block together).
  varargin(1)=[];
  lv=lv-1;
  if lv>=1 & ~isstr(varargin{1}),
    nens=varargin{1};
    varargin(1)=[];
    lv=lv-1;
  end;
end;

% Read optional args
while length(varargin)>0,
 switch varargin{1}(1:3),
	 case 'bas',
	   century = varargin{2};
	 case 'des',
	   if isstr(varargin{2}),
	    if strcmp(varargin{2},'no'), vels='no';
	    else vels=[.3 .3 .3]; end;
	   else
	    vels=varargin{2}; 
	   end;   
	 otherwise,
	   error(['Unknown command line option  ->' varargin{1}]);
   end;
   varargin([1 2])=[];
end;   	          	




% Check file information first

naminfo=dir(name);

if isempty(naminfo),
  fprintf('ERROR******* Can''t find file %s\n',name);
  return;
end;

fprintf('\nOpening file %s\n\n',name);
fd=fopen(name,'r','ieee-le');

% Read first ensemble to initialize parameters

[ens,hdr,cfg]=rd_buffer(fd,-2); % Initialize and read first two records
fseek(fd,0,'bof');              % Rewind
 
if (cfg.prog_ver<16.05 & cfg.prog_ver>5.999) | cfg.prog_ver<5.55,
  fprintf('**************Assuming that the century begins year %d *********** \n\n',century);
else
  century=0;  % century included in clock.  
end;

dats=datenum(century+ens.rtc(1,:),ens.rtc(2,:),ens.rtc(3,:),ens.rtc(4,:),ens.rtc(5,:),ens.rtc(6,:)+ens.rtc(7,:)/100);
t_int=diff(dats);
fprintf('Record begins at %s\n',datestr(dats(1),0));
fprintf('Ping interval appears to be %s\n',datestr(t_int,13));


% Estimate number of records (since I don't feel like handling EOFs correctly,
% we just don't read that far!)


% Now, this is a puzzle - it appears that this is not necessary in
% a firmware v16.12 sent to me, and I can't find any example for
% which it *is* necessary so I'm not sure why its there. It could be
% a leftoever from dealing with the bad WAVESMON/PARSE problem (now
% fixed) that inserted extra bytes.
% ...So its out for now.
%if cfg.prog_ver>=16.05, extrabytes=2; else extrabytes=0; end; % Extra bytes
extrabytes=0;

if length(nens)==1,
  if nens==-1,
    nens=fix(naminfo.bytes/(hdr.nbyte+2+extrabytes));
    fprintf('\nEstimating %d ensembles in this file\nReducing by a factor of %d\n',nens,num_av);  
  else
    fprintf('\nReading %d ensembles in this file\nReducing by a factor of %d\n',nens,num_av); 
  end; 
else
  fprintf('\nReading ensembles %d-%d in this file\nReducing by a factor of %d\n',nens,num_av); 
  fseek(fd,(hdr.nbyte+2+extrabytes)*(nens(1)-1),'bof');
 nens=diff(nens)+1;
end;

if num_av>1,
  if isstr(vels),
     fprintf('\n Simple mean used for ensemble averaging\n');
  else
     fprintf('\n Averaging after outlier rejection with parameters [%f %f %f]\n',vels);
  end;
end;
   
% Number of records after averaging.

n=fix(nens/num_av);

% Structure to hold all ADCP data 
% Note that I am not storing all the data contained in the raw binary file, merely
% things I think are useful.

switch cfg.sourceprog,
  case 'WINRIVER',
    adcp=struct('name','adcp','config',cfg,'mtime',zeros(1,n),'number',zeros(1,n),'BIT',zeros(1,n),'pitch',zeros(1,n),...
        	'roll',zeros(1,n),'heading',zeros(1,n),'pitch_std',zeros(1,n),...
        	'roll_std',zeros(1,n),'heading_std',zeros(1,n),'depth',zeros(1,n),...
        	'temperature',zeros(1,n),'salinity',zeros(1,n),...
        	'pressure',zeros(1,n),'pressure_std',zeros(1,n),...
        	'east_vel',zeros(cfg.n_cells,n),'north_vel',zeros(cfg.n_cells,n),'vert_vel',zeros(cfg.n_cells,n),...
        	'error_vel',zeros(cfg.n_cells,n),'corr',zeros(cfg.n_cells,4,n),...
        	'status',zeros(cfg.n_cells,4,n),'intens',zeros(cfg.n_cells,4,n),...
	        'bt_range',zeros(4,n),'bt_vel',zeros(4,n),...
            'nav_longitude',zeros(1,n),'nav_latitude',zeros(1,n),'adc',zeros(8,n));
  case 'VMDAS',
    adcp=struct('name','adcp','config',cfg,'mtime',zeros(1,n),'number',zeros(1,n),'BIT',zeros(1,n),'pitch',zeros(1,n),...
        	'roll',zeros(1,n),'heading',zeros(1,n),'pitch_std',zeros(1,n),...
        	'roll_std',zeros(1,n),'heading_std',zeros(1,n),'depth',zeros(1,n),...
        	'temperature',zeros(1,n),'salinity',zeros(1,n),...
        	'pressure',zeros(1,n),'pressure_std',zeros(1,n),...
        	'east_vel',zeros(cfg.n_cells,n),'north_vel',zeros(cfg.n_cells,n),'vert_vel',zeros(cfg.n_cells,n),...
        	'error_vel',zeros(cfg.n_cells,n),'corr',zeros(cfg.n_cells,4,n),...
        	'status',zeros(cfg.n_cells,4,n),'intens',zeros(cfg.n_cells,4,n),...
	        'bt_range',zeros(4,n),'bt_vel',zeros(4,n),...
	        'nav_smtime',zeros(1,n),'nav_emtime',zeros(1,n),...
	        'nav_slongitude',zeros(1,n),'nav_elongitude',zeros(1,n),...
	        'nav_slatitude',zeros(1,n),'nav_elatitude',zeros(1,n),'adc',zeros(8,n));
  otherwise 
    adcp=struct('name','adcp','config',cfg,'mtime',zeros(1,n),'number',zeros(1,n),'BIT',zeros(1,n),'pitch',zeros(1,n),...
        	'roll',zeros(1,n),'heading',zeros(1,n),'pitch_std',zeros(1,n),...
        	'roll_std',zeros(1,n),'heading_std',zeros(1,n),'depth',zeros(1,n),...
        	'temperature',zeros(1,n),'salinity',zeros(1,n),...
        	'pressure',zeros(1,n),'pressure_std',zeros(1,n),...
        	'east_vel',zeros(cfg.n_cells,n),'north_vel',zeros(cfg.n_cells,n),'vert_vel',zeros(cfg.n_cells,n),...
        	'error_vel',zeros(cfg.n_cells,n),'corr',zeros(cfg.n_cells,4,n),...
        	'status',zeros(cfg.n_cells,4,n),'intens',zeros(cfg.n_cells,4,n),...
		    'bt_range',zeros(4,n),'bt_vel',zeros(4,n),...
            'adc',zeros(8,n),'xmit_current',zeros(1,n),'xmit_voltage',zeros(1,n),...
            'percent_good',zeros(cfg.n_cells,4,n));
end;


% Calibration factors for backscatter data

clear global ens
% Loop for all records
for k=1:n,

  % Gives display so you know something is going on...
    
  if rem(k,50)==0,  fprintf('\n%d',k*num_av);end;
  fprintf('.');

  % Read an ensemble
  
  ens=rd_buffer(fd,num_av);


  if ~isstruct(ens), % If aborting...
    fprintf('Only %d records found..suggest re-running RDRADCP using this parameter\n',(k-1)*num_av);
    fprintf('(If this message preceded by a POSSIBLE PROGRAM PROBLEM message, re-run using %d)\n',(k-1)*num_av-1);
    break;
  end;
    
  dats=datenum(century+ens.rtc(1,:),ens.rtc(2,:),ens.rtc(3,:),ens.rtc(4,:),ens.rtc(5,:),ens.rtc(6,:)+ens.rtc(7,:)/100);
  adcp.mtime(k)=median(dats);  
  adcp.number(k)      =ens.number(1);
  adcp.heading(k)     =mean(ens.heading);
  adcp.pitch(k)       =mean(ens.pitch);
  adcp.roll(k)        =mean(ens.roll);
  adcp.heading_std(k) =mean(ens.heading_std);
  adcp.pitch_std(k)   =mean(ens.pitch_std);
  adcp.roll_std(k)    =mean(ens.roll_std);
  adcp.depth(k)       =mean(ens.depth);
  adcp.temperature(k) =mean(ens.temperature);
  adcp.salinity(k)    =mean(ens.salinity);
  adcp.pressure(k)    =mean(ens.pressure);
  adcp.pressure_std(k)=mean(ens.pressure_std);
  if isstr(vels),
    adcp.east_vel(:,k)    =nmean(ens.east_vel ,2);
    adcp.north_vel(:,k)   =nmean(ens.north_vel,2);
    adcp.vert_vel(:,k)    =nmean(ens.vert_vel ,2);
    adcp.error_vel(:,k)   =nmean(ens.error_vel,2);
  else
   adcp.east_vel(:,k)    =nmedian(ens.east_vel  ,vels(1),2);
   adcp.north_vel(:,k)   =nmedian(ens.north_vel,vels(1),2);
   adcp.vert_vel(:,k)    =nmedian(ens.vert_vel  ,vels(2),2);
   adcp.error_vel(:,k)   =nmedian(ens.error_vel,vels(3),2);
  end;
  
  adcp.corr(:,:,k)      =nmean(ens.corr,3);        % added correlation RKD 9/00
  adcp.status(:,:,k)	=nmean(ens.status,3);   
  
  adcp.intens(:,:,k)   =nmean(ens.intens,3);
  
  adcp.bt_range(:,k)   =nmean(ens.bt_range,2);
  adcp.bt_vel(:,k)     =nmean(ens.bt_vel,2);
%added by douglas j schillinger Aug 27 2004
  adcp.xmit_current(k) = mean(ens.adc(1,:));
  adcp.xmit_voltage(k) = mean(ens.adc(2,:));
  adcp.percent_good(:,:,k) = nmean(ens.percent,3);
  adcp.adc(:,k) = nmean(ens.adc,2);
  adcp.BIT(k) = mean(ens.BIT);
  %end additions

  switch cfg.sourceprog,
    case 'WINRIVER',
     adcp.nav_longitude(k)=nmean(ens.slongitude);
     adcp.nav_latitude(k)=nmean(ens.slatitude);  
   case 'VMDAS',
     adcp.nav_smtime(k)   =ens.smtime(1);
     adcp.nav_emtime(k)   =ens.emtime(end);
     adcp.nav_slatitude(k)=ens.slatitude(1);
     adcp.nav_elatitude(k)=ens.elatitude(end);
     adcp.nav_slongitude(k)=ens.slongitude(1);
     adcp.nav_elongitude(k)=ens.elongitude(end);
  end;   
end;  

fprintf('\n');
fclose(fd);

%added by Douglas J Schillinger August 27 2004
switch (cfg.coord_sys)
    case ('beam')
        disp('Need to do coord transformation')
        adcp.v1 = adcp.east_vel;
        adcp.v2 = adcp.north_vel;
        adcp.v3 = adcp.vert_vel;
        adcp.v4 = adcp.error_vel;
        %[adcp.east_vel adcp.north_vel adcp.vert_vel adcp.error_vel] = rdi_coordTransform(adcp,cfg);
    case ('instrument')
        adcp.x = adcp.east_vel;
        adcp.y = adcp.north_vel;
        adcp.z = adcp.vert_vel;
        %error velocity is still error velocity
        %[adcp.east_vel adcp.north_vel adcp.vert_vel adcp.error_vel] = rdi_coordTransform(adcp,cfg);
    case ('ship')
    case ('earth')
    otherwise
        disp('Ooops');
end % swtich, end addition


%-------------------------------------
function hdr=rd_hdr(fd);
% Read config data

cfgid=fread(fd,1,'uint16');
if cfgid~=hex2dec('7F7F'),
 error(['File ID is ' dec2hex(cfgid) ' not 7F7F - data corrupted or not a BB/WH raw file?']);
end; 

hdr=rd_hdrseg(fd);

%-------------------------------------
function cfg=rd_fix(fd);
% Read config data

cfgid=fread(fd,1,'uint16');
if cfgid~=hex2dec('0000'),
 warning(['Fixed header ID ' cfgid 'incorrect - data corrupted or not a BB/WH raw file?']);
end; 

cfg=rd_fixseg(fd);



%--------------------------------------
function [hdr,nbyte]=rd_hdrseg(fd);
% Reads a Header

hdr.nbyte          =fread(fd,1,'int16'); 
fseek(fd,1,'cof');
ndat=fread(fd,1,'int8');
hdr.dat_offsets    =fread(fd,ndat,'int16');
nbyte=4+ndat*2;

%-------------------------------------
function opt=getopt(val,varargin);
% Returns one of a list (0=first in varargin, etc.)
if val+1>length(varargin),
	opt='unknown';
else
   opt=varargin{val+1};
end;
   			
%
%-------------------------------------
function [cfg,nbyte]=rd_fixseg(fd);
% Reads the configuration data from the fixed leader

%%disp(fread(fd,10,'uint8'))
%%fseek(fd,-10,'cof');

cfg.name='wh-adcp';
cfg.sourceprog='instrument';  % default - depending on what data blocks are
                              % around we can modify this later in rd_buffer.
cfg.prog_ver       =fread(fd,1,'uint8')+fread(fd,1,'uint8')/100;

if fix(cfg.prog_ver)==4 | fix(cfg.prog_ver)==5,
    cfg.name='bb-adcp';
elseif fix(cfg.prog_ver)==8 | fix(cfg.prog_ver)==16,
    cfg.name='wh-adcp';
else
    cfg.name='unrecognized firmware version'   ;    
end;    

config         =fread(fd,2,'uint8');  % Coded stuff
cfg.config          =[dec2base(config(2),2,8) '-' dec2base(config(1),2,8)];
 cfg.beam_angle     =getopt(bitand(config(2),3),15,20,30);
 cfg.beam_freq      =getopt(bitand(config(1),7),75,150,300,600,1200,2400);
 cfg.beam_pattern   =getopt(bitand(config(1),8)==8,'concave','convex'); % 1=convex,0=concave
 cfg.orientation    =getopt(bitand(config(1),128)==128,'down','up');    % 1=up,0=down
cfg.simflag        =getopt(fread(fd,1,'uint8'),'real','simulated'); % Flag for simulated data
fseek(fd,1,'cof'); 
cfg.n_beams        =fread(fd,1,'uint8');
cfg.n_cells        =fread(fd,1,'uint8');
cfg.pings_per_ensemble=fread(fd,1,'uint16');
cfg.cell_size      =fread(fd,1,'uint16')*.01;	 % meters
cfg.blank          =fread(fd,1,'uint16')*.01;	 % meters
cfg.prof_mode      =fread(fd,1,'uint8');         %
cfg.corr_threshold =fread(fd,1,'uint8');
cfg.n_codereps     =fread(fd,1,'uint8');
cfg.min_pgood      =fread(fd,1,'uint8');
cfg.evel_threshold =fread(fd,1,'uint16');
cfg.time_between_ping_groups=sum(fread(fd,3,'uint8').*[60 1 .01]'); % seconds
coord_sys      =fread(fd,1,'uint8');                                % Lots of bit-mapped info
  cfg.coord=dec2base(coord_sys,2,8);
  cfg.coord_sys      =getopt(bitand(bitshift(coord_sys,-3),3),'beam','instrument','ship','earth');
  cfg.use_pitchroll  =getopt(bitand(coord_sys,4)==4,'no','yes');  
  cfg.use_3beam      =getopt(bitand(coord_sys,2)==2,'no','yes');
  cfg.bin_mapping    =getopt(bitand(coord_sys,1)==1,'no','yes');
cfg.xducer_misalign=fread(fd,1,'int16')*.01;    % degrees
cfg.magnetic_var   =fread(fd,1,'int16')*.01;	% degrees
cfg.sensors_src    =dec2base(fread(fd,1,'uint8'),2,8);
cfg.sensors_avail  =dec2base(fread(fd,1,'uint8'),2,8);
cfg.bin1_dist      =fread(fd,1,'uint16')*.01;	% meters
cfg.xmit_pulse     =fread(fd,1,'uint16')*.01;	% meters
cfg.water_ref_cells=fread(fd,2,'uint8');
cfg.fls_target_threshold =fread(fd,1,'uint8');
fseek(fd,1,'cof');
cfg.xmit_lag       =fread(fd,1,'uint16')*.01; % meters
nbyte=40;

if cfg.prog_ver>=8.14,  % Added CPU serial number with v8.14
  cfg.serialnum      =fread(fd,8,'uint8');
  nbyte=nbyte+8; 
end;

if cfg.prog_ver>=8.24,  % Added 2 more bytes with v8.24 firmware
  cfg.sysbandwidth  =fread(fd,2,'uint8');
  nbyte=nbyte+2;
end;

if cfg.prog_ver>=16.05,                      % Added 1 more bytes with v16.05 firmware
  cfg.syspower      =fread(fd,1,'uint8');
  nbyte=nbyte+1;
end;

% It is useful to have this precomputed.

cfg.ranges=cfg.bin1_dist+[0:cfg.n_cells-1]'*cfg.cell_size;
if cfg.orientation==1, cfg.ranges=-cfg.ranges; end
	
	
%-----------------------------
function [ens,hdr,cfg]=rd_buffer(fd,num_av);

% To save it being re-initialized every time.
global ens hdr

% A fudge to try and read files not handled quite right.
global FIXOFFSET SOURCE

% If num_av<0 we are reading only 1 element and initializing
if num_av<0 | isempty(ens),
 FIXOFFSET=0;   
 SOURCE=0;  % 0=instrument, 1=VMDAS, 2=WINRIVER
 n=abs(num_av);
 pos=ftell(fd);
 hdr=rd_hdr(fd);
 cfg=rd_fix(fd);
 fseek(fd,pos,'bof');
 clear global ens
 global ens
 
 ens=struct('number',zeros(1,n),'rtc',zeros(7,n),'BIT',zeros(1,n),'ssp',zeros(1,n),'depth',zeros(1,n),'pitch',zeros(1,n),...
            'roll',zeros(1,n),'heading',zeros(1,n),'temperature',zeros(1,n),'salinity',zeros(1,n),...
            'mpt',zeros(1,n),'heading_std',zeros(1,n),'pitch_std',zeros(1,n),...
            'roll_std',zeros(1,n),'adc',zeros(8,n),'error_status_wd',zeros(1,n),...
            'pressure',zeros(1,n),'pressure_std',zeros(1,n),...
            'east_vel',zeros(cfg.n_cells,n),'north_vel',zeros(cfg.n_cells,n),'vert_vel',zeros(cfg.n_cells,n),...
            'error_vel',zeros(cfg.n_cells,n),'intens',zeros(cfg.n_cells,4,n),'percent',zeros(cfg.n_cells,4,n),...
            'corr',zeros(cfg.n_cells,4,n),'status',zeros(cfg.n_cells,4,n),'bt_range',zeros(4,n),'bt_vel',zeros(4,n),...
            'smtime',zeros(1,n),'emtime',zeros(1,n),'slatitude',zeros(1,n),...
	        'slongitude',zeros(1,n),'elatitude',zeros(1,n),'elongitude',zeros(1,n),...
	        'flags',zeros(1,n));
  num_av=abs(num_av);
end;

k=0;
while k<num_av,
   
   
   id1=dec2hex(fread(fd,1,'uint16'));
   if ~strcmp(id1,'7F7F'),
	if isempty(id1),  % End of file
	 ens=-1;
	 return;
	end;    
        error(['Not a workhorse/broadband file or bad data encountered: ->' id1]); 
   end;
   startpos=ftell(fd)-2;  % Starting position.
   
   
   % Read the # data types.
   [hdr,nbyte]=rd_hdrseg(fd);      
   byte_offset=nbyte+2;
%%disp(length(hdr.dat_offsets))
   % Read all the data types.
   for n=1:length(hdr.dat_offsets),

    id=fread(fd,1,'uint16');
%%    fprintf('ID=%s\n',dec2hex(id,4));
    
    % handle all the various segments of data. Note that since I read the IDs as a two
    % byte number in little-endian order the high and low bytes are exchanged compared to
    % the values given in the manual.
    %
    
    switch dec2hex(id,4),           
     case '0000',   % Fixed leader
      [cfg,nbyte]=rd_fixseg(fd);
      nbyte=nbyte+2;
      
    case '0080'   % Variable Leader
      k=k+1;
      ens.number(k)         =fread(fd,1,'uint16');
      ens.rtc(:,k)          =fread(fd,7,'uint8');
      ens.number(k)         =ens.number(k)+65536*fread(fd,1,'uint8');
      ens.BIT(k)            =fread(fd,1,'uint16');
      ens.ssp(k)            =fread(fd,1,'uint16');
      ens.depth(k)          =fread(fd,1,'uint16')*.1;   % meters
      ens.heading(k)        =fread(fd,1,'uint16')*.01;  % degrees
      ens.pitch(k)          =fread(fd,1,'int16')*.01;   % degrees
      ens.roll(k)           =fread(fd,1,'int16')*.01;   % degrees
      ens.salinity(k)       =fread(fd,1,'int16');       % PSU
      ens.temperature(k)    =fread(fd,1,'int16')*.01;   % Deg C
      ens.mpt(k)            =sum(fread(fd,3,'uint8').*[60 1 .01]'); % seconds
      ens.heading_std(k)    =fread(fd,1,'uint8');     % degrees
      ens.pitch_std(k)      =fread(fd,1,'int8')*.1;   % degrees
      ens.roll_std(k)       =fread(fd,1,'int8')*.1;   % degrees
      ens.adc(:,k)          =fread(fd,8,'uint8');
      nbyte=2+40;

      if strcmp(cfg.name,'bb-adcp'),
      
          if cfg.prog_ver>=5.55,
              fseek(fd,15,'cof'); % 14 zeros and one byte for number WM4 bytes
	          cent=fread(fd,1,'uint8');            % possibly also for 5.55-5.58 but
	          ens.rtc(:,k)=fread(fd,7,'uint8');    % I have no data to test.
	          ens.rtc(1,k)=ens.rtc(1,k)+cent*100;
	          nbyte=nbyte+15+8;
		  end;
          
      elseif strcmp(cfg.name,'wh-adcp'), % for WH versions.		

          ens.error_status_wd(k)=fread(fd,1,'uint32');
          nbyte=nbyte+4;;

	      if cfg.prog_ver>=8.13,  % Added pressure sensor stuff in 8.13
                  fseek(fd,2,'cof');   
                  ens.pressure(k)       =fread(fd,1,'uint32');  
                  ens.pressure_std(k)   =fread(fd,1,'uint32');
	          nbyte=nbyte+10;  
	      end;

	      if cfg.prog_ver>8.24,  % Spare byte added 8.24
	          fseek(fd,1,'cof');
	          nbyte=nbyte+1;
	      end;

	      if cfg.prog_ver>=16.05,   % Added more fields with century in clock 16.05
	          cent=fread(fd,1,'uint8');            
	          ens.rtc(:,k)=fread(fd,7,'uint8');   
	          ens.rtc(1,k)=ens.rtc(1,k)+cent*100;
	          nbyte=nbyte+8;
	      end;
      end;
  	      
    case '0100',  % Velocities
      vels=fread(fd,[4 cfg.n_cells],'int16')'*.001;     % m/s
      ens.east_vel(:,k) =vels(:,1);
      ens.north_vel(:,k)=vels(:,2);
      ens.vert_vel(:,k) =vels(:,3);
      ens.error_vel(:,k)=vels(:,4);
      nbyte=2+4*cfg.n_cells*2;
      
    case '0200',  % Correlations
      ens.corr(:,:,k)   =fread(fd,[4 cfg.n_cells],'uint8')';
      nbyte=2+4*cfg.n_cells;
      
    case '0300',  % Echo Intensities  
      ens.intens(:,:,k)   =fread(fd,[4 cfg.n_cells],'uint8')';
      nbyte=2+4*cfg.n_cells;

    case '0400',  % Percent good
      ens.percent(:,:,k)   =fread(fd,[4 cfg.n_cells],'uint8')';
      nbyte=2+4*cfg.n_cells;
   
    case '0500',  % Status
         % Note in one case with a 4.25 firmware SC-BB, it seems like
         % this block was actually two bytes short!
      ens.status(:,:,k)   =fread(fd,[4 cfg.n_cells],'uint8')';
      nbyte=2+4*cfg.n_cells;

	case '0600', % Bottom track
                 % In WINRIVER GPS data is tucked into here in odd ways, as long
                 % as GPS is enabled.
      if SOURCE==2,
          fseek(fd,2,'cof');
          long1=fread(fd,1,'uint16');
          fseek(fd,6,'cof');           
          cfac=180/2^31;
          ens.slatitude(k)  =fread(fd,1,'int32')*cfac;
      else    
          fseek(fd,14,'cof'); % Skip over a bunch of stuff
      end;    
      ens.bt_range(:,k)=fread(fd,4,'uint16')*.01; %
      ens.bt_vel(:,k)  =fread(fd,4,'int16');
      if SOURCE==2,
          fseek(fd,12+2,'cof');
          ens.slongitude(k)=(long1+65536*fread(fd,1,'uint16'))*cfac;
          if ens.slongitude(k)>180, ens.slongitude(k)=ens.slongitude(k)-360; end;
          fseek(fd,71-33-16,'cof');
          nbyte=2+68; 
      else    
          fseek(fd,71-33,'cof');
          nbyte=2+68;
      end;    
      if cfg.prog_ver>=5.3,    % Version 4.05 firmware seems to be missing these last 11 bytes.
       fseek(fd,78-71,'cof');  
       ens.bt_range(:,k)=ens.bt_range(:,k)+fread(fd,4,'uint8')*655.36;
       nbyte=nbyte+11;
       if cfg.prog_ver>=16,   % RDI documentation claims these extra bytes were added in v 8.17
           fseek(fd,4,'cof');  % but they don't appear in my 8.33 data.
           nbyte=nbyte+4;
       end;
      end;
     
% The raw files produced by VMDAS contain a binary navigation data
% block. 
      
    case '2000',  % Something from VMDAS.
      cfg.sourceprog='VMDAS';
      SOURCE=1;
      utim  =fread(fd,4,'uint8');
      mtime =datenum(utim(3)+utim(4)*256,utim(2),utim(1));
      ens.smtime(k)     =mtime+fread(fd,1,'uint32')/8640000;
      fseek(fd,4,'cof');
      cfac=180/2^31;
      ens.slatitude(k)  =fread(fd,1,'int32')*cfac;
      ens.slongitude(k) =fread(fd,1,'int32')*cfac;
      ens.emtime(k)     =mtime+fread(fd,1,'uint32')/8640000;
      ens.elatitude(k)  =fread(fd,1,'int32')*cfac;
      ens.elongitude(k) =fread(fd,1,'int32')*cfac;
      fseek(fd,12,'cof');
      ens.flags(k)      =fread(fd,1,'uint16');	
      fseek(fd,30,'cof');
      nbyte=2+76;
       
% The following blocks come from WINRIVER files, they aparently contain
% the raw NMEA data received from a serial port.
%
% Note that for WINRIVER files somewhat decoded data is also available
% tucked into the bottom track block.
    
    case '2100', % $xxDBT  (Winriver addition) 38
      cfg.sourceprog='WINRIVER';
      SOURCE=2;
      str=fread(fd,38,'uchar')';
      nbyte=2+38;

    case '2101', % $xxGGA  (Winriver addition) 94 in maanual but 97 seems to work
      cfg.sourceprog='WINRIVER';
      SOURCE=2;
      str=fread(fd,97,'uchar')';
      nbyte=2+97;
 %     disp(setstr(str(1:80)));
      
    case '2102', % $xxVTG  (Winriver addition) 45
      cfg.sourceprog='WINRIVER';
      SOURCE=2;
      str=fread(fd,45,'uchar')';
      nbyte=2+45;
%      disp(setstr(str));
      
    case '2103', % $xxGSA  (Winriver addition) 60
      cfg.sourceprog='WINRIVER';
      SOURCE=2;
      str=fread(fd,60,'uchar')';
%      disp(setstr(str));
      nbyte=2+60;

    case '2104',  %xxHDT or HDG (Winriver addition) 38
      cfg.sourceprog='WINRIVER';
      SOURCE=2;
      str=fread(fd,38,'uchar')';
%      disp(setstr(str));
      nbyte=2+38;
      
      
        
    case '0701', % Number of good pings
      fseek(fd,4*cfg.n_cells,'cof');
      nbyte=2+4*cfg.n_cells;
    
    case '0702', % Sum of squared velocities
      fseek(fd,4*cfg.n_cells,'cof');
      nbyte=2+4*cfg.n_cells;

    case '0703', % Sum of velocities      
      fseek(fd,4*cfg.n_cells,'cof');
      nbyte=2+4*cfg.n_cells;

% These blocks were implemented for 5-beam systems

    case '0A00', % Beam 5 velocity (not implemented)
      fseek(fd,cfg.n_cells,'cof');
      nbyte=2+cfg.n_cells;

    case '0301', % Beam 5 Number of good pings (not implemented)
      fseek(fd,cfg.n_cells,'cof');
      nbyte=2+cfg.n_cells;

    case '0302', % Beam 5 Sum of squared velocities (not implemented)
      fseek(fd,cfg.n_cells,'cof');
      nbyte=2+cfg.n_cells;
             
    case '0303', % Beam 5 Sum of velocities (not implemented)
      fseek(fd,cfg.n_cells,'cof');
      nbyte=2+cfg.n_cells;
             
    case '020C', % Ambient sound profile (not implemented)
      fseek(fd,4,'cof');
      nbyte=2+4;
             
    otherwise,
      
      fprintf('Unrecognized ID code: %s\n',dec2hex(id,4));
      nbyte=2;
     %% ens=-1;
     %% return;
      
      
    end;
   
    % here I adjust the number of bytes so I am sure to begin
    % reading at the next valid offset. If everything is working right I shouldn't have
    % to do this but every so often firware changes result in some differences.

    %%fprintf('#bytes is %d, original offset is %d\n',nbyte,byte_offset);
    byte_offset=byte_offset+nbyte;   
      
    if n<length(hdr.dat_offsets),
      if hdr.dat_offsets(n+1)~=byte_offset,    
        fprintf('%s: Adjust location by %d\n',dec2hex(id,4),hdr.dat_offsets(n+1)-byte_offset);
        fseek(fd,hdr.dat_offsets(n+1)-byte_offset,'cof');
      end;	
      byte_offset=hdr.dat_offsets(n+1); 
    end;
  end;

  % Now at the end of the record we have two reserved bytes, followed
  % by a two-byte checksum = 4 bytes to skip over.

  readbytes=ftell(fd)-startpos;
  offset=(hdr.nbyte+2)-byte_offset; % The 2 is for the checksum
  if offset ~=4 & FIXOFFSET==0, 
	fprintf('\n*****************************************************\n');
    fprintf('Adjust location by %d (readbytes=%d, hdr.nbyte=%d)\n',offset,readbytes,hdr.nbyte);
    fprintf(' NOTE - THIS IS A PROGRAM PROBLEM, POSSIBLY FIXED BY A FUDGE\n');
    fprintf('        PLEASE REPORT TO rich@ocgy.ubc.ca WITH DETAILS!!\n');
    fprintf('        ATTEMPTING TO RECOVER...\n');
    fprintf('******************************************************\n');
    FIXOFFSET=offset-4;
  end;  
  fseek(fd,4+FIXOFFSET,'cof'); 
   
  % An early version of WAVESMON and PARSE contained a bug which stuck an additional two
  % bytes in these files, but they really shouldn't be there 
  %if cfg.prog_ver>=16.05,    
  %	  fseek(fd,2,'cof');
  %end;
  	   
end;

% Blank out stuff bigger than error velocity
% big_err=abs(ens.error_vel)>.2;
big_err=0;
	
% Blank out invalid data	
ens.east_vel(ens.east_vel==-32.768 | big_err)=NaN;
ens.north_vel(ens.north_vel==-32.768 | big_err)=NaN;
ens.vert_vel(ens.vert_vel==-32.768 | big_err)=NaN;
ens.error_vel(ens.error_vel==-32.768 | big_err)=NaN;




%--------------------------------------
function y=nmedian(x,window,dim);
% Copied from median but with handling of NaN different.

if nargin==2, 
  dim = min(find(size(x)~=1)); 
  if isempty(dim), dim = 1; end
end

siz = [size(x) ones(1,dim-ndims(x))];
n = size(x,dim);

% Permute and reshape so that DIM becomes the row dimension of a 2-D array
perm = [dim:max(length(size(x)),dim) 1:dim-1];
x = reshape(permute(x,perm),n,prod(siz)/n);

% Sort along first dimension
x = sort(x,1);
[n1,n2]=size(x);

if n1==1,
 y=x;
else
  if n2==1,
   kk=sum(finite(x),1);
   if kk>0,
     x1=x(max(fix(kk/2),1));
     x2=x(max(ceil(kk/2),1));
     x(abs(x-(x1+x2)/2)>window)=NaN;
   end;
   x = sort(x,1);
   kk=sum(finite(x),1);
   x(isnan(x))=0;
   y=NaN;
   if kk>0,
    y=sum(x)/kk;
   end;
  else
   kk=sum(finite(x),1);
   ll=kk<n1-2;
   kk(ll)=0;x(:,ll)=NaN;
   x1=x(max(fix(kk/2),1)+[0:n2-1]*n1);
   x2=x(max(ceil(kk/2),1)+[0:n2-1]*n1);

   x(abs(x-ones(n1,1)*(x1+x2)/2)>window)=NaN;
   x = sort(x,1);
   kk=sum(finite(x),1);
   x(isnan(x))=0;
   y=NaN+ones(1,n2);
   if any(kk),
    y(kk>0)=sum(x(:,kk>0))./kk(kk>0);
   end;
  end;
end; 

% Permute and reshape back
siz(dim) = 1;
y = ipermute(reshape(y,siz(perm)),perm);

%--------------------------------------
function y=nmean(x,dim);
% R_NMEAN Computes the mean of matrix ignoring NaN
%         values
%   R_NMEAN(X,DIM) takes the mean along the dimension DIM of X. 
%

kk=finite(x);
x(~kk)=0;

if nargin==1, 
  % Determine which dimension SUM will use
  dim = min(find(size(x)~=1));
  if isempty(dim), dim = 1; end
end;

if dim>length(size(x)),
 y=x;              % For matlab 5.0 only!!! Later versions have a fixed 'sum'
else
  ndat=sum(kk,dim);
  indat=ndat==0;
  ndat(indat)=1; % If there are no good data then it doesn't matter what
                 % we average by - and this avoid div-by-zero warnings.

  y = sum(x,dim)./ndat;
  y(indat)=NaN;
end;

























