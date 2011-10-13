function [M] = rotation_matrix(varargin);
% function [M] = rotation_matrix(varargin);
%
% Provides the rotation matrix for a series of rotations about axes in a
% 3-dimensional, right-handed, cartesian system. 
%
% The axis are labelled as 1,2,3 ([x,y,z], [x1,x2,x3], etc...).
%
% M = rotate(pi/4,1) 
%         produces the rotation matrix for a rotation of pi/4 about the 
%         x1 axis
%
% M = rotate(pi/4,pi/4, [1 3])
%         produces the rotation matrix for a rotation of pi/4 about the
%         x1 axis, followed by a rotation about the (rotated) x3 axis.
%         note that 
%                   rotate(pi/4,pi/4, [1 3]) != rotate(pi/4,pi/4, [3 1])

N = nargin;
if (length(varargin{N}) ~= N-1)
     error(['Number of arguments does not make sense.']);
end
if ~isempty(varargin{N}) & (any(any(varargin{N} ~= fix(varargin{N}))) ...
				| any(any(imag(varargin{N}))))
    error( 'axis of rotation must be 1, 2, or 3'.' );
elseif(max(varargin{N}) > 3 | min(varargin{N} < 1))
     error(['axis of rotation must be 1, 2, or 3']);
end

order = varargin{N};

M = eye(3);
for i = length(order):-1:1
   angle = varargin{i};
   switch order(i)
      case 1,
      Mi = [           1           0           0; ...
                       0  cos(angle)  sin(angle); ...
                       0 -sin(angle)  cos(angle)];
      case 2,
      Mi = [  cos(angle)           0 -sin(angle); ...
                       0           1           0; ...
              sin(angle)           0  cos(angle)];
      otherwise,
      Mi = [  cos(angle)  sin(angle)           0; ...
             -sin(angle)  cos(angle)           0; ...
                       0           0           1];
   end

   M = M * Mi;
end
