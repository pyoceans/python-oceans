function [r, coord] = line_plane_intersect(line_stuff, plane_stuff);
%function [r, coord] = line_plane_intersect(line_stuff, plane_stuff);
%
% Finds the location of a plane intersection for a line of the parametric
% form
%       X = Xo + kr, Y = Yo + lr, Z = Zo + mr
% with a plane of the form
%       AX + BY + CZ = D
%
% Inputs are LINE_STUFF = [Xo Y_o Z_o k l m] (in that order)  
%        and PLANE_STUFF = [A B C D]
%
% Outputs are R and COORD = [Xp Yp Zp]

list = line_stuff;
plst = plane_stuff;

Xo = list(1); Yo = list(2); Zo = list(3);
k = list(4); l = list(5); m = list(6);

A = plst(1); B = plst(2); C = plst(3); D = plst(4);

r = - (A * Xo + B * Yo + C * Zo - D) / (A * k + B * l + C * m);

coord = [Xo + k * r; Yo + l * r; Zo + m * r];
