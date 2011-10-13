function [point] = intersection(line1, line2);
%INTERSECTION
%
% This function returns the point of intersection of two lines.
%
% Usage:
% >> point = intersection(line1, line2);
%
% where line1 and line2 are vectors of the form [x1 y1 x2 y2] where the line
% connects the points (x1, y1) and (x2, y2).

p11 = line1(1:2,:);
p12 = line1(3:4,:);

p21 = line2(1:2,:);
p22 = line2(3:4,:);

clear line1 line2;

m1 = (p12(2,:) - p11(2,:)) ./ (p12(1,:) - p11(1,:));
m2 = (p22(2,:) - p21(2,:)) / (p22(1,:) - p21(1,:));

b1 = -m1 .* p11(1,:) + p11(2,:);
b2 = -m2 .* p21(1,:) + p21(2,:);

point(1,:) = (b2 - b1) ./ (m1 - m2);
point(2,:) = m1 .* point(1,:) + b1;


