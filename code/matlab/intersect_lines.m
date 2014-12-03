%[x,t]=intersect_lines(a,dir)
% Input
%   a [3xN] 3D points that the lines go through
%   dir [3xN] 3D direction vectors for the line
%
%   Line i is described by the equation xi = a(:,i)+t*dir(:,i)
%   where t is the free parameter.
% Output
%   x [3x1] is the intersection (or closest point)
%   t [1xN] is the parameter for the intersection point of each line
function [x,t]=intersect_lines(a,dir)

line_count = size(a,2);

A = zeros(line_count*3,3+line_count);
b = zeros(line_count*3,1);
for i=1:size(a,2)
  A(i*3-2:i*3,1:3) = eye(3);
  A(i*3-2:i*3, 3+i) = -dir(:,i);
  b(i*3-2:i*3) = a(:,i);
end

z = A\b;
x = z(1:3);
t = z(4:end);
