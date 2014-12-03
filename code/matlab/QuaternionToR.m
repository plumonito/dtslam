function R=QuaternionToR(quat)
a = quat(1);
b = quat(2);
c = quat(3);
d = quat(4);

R = zeros(3);
R(1,1) = a^2+b^2-c^2-d^2;
R(1,2) = 2*b*c-2*a*d;
R(1,3) = 2*b*d+2*a*c;

R(2,1) = 2*b*c+2*a*d;
R(2,2) = a^2-b^2+c^2-d^2;
R(2,3) = 2*c*d-2*a*b;

R(3,1) = 2*b*d-2*a*c;
R(3,2) = 2*c*d+2*a*b;
R(3,3) = a^2-b^2-c^2+d^2;
