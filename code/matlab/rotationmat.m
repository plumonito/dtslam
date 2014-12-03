% by Juho Kannala
%Vector form of rotation to matrix form (Rodriguez formula)
function R=rotationmat(t)

t=t(:);
phi=norm(t);

if 0
if phi==0
  sincphi=1;
  sc=0;
else
  sc=(1-cos(phi))/(phi^2);
  sincphi=sin(phi)/phi;
end
R=cos(phi)*eye(3)+sincphi*[0 -t(3) t(2); t(3) 0 -t(1); -t(2) t(1) 0]+sc*t*t';
end

if phi==0
  R=eye(3);
else
  th=t/phi;
  thx=[0 -th(3) th(2); th(3) 0 -th(1); -th(2) th(1) 0];
  R=eye(3)+sin(phi)*thx+(1-cos(phi))*thx*thx;
end
