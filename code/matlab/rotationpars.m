% by Juho Kannala
%Matrix form of rotation to vector form (Rodriguez formula)
function t=rotationpars(R)

[V,D]=eig(R-eye(3));
evs=diag(D);
[minv,mini]=min(evs);
vect=V(:,mini);

cosphi=0.5*(trace(R)-1);
sinphi=0.5*vect'*[ R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)];

%if cosphi==0
%  phi=pi/2;
%else
%  phi=atan(sinphi/cosphi);
%end

phi=atan2(sinphi,cosphi);

t=phi*vect;
