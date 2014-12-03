function [r,t]=absolute_orientation(x,y)
r0 = (bsxfun(@minus,y,mean(y)))'*bsxfun(@minus,x,mean(x));
[u,~,v]=svd(r0);
r = v*u';
% r = eye(3);
t=mean(x)'-r*mean(y)';

% t(2:3)=0;