function res=zssd(a,b)
res = (2*sum(a(:))*sum(b(:))-sum(a(:))^2-sum(b(:))^2)/length(a(:))^2 + (sum(a(:).^2)+sum(b(:).^2)-2*sum(a(:).*b(:)) ) / length(a(:));
% v = var(a(:));
% res = res/v;