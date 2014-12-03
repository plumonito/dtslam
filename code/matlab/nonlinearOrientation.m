%At the end: pb = s*R*pa
function [s,R,t, paa]=nonlinearOrientation(pa,pb)
    [s,R,t] = absoluteOrientationQuaternion(pa, pb, true);

%     options = optimoptions('lsqnonlin','display','none','MaxFunEvals',1e10,'MaxIter',1e10);
%     params0 = [rotationpars(R); t; s];
%     params = lsqnonlin(@(x) errorFunc(x,pa,pb), params0,[],[],options);
%     R = rotationmat(params(1:3));
%     t = params(4:6);
%     s = params(7);
     paa = bsxfun(@plus, s*R*pa, t);
    
%     clf
%     plot3(paa(1,:), paa(2,:), paa(3,:),'r*-')
%     hold on; grid
%     plot3(pb(1,:), pb(2,:), pb(3,:),'b*-')
%     for i=1:size(pb,2); 
%         plot3([pb(1,i),paa(1,i)], [pb(2,i),paa(2,i)], [pb(3,i),paa(3,i)],'-'); 
%     end
%     axis equal
%     rmse = mean(sum((paa-pb).^2)).^0.5;
%     str = sprintf('RMSE: %f',rmse);
%     fprintf('%s\n',str);
%     title(str);
end

function res=errorFunc(params, pa, pb)
    R = rotationmat(params(1:3));
    t = params(4:6);
    s = params(7);
    paa = bsxfun(@plus, s*R*pa, t);
    res = reshape(pb-paa,[],1);
%     ss = sign(res);
%     res = ss.*abs(res).^0.25;
end