function xc = unprojectToWorld(K,kc,uv)
    pcount = size(uv,2);
    uv(3,:) = 1;
    xd = K\uv;
    
    xn = zeros(2,pcount);
    for i=1:pcount
        xn(:,i) = undistort(kc,xd(1:2,i));
    end
    
    xc = xn;
    xc(3,:) = 1;
end
