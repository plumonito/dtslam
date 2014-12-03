function uv=projectFromWorld(K,kc,xc)
    xn = bsxfun(@rdivide, xc(1:2,:),xc(3,:));
    xd = distort(kc,xn);
    xd(3,:) = 1;
    uv = K(1:2,1:3) * xd;
end
