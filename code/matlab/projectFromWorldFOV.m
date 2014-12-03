function uv=projectFromWorldFOV(K,omega,xc)
    xn = bsxfun(@rdivide, xc(1:2,:),xc(3,:));
    xd = distortFOV(omega,xn);
    xd(3,:) = 1;
    uv = K(1:2,1:3) * xd;
end
