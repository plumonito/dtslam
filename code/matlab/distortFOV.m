function xd = distortFOV(omega,xn)
    r = sum(xn.^2).^0.5;
    rp = 1/omega * atan(2*r*tan(omega/2));
    factor = rp./r;
    
    xd = bsxfun(@times, factor, xn);
end