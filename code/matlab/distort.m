function xd = distort(kc,xn)
    r2 = sum(xn.^2);
    r4 = r2.*r2;
    factor = 1 + kc(1)*r2 + kc(2)*r4;
    
    xd = bsxfun(@times, factor, xn);
end