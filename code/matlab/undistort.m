function xn = undistort(kc, xd)
	kMaxIters = 11;
	x = xd(1);
	x0 = x;
	y = xd(2);
	y0 = y;
    for j=1:kMaxIters
        r2 = x*x + y*y;
        icdist = 1 + (kc(2)*r2 + kc(1))*r2;
        x = (x0 )/icdist;
        y = (y0 )/icdist;
    end
    xn = [x;y];
end