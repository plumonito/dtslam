function [ res ] = CauchyLoss( s, th )
    b = th*th;
    c = 1/b;

    ss = 1+s*c;
    res = b*log(ss);
end

