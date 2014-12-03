function [ res ] = HuberLoss( s, th )
    a = th;
    b = th*th;
    isOverOne =s > b;
    r = s.^0.5;
    
    res = zeros(size(s));
    res(isOverOne) = 2*a*r(isOverOne)-b;
    res(~isOverOne) = s(~isOverOne);
end

