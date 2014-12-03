function distXn=getXnSegmentDistance(xnA, xnB, xnPoint)
    xnPoint = xnPoint/norm(xnPoint);
    
    lineDir = xnB-xnA;
%     lineDir = lineDir/norm(lineDir);
    
    diffB = xnPoint-xnB;
    signB = dot(lineDir,diffB);
    if(signB>0)
        distXn = diffB;
    else
        diffA = xnPoint-xnA;
        signA = -dot(lineDir,diffA);
        if(signA>0)
            distXn = diffA;
        else
            N = cross(xnA,xnB);
            N = N/norm(N);
            distXn = N*dot(N,xnPoint);
        end
    end
   
end