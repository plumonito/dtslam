mcount = length(imgPos);
isValid = false(1,mcount);
isValid2 = false(1,mcount);
isValid3 = false(1,mcount);
for i=1:mcount
    relR = imgPoseR{i}*refPoseR{i}';
    relT = imgPoseT{i}'-relR*refPoseT{i}';
    
    %Oriented epipolar
    E = SkewSymmetric(relT)*relR;
    imgLine=E*refXn{i}';
    imgEpipole = relT / norm(relT);
    
    left = cross(imgEpipole, imgXn{i}');
    right = imgLine;
    
    if(sign(left(1))==sign(right(1)) && sign(left(2))==sign(right(2)))
        isValid(i) = true;
    end
    
    %Triangulate
    centers(1:3, 1) = [0,0,0];
    centers(1:3, 2) = -relR'*relT;
    dirs(1:3, 1) = refXn{i}';
    dirs(1:3, 2) = relR'*imgXn{i}';
    [x,t] = intersect_lines(centers,dirs);
    if(all(t>0))
        isValid2(i) = true;
    else
        isValid2(i) = false;
    end
    
    %Angle constraint
    imgInfXn = relR * refXn{i}';

    vecInf = imgInfXn - imgEpipole;
    vecInf = vecInf/norm(vecInf);
    
    vecXn = imgXn{i}' - imgEpipole;
    vecXn2 = dot(vecXn, vecInf)*vecInf;
    xn = vecXn2+imgEpipole;
    xn = xn/norm(xn);
    
    maxAngle = acos(dot(imgEpipole, imgInfXn));
    angle1 = acos(dot(imgEpipole, xn));
    angle2 = acos(dot(imgInfXn, xn));
    if(angle1 <= maxAngle && angle2 <= maxAngle)
        isValid3(i) = true;
    else
        isValid3(i) = false;
    end
end

sum(isValid)
sum(isValid2)
sum(isValid3)