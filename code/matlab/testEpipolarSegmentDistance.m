%%
imageSize = [1080,1920];
K = [1759.95830,   0, 963.80029; 0, 1758.22390, 541.04811; 0 0 1];

imageSize = imageSize/4;
K(1:2,:) = K(1:2,:)/4;

kc = [0.16262,-0.67445];

X = [3,0,10]';

Ra = eye(3);
ta = zeros(3,1);
centerA = -Ra'*ta;

Rb = eye(3);
tb = [-1,0,0]';

Rrel = Rb*Ra';

ma = projectFromWorld(K,kc,Ra*X+ta);
mb = projectFromWorld(K,kc,Rb*X+tb);

minDepth = 2;

rayA = Ra*X+ta;
rayA = rayA/norm(rayA);
XminDepthA = minDepth*rayA;
XminDepth = Ra'*(XminDepthA - ta);

epipoleXn = tb/norm(tb);
minDepthXn = Rb*XminDepth + tb;
minDepthXn = minDepthXn/norm(minDepthXn);
infiniteXn = Rrel*(Ra*X+ta);
infiniteXn = infiniteXn / norm(infiniteXn);

%%
distXn = zeros(imageSize);
distImage = zeros(imageSize);
fprintf('Generating distance image...');
for v=0:imageSize(1)
    for u=0:imageSize(2)
        xn = unprojectToWorld(K,kc,[u;v]);        
        diff = getXnSegmentDistance(minDepthXn,infiniteXn,xn);
        distXn(v+1,u+1) = norm(diff);
        
        ujac = [K(1,1)/xn(3); 0; -K(1,1)*xn(1)/xn(3)^2];
        vjac = [0; K(2,2)/xn(3); -K(2,2)*xn(2)/xn(3)^2];
        
        du = dot(diff,ujac);
        dv = dot(diff,vjac);
        distImage(v+1,u+1) = norm([du,dv]);
    end
end
fprintf('done.\n');
%%
figure(1);
clf;
imshow(distXn,[])
hold on

infiniteUV = projectFromWorld(K,kc,infiniteXn);
plot(infiniteUV(1),infiniteUV(2),'*g')

minDepthUV = projectFromWorld(K,kc,minDepthXn);
plot(minDepthUV(1),minDepthUV(2),'*r')
