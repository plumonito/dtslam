%%
CameraFx=1803.70267428;
CameraFy=1803.70267428;
CameraU0=816.961496961;
CameraV0=586.608368412;
CameraK1=-0.228728298249;
CameraK2=0.204354044984;
CameraWidth=1600;
CameraHeight=1200;

K = [CameraFx, 0, CameraU0; 0, CameraFy, CameraV0; 0,0,1];
kc = [CameraK1, CameraK2];

minUv = [0;0];
minXd = K\[minUv;1];

maxUv = [CameraWidth;CameraHeight];
maxXd = K\[maxUv;1];

step = 0.05;
[xx,yy] = meshgrid(minXd(1):step:maxXd(1), minXd(2):step:maxXd(2));

xc = [xx(:), yy(:)]';
xc(3,:) = 1;
uv = projectFromWorld(K,kc,xc);

%%
options = optimoptions('lsqnonlin','display','iter');
params0 = [K(1,1),K(2,2),K(1,3),K(2,3),0,0];
params = lsqnonlin(@(x) reshape(uv - projectFromWorld(params2K(x(1:4)),x(5:6),xc),[],1), params0,[],[],options);
Kp = params2K(params(1:4));
kcp = params(5:6);

%%
options = optimoptions('lsqnonlin','display','iter','MaxFunEvals',1e10,'MaxIter',1e10);
params0 = [K(1,1),K(2,2),K(1,3),K(2,3)];
params = lsqnonlin(@(x) reshape(uv - projectFromWorld(params2K(x(1:4)),[0,0],xc),[],1), params0,[],[],options);
Knull = params2K(params(1:4));

%%
options = optimoptions('lsqnonlin','display','iter','MaxFunEvals',1e10,'MaxIter',1e10);
omega0 = 3;
params0 = [Knull(1,1),Knull(2,2),Knull(1,3),Knull(2,3),omega0];
params = lsqnonlin(@(x) reshape(uv - projectFromWorldFOV(params2K(x(1:4)),x(5),xc),[],1), params0,[],[],options);
Kfovp = params2K(params(1:4));
omega = params(5);

%%
figure(1);
clf;
plot(uv(1,:),uv(2,:),'b*','DisplayName','Ground truth');
hold on

uv1 = projectFromWorld(K,kc,xc);
plot(uv1(1,:),uv1(2,:),'g*','DisplayName','Reprojected using ground truth');

uv2 = projectFromWorldFOV(Kfovp,omega,xc);
plot(uv2(1,:),uv2(2,:),'r*','DisplayName','Reprojected using FOV model');

uv3 = projectFromWorld(Knull,[0,0],xc);
plot(uv3(1,:),uv3(2,:),'k*','DisplayName','Reprojected using no distortion');

%%
fprintf('\n');
fprintf('For PTAM orig:    [%f %f %f %f %f]\n', K(1,1)/CameraWidth, K(2,2)/CameraHeight, K(1,3)/CameraWidth, K(2,3)/CameraHeight, 0);
fprintf('For PTAM no dist: [%f %f %f %f %f]\n', Knull(1,1)/CameraWidth, Knull(2,2)/CameraHeight, Knull(1,3)/CameraWidth, Knull(2,3)/CameraHeight, 0);
fprintf('For PTAM:         [%f %f %f %f %f]\n', Kfovp(1,1)/CameraWidth, Kfovp(2,2)/CameraHeight, Kfovp(1,3)/CameraWidth, Kfovp(2,3)/CameraHeight, omega);

fprintf('For slamRT no dist:\n -CameraFx=%f\n -CameraFy=%f\n -CameraU0=%f\n -CameraV0=%f\n -CameraK1=0\n -CameraK2=0\n -CameraWidth=%d\n -CameraHeight=%d\n', ...
    Knull(1,1), Knull(2,2), Knull(1,3), Knull(2,3), CameraWidth, CameraHeight);
