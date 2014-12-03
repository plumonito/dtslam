%%
imageSize = [1080,1920];
K = [1759.95830,   0, 963.80029; 0, 1758.22390, 541.04811; 0 0 1];

% imageSize = imageSize/4;
% K(1:2,:) = K(1:2,:)/4;

kc = [0.16262,-0.67445];

im = imread('C:\code\dslam\datasets\cityOfSights\CS_BirdsView_L0\Frame_01890.jpg');

refR = eye(3);
refT = zeros(3,1);
refCenter = -refR'*refT;

imgR = eye(3);
imgT = [1;1;10];

refPos = [500;500];
refXn = unprojectToWorld(K,kc,refPos);

worldDir = refR'*refXn;


epipole = imgR*refCenter + imgT;
epipoleUv = projectFromWorld(K,kc,epipole);

infiniteUv = projectFromWorld(K,kc,imgR*refR'*refXn);

depths = [5 10 20 40];
pointDepth = zeros(2,length(depths));
for i=1:length(depths)
    X = refCenter + depths(i)*worldDir;
    pointDepth(:,i) = projectFromWorld(K,kc, imgR*X+imgT);
end

figure(1);
hold off
imshow(im);
hold on;
plot(epipoleUv(1), epipoleUv(2), 'g*')
plot(infiniteUv(1), infiniteUv(2), 'g*')

plot(pointDepth(1,:), pointDepth(2,:), '+g');

plot([epipoleUv(1) pointDepth(1,2)], [epipoleUv(2) pointDepth(2,2)], '-y')
plot([infiniteUv(1) pointDepth(1,2)], [infiniteUv(2) pointDepth(2,2)], '-g')

lineDir = infiniteUv - epipoleUv;
lineDir = lineDir/norm(lineDir);

pminus = epipoleUv - 1000*lineDir
plot([epipoleUv(1) pminus(1)], [epipoleUv(2) pminus(2)], '-r')

pminus = infiniteUv + 1000*lineDir
plot([infiniteUv(1) pminus(1)], [infiniteUv(2) pminus(2)], '-r')


textOffset = 25;
for i=1:length(depths)
    t = num2str(depths(i)/10,'%.1f');
%     t = num2str(depths(i));
    text(pointDepth(1,i), pointDepth(2,i)-textOffset, t, 'Color', 'green','FontWeight','bold','FontSize',14);
end
text(epipoleUv(1),epipoleUv(2)-textOffset,'Epipole','Color','Green','FontWeight','bold','FontSize',14);
text(infiniteUv(1)-2*textOffset,infiniteUv(2)-textOffset,'Infinite','Color','Green','FontWeight','bold','FontSize',14);

margin = 175;
xlim([min(epipoleUv(1),infiniteUv(1))-margin, max(epipoleUv(1),infiniteUv(1))+margin]);
ylim([min(epipoleUv(2),infiniteUv(2))-margin, max(epipoleUv(2),infiniteUv(2))+margin]);

%%
points = [602.5366  499.4450
  761.1289  626.3188
  868.6193  584.0275
  965.5368  709.1393
  405.1772  510.0178
  965.5368  609.1393]';

% perpDir = [lineDir(2),-lineDir(1)];
% pa = infiniteUv-1000*lineDir;
% pa = infiniteUv-1000*lineDir;

figure(2);
hold off;
imshow(im);
hold on;

plot([infiniteUv(1) pointDepth(1,2)], [infiniteUv(2) pointDepth(2,2)], '-g')
plot(points(1,:), points(2,:),'*y','MarkerSize',10,'LineWidth',1.5)

xlim([min(epipoleUv(1),infiniteUv(1))-margin, max(epipoleUv(1),infiniteUv(1))+margin]);
ylim([min(epipoleUv(2),infiniteUv(2))-margin, max(epipoleUv(2),infiniteUv(2))+margin]);
