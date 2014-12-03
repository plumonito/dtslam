function plotPoseResults(timestamps, centers, titleStr, color)

if(~exist('titleStr','var'))
  
    titleStr = '';
end

%%
%Plot
figure(1);
plot3(centers(1,:), centers(2,:), centers(3,:), ['-' color], 'DisplayName', titleStr);
hold on

axis equal
xlabel('x');
ylabel('y');
zlabel('z');

%%
figure(2);

% xrange = [200 2200];
% xrange = [1250 2100];

subplot(3,1,1);
plot(timestamps, centers(1,:), ['-' color], 'DisplayName', titleStr);
hold on
ylabel('x');

subplot(3,1,2);
plot(timestamps, centers(2,:), ['-' color], 'DisplayName', titleStr);
hold on
ylabel('y');

subplot(3,1,3);
plot(timestamps, centers(3,:), ['-' color], 'DisplayName', titleStr);
hold on
ylabel('z');

% xlabel('FrameID');
% % ylim([202 212]);

legend('Location','SouthEast')
%end
