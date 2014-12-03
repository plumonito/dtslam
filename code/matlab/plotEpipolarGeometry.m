clf;

%Centers
center{1} = [0,0,0]';
center{2} = -relR*relT;
plot3(center{1}(1),center{1}(2),center{1}(3),'*k');
hold on
plot3(center{2}(1),center{2}(2),center{2}(3),'*k');

%Measurements
m = mxn{1}'+center{1};
plot3(m(1),m(2),m(3), '*r');
plot3([center{1}(1),m(1)],[center{1}(2),m(2)],[center{1}(3),m(3)], '-r');

m = mxn{2}'+center{2};
plot3(m(1),m(2),m(3), '*r');
plot3([center{2}(1),m(1)],[center{2}(2),m(2)],[center{2}(3),m(3)], '-r');

axis equal