%function evalCityOfSightsPose
%%
% Load
clear
realData = importdata('C:\code\dslam\datasets\cityOfSights\Poses_CS_BirdsView_L2_L0.csv');
realId = realData.data(:,9);
realCenterAll = realData.data(:,2:4)';
realQuatAll = realData.data(:,[8,5,6,7])';
for i=1:size(realCenterAll,2)
    R = qGetR(realQuatAll(:,i));
    t = realCenterAll(:,i);
    P = [R, t; 0 0 0 1];

    realCenterAll(:,i) = -R'*t;
end

usePTAM=false;
% usePTAM=true;

if(usePTAM)
    expData = importdata('C:\code\PTAM-Windows\release\pose.txt');

    valid = expData(:,1) == expData(:,2);
    timestamps= expData(valid,1);
    expT = expData(valid,3:5)';
    expRparams = expData(valid,6:8)';
    
    poseCenters = zeros(3,size(expT,2));
    for i=1:size(expT,2)
        R = rotationmat(expRparams(:,i));
        t = expT(:,i);

         poseCenters(:,i) = -R'*t;
    end
else
    expData = importdata('C:\code\build\dslam\nvslam_desktop\pose.txt');

    valid = expData(:,1) == expData(:,2);
    timestamps = expData(valid,1);
    poseCenters = expData(valid,3:5)';
end

%%
minTimestamp = 300;
valid = timestamps > minTimestamp;
timestamps = timestamps(valid);
poseCenters = poseCenters(:,valid);

%%
%Discard first frames
refpoints = zeros(size(poseCenters));
for j=1:size(refpoints,2)
    match = find(realId==timestamps(j));
    if(isempty(match))
        error('Id not found!');
    end
    refpoints(:,j) = realCenterAll(:,match);
end

[~,~,~,aligned] = nonlinearOrientation(poseCenters, refpoints);

%%
if(usePTAM)
    systemStr = 'PTAM';
else
    systemStr = 'slamRT';
end
%plotPoseResults(validTimestamps, realCenters, expCentersAligned, systemStr);
plotPoseResults(timestamps, refpoints, aligned, systemStr)
%plotAlignedPoints(validMaps{end}.refpoints, validMaps{end}.aligned)
