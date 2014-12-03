%function evalCityOfSightsPose
%%
% Load
% clear
% realDataFilename = 'C:\code\dslam\datasets\cityOfSights\Poses_CS_BirdsView_L2_L0.csv';
% realDataFilename = 'C:\code\dslam\datasets\cityOfSights\Poses_CS_StreetView1_L0.csv';
realDataFilename = 'C:\code\dslam\datasets\cityOfSights\Poses_CS_TopView_L0.csv';
realData = importdata(realDataFilename);
realId = realData.data(:,9);
realCenterAll = realData.data(:,2:4)';
realQuatAll = realData.data(:,[8,5,6,7])';
for i=1:size(realCenterAll,2)
    R = qGetR(realQuatAll(:,i));
    t = realCenterAll(:,i);
    P = [R, t; 0 0 0 1];

    realCenterAll(:,i) = -R'*t;
end

minTimestamp = 00;

figure(1);
clf;
figure(2);
clf;
valid = realId > minTimestamp;
plotPoseResults(realId(valid), realCenterAll(:,valid), 'Ground truth', 'b');

%%
for usePTAM = [true, false]
% for usePTAM = [true]

    if(usePTAM)
        systemStr = 'PTAM';
        color = 'r';
        filename = 'C:\code\dslam\results\ptam_bad.txt';
%         filename = 'C:\code\dslam\results\ptam_cos_street_bad085.txt';
        filename = 'C:\code\PTAM-Windows\release\pose_top_bad.txt';
        expData = importdata(filename);

        mapTimestamps = unique(expData(:,1));
        maps = cell(1,length(mapTimestamps));
        for i=1:length(mapTimestamps)
            valid = expData(:,1) == mapTimestamps(i);
            maps{i}.time = mapTimestamps(i);
            maps{i}.frameTime = expData(valid,2)';
            maps{i}.tparams = expData(valid,3:5)';
            maps{i}.rparams = expData(valid,6:8)';

            pcount = length(maps{i}.frameTime);
            maps{i}.poseCenter = zeros(3,pcount);
            for j=1:pcount
                R = rotationmat(maps{i}.rparams(:,j));
                t = maps{i}.tparams(:,j);
                maps{i}.poseCenter(:,j) = -R'*t;
            end

            valid = maps{i}.frameTime > minTimestamp;
            maps{i}.frameTime = maps{i}.frameTime(valid);
            maps{i}.poseCenter = maps{i}.poseCenter(:,valid);
        end

    %     expId = expData(:,1);
    %     expCenter = expData(:,2:4)';
    % 
    %     expRparams = expData(:,5:7)';
    %     for i=1:size(expCenter,2)
    %         R = rotationmat(expRparams(:,i));
    %         t = expCenter(:,i);
    % 
    %          expCenter(:,i) = -R'*t;
    %     end
    else
        color = 'g';
        filepath = 'C:\code\dslam\results\';
    %     filename = 'slamrt_200_multi_no2D.txt';
    %     filename = 'slamrt_200_multi_2D.txt';
    %     filename = 'slamrt_500_single_no2D.txt';
    %     filename = 'slamrt_500_single_2D.txt';
%         filename = 'slamrt_cos_birds_500_multi_2d.txt';
    %     filename = 'slamrt_cos_street_500_single_2D.txt';
%         filename = 'slamrt_cos_street_500_multi_2D.txt';
        filename = 'slamrt_cos_top_200_multi_2D.txt';

%         filepath = 'C:\code\build\dslam_vs\nvslam_desktop\Release\';
%         filename = 'pose.txt';

        systemStr = strrep(filename,'_','\_');
        systemStr = 'SLAM-RT';
        expData = importdata([filepath filename]);

        mapTimestamps = unique(expData(:,1));
        maps = cell(1,length(mapTimestamps));
        for i=1:length(mapTimestamps)
            valid = expData(:,1) == mapTimestamps(i);
            maps{i}.time = mapTimestamps(i);
            maps{i}.frameTime = expData(valid,2)';
            maps{i}.poseCenter = expData(valid,3:5)';

            valid = maps{i}.frameTime > minTimestamp;
            maps{i}.frameTime = maps{i}.frameTime(valid);
            maps{i}.poseCenter = maps{i}.poseCenter(:,valid);
        end
    end

    %%
    %Discard first frames
    %valid = mapTimestamps>minTimestamp;
    %valid = mapTimestamps==mapTimestamps(end);
    valid = cellfun(@(x) size(x.poseCenter,2)>=4, maps);
    validMaps = maps(valid);
    validTimestamps = mapTimestamps(valid);

    expCentersAligned = zeros(3,length(validMaps));
    realCenters = zeros(3,length(validMaps));

    fprintf('Aligning...');
    for i=1:length(validMaps)
        mapi = validMaps{i};

        mapi.refpoints = zeros(size(mapi.poseCenter));
        for j=1:size(mapi.refpoints,2)
            match = find(realId==mapi.frameTime(j));
            if(isempty(match))
                error('Id not found!');
            end
            mapi.refpoints(:,j) = realCenterAll(:,match);
        end

        [~,~,~,mapi.aligned] = nonlinearOrientation(mapi.poseCenter, mapi.refpoints);

        expCentersAligned(:,i) = mapi.aligned(:,end);
        realCenters(:,i) = mapi.refpoints(:,end);

        validMaps{i} = mapi;
    end
    fprintf('done\n');

    %%
    errorSignal = sum((realCenters-expCentersAligned).^2).^0.5;
    errorRMSE = mean(errorSignal.^2).^0.5;
    titleStr = sprintf('%s, RMSE=%.2f', systemStr, errorRMSE);
    plotPoseResults(validTimestamps, expCentersAligned, titleStr, color);

    figure(3);
    hold off
    plot(validTimestamps,errorSignal);
    title('Error');
    xlabel('Time');

    %Fixed size for paper:
    set(2,'Position',[ 1228         484         383         421])

    %CamStudio options:
    % Left:909 Top:42 Width:640 Height:480
    set(1,'Position',[909   560   643   426]);
    
    % plotPoseResults(validMaps{end}.frameTime, validMaps{end}.refpoints, validMaps{end}.aligned, systemStr)
    %plotAlignedPoints(validMaps{end}.refpoints, validMaps{end}.aligned)
end
