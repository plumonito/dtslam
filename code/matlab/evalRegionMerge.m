%%
filepath = 'C:\code\build\dslam_release\nvslam_desktop\'
filename = 'pose.txt';

systemStr = strrep(filename,'_','\_');
expData = importdata([filepath filename]);

mapTimestamps = unique(expData(:,1));
maps = cell(1,length(mapTimestamps));
for i=1:length(mapTimestamps)
    valid = expData(:,1) == mapTimestamps(i);
    maps{i}.time = mapTimestamps(i);
    maps{i}.frameTime = expData(valid,2)';
    maps{i}.poseCenter = expData(valid,3:5)';
end

