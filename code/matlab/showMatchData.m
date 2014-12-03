function showMatchData(data, showCount)

if(nargin<2)
  showCount = 2;
end

figure(1); 
subplot(1,showCount+1,1);
imshow(data.refPatch/255); 
title('ref');

%Update scores
for i=1:length(data.patches)
  patch = data.patches(i).patch;
%    patch(2:end,:) = patch(1:end-1,:);
%     patch(1:end-1,:) = patch(2:end,:);

  data.patches(i).score = zssd(data.refPatch, patch);
  data.patches(i).patch = patch;
end

scores = [data.patches(:).score];
[scoresSorted,scoresIdx] = sort(scores);

for i=1:showCount
  %Calculate score ourselves
  patch = data.patches(scoresIdx(i)).patch;
%    patch(:,2:end) = patch(:,1:end-1);
  s = zssd(data.refPatch, patch);
  
  subplot(1,showCount+1,1+i);
  imshow(patch/255);
  %title(sprintf('score=%f',scoresSorted(i)));
  title(sprintf('#%d, score=%f',scoresIdx(i),s));
end

figure(2); 
hist(scores,0:100:3000)