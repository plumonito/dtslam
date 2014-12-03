function [minP,maxP]=getMinMax(patch)

ff = ones(3)/9;
%patchSmooth = imfilter(patch,ff,'replicate');
patchSmooth = patch;
patchSmall = patchSmooth(2:end-1,2:end-1);

[~,ind]=min(patchSmall(:));
[minP(2),minP(1)] = ind2sub(size(patchSmall),ind);
minP = minP+1;

[~,ind]=max(patchSmall(:));
[maxP(2),maxP(1)] = ind2sub(size(patchSmall),ind);
maxP = maxP+1;