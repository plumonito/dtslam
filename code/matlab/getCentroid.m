function p=getCentroid(patch)
[uu,vv] = meshgrid(1:size(patch,2),1:size(patch,1));
p(1) = sum(uu(:).*patch(:)) / sum(patch(:));
p(2) = sum(vv(:).*patch(:)) / sum(patch(:));