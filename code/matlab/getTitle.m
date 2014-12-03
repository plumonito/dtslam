function str=getTitle(ax)

if(~exist('ax','var'))
    ax = gca;
end

h = get(ax,'Title');
str = get(h,'String');
