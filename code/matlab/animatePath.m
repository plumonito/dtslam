function animatePath(points,color)
    set(gcf,'Position',[909   560   643   426]);
    h = plot3(points(1,1),points(2,1),points(3,1),color);
    for i=1:size(points,2)
        delete(h);
        h = plot3(points(1,1:i),points(2,1:i),points(3,1:i),color);
        title(sprintf('Frame %d',i));
        
        pause(0.033);
    end
end