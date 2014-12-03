function plotAlignedPoints(pa,pb)
    plot3(pa(1,:), pa(2,:), pa(3,:),'r*')
    hold on; grid on
    plot3(pb(1,:), pb(2,:), pb(3,:),'b*')
    for i=1:size(pb,2); 
        plot3([pb(1,i),pa(1,i)], [pb(2,i),pa(2,i)], [pb(3,i),pa(3,i)],'-'); 
    end
    axis equal
end
