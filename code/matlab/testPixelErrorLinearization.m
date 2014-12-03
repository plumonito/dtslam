imageSize = [480,640];
K =[618.8956         0  316.4595; 
    0  619.2835  238.4590;
     0         0    1.0000];
kc = [0.1204, -0.2117];

angle = zeros(imageSize);
for v=1:imageSize(1)
    for u=1:imageSize(2)
        %points = [u, u+2^0.5;v, v+2^0.5];
        points = [u,v; u+1,v; u,v+1; u+2^0.5,v+2^0.5; u+2^0.5,v-2^0.5;]';
        Xc = unprojectToWorld(K,kc,points);
        a = Xc(:,1);
        b = Xc(:,2);
        angles_i(1) = acos(dot(a,b)/(norm(a)*norm(b)));
        a = Xc(:,1);
        b = Xc(:,3);
        angles_i(2) = acos(dot(a,b)/(norm(a)*norm(b)));
        a = Xc(:,1);
        b = Xc(:,4);
        angles_i(3) = acos(dot(a,b)/(norm(a)*norm(b)));
        a = Xc(:,1);
        b = Xc(:,5);
        angles_i(4) = acos(dot(a,b)/(norm(a)*norm(b)));
        angle(v,u) = max(angles_i);
    end
end
imtool(angle*180/pi*100,[])