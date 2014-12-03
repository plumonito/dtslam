function res=gradientHistogram(im)
    im = rgb2gray(im);
    
    kMaxTopLevelWidth = 240;
    octaveCount = 1;
    width = size(im,2);
    while(width > kMaxTopLevelWidth)
        octaveCount = octaveCount+1;
        width = ceil(width/2);
    end
    
    pyramid = cell(1,octaveCount);
    pyramid{1} = im;
    for octave=2:octaveCount
        pyramid{octave} = impyramid(pyramid{octave-1}, 'reduce');
    end
    
    %
    histEdges = 0:255;
    histograms = cell(1,octaveCount);
    for octave=1:octaveCount
        [dx,dy] = imgradientxy(pyramid{octave});
        dm = (dx.^2 + dy.^2).^0.5;
        histograms{octave} = histc(dm(:), histEdges);
    end
    
    subplot(1,2,1);
    imshow(pyramid{2});
    subplot(1,2,2);
    plot(histograms{2});
    grid on;
    ylim([0,500]);
    xlim([0,255]);
    
    res = histograms;
end