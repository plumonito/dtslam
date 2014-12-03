function convertImages(formatIn, formatOut)
    idx = 0;
    fileFound = true;
    while(fileFound)
        fileIn = sprintf(formatIn,idx);
        if(~exist(fileIn,'file'))
            fileFound = false;
            break;
        end
        
        im = imread(fileIn);
        
        fileOut = sprintf(formatOut,idx);
        imwrite(im,fileOut);
        
        idx=idx+1;
    end
end