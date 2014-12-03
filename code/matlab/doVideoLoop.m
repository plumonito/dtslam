%doVideoLoop
%Processes a video sequence (either from disk or from a webcam) and calls
%the given function for each frame. The user can cleanly stop the process by
%clicking a stop button.

%Inputs:
% processFun - [function] pointer to a function that takes a single
% argument, the current image, and returns true if processing should
% continue.
% videoSource - [string] source of the video stream, can be a filename
% (avi) or an adaptor name from the image acquisition toolbox.
% deviceId - [int] device id for the image acquisition toolbox (see imaqhwinfo()).
% videoFormat - [string] format for the image acquisition toolbox (see imaqhwinfo()).
function res=doVideoLoop(processFun, videoSource, deviceId, videoFormat, isUniformOutput)
    if(~exist('isUniformOutput','var'))
        isUniformOutput = true;
    end
    
    isVideo = (strendswith(videoSource,'.avi') || strendswith(videoSource,'.mp4'));
    if(isVideo)
        res=aviLoop(processFun, videoSource, isUniformOutput);
    else
        if(~exist('deviceId','var') || isempty(deviceId))
            deviceId = 1;
        end
        if(~exist('videoFormat','var') || isempty(videoFormat))
            info = imaqhwinfo(videoSource);
            videoFormat = info.DeviceInfo(deviceId).DefaultFormat;
        end
        
        res=webcamLoop(processFun, videoSource, deviceId, videoFormat, isUniformOutput);
    end
end

function res=webcamLoop(processFun, videoSource, deviceId, videoFormat, isUniformOutput)
    fs = stoploop('Stop video');

    imaqreset

    fprintf('Starting loop from webcam (click STOP button to finish)\n');
    
    videoObj = videoinput(videoSource, deviceId, videoFormat);
    set(videoObj,'framesperTrigger',10,'TriggerRepeat',Inf);
    start(videoObj);
    
    frameId = 0;
    if(isUniformOutput)
        res = [];
    else
        res = {};
    end
    
    while(~fs.Stop())
        data=getdata(videoObj,1);
        flushdata(videoObj);
        frameId = frameId+1;
        
        if(size(data,4) < 1)
            %End of video
            break;
        end
        
        im = data(:,:,:,1);
        resi = processFun(frameId, im);
        if(isUniformOutput)
            res = [res;resi];
        else
            res{frameId} = resi;
        end
    end
    
    fprintf('Stopping webcam.\n');
   
    stop(videoObj);
    delete(videoObj);

    fs.Clear();
    clear fs;
end

function res=aviLoop(processFun, videoSource, isUniformOutput)
    fs = stoploop('Stop video');

    aviObj = VideoReader(videoSource);
    
    frameId = 1;
    fcount = aviObj.NumberOfFrames;

    fprintf('Starting loop from video file (%d frames)\n', fcount);

    if(isUniformOutput)
        res = [];
    else
        res = {};
    end

    while(~fs.Stop() && frameId<= fcount)
        %im=read(aviObj,i);
        im = read(aviObj,frameId);
        resi = processFun(frameId, im);
        if(isUniformOutput)
            res = [res;resi];
        else
            res{frameId} = resi;
        end

        frameId=frameId+1;
    end
    fprintf('Video finished\n');
   
    fs.Clear();
    clear fs;
end

function b = strendswith(s, pat)
%STRENDSWITH Determines whether a string ends with a specified pattern
%
%   b = strstartswith(s, pat);
%       returns whether the string s ends with a sub-string pat.
%

%   History
%   -------
%       - Created by Dahua Lin, on Oct 9, 2008
%

%% main

sl = length(s);
pl = length(pat);

b = (sl >= pl && strcmp(s(sl-pl+1:sl), pat)) || isempty(pat);
end