% We want to test syncing by turning the input 'bright' and checking we get
% the bright image then.

% clear
% close all; 

%% Settings
brightLevel = 255;
darkLevel = 0;
filterLevel = 255; 
preChangeImages = 10;
postChangeImages = 10; 
sz = [1000,1000];

%% Connect to the hardware
if ~exist('camera') || ~exist('slm')
    [camera,slm] = setupOpticalCorrelate();
end

%% Preprocessing
darkInput = zeros(sz,'uint8');
brightInput = 255*ones(sz,'uint8');
filter = 255*ones(sz,'uint8');

totalImages = preChangeImages + postChangeImages;
intensities = zeros(totalImages,1);

c = 0;
for t = 1:preChangeImages
    c = c+1;
    opticalOut = opticalCorrelate(camera, slm, darkInput, filter);
    intensities(c) = sum(double(opticalOut(:)));
end
for t = 1:postChangeImages
    c = c+1;
    opticalOut = opticalCorrelate(camera,slm, brightInput, filter); 
    intensities(c) = sum(double(opticalOut(:)));
end


plot((1:c)-preChangeImages-0.5,intensities,'x'); 
title('Changeover should be around 0'); 
