function imageAcquired = BacteriaScan(inputImg, filterImg)

% clear all;
% close all;
addpath('./common');
if ~exist('camera','var') || ~exist('slm','var') || exist('json_file','var')
    [camera,slm,json_file] = setupOpticalCorrelate();
end


% inputImg = imread('Input_Bacteria.png');
% filterImg = imread('Test_Filter[302,653].png');

%% For testing correlator
% inputImg = imread('8.png');
% filterImg = imread('Filter.png');

imageAcquired = opticalCorrelate(camera,slm,json_file, inputImg, filterImg);



% figure(1);
% subplot(2,2,1);
% imagesc(inputImg)
% %daspect([1,1,1])
% 
% subplot(2,2,2);
% imagesc(filterImg)
% %daspect([1,1,1])
% 
% 
% subplot(2,2,3)
% imagesc(imageAcquired)
% colorbar
% %daspect([1,1,1])
% 
% subplot(2,2,4)
% mesh(flipdim(imageAcquired, 1))
% axis tight
% colormap jet
% colorbar


end