function [inputImage, target, spotSizeInTargetPx] = putSpotInFourierDomain(freqX,freqY,spotSize,inputSize)
%% putSpotInFourierDomain
% Put a spot at a given location in the Fourier domain, using a given input
% image

% Input arguments
% - freqX,freqY: X,Y frequency coordinates. These are in Nyquist units;
%   they run from -1/2 to 1/2. 
% - spotSize: The width of the target spot in the same Nyquist units.
%   (Default 1/1000 = 1px)
% - inputSize: The size of the filter (default [1000,1000])

% Output arguments: 
% - inputImage: the input to write to produce the given filter distribution
% - target: The optical distribution we are aiming for
% - spotSizeInTargetPx: The spot size we end up with in targetpx [r,c]

if nargin<4; inputSize = [1000,1000]; end
if nargin<3; spotSize = 1/1000; end
addpath('../common');


% Define Nyquist coordinates 
fx = linspace(-1/2,1/2,inputSize(2));
fy = linspace(-1/2,1/2,inputSize(1)); 

% Determine which spots fall inside the 
distanceFromCentre_xx = abs(fx-freqX);
distanceFromCentre_yy = abs(fy-freqY);
validSpot_xx = distanceFromCentre_xx<spotSize/2;
validSpot_yy = distanceFromCentre_yy<spotSize/2; 
validSpot = single(validSpot_yy') * single(validSpot_xx);     % Determine by outer product 
target = validSpot; 

% Find the width of the spotSize
spotWidth = sum(validSpot_xx);
spotHeight = sum(validSpot_yy);
spotSizeInTargetPx = [spotHeight,spotWidth]; 

% Create the filter
inputImage = createSimpleBinaryFilter(target);

end