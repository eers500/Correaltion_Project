function [filterImage, spotImage, spotSizeInTargetPx] = putSpotInFilter(freqX,freqY,spotSize,filterSize,offLevel,onLevel)
%% putSpotInFilter
% Put a spot on the filter
% centreX and centreY are the coordinates of the spot in freq

% Input arguments
% - freqX,freqY: X,Y frequency coordinates. These are in Nyquist units;
%   they run from -1/2 to 1/2. 
% - spotSize (opt): The width of the target spot in the same Nyquist units.
%   (default 1/1000 = 1px)
% - filterSize: The size of the filter (default [1000,1000])
% - offLevel (opt): The GL to represent logical 0. (default 0)
% - onLevel (opt): The GL to represent logical 1. (default 255)

% Output arguments:
% - filterImage: The filter to write to draw the correct window
%   the spot image itself 
% - spotImage: The logical spot window used 
% - spotSizeInTargetPx: The spot size we end up with in targetpx [r,c]


if nargin<6; onLevel = 255; end
if nargin<5; offLevel = 0; end
if nargin<4; filterSize = [1000,1000]; end
if nargin<3; spotSize = 1/1000; end

% Define Nyquist coordinates 
fx = linspace(-1/2,1/2,filterSize(2));
fy = linspace(-1/2,1/2,filterSize(1)); 

% Determine which spots fall inside the 
distanceFromCentre_xx = abs(fx-freqX);
distanceFromCentre_yy = abs(fy-freqY);
validSpot_xx = distanceFromCentre_xx<spotSize/2;
validSpot_yy = distanceFromCentre_yy<spotSize/2; 
validSpot = single(validSpot_yy') * single(validSpot_xx);     % Determine by outer product 

% Find the width of the spotSize
spotWidth = sum(validSpot_xx);
spotHeight = sum(validSpot_yy);
spotSizeInTargetPx = [spotHeight,spotWidth]; 

% Map onto the output
spotImage = logical(validSpot); 
filterImage = zeros(filterSize); 
filterImage(spotImage) = onLevel;
filterImage(~spotImage) = offLevel; 
filterImage = uint8(filterImage);
end