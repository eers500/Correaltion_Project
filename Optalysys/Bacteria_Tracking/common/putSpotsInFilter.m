function [filterImage, spotImage, spotSizesInTargetPx] = putSpotsInFilter(freqsXY,spotSize,filterSize,offLevel,onLevel)
%% putSpotInFilter
% Put a spot on the filter
% centreX and centreY are the coordinates of the spot in freq

% Input arguments
% - freqsXY,freqY: X,Y frequency coordinates, in a N*2 list [X,Y].
%   These are in Nyquist units;
%   they run from -1/2 to 1/2.
% - spotSize (opt): The width of the target spot in the same Nyquist units.
%   (default 1/1000 = 1px)
% - filterSize: The size of the filter (default [1000,1000])
% - offLevel (opt): The GL to represent logical 0. (default 0)
% - onLevel (opt): The GL to represent logical 1. (default 255)
%   This can either be a 
%   - scalar, in which case it is applied to each window
%   - vector, of same length as freqsXY, in which case it is applied to
%     each window

% Output arguments:
% - filterImage: The filter to write to draw the correct window
%   the spot image itself
% - spotImage: The logical spot window used
% - spotSizeInTargetPx: The spot size we end up with in targetpx [r,c]


if nargin<5; onLevel = 255; end
if nargin<4; offLevel = 0; end
if nargin<3; filterSize = [1000,1000]; end
if nargin<2; spotSize = 1/1000; end
N = size(freqsXY,1);
if isscalar(onLevel)
    onLevel = onLevel*ones(N,1); % Convert to on level for each window
end
onLevel =onLevel(:); % Conver to vector
if size(onLevel) ~= [N,1]
    error('On level should either be scalar, or N-length vector');
end


% Define Nyquist coordinates
fx = linspace(-1/2,1/2,filterSize(2));
fy = linspace(-1/2,1/2,filterSize(1));

combinedWindow = ones(filterSize)*double(offLevel);
spotSizesInTargetPx = zeros(N,2);
spotImage = zeros(filterSize,'logical');
for n = 1:N
    freqX = freqsXY(n,1);
    freqY = freqsXY(n,2);
    % Determine which spots fall inside the
    distanceFromCentre_xx = abs(fx-freqX);
    distanceFromCentre_yy = abs(fy-freqY);
    validSpot_xx = distanceFromCentre_xx<spotSize/2;
    validSpot_yy = distanceFromCentre_yy<spotSize/2;
    validSpot = logical(single(validSpot_yy') * single(validSpot_xx));     % Determine by outer product
    combinedWindow(validSpot) = onLevel(n); % Update combined window
    spotImage = or(spotImage,validSpot);
    
    % Find the width of the spotSize
    spotWidth = sum(validSpot_xx);
    spotHeight = sum(validSpot_yy);
    spotSizesInTargetPx(n,:) = [spotHeight,spotWidth];
    
end

filterImage = combinedWindow;

end