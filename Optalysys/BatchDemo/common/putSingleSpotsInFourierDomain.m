function [inputImage, target] = putSpotsInFourierDomain(freqsXY,inputSize)
%% putSpotInFourierDomain
% Put a set of spots at a given location in the Fourier domain, using a given input
% image

% Input arguments
% - freqsXY,freqY: X,Y frequency coordinates, in a N*2 list [X,Y].
%   These are in Nyquist units;
%   they run from -1/2 to 1/2.
% - spotSize: The width of the target spot in the same Nyquist units.
%   (Default 1/1000 = 1px)
% - inputSize: The size of the filter (default [1000,1000])

% Output arguments:
% - inputImage: the input to write to produce the given filter distribution
% - target: The optical distribution we are aiming for
% - spotSizeInTargetPx: The spot size we end up with in targetpx [r,c]

if nargin<2; inputSize = [1000,1000]; end
if nargin<1; spotSize = 1/1000; end
addpath('../common');

N = size(freqsXY,1);

% Define Nyquist coordinates
fx = linspace(-1/2,1/2,inputSize(2));
fy = linspace(-1/2,1/2,inputSize(1));

combinedTarget = zeros(inputSize);
spotSizesInTargetPx = zeros(N,2);
for n = 1:N
    freqX = freqsXY(n,1);
    freqY = freqsXY(n,2);
    % Determine which spot is nearest the tarsget location
    distanceFromCentre_xx = abs(fx-freqX);
    distanceFromCentre_yy = abs(fy-freqY);
    validSpot_xx = distanceFromCentre_xx == min(distanceFromCentre_xx(:));
    validSpot_yy = distanceFromCentre_yy == min(distanceFromCentre_yy(:));  % TODO there will be a more elegant way to do this 
    validSpot = single(validSpot_yy') * single(validSpot_xx);     % Determine by outer product
    target = validSpot;
    combinedTarget = combinedTarget+target;
    
    % Find the width of the spotSize
    spotWidth = sum(validSpot_xx);
    spotHeight = sum(validSpot_yy);
    spotSizesInTargetPx(n,:) = [spotHeight,spotWidth];
    
end

% Create the filter
inputImage = createSimpleBinaryFilter(combinedTarget);

end