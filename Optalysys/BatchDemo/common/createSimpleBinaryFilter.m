function [binaryFilter] = createSimpleBinaryFilter(target, filterLevel, targetSize)

%% A prototype for the code used to create filters for the correlator

% This is a first-step filter generation code.
% It makes the following assumptions
% - The filter is a binary phase filter
% - The levels used to represent the filter are constant across the SLM
% - One of the levels is 0; the other is filterLevel
% - The target size of the filter can be specified, which may be larger
% than the filter

% target is what the filter is designed to search for
% (opt) Filter level should be a single number specficing the maximum filter
% level
% (opt) Targetsize should be a [R,C] vector specifying the target size of the
% flter.

% N.B. This should be compatible with Octave

% Establish defaults
if nargin<3; targetSize = size(target); end
if nargin<2; filterLevel = 255; end

% Check the targetsize is an [R,C] vector
if ~isequal(size(targetSize),[1,2])
    error('Targetsize is wrong size');
end

% Check we are not making the target smaller
if or(size(target,1)>targetSize(1) , size(target,2)>targetSize(2))
    error('targetSize is smaller than target. This is invalid');
end

% Check filterLevel is a scalar
if ~isscalar(filterLevel)
    error('filterLevel is wrong size');
end

%% Set the target to the correct size
target = embedTargetInTargetSize(target,targetSize);


%% Compute the filter
% We assume that we are creating a binary phase filter
% The fftshifts are to ensure that the zero frequency is in the middle
targetFT = fftshift( fft2( ifftshift( target ) ) );
conjugatePhaseTargetFT = -angle(targetFT);
binaryConjugatePhaseTargetFT = conjugatePhaseTargetFT > 0; % A logical array for phase>0
binaryFilter = uint8(filterLevel)*uint8(binaryConjugatePhaseTargetFT);

end


function [target] = embedTargetInTargetSize(target,targetSize)
%% Embed the function within the target
% This is essentially a re-implementation of padarray
if ~isequal(size(target),targetSize)
    resizedTarget = zeros(targetSize);
    startIndices = floor(0.5*(targetSize-size(target)));
    resizedTarget(startIndices(1)+1:startIndices(1)+size(target,1) , ... % Place the target inside the template
        startIndices(2)+1:startIndices(2)+size(target,2) ) = target;
    target = resizedTarget; % Overwrite the target image.
end
end