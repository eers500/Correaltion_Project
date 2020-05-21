function [reinterpolated,reinterpolatedLocations] = reinterpolateOutliers(inputData,tol)
%% reinterpolateOutliers
% For gridded input data, identify outliers and reinterpolate them based on
% the neighbords. 
% Inputs:
% - inputData: Gridded input data
% - tol (optional): The tol to use (default 1.6)

if nargin<2; tol = 1.4; end

x = 1:size(inputData,2);
y = 1:size(inputData,1); 

% Identify the outliers 
outliersR = isoutlier(inputData,'movmean',5,1,'ThresholdFactor',tol);
outliersC = isoutlier(inputData,'movmean',5,2,'ThresholdFactor',tol);
outliers = and(outliersR,outliersC);
reinterpolatedLocations = outliers;

% Fix these outliers 
RCoutliers = getRCofLogical(outliers);
YXoutliers = [RCoutliers(:,2),RCoutliers(:,1)];

reinterpolated = reinterpolateTerm2d(inputData,x,y,YXoutliers);


end


function [RC] = getRCofLogical(logicalArray)
%% getRCofLogical
% Get the RC of some logical points

logicalArrayCol = logicalArray(:);
colIndices = find(logicalArrayCol);
[R,C] = ind2sub(size(logicalArray),colIndices);

RC = [R(:),C(:)];

end