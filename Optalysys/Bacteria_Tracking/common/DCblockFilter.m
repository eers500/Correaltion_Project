function [filter] = DCblockFilter(filter, radius,level)
%% DCblockFilter
% Add a DC block to a filter
    
filterMiddle = floor(size(filter)/2);

filter(filterMiddle(1)-radius:filterMiddle(1)+radius , filterMiddle(2)-radius:filterMiddle(2)+radius) = level;

end