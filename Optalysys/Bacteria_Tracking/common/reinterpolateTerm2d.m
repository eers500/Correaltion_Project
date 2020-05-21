function [reinterpolatedFunction] = reinterpolateTerm2d(func,x,y,reinterpCoords)
%% reinterpolateTerm2d
% A function to reinterpolate some points in some data based from
% surrounding values to replace the current value

% Inputs:
% - func: A 2d grid of data
% - x: A vector of X (col) coords of the data
% - y: A vector of Y (row) coords of the data
% - reinterpCoords: A list of [x1,y1;x2,y2; ... ] coords to replace

if ~ismatrix(func)
    error('func should be a matrix'); 
end

N = size(reinterpCoords,1);  % Number of coords to replace 

[xx,yy] = meshgrid(x,y); 

reinterpolatedFunction = func;
for n = 1:N
    locationToUpdate = and(xx==reinterpCoords(n,1) , yy==reinterpCoords(n,2));
    % Remove the old value from the data
    xx_ = xx(:); xx_(locationToUpdate(:)) = [];
    yy_ = yy(:); yy_(locationToUpdate(:)) = []; 
    func_ = func(:); 
	func_(locationToUpdate(:)) = []; 
    interpFunc = scatteredInterpolant(xx_,yy_,func_);
    reinterpolatedFunction(locationToUpdate) = interpFunc(reinterpCoords(n,1),reinterpCoords(n,2)); 
end

end
