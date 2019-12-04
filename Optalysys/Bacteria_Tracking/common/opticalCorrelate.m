    function [opticalOut] = opticalCorrelate(camera,slm,json_file, input,filter)
%% opticalCorrelate
% Show an input and filter on the optical correlator, and return the
% optical output
% Currently works for a 1by1 Sony system 

%% THIS IS EFFECTIVELY LIFED FROM test_octy_call_device.m

% Quiet warnings
warning('off','all');

% Cast the input and filter to make sure they are uint 8
input = uint8(input);
filter = uint8(filter); 

%% Prepare the images
% Assign the images to the SLMs
slm(1).workingImage = input;
slm(2).workingImage = filter;

% Transform the images 
for i = 1:numel(slm)
    
    % fill final image background
    slm(i).image = zeros(slm(i).height, slm(i).width, slm(i).depth, slm(i).expectedType);

    if ~strcmp(slm(i).expectedType, class(slm(i).workingImage))
       warning('Supplied image for slm %d has a data type of %s, but the SLM requires type %s. Casting to required type.', i, class(slm(i).workingImage), slm(i).expectedType);
       slm(i).workingImage = cast(slm(i).workingImage, slm(i).expectedType);
    end

    % flip left-right
    if slm(i).fliplr
        slm(i).workingImage = fliplr(slm(i).workingImage);
    end

    % flip up-down
    if slm(i).flipud
        slm(i).workingImage = flipud(slm(i).workingImage);
    end

    % rotate
    slm(i).workingImage = rot90(slm(i).workingImage, slm(i).rot90x);

    % scale
    scaleAlgorithm = lower(strrep(slm(i).scaleAlgorithm, ' ', ''));
    if slm(i).depth > 1 && ~strcmp(scaleAlgorithm, 'nearest')
        warning('Suggested scale algorithm for slm %d with bit depth (%d) should be "nearest" neighbor, but is "%s".', i, slm(i).depth, slm(i).scaleAlgorithm);
    end

    slm(i).workingImage =  imresize(slm(i).workingImage, slm(i).scaleFactor);

    % set bit depth appropriately
    if size(slm(i).workingImage, 3) < slm(i).depth
        temp = zeros([size(slm(i).workingImage, 1) size(slm(i).workingImage, 2) slm(i).depth]);
        for j = 1:slm(i).depth
            temp(:,:,j) = slm(i).workingImage;
        end
        slm(i).workingImage = temp;
    elseif size(slm(i).workingImage, 3) > slm(i).depth
        warning('Image %d depth (%d) is greater than SLM depth. Using only %d channels.', i, size(slm(i).workingImage, 3), slm(i).depth, slm(i).depth);
        temp = slm(i).workingImage(:,:,1:slm(i).depth);
        slm(i).workingImage = temp;
    end

    % displace
    imageStartRow = 1;
    targetStartRow = 1 + floor((size(slm(i).image, 1) - size(slm(i).workingImage, 1)) / 2) - slm(i).offsetY;
    if targetStartRow < 1
        imageStartRow = abs(targetStartRow);
        targetStartRow = 1;
	end
	if imageStartRow == 0
		imageStartRow = 1;
		
	end
    targetStopRow = targetStartRow + size(slm(i).workingImage, 1) - imageStartRow;
    if targetStopRow > slm(i).height
        targetStopRow = slm(i).height;
    end
    imageStopRow = imageStartRow + (targetStopRow - targetStartRow);

    imageStartCol = 1;
    targetStartCol = 1 + floor((size(slm(i).image, 2) - size(slm(i).workingImage, 2)) / 2) + slm(i).offsetX;
    if targetStartCol < 1
        imageStartCol = abs(targetStartCol);
        targetStartCol = 1;
    end
    targetStopCol = targetStartCol + size(slm(i).workingImage, 2) - imageStartCol;
    if targetStopCol > slm(i).width
        targetStopCol = slm(i).width;
	end

    imageStopCol = imageStartCol + (targetStopCol - targetStartCol);
	
    slm(i).image(targetStartRow:targetStopRow,targetStartCol:targetStopCol,:) = slm(i).workingImage(imageStartRow:imageStopRow,imageStartCol:imageStopCol,:);
end

[camera(1).photo] = octy(json_file, slm(1).image, slm(2).image);

%% Process camera images
for i = 1:numel(camera)
    
    % flip left-right
    if camera(i).fliplr
        camera(i).photo = fliplr(camera(i).photo);
    end
    
    % flip up-down
    if camera(i).flipud
        camera(i).photo = flipud(camera(i).photo);
    end
    
    % rotate
    camera(i).photo = rot90(camera(i).photo, camera(i).rot90x);
end

opticalOut = camera(1).photo;

end