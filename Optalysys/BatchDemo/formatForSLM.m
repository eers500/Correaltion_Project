function [FormattedBatch] = batchFormat(camera,slm,json_file, filledBatch, SLM_Type)
%% opticalCorrelate
% Show an input and filter on the optical correlator, and return the
% optical output
% Currently works for a 1by1 Sony system

%% THIS IS EFFECTIVELY LIFED FROM test_octy_call_device.m

% Quiet warnings
warning('off','all');

% Cast the input and filter to make sure they are uint 8
filledBatch = uint8(filledBatch);


%% Prepare the images
% Assign the images to the SLMs
if strcmp(SLM_Type, 'input')
	slmNo = 1;
elseif strcmp (SLM_Type, 'filter')
	
	slmNo = 2;
end
%disp(['SLM No ', num2str(slmNo)]);


% Figure out which dimension to loop over. 
% For 4DDs, the dimensions are [x,y,3,N]. For Sonys, [x,y,N]
% But N = 1, so the dimensions are [x,y,3] and [x,y] respectively.
% Find N 

if ndims(filledBatch)==2  %this will be a sony with a single image
	iter = 1;
elseif ndims(filledBatch)==3  % this will be sony with >1 images
	iter = size(filledBatch, 3);
end

	


% Transform the images
for ii=1:iter
	%pause(0.01);
	%disp(['Formatting image ', num2str(ii) , ' of ', num2str(size(filledBatch,4))]);
	% Assign the images to the SLMs
	if ndims(filledBatch)==2
		slm(slmNo).workingImage = filledBatch(:,:);
	elseif ndims(filledBatch)==3
		slm(slmNo).workingImage = filledBatch(:,:,ii);
	end
		
	% fill final image background
	slm(slmNo).image = zeros(slm(slmNo).height, slm(slmNo).width, slm(slmNo).depth, slm(slmNo).expectedType);
	
	if ~strcmp(slm(slmNo).expectedType, class(slm(slmNo).workingImage))
		warning('Supplied image for slm %d has a data type of %s, but the SLM requires type %s. Casting to required type.', slmNo, class(slm(slmNo).workingImage), slm(slmNo).expectedType);
		slm(slmNo).workingImage = cast(slm(slmNo).workingImage, slm(slmNo).expectedType);
	end
	
	% flip left-right
	if slm(slmNo).fliplr
		slm(slmNo).workingImage = fliplr(slm(slmNo).workingImage);
	end
	
	% flip up-down
	if slm(slmNo).flipud
		slm(slmNo).workingImage = flipud(slm(slmNo).workingImage);
	end
	
	% rotate
	slm(slmNo).workingImage = rot90(slm(slmNo).workingImage, slm(slmNo).rot90x);
	
	% scale
	scaleAlgorithm = lower(strrep(slm(slmNo).scaleAlgorithm, ' ', ''));
	if slm(slmNo).depth > 1 && ~strcmp(scaleAlgorithm, 'nearest')
		warning('Suggested scale algorithm for slm %d with bit depth (%d) should be "nearest" neighbor, but is "%s".', slmNo, slm(slmNo).depth, slm(slmNo).scaleAlgorithm);
	end
	
	slm(slmNo).workingImage =  imresize(slm(slmNo).workingImage, slm(slmNo).scaleFactor);
	
	% set bit depth appropriately
	if size(slm(slmNo).workingImage, 3) < slm(slmNo).depth
		temp = zeros([size(slm(slmNo).workingImage, 1) size(slm(slmNo).workingImage, 2) slm(slmNo).depth]);
		for j = 1:slm(slmNo).depth
			temp(:,:,j) = slm(slmNo).workingImage;
		end
		slm(slmNo).workingImage = temp;
	elseif size(slm(slmNo).workingImage, 3) > slm(slmNo).depth
		warning('Image %d depth (%d) is greater than SLM depth. Using only %d channels.', slmNo, size(slm(slmNo).workingImage, 3), slm(slmNo).depth, slm(slmNo).depth);
		temp = slm(slmNo).workingImage(:,:,1:slm(slmNo).depth);
		slm(slmNo).workingImage = temp;
	end
	
	% displace
	imageStartRow = 1;
	targetStartRow = 1 + floor((size(slm(slmNo).image, 1) - size(slm(slmNo).workingImage, 1)) / 2) - slm(slmNo).offsetY;
	if targetStartRow < 1
		imageStartRow = abs(targetStartRow);
		targetStartRow = 1;
	end
	targetStopRow = targetStartRow + size(slm(slmNo).workingImage, 1) - imageStartRow;
	if targetStopRow > slm(slmNo).height
		targetStopRow = slm(slmNo).height;
	end
	imageStopRow = imageStartRow + (targetStopRow - targetStartRow);
	
	imageStartCol = 1;
	targetStartCol = 1 + floor((size(slm(slmNo).image, 2) - size(slm(slmNo).workingImage, 2)) / 2) + slm(slmNo).offsetX;
	if targetStartCol < 1
		imageStartCol = abs(targetStartCol);
		targetStartCol = 1;
	end
	targetStopCol = targetStartCol + size(slm(slmNo).workingImage, 2) - imageStartCol;
	if targetStopCol > slm(slmNo).width
		targetStopCol = slm(slmNo).width;
	end
	imageStopCol = imageStartCol + (targetStopCol - targetStartCol);
	
	slm(slmNo).image(targetStartRow:targetStopRow,targetStartCol:targetStopCol,:) = slm(slmNo).workingImage(imageStartRow:imageStopRow,imageStartCol:imageStopCol,:);
	

    
	if ndims(filledBatch)==2 
		FormattedBatch(:,:) = slm(slmNo).image;
    elseif ndims(filledBatch)==3
		FormattedBatch(:,:,ii) = slm(slmNo).image;
    end
    
	
end
end



% [camera(1).photo] = octy(json_file, slm(1).image, slm(2).image);
%
% %% Process camera images
% for i = 1:numel(camera)
%
%     % flip left-right
%     if camera(i).fliplr
%         camera(i).photo = fliplr(camera(i).photo);
%     end
%
%     % flip up-down
%     if camera(i).flipud
%         camera(i).photo = flipud(camera(i).photo);
%     end
%
%     % rotate
%     camera(i).photo = rot90(camera(i).photo, camera(i).rot90x);
% end
%
% opticalOut = camera(1).photo;

%end