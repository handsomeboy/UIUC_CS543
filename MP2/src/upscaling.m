% Assignment 2: Blob Detection using Scale-space 
% Zhenye Na (zna2)
% 3/6/2018

% img filename: butterfly.jpg, einstein.jpg, fishes.jpg, flower.jpg
% frog.jpg, gta5.jpg, sunflowers.jpg, tnj.jpg, music.jpg

% Turn off this warning "Warning: Image is too big to fit on screen; displaying at 33% "
% To set the warning state, you must first know the message identifier for the one warning you want to enable. 
% Query the last warning to acquire the identifier.  For example: 
% warnStruct = warning('query', 'last');
% msgid_integerCat = warnStruct.identifier
% MATLAB:concatenation:integerInteraction
warning('off', 'Images:initSize:adjustingMag');

% clear all
% Read in image and convert to double and then grayscale
[img, map] = imread('../data/einstein.jpg');
img = rgb2gray(img);
img = im2double(img);


[h, w] = size(img);             % Image size
threshold = 0.25;               % Define threshold
k = 1.25;                       % Increasing factor of k
levels = 12;                    % Define number of iterations
initial_sigma = 2;              % Define scale for LoG
sigma = 2;


% Perform LoG filter to image for several levels
% [h,w] - dimensions of image, n - number of levels in scale space
scale_space = zeros(h, w, levels);
tic
for i = 1:levels
    % Generate a Laplacian of Gaussian filter / scale normalization
    LoG = fspecial('log', 2 * ceil(3 * sigma) + 1, sigma);
    % Filter the img with LoG
    scale_space(:,:,i) = (imfilter(img, LoG, 'replicate', 'same').*(sigma^2)).^2;
    % Increase scale by a factor k
    sigma = sigma * k;
    % hsize = 2 * ceil(sigma) + 1;
    
%     % GENERATE GIF IN MATLAB
%     % Capture the plot as an image
%     h = figure;
%     imshow(scale_space(:,:,i), map);
%     frame = getframe(h);
%     im = frame2im(frame);
%     [imind, cm] = rgb2ind(im, 256);
%     
%     % WRITE TO THE GIF FILE
%     if i == 1
%         imwrite(imind, cm, 'music.gif', 'gif','LoopCount',Inf,'DelayTime',1)
%     else
%         imwrite(imind, cm, 'music.gif', 'gif','WriteMode','append','DelayTime',1);
%     end

end
toc

% figure;surf(LoG);title('filter size = %d, sigma = %d',2 * ceil(3 * sigma) + 1, sigma);

% Perform nonmaximum suppression in each 2D slice
suppressed_space = zeros(h,w,levels);
for i = 1:levels
    suppressed_space(:,:,i) = ordfilt2(scale_space(:,:,i),9,ones(3,3)); 
end


% Perform nonmaximum suppression in scale space and apply threshold
% nonmax_space(:,:,num) = suppressed_space(:,:,num) .* (suppressed_space(:,:,num) >= threshold);

% for num = 1:levels
%     if num == 1
%         maxima_space(:,:,num) = max(suppressed_space(:, :, num:num + 1), [], 3);
%     elseif num == levels
%         maxima_space(:,:,num) = max(suppressed_space(:, :, num - 1:num), [], 3);
%     else
%         maxima_space(:,:,num) = max(suppressed_space(:, :, num - 1:num + 1), [], 3);
%     end
%     maxima_space(:,:,num) = ((maxima_space(:,:,num) == scale_space(:,:,num)) .* img);
% end

% maxima_space = zeros(h, w, levels);
maxima_space = max(suppressed_space, [], 3);
survive_space = zeros(h,w,levels);
for i = 1:levels
    survive_space(:,:,i) = (maxima_space == scale_space(:,:,i));
    survive_space(:,:,i) = survive_space(:,:,i) .* img;
end
% survive_space = 



% Find all the coordinates and corresponding sigma
% % Find index and corresponding radius


for num = 1:levels
    [c,r] = find(survive_space(:,:,num) >= threshold);
    rad = sqrt(2) * initial_sigma * k^(num-1);
    if num == 1
        cx = c;
        cy = r;
        radius = rad .* ones(size(r,1), 1);
    else
        cx = [cx; c];
        cy = [cy; r];
        radius = [radius; rad .* ones(size(r,1), 1)];
    end
end
%[c,r] = find(max_space);


show_all_circles(img, cy, cx, radius, threshold, initial_sigma, k);