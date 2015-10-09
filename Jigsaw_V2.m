clear all;

% Read image
I = double(imread('C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\128_Dog.png'))/255;

%Get sizes of image
ISizeRGB = size(I); %128x128x3
ISize2D = size(I(:,:,1)); %128x128
ISize1D = ISizeRGB (1);

%Draw original image
figure, imagesc(I), title('Original Image');

%Set size of Jigsaw Matrix
j1D = 32;
jSize = [3 j1D j1D];

%Initialize jigsaw mean, jigsaw variance and offset map L matrices
jMean = zeros(jSize) - 1;
jVar = zeros(jSize);
L = zeros (ISize2D); % size of 128 x 128

%Initialize reconstructed images 
reconsImage = zeros (ISizeRGB); %128x128x3
reconsNoisyImage = zeros (ISizeRGB); %128x128x3

%Set constants
beta = 1;
mean_0 = 0.5;

%Calculate mean and variance of image elements in order to find a and b
%constants

sumX = sum(sum(I, 1), 2);
sumXX = sum(sum(I.^2, 1), 2);
pixelMean = sumX ./ prod(ISize2D);
pixelVar = sumXX ./ prod(ISize2D) - pixelMean.^2;    
pixelStd = sqrt(sumXX ./ prod(ISize2D) - pixelMean.^2);

%Initialize Gamma distribution parameters
b = 3 * (1 / pixelVar);
a = b.^2;

%Initialize Jigsaw Variance
for i = 1 : 3
    jVar(i,:,:) = b(i) / a(i);
end

%Initialize Jigsaw Mean and making sure to keep the pixels between 0 and 1
done = false;
while(~done)
    done = true;
    
    for i = 1 : 3
        idx = jMean (i,:,:) < 0 | jMean (i,:,:) > 1;
        N = sum(idx(:));
        
        if(N > 0)
            done = false;
            jMean(i,idx) = randn(N,1) .* pixelStd(1,i) + pixelMean(1,i);
        end
    end
end

%Convert [3 32 32] jigsaw to [32 32 3] jigsaw
jMean = permute(jMean,[2 3 1]);
jVar = permute(jVar,[2 3 1]);
jSize = [j1D j1D 3];

%Find label size
labelSize = (j1D * j1D) + (((ISize1D - 1)*j1D)*2) + ((ISize1D - 1)^2);
pixelSize = ISize1D^2;

%Set 4 connected grid of image pixels. 4 connected grid defined in
%Isize1DxIsize1D matrix (eg. if image size is 128x128, 4 conected grid 
%defined in 16384x16384 matrix) because alpha expansion grap cut code
%define neigberhood of pixels as this way.
[r,c] = size(I(:,:,1));                     %# Get the matrix size
diagVec1 = repmat([ones(c-1,1); 0],r,1);    %# Make the first diagonal vector
                                            %# (for horizontal connections)
diagVec1 = diagVec1(1:end-1);               %# Remove the last value
diagVec2 = ones(c*(r-1),1);                 %# Make the second diagonal vector
                                            %#   (for vertical connections)
adj = diag(diagVec1,1)+...                  %# Add the diagonals to a zero matrix
      diag(diagVec2,c);
adj = adj+adj.';                            %# Add the matrix to a transposed
                                            %# copy of itself to make it
                                            %# symmetric
%Get upper triangular parrt of adj matrix 
triuAdj = triu(adj);

%Set weights of neighbourhood edges
w = 5;
triuAdj = triuAdj .* w;

%Alpha expansion code uses sparse matrix
Sadj = sparse(triuAdj);

%Convert image to 1x(ISize1DxISize1D)x3 matrix for graph cut code
%Eg. if image size is 128x128x3, converted matrix equals to 1x16384x3
I1DArray = reshape (I,[1,pixelSize,3]);

%Set Label offset matrix
offset = zeros(labelSize,2);

%set offset values for I(1,1)
%Namely, finding offset values of assigning pixel(1,1) each jigsaw pixel.
index = 1; 
for i = 1 : j1D
    for j = 1 : j1D
        offset (index,1) = 1 - i;
        offset (index,2) = 1 - j;
        index = index + 1;
    end
end

%set offset values for I(1,2:end)
for i = 2 : ISize1D
    for j = 1 : j1D
        offset(index,1) = 1 - j; 
        offset(index,2) = i -1;
        index = index + 1;
    end
end

%set offset values for I(2:end,1)
for i = 2 : ISize1D
    for j = 1 : j1D
        offset(index,1) = i - 1;
        offset(index,2) = 1 - j;
        index = index + 1;
    end
end

%set offset matrix for rest of the image pixels (I(2:end,2:end)) 
for i = 2 : ISize1D
    for j = 2 : ISize1D
        offset(index,1) = i - 1;
        offset(index,2) = j - 1;
        index = index + 1;
    end
end

%Test offset matrix has same offset value;
% disp('start  to check labels');
% for i = 1 : labelSize
%     for j = 1 : labelSize
%         if ((offset(i,1)==offset(j,1)) && (offset(i,2) == offset (j,2)) && (i ~= j))
%             disp('label error');
%             i
%             j
%             break
%         end
%     end
% end
% disp('end of cheeck label');
% i

%Beginning of EM algorithm
labelSize = 100;
for em = 1 : 1 %Beginning of EM iteration
    disp ('EM iteration nu=');
    em
    
    %Set data cost matrix
    disp ('setting data cost matrix');
    dataCost = zeros (labelSize, pixelSize);
    for i = 1 : labelSize
        disp('Label =');
        i
        for j = 1 : pixelSize
            %Convert 1D pixel to image 2D index
            [IX,IY] = ind2sub(ISize2D,j);
            %Convert offset value to jigsaw index
            jX = mod ((IX - offset (i,1)),j1D);
            if (jX == 0) 
                jX = j1D;
            end
            jY = mod ((IY - offset (i,2)),j1D);
            if (jY == 0) 
                jY = j1D;
            end
            for k = 1 : 3   
                if (jX < 0 || jX >32 || jY < 0 || jY>32)
                    disp ('index error');
                end
                dataCost(i,j) = (I1DArray(1,j,k) - jMean(jX,jY,k))^2 + dataCost(i,j);
            end
            dataCost(i,j) = dataCost(i,j) * 100;
        end
    end
    
    %Alpha expansion graphcut uses int32
    dataCost = int32(dataCost);
    disp ('setting data cost matrix finished');
    
    %Create graph cut handle
    disp('Create grap cut handle');
    h = GCO_Create(pixelSize,labelSize);
    
    %Set data cost matrix for alpha expansion graph cut
    disp ('setting data cost matrix');
    GCO_SetDataCost(h,dataCost);
    
    %Since we use Pott's model we dont call GCO_SetSmootthCost
    
    %Setting Neighborhood relation
    disp ('setting neighborhood relation matrix');
    GCO_SetNeighbors(h,Sadj);
    
    %Start expanssion so as to assign labels
    disp('Expansion step begin');
    GCO_Expansion(h);
    
    %Assign optimized label values for each pixel to label matrix (16384x1)
    disp('Show labels');
    label = GCO_GetLabeling(h);
    
    %Get optimized energy
    disp('Get optimized energy');
    [E D S] = GCO_ComputeEnergy(h)
    
    %Update Jigsaw mean and variance step
    %jigsawLabel(:,:,1) = how many times this label assigned to a pixel
    %jigsawLabel(:,:,2) = total value of all assigned pixels to this jigsaw
    %pixel
    %jigsawLabel(:,:,3) = sum of squares of assigned pixel values
    disp('Updating jMean and jVar');
    jigsawLabel = zeros (j1D,j1D);
    jigsawAssignedPixel = zeros (j1D,j1D,3);
    jigsawAssignedPixel2 = zeros (j1D,j1D,3);
    for i = 1 : pixelSize
        %Convert 1D pixel to image 2D index
        [IX,IY] = ind2sub(ISize2D,j);
        zX = mod((IX - offset(label(i),1)),j1D);
        if (zX == 0) 
            zX = j1D;
        end
        zY = mod((IY - offset(label(i),2)),j1D);
        if (zY == 0) 
            zY = j1D;
        end
        jigsawLabel(zX,zY) = jigsawLabel(zX,zY) + 1;
        for j = 1 :3
            jigsawAssignedPixel(zX,zY,j) = jigsawAssignedPixel(zX,zY,j) + I(IX,IY,j);
            jigsawAssignedPixel2(zX,zY,j) = jigsawAssignedPixel2(zX,zY,j) + (I(IX,IY,j)^2);
        end
    end
    
    %Update Jigsaw mean and variance
    for i = 1 : j1D
        for j = 1 : j1D
            for k = 1 : 3
                jMean(i,j,k) = (((beta * mean_0) + jigsawAssignedPixel(i,j,k)) / (beta + jigsawLabel(i,j)));
                jVar(i,j,k) = ((b(:,:,k) + (beta * mean_0^2) - ((beta + jigsawLabel(i,j)) * jMean(i,j,k)^2) + jigsawAssignedPixel2(i,j,k) ) / (a(:,:,k) + jigsawLabel(i,j)));
            end
        end
    end
    disp('jMean and jVar is updated');
end

%Reconstruct image
disp('Starting to create reconstructed image');
for i =  1 : pixelSize
    %Convert 1D pixel to image 2D index
    [IX,IY] = ind2sub(ISize2D,i);
    %Convert offset value to jigsaw index
    jX = mod ((IX - offset (label(i),1)),j1D);
    if (jX == 0) 
        jX = j1D;
    end
    jY = mod ((IY - offset (label(i),2)),j1D);
    if (jY == 0) 
        jY = j1D;
    end
    for j = 1 : 3
        reconsImage(IX,IY,j) = jMean(jX,jY);
        reconsNoisyImage(IX,IY,j) = imnoise(reconsImage(IX,IY,j),'localvar',jVar(jX,jY,j));
    end 
end
disp('Reconstructed images created');

%Draws reconstructed images
figure, imagesc(reconsImage), title('Reconstructed Image');
figure, imagesc(reconsNoisyImage), title('Reconstructed Noisy Image');

%Save jmean, jvar, reconsImage andte reconsNoisyImage
disp('save jMean as jigsaw_mean.png');
imwrite(jMean,'jigsaw_mean.png');
disp('jMean saved');

disp('save j jVar as jigsaw_var.png');
imwrite(jVar,'jigsaw_var.png');
disp('jVar saved');

disp('save reconsImage as reconstructed_wo_noise.png');
imwrite(reconsImage,'reconstructed_wo_noise.png');
disp('reconsImage saved');

disp('save reconsNoisyImage as reconstructed_noisy.png');
imwrite(reconsNoisyImage,'reconstructed_noisy.png');
disp('reconsNoisyImage saved');
