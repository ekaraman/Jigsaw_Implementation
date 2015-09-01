clear all;

% read in an image
I = double(imread('C:\Users\KARAMAN\Google Drive\RESEARCH\Jigsaw_Epitome\Jigsaw_Implementation\128_Dog.png'))/255;

%constants
beta = 1;
mean_0 = 0.5;
% the size of the jigsaw 
jSize = [32 32 3];

%Initialize jigsaw mean, jigsaw variance and offset map L matrices
jMean = zeros(jSize);
jVar = zeros(jSize);
L = zeros (size(I));

%Mean and variance of image elements
Isize = size(I);
sumX = sum(sum(I, 1), 2);
sumXX = sum(sum(I.^2, 1), 2);
pixelMean = sumX ./ prod(Isize(1:end-1));
pixelVar = sumXX ./ prod(Isize(1:end-1)) - pixelMean.^2;
pixelStd = sqrt(sumXX ./ prod(Isize(1:end-1)) - pixelMean.^2);


%Initialize Gamma distribution parameters
b = 3 * (1 / pixelVar);
a = b.^2;

%Initialize Jigsaw Variance
%Initialize Jigsaw Mean
for i = 1 : Isize(end)
    jVar(:,:,i) = b(i) / a(i);
end

%Initialize Jigsaw Mean
for i = 1 : Isize(end)
    jMean (:,:,i) = random('norm', pixelMean(1,i), pixelVar(1,i), jSize(1) , jSize(1));
end

%Initialize Data Cost matrix
iASize = Isize(1)^2;
iALSize = jSize (1)^2;
dataCost = zeros (iALSize, iASize);

%Calculate Data Cost for eeach pixel
Iarray = reshape (I,[1,iASize,3]);
jarray = reshape (jMean,[1,iALSize,3]);

disp ('setting data cost matrix');
for i = 1 : iALSize
    for j = 1 : iASize
        for k = 1 : 3
            dataCost(i,j) = (Iarray(1,j,k) - jarray(1,i,k))^2 + dataCost(i,j);
        end
        dataCost(i,j) = dataCost(i,j) * 1000;
    end
    i
end


dataCost = int32(dataCost);

h = GCO_Create(iASize,iALSize);

disp ('setting data cost matrix');
GCO_SetDataCost(h,dataCost);

disp ('setting smooth cost matrix');
%Calculate Smooth Cost
smoothCost = int32(zeros (iALSize, iALSize));
w = 0.2; %weight
for i = 1 : iALSize
    for j = 1 : iALSize
        if (i == j) 
            smoothCost(i,j) = 0;
        else
            smoothCost(i,j)  = 5;
        end
        
    end
end

disp ('setting smooth cost');
GCO_SetSmoothCost(h,smoothCost);

%Set neighbours

[r,c] = size(I(:,:,1));     %# Get the matrix size
diagVec1 = repmat([ones(c-1,1); 0],r,1);  %# Make the first diagonal vector
                                          %#   (for horizontal connections)
diagVec1 = diagVec1(1:end-1);             %# Remove the last value
diagVec2 = ones(c*(r-1),1);               %# Make the second diagonal vector
                                          %#   (for vertical connections)
adj = diag(diagVec1,1)+...                %# Add the diagonals to a zero matrix
      diag(diagVec2,c);
adj = adj+adj.';                         %'# Add the matrix to a transposed
                                          %#   copy of itself to make it
                                          %#   symmetric
%adj
triuAdj = triu(adj);
Sadj = sparse(triuAdj);
disp ('setting neighborhood matrix');
GCO_SetNeighbors(h,Sadj);
disp('Expansion step begin');
GCO_SetVerbosity(h,2);
GCO_Expansion(h);
disp('Show labels');
label = GCO_GetLabeling(h);

%Update Jigsaw Mean
disp('Update Jigsaw Mean');
%Find the set of image pixels that are mapped to the jigsaw pixel z
sI = [Isize(1), Isize(1)];
sJ = [jSize(1), jSize(1)];

for i = 1: iALSize 
    xIndex = find (label == i);
   [Xj,Yj] = ind2sub(sJ,i);
    if (isEmpty(xIndex) == 0)
        xDim = size (xIndex,1);
        for j = 1 : 3
            xZ = 0;
            for k = 1 : xDim
                %Convert 1D label index to 2D image index 
                [Xi,Yi] = ind2sub(sI,xIndex(k));
                xZ = xZ + I (Xi,Yi,j);
            end
            jMean(Xj,Yj,j) = (((beta * mean_0) + xZ) / (beta +xDim));
        end
    end
end

jMean