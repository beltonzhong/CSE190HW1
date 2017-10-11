imagedata = loadMNISTImages('train-images.idx3-ubyte');
labeldata = loadMNISTLabels('train-labels.idx1-ubyte');

images = imagedata(:, 1:20000);
images = [images; ones(1, 20000)];
labels = labeldata(1:20000,:);
labels1hot = zeros(20000, 10);
for i=1:10
    labels1hot(labels == i, i) = 1;
end


testimagedata = loadMNISTImages('t10k-images.idx3-ubyte');
testlabeldata = loadMNISTLabels('t10k-labels.idx1-ubyte');

testimages = testimagedata(:, 1:2000);
testlabels = testlabeldata(1:2000, :);

w = zeros(785, 10);
alpha = 1;
numEpochs = 1000;
for i=1:numEpochs
    [values indices] = max(w' * images);
    predictions = indices - 1;
    predictions = predictions';
    correct = sum(labels == predictions);
    message = strcat({'Training error rate: '}, {num2str((20000 - correct) / 20000 * 100)}, {'%'});
    disp(message);
    predictions1hot = zeros(20000, 10);
    for j=1:10
        predictions1hot(predictions == j - 1, j) = 1;
    end
    predictions1 = labels1hot - predictions1hot;
    imageBuf = images;
    gradient = imageBuf * predictions1;
    w = w - alpha * gradient;
end

%for i=1:20000
%   input_vec = images(:, i)
%   w' * input_vec
%end


