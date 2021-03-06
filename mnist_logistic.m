imagedata = loadMNISTImages('train-images.idx3-ubyte');
labeldata = loadMNISTLabels('train-labels.idx1-ubyte');

images = imagedata(:, 1:20000);
images = [images; ones(1, 20000)];
labels = labeldata(1:20000,:);
labels1hot = zeros(20000, 10);
for i=1:10
    labels1hot(labels == i - 1, i) = 1;
end


testimagedata = loadMNISTImages('t10k-images.idx3-ubyte');
testlabeldata = loadMNISTLabels('t10k-labels.idx1-ubyte');

testimages = testimagedata(:, 1:2000);
testimages = [testimages; ones(1, 2000)];
testlabels = testlabeldata(1:2000, :);
testlabels1hot = zeros(2000, 10);
for i=1:10
    testlabels1hot(testlabels == i - 1, i) = 1;
end

w = ones(785, 10);
alpha = .05;
numEpochs = 50;
errorRates = zeros(numEpochs, 1);
for i=1:numEpochs
    g = 1 ./ (exp(-(w' * images)) + 1);
    [values indices] = max(g);
    predictions = indices' - 1;
    correct = sum(labels == predictions);
    errorRates(i) = (20000 - correct) / 200;
    predictions1hot = zeros(20000, 10);
    for j=1:10
        predictions1hot(predictions == j - 1, j) = 1;
    end
    predictions1 = labels1hot - predictions1hot;
    for j=1:10
        c = predictions1(:, j);
        c = c';
        gradient = sum(images .* c, 2);
        w(:, j) = w(:, j) + alpha * gradient;
    end
end

figure
scatter([1:numEpochs], errorRates);

testcorrect = 0;
for i=1:2000
    input = testimages(:, i);
    g = 1 ./ (exp(-(w' * input)) + 1);
    [values y] = max(g);
    y = y - 1;
    if y == testlabels(i)
        testcorrect = testcorrect + 1;
    end
end
disp(testcorrect);


