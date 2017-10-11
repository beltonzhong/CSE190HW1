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
alpha = .01;
numEpochs = 50;
errorRates = zeros(50, 1);
for i=1:numEpochs
    g0 = w(:, 1)' * images;
    g1 = w(:, 2)' * images;
    g2 = w(:, 3)' * images;
    g3 = w(:, 4)' * images;
    g4 = w(:, 5)' * images;
    g5 = w(:, 6)' * images;
    g6 = w(:, 7)' * images;
    g7 = w(:, 8)' * images;
    g8 = w(:, 9)' * images;
    g9 = w(:, 10)' * images;
    gs = g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9;
    h0 = g0 ./ gs;
    h1 = g1 ./ gs;
    h2 = g2 ./ gs;
    h3 = g3 ./ gs;
    h4 = g4 ./ gs;
    h5 = g5 ./ gs;
    h6 = g6 ./ gs;
    h7 = g7 ./ gs;
    h8 = g8 ./ gs;
    h9 = g9 ./ gs;
    
    g = [h0;h1;h2;h3;h4;h5;h6;h7;h8;h9];
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
scatter([1:50], errorRates);

testcorrect = 0;
for i=1:2000
    input = testimages(:, i);
    g0 = w(:, 1)' * input;
    g1 = w(:, 2)' * input;
    g2 = w(:, 3)' * input;
    g3 = w(:, 4)' * input;
    g4 = w(:, 5)' * input;
    g5 = w(:, 6)' * input;
    g6 = w(:, 7)' * input;
    g7 = w(:, 8)' * input;
    g8 = w(:, 9)' * input;
    g9 = w(:, 10)' * input;
    gs = g0 + g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8 + g9;
    h0 = g0 / gs;
    h1 = g1 / gs;
    h2 = g2 / gs;
    h3 = g3 / gs;
    h4 = g4 / gs;
    h5 = g5 / gs;
    h6 = g6 / gs;
    h7 = g7 / gs;
    h8 = g8 / gs;
    h9 = g9 / gs;
    
    g = [h0;h1;h2;h3;h4;h5;h6;h7;h8;h9];
    [values y] = max(g);
    y = y - 1;
    if y == testlabels(i)
        testcorrect = testcorrect + 1;
    end
end
disp(testcorrect);


