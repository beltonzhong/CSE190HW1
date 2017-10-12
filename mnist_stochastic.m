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
errorRates = zeros(1, 50);
for i=1:numEpochs
    correct = 0;
    p = randperm(20000);
    for j=1:20000
        image = images(:, p(j));
        g0 = exp(w(:, 1)' * image);
        g1 = exp(w(:, 2)' * image);
        g2 = exp(w(:, 3)' * image);
        g3 = exp(w(:, 4)' * image);
        g4 = exp(w(:, 5)' * image);
        g5 = exp(w(:, 6)' * image);
        g6 = exp(w(:, 7)' * image);
        g7 = exp(w(:, 8)' * image);
        g8 = exp(w(:, 9)' * image);
        g9 = exp(w(:, 10)' * image);
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
        label = labels(p(j));
        correct = correct + (label == predictions);
        teachingLabel = labels1hot(p(j), :);
        predictions1hot = zeros(1, 10);
        for k=1:10
            predictions1hot(predictions == k - 1, k) = 1;
        end
        predictions1 = teachingLabel - predictions1hot;
        for k=1:10
            c = predictions1(:, k);
            gradient = image * c;
            w(:, k) = w(:, k) - alpha * gradient;
        end
    end
    errorRates(1, i) = (20000 - correct) / 200;
end

figure
scatter([1:50], errorRates);

testcorrect = 0;
for i=1:2000
    input = testimages(:, i);
    g0 = exp(w(:, 1)' * input);
    g1 = exp(w(:, 2)' * input);
    g2 = exp(w(:, 3)' * input);
    g3 = exp(w(:, 4)' * input);
    g4 = exp(w(:, 5)' * input);
    g5 = exp(w(:, 6)' * input);
    g6 = exp(w(:, 7)' * input);
    g7 = exp(w(:, 8)' * input);
    g8 = exp(w(:, 9)' * input);
    g9 = exp(w(:, 10)' * input);
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


