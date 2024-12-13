clear;
clc;

[x1, x2] = meshgrid(0:0.05:1, 0:0.05:1); 
x1 = x1(:);
x2 = x2(:); 

Y_desired = (1 + 0.6 * sin(2 * pi * x1 / 0.7) .* sin(2 * pi * x2 / 0.7)) / 2;

num_inputs = 2; 
num_hidden_neurons = 8;
num_outputs = 1;

W1 = randn(num_hidden_neurons, num_inputs); % Weights for input to hidden layer
b1 = randn(num_hidden_neurons, 1);          % Biases for hidden layer
W2 = randn(num_outputs, num_hidden_neurons);% Weights for hidden to output layer
b2 = randn(num_outputs, 1);                 % Bias for output layer

learning_rate = 0.1;
error_threshold = 1;

sigmoid = @(z) 1 ./ (1 + exp(-z));
dsigmoid = @(z) sigmoid(z) .* (1 - sigmoid(z));

epoch = 0;
total_error = inf; 

while total_error > error_threshold
    total_error = 0;
    epoch = epoch + 1;
    
    for i = 1:length(x1)
        input = [x1(i); x2(i)];
        y_true = Y_desired(i);
        z1 = W1 * input + b1;
        a1 = sigmoid(z1);
        z2 = W2 * a1 + b2;
        y_pred = z2; 

        error = y_true - y_pred;
        total_error = total_error + error^2; 

        delta2 = error;
        delta1 = (W2' * delta2) .* dsigmoid(z1);

        W2 = W2 + learning_rate * delta2 * a1';
        b2 = b2 + learning_rate * delta2;
        W1 = W1 + learning_rate * delta1 * input';
        b1 = b1 + learning_rate * delta1;
    end

    fprintf('Epoch %d, Total Error: %.4f\n', epoch, total_error);
end

fprintf('Training completed in %d epochs with final error %.4f\n', epoch, total_error);


Y_pred = zeros(size(Y_desired));
for i = 1:length(x1)
    input = [x1(i); x2(i)];
    z1 = W1 * input + b1;
    a1 = sigmoid(z1);
    z2 = W2 * a1 + b2;
    
    Y_pred(i) = z2; 
end

x1_grid = reshape(x1, sqrt(length(x1)), []);
x2_grid = reshape(x2, sqrt(length(x2)), []);
Y_desired_grid = reshape(Y_desired, size(x1_grid));
Y_pred_grid = reshape(Y_pred, size(x1_grid));

% Plot results
figure;
surf(x1_grid, x2_grid, Y_desired_grid, 'FaceColor', 'red'); 
hold on;
surf(x1_grid, x2_grid, Y_pred_grid);
legend('Desired Surface', 'MLP Approximation');
title('MLP Surface Approximation');
xlabel('Input x1');
ylabel('Input x2');
zlabel('Output Y');
