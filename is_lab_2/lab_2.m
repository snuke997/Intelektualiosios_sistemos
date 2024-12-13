clear;
clc;

X = 0:0.05:1;

Y_desired = (1 + 0.6 * sin(2 * pi * X / 0.7) + 0.3 * sin(2 * pi * X)) / 2;

num_inputs = 1;
num_hidden_neurons = 6;
num_outputs = 1;

W1 = randn(num_hidden_neurons, num_inputs); % Weights for input to hidden layer
b1 = randn(num_hidden_neurons, 1);          % Biases for hidden layer
W2 = randn(num_outputs, num_hidden_neurons);% Weights for hidden to output layer
b2 = randn(num_outputs, 1);                 % Bias for output layer

learning_rate = 0.1;
error_threshold = 0.001; 

sigmoid = @(z) 1 ./ (1 + exp(-z));
dsigmoid = @(z) sigmoid(z) .* (1 - sigmoid(z));

epoch = 0;
total_error = 1;

while total_error > error_threshold
    total_error = 0;
    epoch = epoch + 1;
    
    for i = 1:length(X)
        x = X(i);
        y_true = Y_desired(i);
        z1 = W1 * x + b1;
        a1 = sigmoid(z1);
        z2 = W2 * a1 + b2;
        y_pred = z2;

        error = y_true - y_pred;
        total_error = total_error + error^2;

        delta2 = error;
        delta1 = (W2' * delta2) .* dsigmoid(z1);

        W2 = W2 + learning_rate * delta2 * a1';
        b2 = b2 + learning_rate * delta2;
        W1 = W1 + learning_rate * delta1 * x;
        b1 = b1 + learning_rate * delta1;
    end
    
    fprintf('Epoch %d, Total Error: %.4f\n', epoch, total_error);
end

fprintf('Training completed in %d epochs with final error %.4f\n', epoch, total_error);

% Testing
x_test = 1:0.05:1.3;
Y_desired_test = (1 + 0.6 * sin(2 * pi * x_test / 0.7) + 0.3 * sin(2 * pi * x_test)) / 2;
total_error_test = 0;
for i = 1:length(x_test)
    x = x_test(i);
    y_true = Y_desired_test(i);
    
    z1 = W1 * x + b1;
    a1 = sigmoid(z1);
    z2 = W2 * a1 + b2;
    y_pred = z2;
    
    fprintf('y_pred = %.4f, y_true = %.4f\n', y_pred, y_true);
    
    error = y_true - y_pred;
    total_error_test = total_error_test + error^2;
end

fprintf('testing complete with final error %.4f\n', total_error_test);

% Data for plot
Y_pred = zeros(1, length(X));
for i = 1:length(X)
    x = X(i);
    
    z1 = W1 * x + b1;
    a1 = sigmoid(z1);
    z2 = W2 * a1 + b2;
    Y_pred(i) = z2;
end

% Plot results
figure;
plot(X, Y_desired, 'r-', 'LineWidth', 2); hold on;
plot(X, Y_pred, 'b--', 'LineWidth', 2);
legend('Desired Output', 'MLP Output');
title('MLP Approximation');
xlabel('Input X');
ylabel('Output Y');
