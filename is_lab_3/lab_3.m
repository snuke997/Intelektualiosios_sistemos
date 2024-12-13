x = 0.1:1/22:1; 
y = ((1 + 0.6 * sin(2 * pi * x / 0.7)) + 0.3 * sin(2 * pi * x)) / 2;

c1 = 0.5; r1 = 0.1;
c2 = 0.65; r2 = 0.2;

F1 = exp(-((x - c1).^2) / (2 * r1^2));
F2 = exp(-((x - c2).^2) / (2 * r2^2)); 

Phi = [ones(length(x), 1), F1', F2'];
Y = y';
weights = rand(3, 1);

learning_rate = 0.01; 
error_threshold = 0.01; 
max_iterations = 5000;
iteration = 0;

while true
    iteration = iteration + 1;
    error_sum = 0;
    
    for i = 1:length(x)
        y_hat = Phi(i, :) * weights;
        error = Y(i) - y_hat;
        
        weights = weights + learning_rate * error * Phi(i, :)';
        
        error_sum = error_sum + abs(error);
    end
    
    mae = error_sum / length(x);
    
    if mae < error_threshold
        fprintf('Training stopped after %d iterations. MAE: %.4f\n', iteration, mae);
        break;
    end
    
    if iteration >= max_iterations
        fprintf('Max iterations reached. Final MSE: %.4f\n', mae);
        break;
    end
end

y_train_hat = Phi * weights;

figure;
plot(x, y, 'b', 'LineWidth', 1.5, 'DisplayName', 'Desired Output (Training)');
hold on;
plot(x, y_train_hat, 'r--', 'LineWidth', 1.5, 'DisplayName', 'RBF Network Output (Training)');
legend('show');
xlabel('Input x');
ylabel('Output y');
title('RBF Network with Perceptron Training (Training Results)');
grid on;
hold off;

% test
x_test = 1.01:0.01:1.3;
F1_test = exp(-((x_test - c1).^2) / (2 * r1^2));
F2_test = exp(-((x_test - c2).^2) / (2 * r2^2));
Phi_test = [ones(length(x_test), 1), F1_test', F2_test'];

y_test_hat = Phi_test * weights;
y_test = ((1 + 0.6 * sin(2 * pi * x_test / 0.7)) + 0.3 * sin(2 * pi * x_test)) / 2;

test_error = sum(mean(abs(y_test - y_test_hat)));
fprintf('Testing Error (MAE): %.4f\n', test_error);

figure;
plot(x_test, y_test, 'b', 'LineWidth', 1.5, 'DisplayName', 'Desired Output (Testing)');
hold on;
plot(x_test, y_test_hat, 'r--', 'LineWidth', 1.5, 'DisplayName', 'RBF Network Output (Testing)');
legend('show');
xlabel('Input x');
ylabel('Output y');
title('RBF Network with Perceptron Training (Testing Results)');
grid on;
hold off;
