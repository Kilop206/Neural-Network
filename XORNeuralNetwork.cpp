#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

// Activation Functions
double ReLU(double x) {
    return max(0.0, x);
}

double ReLU_derivative(double x) {
    return (x > 0) ? 1 : 0;
}

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double a) {
    return a * (1 - a);
}

// LOSS function
double meanSquaredError(const vector<double>& predictions, const vector<double>& targets) {
    double sum = 0;
    for (size_t i = 0; i < predictions.size(); i++) {
        sum += pow(predictions[i] - targets[i], 2);
    }
    return sum / predictions.size();
}

class NeuralNetwork {
    public:
        vector<vector<double>> weights1, weights2;
        vector<double> bias1, bias2;

        NeuralNetwork(int inputSize, int hiddenSize, int outputSize) 
        {
            weights1.resize(inputSize, vector<double>(hiddenSize));
            weights2.resize(hiddenSize, vector<double>(outputSize));

            bias1.resize(hiddenSize, 0.1);
            bias2.resize(outputSize, 0.1);

            initWeights();
        }
        
        void initWeights() 
        {
            for(auto &row : weights1) {
                for(double &w : row) {
                    w = (double) rand() / RAND_MAX - 0.5;
                }
            }

            for(auto &row : weights2) {
                for(double &w : row) {
                    w = (double) rand() / RAND_MAX - 0.5;
                }
            }
        }

        // Forward Propagation
        vector<double> forward(const vector<double>& input, vector<double> &hiddenLayer) 
        {
            hiddenLayer.assign(weights1[0].size(), 0);

            for(size_t i = 0; i < weights1.size(); i++) {
                for(size_t j = 0; j < weights1[0].size(); j++) {
                    hiddenLayer[j] += input[i] * weights1[i][j];
                }
            }

            for(size_t j = 0; j < hiddenLayer.size(); j++) {
                hiddenLayer[j] = ReLU(hiddenLayer[j] + bias1[j]);
            }

            vector<double> output(weights2[0].size(), 0);

            for(size_t i = 0; i < weights2.size(); i++) {
                for(size_t j = 0; j < weights2[0].size(); j++) {
                    output[j] += hiddenLayer[i] * weights2[i][j];
                }
            }

            for(size_t j = 0; j < output.size(); j++) {
                output[j] = sigmoid(output[j] + bias2[j]);
            }

            return output;
        }

        vector<double> predict(const vector<double>& input) {
            vector<double> hiddenLayer;
            return forward(input, hiddenLayer);
        }

        // Training with Backpropagation
        void train(const vector<vector<double>>& inputs, const vector<vector<double>>& targets, double learningRate, int epochs) 
        {
            for(int e = 0; e < epochs; e++) {
                double totalLoss = 0;

                for(size_t i = 0; i < inputs.size(); i++) {
                    vector<double> hiddenLayer;
                    vector<double> output = forward(inputs[i], hiddenLayer);
                    double loss = meanSquaredError(output, targets[i]);
                    totalLoss += loss;

                    // Compute Gradients (Backpropagation)
                    vector<double> outputGradients(output.size());
                    for (size_t j = 0; j < output.size(); j++) {
                        outputGradients[j] = (output[j] - targets[i][j]) * sigmoid_derivative(output[j]);
                    }

                    vector<double> hiddenGradients(hiddenLayer.size());
                    for(size_t j = 0; j < hiddenLayer.size(); j++) {
                        hiddenGradients[j] = 0;
                        for(size_t k = 0; k < output.size(); k++) {
                            hiddenGradients[j] += outputGradients[k] * weights2[j][k];
                        }
                        hiddenGradients[j] *= ReLU_derivative(hiddenLayer[j]);
                    }

                    // Update weights and Biases
                    for(size_t j = 0; j < weights2.size(); j++) {
                        for(size_t k = 0; k < weights2[j].size(); k++) {
                            weights2[j][k] -= learningRate * outputGradients[k] * hiddenLayer[j];
                        }
                    }

                    for(size_t j = 0; j < bias2.size(); j++) {
                        bias2[j] -= learningRate * outputGradients[j];
                    }

                    for(size_t j = 0; j < weights1.size(); j++) {
                        for(size_t k = 0; k < weights1[0].size(); k++) {
                            weights1[j][k] -= learningRate * hiddenGradients[k] * inputs[i][j];
                        }
                    }

                    for(size_t j = 0; j < bias1.size(); j++) {
                        bias1[j] -= learningRate * hiddenGradients[j];
                    }
                }

                if (e % 1000 == 0) {
                    cout << " EPOCH : " << e << " LOSS : " << totalLoss / inputs.size() << endl;
                }
            }
        }
};

int main() {

    NeuralNetwork nn(2, 3, 1); 

    // XOR Training data
    vector<vector<double>> inputs = {{0,0}, {0,1}, {1,0}, {1,1}}; 
    vector<vector<double>> targets = {{0}, {1}, {1}, {0}}; 

    nn.train(inputs, targets, 0.1, 10000);

    cout << "\nTraining complete\n" << endl;

    cout << "Predictions after training: " << endl;
    for (const auto &input : inputs) {
        vector<double> prediction = nn.predict(input);
        cout << "Input : [" << input[0] << " , " << input[1] << "] -> Prediction : " << prediction[0] << endl;
    }

    return 0;
}