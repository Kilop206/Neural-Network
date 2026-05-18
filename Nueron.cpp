#include <iostream>
#include <vector>

using namespace std;

double ReLU(double x) {
    return max(0.0, x);
}

class Neuron {
    private:
        vector<double> weights;
        double bias;

    public:
        Neuron(const vector<double>& intiialWeghts, double initialBias)
            : weights(intiialWeghts),
            bias(initialBias) {}

        double forward(const vector<double>& inputs, double (*activationFunc)(double)) {

            if(inputs.size() != weights.size()) {
                throw invalid_argument("Inputs size must match weights size.");
            }

            double weightedSum = 0.0;
            for(size_t i = 0; i < inputs.size(); i++) {
                weightedSum += inputs[i] * weights[i];
            }

            weightedSum += bias;

            return activationFunc(weightedSum);
        }
};

int main() {
    vector<double> weights = {0.5, -1.2, 0.3};
    double bias = 0.7;

    Neuron neuron(weights, bias);

    vector<double> inputs = {1.0, 0.5, -1.0};

    double output = neuron.forward(inputs, ReLU);

    cout << "Output with ReLU" << output << endl;

    return 0;
}