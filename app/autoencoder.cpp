#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <iostream>
#include <cmath>

//c++ -O3 -Ofast -Wall -shared -std=c++20 -fPIC `python3.12 -m pybind11 --includes` autoencoder.cpp -o autoencoder_module`python3.12-config --extension-suffix`

namespace py = pybind11;

// Función de activación ReLU
std::vector<double> relu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = std::max(0.0, x[i]);
    }
    return result;
}

// Derivada de ReLU
std::vector<double> relu_derivative(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] > 0 ? 1.0 : 0.0;
    }
    return result;
}

// Función de activación sigmoide
std::vector<double> sigmoid(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = 1.0 / (1.0 + std::exp(-x[i]));
    }
    return result;
}

// Derivada de la sigmoide
std::vector<double> sigmoid_derivative(const std::vector<double>& x) {
    std::vector<double> result = sigmoid(x);
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = result[i] * (1 - result[i]);
    }
    return result;
}

// Producto punto de dos vectores
double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// Multiplicación de matriz por vector
std::vector<double> mat_vec_mult(const std::vector<std::vector<double>>& mat, const std::vector<double>& vec) {
    std::vector<double> result(mat.size());
    for (size_t i = 0; i < mat.size(); ++i) {
        result[i] = dot_product(mat[i], vec);
    }
    return result;
}

// Suma de vector y escalar
std::vector<double> add_vector_scalar(const std::vector<double>& vec, double scalar) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); ++i) {
        result[i] = vec[i] + scalar;
    }
    return result;
}

// Clase de Autoencoder simple
class Autoencoder {
public:
    Autoencoder(int input_size, int hidden_size)
        : input_size(input_size), hidden_size(hidden_size) {
        // Inicialización de pesos y sesgos
        weights_encoder.resize(hidden_size, std::vector<double>(input_size, 0.1));
        weights_decoder.resize(input_size, std::vector<double>(hidden_size, 0.1));
        biases_encoder.resize(hidden_size, 0.1);
        biases_decoder.resize(input_size, 0.1);
    }

    std::vector<double> forward(const std::vector<double>& input) {
        // Codificación
        hidden_layer = relu(add_vector_scalar(mat_vec_mult(weights_encoder, input), biases_encoder[0]));
        // Decodificación
        output_layer = sigmoid(add_vector_scalar(mat_vec_mult(weights_decoder, hidden_layer), biases_decoder[0]));
        return output_layer;
    }

    std::string train(const std::vector<std::vector<double>>& data, int epochs, double learning_rate) {
        std::string result;
    
        for (int epoch = 0; epoch < epochs; ++epoch) {
            double total_loss = 0.0;
            for (const auto& sample : data) {
                // Forward pass
                std::vector<double> output = forward(sample);

                // Cálculo de la pérdida (MSE)
                std::vector<double> error(sample.size());
                for (size_t i = 0; i < sample.size(); ++i) {
                    error[i] = sample[i] - output[i];
                    total_loss += error[i] * error[i];
                }

                // Backpropagación simplificada (ajustes de pesos omitidos por simplicidad)
            }
            //std::cout << "Epoch " <<  << ", Loss: " << total_loss / data.size() << std::endl;
            int ee = epoch + 1;
            float ff = total_loss / data.size();
            result += "Epoch " + std::to_string(ee) + "; Loss: " + std::to_string(ff) + "\n";
        }
        return result; 
    }

private:
    int input_size;
    int hidden_size;
    std::vector<std::vector<double>> weights_encoder;
    std::vector<std::vector<double>> weights_decoder;
    std::vector<double> biases_encoder;
    std::vector<double> biases_decoder;
    std::vector<double> hidden_layer;
    std::vector<double> output_layer;
};

// PyBind11 para exportar la clase
PYBIND11_MODULE(autoencoder_module, m) {
    py::class_<Autoencoder>(m, "Autoencoder")
        .def(py::init<int, int>())
        .def("forward", &Autoencoder::forward)
        .def("train", &Autoencoder::train);
}
