#include "neuron.h"
#include <limits>

namespace ActivationFunction
{
    double unipolar(double input, double *params)
    {
        return __normalize((1.0 / (1.0 + pow(M_E, -((*params) * input)))));
    }

    double unipolar_derivative(double input, double *params)
    {
        return __normalize((*params) * input * (1.0 - input));
    }

    double bipolar(double input, double *params)
    {
        return __normalize(tanh((*params) * input));
    }

    double bipolar_derivative(double input, double *params)
    {
        return __normalize((*params) * (1.0 - pow(input, 2.0)));
    }
};

Neuron::Neuron(uint32_t inputs, uint32_t ID, Layer* prev_layer)
{
    this->ID = ID;
    this->prev_layer = prev_layer;

    this->activation = ActivationFunction::unipolar;
    this->derivative = ActivationFunction::unipolar_derivative;

    this->weights = data_row(inputs, 0);
    this->beta_param = 1.0;
}

Neuron::~Neuron() {}

double &Neuron::feed(data_row &input)
{
    // This is input layer
    if (this->weights.size() == 0)
    {
        // this->input = input[]
    }
    else
    {
        this->input = bias;
        for (uint32_t i{0}; i < weights.size(); i++)
            this->input += (input[i] * weights[i]);
    }

    this->output = this->activation(this->input, &(this->beta_param));
    this->derivative_output = this->derivative(this->output, &(this->beta_param));

    return this->output;
}