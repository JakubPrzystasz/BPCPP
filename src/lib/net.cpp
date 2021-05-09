#include "net.h"

void Net::read_file(std::string filename, pattern_set &input_data)
{
    /**
Output:    
    0) Class
Input:
    1) Alcohol
 	2) Malic acid
 	3) Ash
	4) Alcalinity of ash  
 	5) Magnesium
	6) Total phenols
 	7) Flavanoids
 	8) Nonflavanoid phenols
 	9) Proanthocyanins
	10)Color intensity
 	11)Hue
 	12)OD280/OD315 of diluted wines
 	13)Proline
     */
    std::ifstream input_file(filename);

    if (!input_file)
    {
        std::cout << "Unable to open input data file" << std::endl;
        exit(-1);
    }

    /**
     * input_size - size of input layer
     * output_size - size of output layer
     * output_position - 0 means that output data is stored before input data at single text row
     */
    uint32_t input_size, output_size, output_position;

    {
        //Read first line, that stores size of input and output
        std::string line;
        std::getline(input_file, line);
        std::stringstream steam = std::stringstream(line);
        steam >> input_size >> output_size >> output_position;
    }

    double tmp_value;

    for (std::string line; std::getline(input_file, line);)
    {
        std::stringstream stream = std::stringstream(line);
        //Workaround of invalid reading first value;
        stream >> tmp_value;

        Pattern sample = Pattern();

        //Output vector is before input so, read it first
        if (!output_position)
        {
            for (uint32_t i{0}; i < output_size; i++, stream >> tmp_value)
                sample.output.push_back(tmp_value);

            for (uint32_t i{0}; i < input_size; i++, stream >> tmp_value)
                sample.input.push_back(tmp_value);
        }
        else
        {
            for (uint32_t i{0}; i < input_size; i++, stream >> tmp_value)
                sample.input.push_back(tmp_value);

            for (uint32_t i{0}; i < output_size; i++, stream >> tmp_value)
                sample.output.push_back(tmp_value);
        }

        //append sample to data set
        input_data.push_back(sample);
    }
}

Net::Net(pattern_set &input_data)
{
    this->output_layer_range = std::make_pair(-0.5,0.5);

    /*
        Validate input data
    */
    if (input_data.size() == 0)
        throw std::invalid_argument("Input data set is empty");

    this->input_size = input_data.front().input.size();
    this->output_size = input_data.front().output.size();

    uint32_t it{0};
    for (auto &set : input_data)
    {
        if (set.input.size() != this->input_size)
            throw std::invalid_argument(std::string("Invalid input pattern size at: ") + std::to_string(it));

        if (set.output.size() != this->output_size)
            throw std::invalid_argument(std::string("Invalid output pattern size at: ") + std::to_string(it));

        it++;
    }

    this->input_data = input_data;
}

void Net::setup(std::vector<uint32_t> &hidden_layers)
{
    //Setup layers
    this->layers = std::vector<Layer>();
    //input layer
    this->layers.push_back(Layer(this->input_size, this->input_size, learning_rate, momentum_constans));

    auto range = std::make_pair(-1.0,1.0);

    for (uint32_t i{0}; i < hidden_layers.size(); i++)
        this->layers.push_back(Layer(hidden_layers[i], (i > 0 ? hidden_layers[i - 1] : input_size), learning_rate, momentum_const,range));

    //output layer
    this->layers.push_back(Layer(this->output_size, this->layers.back().neurons.size(), learning_rate, momentum_const, this->output_layer_range));
}

Net::~Net() {}

void Net::feed(uint32_t data_row_num)
{

    //First set input layer
    auto &layer = this->layers[0];
    auto &data = this->input[data_row_num];

    for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
    {
        auto &neuron = layer.neurons[neuron_it];

        neuron.input = neuron.output = neuron.derivative_output = data[neuron_it];
    }

    //Set hidden layers and output layer
    for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
    {
        auto &layer = this->layers[layer_it];
        auto &prev_layer = this->layers[layer_it - 1];

        for (auto &neuron : layer.neurons)
        {
            neuron.output = neuron.bias;
            for (uint32_t weight_it{0}; weight_it < neuron.weights.size(); weight_it++)
                neuron.output += (neuron.weights[weight_it] * prev_layer.neurons[weight_it].output);
            neuron.output = neuron.activation(neuron.output, &(neuron.beta_param));
            neuron.derivative_output = neuron.derivative(neuron.output, &(neuron.beta_param));
        }
    }
}

void Net::train(uint32_t data_row_num)
{
    this->feed(data_row_num);

    //Find delta for each neuron
    //Calculate delta of the last layer
    auto &last_layer = this->layers.back();
    for (uint32_t neuron_it{0}; neuron_it < last_layer.neurons.size(); neuron_it++)
    {
        auto &neuron = last_layer.neurons[neuron_it];
        neuron.delta = (neuron.output - target[data_row_num][neuron_it]) * neuron.derivative_output;
    }

    for (uint32_t layer_it{static_cast<uint32_t>(this->layers.size()) - 2}; layer_it > 0; layer_it--)
    {
        auto &layer = this->layers[layer_it];
        for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
        {
            auto &neuron = layer.neurons[neuron_it];

            double sum = 0;
            for (uint32_t next{0}; next < this->layers[layer_it + 1].neurons.size(); next++)
            {
                auto &next_neuron = this->layers[layer_it + 1].neurons[next];
                sum += next_neuron.weights[neuron_it] * next_neuron.delta;
            }

            //calculate the delta for current neuron
            neuron.delta = sum * neuron.derivative_output;
        }
    }

    static double delta;

    //Update weights and biases
    for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
    {
        auto &layer = this->layers[layer_it];
        for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
        {
            auto &neuron = layer.neurons[neuron_it];

            delta = -1.0 * neuron.delta * layer.learning_rate;

            neuron.bias += delta;

            for (uint32_t weights_it{0}; weights_it < neuron.weights.size(); weights_it++)
                neuron.weights[weights_it] += delta * this->layers[layer_it - 1].neurons[weights_it].output;
        }
    }

    //Calculate error and SSE
    auto &target = this->target[data_row_num];
    static double error = 0;

    for (uint32_t neuron_it{0}; neuron_it < last_layer.neurons.size(); neuron_it++)
        error += (target[neuron_it] - last_layer.neurons[neuron_it].output) * (target[neuron_it] - last_layer.neurons[neuron_it].output);

    this->SSE += (error / last_layer.neurons.size());
}
