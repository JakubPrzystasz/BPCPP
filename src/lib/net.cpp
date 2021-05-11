#include "net.h"

void Net::read_file(std::string filename, pattern_set &input_data)
{
    std::ifstream input_file(filename);

    if (!input_file)
        throw std::invalid_argument("Error in opening input file");

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

void Net::train(double max_epoch, double error_goal)
{
/**
 * 
 * Make vector of indexes for trainig, validate, and test sets
 * After each epoch make test
 * after tarining make validation
 * https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets
 */


    for (uint32_t epoch{0}; epoch < max_epoch; epoch++)
    {
        for (this->batch_it = 0; this->batch_it < input_data.size(); this->batch_it++)
        {
            this->feed(batch_it);
            //Get delta and SSE
            this->get_delta(batch_it);
        }

        if ((epoch % 10) == 0)
            std::cout << "Epoch: " << epoch << "  SSE: " << this->SSE << "  MSE:" << this->SSE / static_cast<double>(input_data.size()) << std::endl;

        if (this->SSE <= error_goal)
            break;

        //Adaptive learning rate:
        //uses bold driver method
        if (this->learning_accelerating_constans > 0 && this->learning_decelerating_constans > 0)
        {
            if (this->SSE_previous > this->SSE)
                this->learning_rate = this->learning_rate * this->learning_accelerating_constans;
            else
                this->learning_rate = this->learning_rate * this->learning_decelerating_constans;
        }
        this->SSE_previous = this->SSE;
        this->SSE = 0;
    }
}

void Net::feed(uint32_t sample_number)
{

    //First set input layer
    auto &layer = this->layers.front();
    auto &data = this->input_data[sample_number].input;

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
            neuron.output = neuron.activation(neuron.output, &(neuron.learn_parameters.beta_param));
            neuron.derivative_output = neuron.derivative(neuron.output, &(neuron.learn_parameters.beta_param));
        }
    }
}

void Net::get_delta(uint32_t sample_number)
{
    //Find delta for each neuron in each layer
    //Calculate delta of the output layer
    auto &last_layer = this->layers.back();
    for (uint32_t neuron_it{0}; neuron_it < last_layer.neurons.size(); neuron_it++)
    {
        auto &neuron = last_layer.neurons[neuron_it];
        neuron.delta = (neuron.output - this->input_data[sample_number].output[neuron_it]) * neuron.derivative_output;
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
}

void Net::setup(std::vector<uint32_t> hidden_layers, LearnParams params)
{
    this->learn_parameters = params;

    //If params.batch_size is 0, then set batch_size equal to size of whole training set
    if (this->learn_parameters.batch_size == 0)
        this->learn_parameters.batch_size = this->input_data.size() * this->subsets_ratio.front();

    //Setup layers
    this->layers = std::vector<Layer>();

    //input layer
    this->layers.push_back(Layer(this->input_size, this->input_size, params));

    //Hidden layers
    for (uint32_t i{0}; i < hidden_layers.size(); i++)
        this->layers.push_back(Layer(hidden_layers[i], (i > 0 ? hidden_layers[i - 1] : input_size), params));

    //output layer
    this->layers.push_back(Layer(this->output_size, this->layers.back().neurons.size(), params));
}

Net::Net(pattern_set &input_data, std::array<double,3> subsets_ratio)
{
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
            throw std::invalid_argument(std::string("Invalid input pattern size at: ") + std::to_string(it + 1));

        if (set.output.size() != this->output_size)
            
        it++;
    }
    

    //Assign given parameters
    this->input_data = input_data;

    //Verify subsets_ratio
    {
        double tmp = 0;
        for(auto &value: subsets_ratio)
            tmp += value;
        if(tmp != 1)
            throw std::invalid_argument(std::string("Sum of subsets ratio is not equal 1"));
    }

    this->subsets_ratio = subsets_ratio;
}

Net::~Net() {}
