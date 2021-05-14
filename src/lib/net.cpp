#include "net.h"

void Net::update_learning_rate(LearningRateUpdate value)
{
    for (uint32_t i{1}; i < this->layers.size(); i++)
    {
        for (auto &neuron : this->layers[i].neurons)
        {
            switch (value)
            {
            case LearningRateUpdate::Increase:
                neuron.learn_parameters.learning_rate *= neuron.learn_parameters.learning_accelerating_constans;
                break;
            default:
                neuron.learn_parameters.learning_rate *= neuron.learn_parameters.learning_decelerating_constans;
                break;
            };
        }
    }
}

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

LearnOutput Net::train(double max_epoch, double error_goal)
{
    LearnOutput out;
    //shuffle values
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(this->indexes.begin(), this->indexes.end(), g);

        for (uint32_t i{0}; i < train_set.size(); i++)
            this->train_set[i] = this->indexes[i];

        for (uint32_t i{0}; i < test_set.size(); i++)
            this->test_set[i] = this->indexes[train_set.size() + i];
    }

    std::vector<bool> classification_test = std::vector<bool>(test_set.size(), 0);
    std::vector<bool> classification_train = std::vector<bool>(train_set.size(), 0);

    double classification_accuracy{0};
    double prev_SSE;
    bool decrease_lr = false;
    uint32_t epoch{0};
    //Make first run to get SSE for adaptive learning method:

    this->SSE = 10000000000;

    for (; epoch < max_epoch; epoch++)
    {
        this->batch_it = 1;
        prev_SSE = this->SSE;
        this->SSE = 0;
        //Make full run over training data
        std::cout << epoch << "  " << prev_SSE << std::endl;

        for (uint32_t it{0}; it < train_set.size(); it++)
        {
            uint32_t index = train_set[it];
            this->learn(index);
            //if batch ends - update weights
            if ((this->batch_it % this->learn_parameters.batch_size) == 0)
            {
                this->update_weights();
                this->batch_it = 1;
            }
            else
                batch_it++;

            classification_train[it] = this->get_classification_succes(index);
        }
        //close open batch
        if (this->batch_it > 1)
            this->update_weights();

        //So now do stuff with adaptive learning rate, and momentum deltas
        if ((this->SSE / prev_SSE) > this->error_ratio)
        {
            decrease_lr = true;
            this->update_learning_rate(LearningRateUpdate::Decrease);
            //discard all deltas:
            for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
            {
                auto &layer = this->layers[layer_it];
                for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
                {
                    auto &neuron = layer.neurons[neuron_it];
                    neuron.bias -= neuron.bias_update;

                    for (uint32_t weight_it{0}; weight_it < neuron.weights.size(); weight_it++)
                        neuron.weights[weight_it] -= neuron.weight_update[weight_it];
                }
            }
        }
        else
        {
            decrease_lr = false;
            this->update_learning_rate(LearningRateUpdate::Increase);
        }

        if (decrease_lr)
        {
            for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
            {
                for (auto &neuron : this->layers[layer_it].neurons)
                {
                    if(!neuron.learn_parameters.momentum_delta_vsize)
                        continue;
                    
                    for(uint32_t weight_it{0};weight_it < neuron.weights.size();weight_it++)
                        neuron.weights_deltas[weight_it][(epoch+1) % neuron.learn_parameters.momentum_delta_vsize] = 0;
                }
            }
        }
        else
        {
            for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
            {
                for (auto &neuron : this->layers[layer_it].neurons)
                {
                    if(!neuron.learn_parameters.momentum_delta_vsize)
                        continue;
                    
                    for(uint32_t weight_it{0};weight_it < neuron.weights.size();weight_it++)
                        neuron.weights_deltas[weight_it][(epoch+1) % neuron.learn_parameters.momentum_delta_vsize] = neuron.weight_update[weight_it];
                }
            }
        }

        //Clear weight and bias update vectors
        for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
        {
            for (auto &neuron : this->layers[layer_it].neurons)
            {
                neuron.bias_update = 0;
                for (auto &weight : neuron.weight_update)
                    weight = 0;
            }
        }

        //Stuff for later analisys
        out.train_set_SSE.push_back(this->SSE);
        out.train_set_MSE.push_back(this->SSE / static_cast<double>(train_set.size()));

        classification_accuracy = 0;
        for (auto value : classification_train)
            classification_accuracy += static_cast<double>(value);
        out.train_set_accuracy.push_back(classification_accuracy / static_cast<double>(train_set.size()));

        if (this->SSE <= error_goal)
            break;
    }

    if (epoch == max_epoch)
        out.result = TrainResult::MaxEpochReached;
    else
        out.result = TrainResult::ErrorGoalReached;

    return out;
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

double Net::get_cost(uint32_t sample_number)
{
    double error{0}, tmp;
    auto &target = this->input_data[sample_number].output;
    auto &layer = this->layers.back();

    for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
    {
        tmp = (target[neuron_it] - layer.neurons[neuron_it].output);
        error += tmp * tmp;
    }
    return error / static_cast<double>(layer.neurons.size());
}

void Net::update_weights()
{
    for (uint32_t i{1}; i < this->layers.size(); i++)
    {
        for (auto &neuron : this->layers[i].neurons)
            neuron.update_weights();
    }
}

void Net::learn(uint32_t sample_number)
{
    uint32_t batch_it = this->batch_it - 1;
    double weight_deltas, delta;
    this->feed(sample_number);
    this->get_delta(sample_number);

    ///Calculate new deltas
    for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
    {
        auto &layer = this->layers[layer_it];
        for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
        {
            auto &neuron = layer.neurons[neuron_it];

            delta = -1.0 * neuron.delta * neuron.learn_parameters.learning_rate;

            neuron.batch.bias_deltas[batch_it] = delta;

            for (uint32_t weights_it{0}; weights_it < neuron.weights.size(); weights_it++)
            {
                //Momentum method:
                if (neuron.learn_parameters.momentum_delta_vsize)
                    weight_deltas = std::accumulate(neuron.weights_deltas[weights_it].begin(), neuron.weights_deltas[weights_it].end(), 0.0);
                neuron.batch.weights_deltas[weights_it][batch_it] = delta * this->layers[layer_it - 1].neurons[weights_it].output + neuron.learn_parameters.momentum_constans * weight_deltas;
            }
        }
    }

    this->SSE += this->get_cost(sample_number);
}

bool Net::get_classification_succes(uint32_t sample_number)
{
    //TODO:
    //I've got 3 classes in range -1.0 - 1.0
    //For general purpose it needs to be changed
    double out_value = this->layers.back().neurons.front().output;
    double sample_output = this->input_data[sample_number].output.front();
    if (out_value > 0.5 && sample_output == 1.0)
        return true;
    if (out_value < -0.5 && sample_output == -1.0)
        return true;
    if (out_value >= -0.5 && out_value <= 0.5 && sample_output == 0.0)
        return true;
    return false;
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

Net::Net(pattern_set &input_data, std::array<double, 2> subsets_ratio)
{
    /*
        Validate input data
    */
    if (input_data.size() == 0)
        throw std::invalid_argument("Input data set is empty");

    this->input_size = input_data.front().input.size();
    this->output_size = input_data.front().output.size();

    {
        uint32_t it{0};
        for (auto &set : input_data)
        {
            if (set.input.size() != this->input_size)
                throw std::invalid_argument(std::string("Invalid input pattern size at: ") + std::to_string(it + 1));

            if (set.output.size() != this->output_size)
                it++;
        }
    }

    //Assign given parameters
    this->input_data = input_data;

    //Verify subsets_ratio
    {
        double tmp = 0;
        for (auto &value : subsets_ratio)
            tmp += value;
        if (tmp != 1)
            throw std::invalid_argument(std::string("Sum of subsets ratio is not equal 1"));
    }

    this->subsets_ratio = subsets_ratio;

    this->train_set = std::vector<uint32_t>(static_cast<uint32_t>(ceil(this->subsets_ratio.front() * input_data.size())));
    this->test_set = std::vector<uint32_t>(static_cast<uint32_t>(floor(this->subsets_ratio.back() * input_data.size())));

    if (!((train_set.size() + test_set.size()) == input_data.size()))
        throw std::invalid_argument(std::string("I do math wrong ;("));

    for (uint32_t i{0}; i < input_data.size(); i++)
        this->indexes.push_back(i);

    //Mr.Miłosz told me that will be fine
    this->error_ratio = 1.05;
}

Net::~Net() {}
