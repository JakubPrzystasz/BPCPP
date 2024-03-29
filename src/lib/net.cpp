#include "net.h"

std::vector<std::pair<double, std::vector<uint32_t>>>::iterator Net::get_class_vector(double class_value)
{
    std::vector<std::pair<double, std::vector<uint32_t>>>::iterator it;

    for (it = this->class_sets.begin(); it < this->class_sets.end(); it++)
    {
        if (it->first == class_value)
            return it;
    }

    return this->class_sets.end();
}

void Net::update_learning_rate(LearningRateUpdate value)
{
    switch (value)
    {
    case LearningRateUpdate::Increase:
        this->learn_parameters.learning_rate *= this->learn_parameters.learning_accelerating_constans;
        break;
    default:
        this->learn_parameters.learning_rate *= this->learn_parameters.learning_decelerating_constans;
        break;
    };

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

void Net::save_output(std::string filename, LearnOutput &output, SaveMode mode)
{
    std::fstream outfile;
    nlohmann::json json_output = output;

    switch (mode)
    {
    case SaveMode::Append:
        outfile.open(filename, std::ios_base::app);
        outfile << json_output << ',' << std::endl
                << std::endl;
        break;
    default:
        outfile.open(filename, std::ios_base::out);
        outfile << json_output << std::endl;
        break;
    };
}

void Net::open_file(std::string filename)
{
    std::fstream outfile;
    outfile.open(filename, std::ios_base::out);
    outfile << '[' << std::endl;
}

void Net::close_file(std::string filename)
{
    std::fstream outfile;
    outfile.open(filename, std::ios_base::app);
    outfile << "{ }" << std::endl
            << ']' << std::endl;
}

LearnOutput Net::train(double max_epoch, double error_goal)
{
    LearnOutput out;

    out.input_params = this->learn_parameters;
    out.input_layers = std::vector<Layer>(this->layers.begin() + 1, this->layers.end());
    out.train_set_ratio = static_cast<double>(this->train_set.size()) / static_cast<double>(this->input_data.size());
    out.test_set_ratio = static_cast<double>(this->test_set.size()) / static_cast<double>(this->input_data.size());

    //shuffle values
    {
        std::mt19937_64 rng;
        uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed >> 32)};
        rng.seed(ss);
        std::mt19937 g(rng());

        for (auto &class_set : this->class_sets)
            std::shuffle(VEC_RANGE(class_set.second), g);

        uint32_t i{0};

        for (auto &class_set : this->class_sets)
        {
            for (uint32_t x{0}; x < static_cast<uint32_t>(std::ceil(class_set.second.size() * this->subsets_ratio.front())); x++)
            {
                this->train_set[i] = class_set.second[x];
                i++;
            }
        }

        i = 0;

        for (auto &class_set : this->class_sets)
        {
            for (uint32_t x{0}; x < static_cast<uint32_t>(std::floor(class_set.second.size() * this->subsets_ratio.back())); x++)
            {
                this->test_set[i] = class_set.second[x];
                i++;
            }
        }
    }

    std::vector<bool> classification_test = std::vector<bool>(test_set.size(), 0);
    std::vector<bool> classification_train = std::vector<bool>(train_set.size(), 0);

    double classification_accuracy, epoch_SSE;
    uint32_t index{0};
    this->epoch = 0;
    this->SSE = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (; this->epoch < max_epoch; this->epoch++)
    {
        this->batch_it = 1;
        epoch_SSE = 0;

        //std::cout << epoch << std::endl;
        
        //Make full run over training data
        for (uint32_t it{0}; it < train_set.size(); it++)
        {
            index = train_set[it];
            this->learn(index);
            classification_train[it] = this->get_classification_succes(index);

            //if batch ends - update weights and biases
            if ((this->batch_it % this->learn_parameters.batch_size) == 0)
            {
                this->SSE *= 0.5;
                this->prev_SSE = this->SSE;
                epoch_SSE += this->SSE;
                this->update_weights();
                this->batch_it = 1;
                this->SSE = 0.0;
            }
            else
                batch_it++;
        }

        //Stuff for later analyse
        out.train_set_SSE.push_back(epoch_SSE);
        out.train_set_MSE.push_back(epoch_SSE / static_cast<double>(train_set.size()));

        classification_accuracy = 0;
        for (auto value : classification_train)
            classification_accuracy += static_cast<double>(value);
        out.train_set_accuracy.push_back(classification_accuracy / static_cast<double>(train_set.size()));

        epoch_SSE = 0;
        for (uint32_t it{0}; it < test_set.size(); it++)
        {
            index = test_set[it];
            this->feed(index);
            epoch_SSE += this->get_loose(index);
            classification_test[it] = this->get_classification_succes(index);
        }
        out.test_set_SSE.push_back(epoch_SSE);
        out.test_set_MSE.push_back(epoch_SSE / static_cast<double>(test_set.size()));

        classification_accuracy = 0;
        for (auto value : classification_test)
            classification_accuracy += static_cast<double>(value);
        out.test_set_accuracy.push_back(classification_accuracy / static_cast<double>(test_set.size()));

        if (out.train_set_SSE.back() < error_goal)
            break;
    }

    auto stop_time = std::chrono::high_resolution_clock::now();
    out.time = std::chrono::duration_cast<std::chrono::seconds>(stop_time - start_time).count();
    out.epoch_count = epoch;
    out.output_params = this->learn_parameters;
    out.output_layers = std::vector<Layer>(this->layers.begin() + 1, this->layers.end());
    out.error_goal = error_goal;

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
        layer.neurons[neuron_it].output = data[neuron_it];

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
        neuron.delta = (this->input_data[sample_number].output[neuron_it] - neuron.output) * neuron.derivative_output * -1.0;
    }

    //Hidden layers
    for (uint32_t layer_it{static_cast<uint32_t>(this->layers.size()) - 2}; layer_it > 0; layer_it--)
    {
        auto &layer = this->layers[layer_it];
        for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
        {
            auto &neuron = layer.neurons[neuron_it];

            neuron.delta = 0;
            for (uint32_t next{0}; next < this->layers[layer_it + 1].neurons.size(); next++)
            {
                auto &next_neuron = this->layers[layer_it + 1].neurons[next];
                neuron.delta += next_neuron.weights[neuron_it] * next_neuron.delta;
            }

            //calculate the delta for current neuron
            neuron.delta *= neuron.derivative_output;
        }
    }
}

double Net::get_loose(uint32_t sample_number)
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
    //Adaptive learning rate:
    if (this->epoch && this->learn_parameters.learning_accelerating_constans > 0 && this->learn_parameters.learning_decelerating_constans > 0)
    {
        double error_r = this->SSE / this->prev_SSE;
        if (error_r > this->learn_parameters.error_ratio)
            this->update_learning_rate(LearningRateUpdate::Decrease);
        else if (error_r < 1)
            this->update_learning_rate(LearningRateUpdate::Increase);
    }

    //Accumulate weight and bias updates form each run
    for (uint32_t i{1}; i < this->layers.size(); i++)
    {
        for (auto &neuron : this->layers[i].neurons)
        {
            neuron.bias_update = std::accumulate(VEC_RANGE(neuron.batch.bias_deltas), 0.0) * neuron.learn_parameters.learning_rate;

            for (uint32_t it{0}; it < neuron.weights.size(); it++)
                neuron.weight_update[it] = std::accumulate(VEC_RANGE(neuron.batch.weights_deltas[it]), 0.0) * neuron.learn_parameters.learning_rate;
        }
    }

    //Momentum:
    if (learn_parameters.momentum_delta_vsize && learn_parameters.momentum_constans > 0)
    {
        uint32_t index = (this->epoch + 1) % this->learn_parameters.momentum_delta_vsize;
        for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
        {
            for (auto &neuron : this->layers[layer_it].neurons)
            {
                neuron.bias_update = (neuron.bias_update * (1.0 - neuron.learn_parameters.momentum_constans)) + (std::accumulate(VEC_RANGE(neuron.bias_deltas), 0.0) * neuron.learn_parameters.momentum_constans);
                neuron.bias_deltas[index] = neuron.bias_update;

                for (uint32_t it{0}; it < neuron.weights.size(); it++)
                {
                    neuron.weight_update[it] = (neuron.weight_update[it] * (1.0 - neuron.learn_parameters.momentum_constans)) + (std::accumulate(VEC_RANGE(neuron.weights_deltas[it]), 0.0) * neuron.learn_parameters.momentum_constans);
                    neuron.weights_deltas[it][index] = neuron.weight_update[it];
                }
            }
        }
    }

    //Finally update weights and biases
    for (uint32_t i{1}; i < this->layers.size(); i++)
    {
        for (auto &neuron : this->layers[i].neurons)
        {
            neuron.bias -= neuron.bias_update;

            for (uint32_t it{0}; it < neuron.weights.size(); it++)
                neuron.weights[it] -= neuron.weight_update[it];
        }
    }
}

void Net::learn(uint32_t sample_number)
{
    uint32_t batch_it = this->batch_it - 1;
    this->feed(sample_number);
    this->get_delta(sample_number);
    this->SSE += this->get_loose(sample_number);

    ///Calculate gradient
    for (uint32_t layer_it{1}; layer_it < this->layers.size(); layer_it++)
    {
        auto &layer = this->layers[layer_it];
        for (uint32_t neuron_it{0}; neuron_it < layer.neurons.size(); neuron_it++)
        {
            auto &neuron = layer.neurons[neuron_it];
            neuron.batch.bias_deltas[batch_it] = neuron.delta;

            for (uint32_t weights_it{0}; weights_it < neuron.weights.size(); weights_it++)
                neuron.batch.weights_deltas[weights_it][batch_it] = neuron.delta * this->layers[layer_it - 1].neurons[weights_it].output;
        }
    }
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

void Net::setup(std::vector<uint32_t> hidden_layers, LearnParams params, std::array<double, 2> subsets_ratio)
{
    this->learn_parameters = params;

    //Verify subsets_ratio
    if (static_cast<uint32_t>(std::accumulate(VEC_RANGE(subsets_ratio), 0.0)) != 1)
        throw std::invalid_argument(std::string("Sum of subsets ratio is not equal 1"));

    this->subsets_ratio = subsets_ratio;

    this->train_set = std::vector<uint32_t>(static_cast<uint32_t>(ceil(this->subsets_ratio.front() * input_data.size())));
    this->test_set = std::vector<uint32_t>(static_cast<uint32_t>(floor(this->subsets_ratio.back() * input_data.size())));

    if (!((train_set.size() + test_set.size()) == input_data.size()))
        throw std::invalid_argument(std::string("I do math wrong ;("));

    //If params.batch_size is 0, then set batch_size equal to size of whole training set
    if (this->learn_parameters.batch_size == 0)
        this->learn_parameters.batch_size = this->train_set.size();

    if ((this->train_set.size() % this->learn_parameters.batch_size) != 0)
        throw std::invalid_argument("Batch size is not divisible by size of train set");

    //Setup layers
    this->layers = std::vector<Layer>();

    //input layer
    this->layers.push_back(Layer(this->input_size, this->input_size, this->learn_parameters));

    //Hidden layers
    for (uint32_t i{0}; i < hidden_layers.size(); i++)
        this->layers.push_back(Layer(hidden_layers[i], (i > 0 ? hidden_layers[i - 1] : input_size), this->learn_parameters));

    //output layer
    this->layers.push_back(Layer(this->output_size, this->layers.back().neurons.size(), this->learn_parameters));
    for (auto &neuron : this->layers.back().neurons)
        neuron.bias = 0;
}

Net::Net(pattern_set &input_data)
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

    //assign class sets
    for (uint32_t index{0}; index < this->input_data.size(); index++)
    {
        //TODO: this is not what i want          \/
        auto &value = this->input_data[index].output[0];
        auto it = get_class_vector(value);
        if (it == this->class_sets.end())
            this->class_sets.push_back(std::make_pair(value, std::vector<uint32_t>(1, index)));
        else
            it->second.push_back(index);
    }

    for (uint32_t i{0}; i < input_data.size(); i++)
        this->indexes.push_back(i);
}

Net::~Net() {}

void to_json(json &j, const LearnOutput &lo)
{
    j = json{
        {"train_set_SSE", lo.train_set_SSE},
        {"train_set_MSE", lo.train_set_MSE},
        {"train_set_accuracy", lo.train_set_accuracy},
        {"test_set_SSE", lo.test_set_SSE},
        {"test_set_MSE", lo.test_set_MSE},
        {"test_set_accuracy", lo.test_set_accuracy},
        {"result", lo.result == TrainResult::ErrorGoalReached ? "goal" : "max_epoch"},
        {"epoch_count", lo.epoch_count},
        {"time", lo.time},
        {"input_params", lo.input_params},
        {"output_params", lo.output_params},
        {"input_layers", lo.input_layers},
        {"output_layers", lo.output_layers},
        {"train_set_ratio", lo.train_set_ratio},
        {"test_set_ratio", lo.test_set_ratio},
    };
}

void to_json(json &j, const LearnParams &lp)
{
    j = json{
        {"batch_size", lp.batch_size},
        {"learning_rate", lp.learning_rate},
        {"learning_accelerating_constans", lp.learning_accelerating_constans},
        {"learning_decelerating_constans", lp.learning_decelerating_constans},
        {"error_ratio", lp.error_ratio},
        {"momentum_constans", lp.momentum_constans},
        {"momentum_delta_vsize", lp.momentum_delta_vsize},
        {"beta_param", lp.beta_param},
        {"weights_range", lp.weights_range},
        {"bias_range", lp.bias_range},
        {"init_function", lp.init_function == InitFunction::rand ? "random" : lp.init_function == InitFunction::nw ? "nw"
                                                                                                                   : "const_rand"},
        {"activation", lp.activation == ActivationFunction::bipolar ? "bipolar" : lp.activation == ActivationFunction::unipolar ? "unipolar"
                                                                                                                                : "linear"},
    };
}

void to_json(json &j, const Neuron &n)
{
    j = json{
        {"learn_parameters", n.learn_parameters},
        {"bias", n.bias},
        {"weights", n.weights},
    };
}

void to_json(json &j, const Layer &l)
{
    j = json{
        {"learn_parameters", l.learn_parameters},
        {"neurons", l.neurons},
    };
}