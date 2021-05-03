#include <lib/net.h>

void read_data(data_set &input, data_set &target)
{
    /**
    0) Class
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
    std::ifstream input_file("INPUT_DATA.txt");
    for (std::string line; getline(input_file, line);)
    {
        data_row input_data = std::vector<double>(13, 0);
        data_row output_data = std::vector<double>(1, 0);
        sscanf(line.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
               &output_data[0], &input_data[0], &input_data[1], &input_data[2], &input_data[3], &input_data[4],
               &input_data[5], &input_data[6], &input_data[7], &input_data[8], &input_data[9], &input_data[10],
               &input_data[11], &input_data[12]);
        input.push_back(input_data);
        target.push_back(output_data);
    }
}

int main()
{
    data_set input, target;
    read_data(input, target);

    auto myNet = Net(input, target);
    myNet.add_layer(3);
    myNet.add_layer(2);
    myNet.add_layer(3);

    // for(auto &layer: myNet.layers){
    //     for(auto &neuron: layer.neurons){
    //         neuron.bias = 0.0;
    //         for(auto &weight: neuron.weights){
    //             weight = 0.4;
    //         }
    //     }
    // }

    //print debug info
    // std::cout << std::endl;
    // std::cout <<
    //     myNet.cost[0] << std::endl <<
    //     myNet.cost[1] << std::endl <<
    //     myNet.cost[2] << std::endl <<

    //     myNet.neurons_inputs[2][0] << std::endl <<
    //     myNet.neurons_inputs[2][1] << std::endl <<
    //     myNet.neurons_inputs[2][2] << std::endl <<

    //     myNet.neurons_inputs[1][0] << std::endl <<
    //     myNet.neurons_inputs[1][1] << std::endl <<

    //     myNet.neurons_inputs[0][2] << std::endl <<

    //     myNet.layers[2].neurons[0].weights[0] << std::endl <<
    //     myNet.layers[2].neurons[1].weights[0] << std::endl <<
    //     myNet.layers[2].neurons[2].weights[0] << std::endl <<
    //     myNet.layers[2].neurons[0].weights[1] << std::endl <<
    //     myNet.layers[2].neurons[1].weights[1] << std::endl <<
    //     myNet.layers[2].neurons[2].weights[1] << std::endl <<

    //     myNet.layers[1].neurons[0].weights[2] << std::endl <<
    //     myNet.layers[1].neurons[1].weights[2] << std::endl;

    // std::cout << std::endl;

    const uint32_t runs = 1000000;
    std::array<uint32_t, runs> run_time = std::array<uint32_t, runs>();

    for (uint32_t i{0}; i < runs; i++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();
        myNet.get_cost(i % input.size());
        for (uint32_t layers{0}; layers < myNet.layers_count; layers++)
        {
            for (uint32_t neurons{0}; neurons < myNet.layers[layers].neuron_count; neurons++)
            {
                myNet.get_delta(layers, neurons);
            }
        }
        auto end_time = std::chrono::high_resolution_clock::now();
        auto time = end_time - start_time;
        run_time[i] = time / std::chrono::microseconds(1);
    }

    double mean = 0;
    for (auto &value : run_time)
        mean += value;

    mean /= runs;

    std::cout << mean << " microseconds for get_delta for each neuron " << runs << "times" << std::endl;

    return 0;
}