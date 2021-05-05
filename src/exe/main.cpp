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
    myNet.add_layer(13);
    myNet.add_layer(6);
    myNet.add_layer(1);

    for(auto &layer:myNet.layers){
        for(auto &neuron: layer.neurons){
            neuron.bias = 0.5;
            for(auto &weight:neuron.weights)
                weight = 0.5;
        }
    }

    myNet.set_batch_size(input.size());

    std::vector<uint32_t> input_vec = std::vector<uint32_t>(input.size());
    for(uint32_t i{0};i<input.size();i++)
        input_vec[i] = i;
        
    data_row costs(input.size(),0);

    for(uint32_t i{0};i<1000;i++){
        myNet.train(input_vec,costs);
        myNet.fit();
        double SSE = 0;
        for(auto &value: costs)
            SSE += value;
        std::cout << i << std::endl;
        std::cout << (SSE/input.size()) << std::endl;
    }

    return 0;
}