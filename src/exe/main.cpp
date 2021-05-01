#include <lib/net.h>
#include <fstream>

int main()
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

    data_set input;
    data_set output;

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
        output.push_back(output_data);
    }

    auto myNet = Net(input,output);
    myNet.add_layer(13);
    myNet.add_layer(6);
    myNet.add_layer(1);

    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << myNet.feed(1) << std::endl;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time = end_time - start_time;
    std::cout << time / std::chrono::microseconds(1) << " microseconds to run." << std::endl;

    return 0;
}