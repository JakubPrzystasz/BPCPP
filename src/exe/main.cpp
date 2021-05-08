#include <lib/net.h>
#include <sciplot/sciplot.hpp>

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

    std::vector<uint32_t> layers{3};

    auto myNet = Net(input, target, layers);

    data_row x, target_plot, output_plot;
    output_plot = data_row(input.size());
    target_plot = data_row(input.size());

    std::vector<uint32_t> index(input.size(),0);
    for(uint32_t i{0};i<input.size();i++)
        index[i] = i;

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    //shuffle (index.begin(), index.end(), std::default_random_engine(seed));

    uint32_t it{0};

    auto &out = myNet.layers.back().neurons;

    while (true)
    {
        myNet.SSE = 0;
        for (uint32_t i{0}; i < input.size(); i++)
            myNet.train(index[i]);

        myNet.SSE = myNet.SSE / static_cast<double>(input.size());
        if(it % 10000 == 0){
            std::cout << it << "  SSE: " << myNet.SSE << std::endl;        
        }
       
        if(myNet.SSE < 0.05)
            break;
        if (it == 1000000)
            break;
        it++;
    }

    for (uint32_t i{0}; i < input.size(); i++)
    {
        myNet.train(i);
        output_plot[i] = out[0].output;
    }


    sciplot::Plot plot0;
    for (uint32_t i{0}; i < input.size(); i++)
        x.push_back(i);

    plot0.drawDots(x, output_plot);

    plot0.legend().hide();
    plot0.save("test1.png");

    return 0;
}