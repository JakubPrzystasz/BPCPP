#include <lib/net.h>
#include <sciplot/sciplot.hpp>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    //Define hidden layers
    std::vector<uint32_t> layers{6};

    auto myNet = Net(input);
    myNet.setup(layers, 1);

    data_row x, target_plot, output_plot;
    output_plot = data_row(input.size());
    target_plot = data_row(input.size());

    std::vector<uint32_t> index(input.size(), 0);
    for (uint32_t i{0}; i < input.size(); i++)
        index[i] = i;

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    //shuffle(index.begin(), index.end(), std::default_random_engine(seed));

    uint32_t it{0};

    auto begin = std::chrono::high_resolution_clock::now();
    while (true)
    {
        myNet.batch_it = 0;
        for (uint32_t i{0}; i < input.size(); i++, myNet.batch_it++){
            myNet.train(index[i]);
            std::cout << myNet.SSE << std::endl;
        }

//        myNet.SSE = myNet.SSE / static_cast<double>(input.size());

        if(myNet.SSE < 0.1)
            break;

        if (it == 1000)
            break;
        it++;
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "ms" << std::endl;

    auto &out = myNet.layers.back().neurons;

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
    plot0.show();

    return 0;
}