#include <lib/net.h>
#include <sciplot/sciplot.hpp>
#include <chrono>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    auto myNet = Net(input, {0.8,0.2});

    LearnParams netParams = LearnParams();

    myNet.setup({13, 3}, netParams);
    auto start = std::chrono::high_resolution_clock::now();
    auto test = myNet.train(1000);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);

    // To get the value of duration use the count()
    // member function on the duration object
    //std::cout << duration.count() << std::endl;

    uint32_t i = 0;
    double best = test.train_set_SSE.front();
    uint32_t best_id = 0;
    for (auto &result : test.train_set_SSE)
    {
        if (result < best)
        {
            best = result;
            best_id = i;
        }
        i++;
    }
    std::cout << "best result: " << best << "   at: " << best_id << std::endl;
    i =0;
    for (auto &epoch : test.train_set_SSE){
        std::cout << i << "  " << epoch << std::endl;
    i++;
        }

    return 0;
}