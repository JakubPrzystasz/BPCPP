#include <lib/net.h>
#include <sciplot/sciplot.hpp>
#include <chrono>

int main()
{
    pattern_set input;
    Net::read_file(std::string("INPUT_DATA.txt"), input);

    auto myNet = Net(input, {1, 0});

    LearnParams netParams = LearnParams();
    netParams.batch_size = 1;

    myNet.setup({3}, netParams);
    auto start = std::chrono::high_resolution_clock::now();
    auto test = myNet.train(1000);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
  
    // To get the value of duration use the count()
    // member function on the duration object
    std::cout << duration.count() << std::endl;

    uint32_t i=0;
    double best = test.train_set_accuracy.front();
    uint32_t best_id = 0;
    for(auto &result: test.train_set_accuracy){
        if(result > best){
            best = result;
            best_id = i;
        }
        i++;
    }

    std::cout << "best result: " << best << "   at: " << best_id << std::endl;

    //TODO:TEST SET


    return 0;
}