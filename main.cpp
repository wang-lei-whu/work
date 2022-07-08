#include "pso.h"
#include "tools.h"
#include <chrono>
#include "srccuda/testgpu.cuh"
using namespace std;
using namespace chrono;

// extern "C" void gputest(); 

int main()
{
     typedef std::chrono::high_resolution_clock Clock;
     auto t1 = Clock::now(); //计时开始

     //维度和定义域
     const int dim = 2;
     const auto lb = MatrixXd::Constant(1, dim, -2);
     const auto ub = MatrixXd::Constant(1, dim, 2);

     for (int i = 0; i < 1; i++)
     {
          // PSO::PSO(MatrixXd (*func)(MatrixXd X), int n_dim, int pop, int max_iter, MatrixXd lb, MatrixXd ub, float w0, float w_min, float c1=0.6, float c2=0.5, bool verbose)
          PSO pso(toolFunc::rosenBrock, dim, 50, 10, lb, ub, 0.8, 0.3, 0.6, 0.5, false);
          pso.run();
          cout << "Best value with PSO:\n"
               << pso.gbest_y
               << "\nX for best value:\n"
               << pso.gbest_x << endl;
     };
     auto t2 = Clock::now(); //计时结束
     std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()/1e+6 <<" ms"<< '\n';
     
     gputest();
     return 0;
}
