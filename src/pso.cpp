#include "pso.h"
#include "tools.h"
#include "float.h"
#include "cassert"
#include <ctime>
#include <chrono>
#define inf DBL_MAX

using namespace std;
using namespace chrono;
typedef std::chrono::high_resolution_clock Clock;
// PSO类的构造函数
PSO::PSO(MatrixXd (*func)(MatrixXd X), int n_dim, int pop, int max_iter, MatrixXd lb, MatrixXd ub, float w0, float w_min, float c1, float c2, bool verbose)
{
    this->func = func;
    this->n_dim = n_dim; // this指针解决参数和变量同名问题
    this->pop = pop;
    this->max_iter = max_iter;
    this->lb = lb;
    this->ub = ub;
    this->w0 = w0;
    this->w = w0;
    this->w_min = w_min;
    cp = c1;
    cg = c2;
    this->verbose = verbose;
    X = toolFunc::random(pop, n_dim, lb, ub); //初始化X
    MatrixXd v_highest = ub - lb;
    // assert(v_highest.unaryExpr);  //应当检查输入正确性；
    V = MatrixXd::Zero(pop, n_dim);
    V = toolFunc::random(pop, n_dim, -v_highest, v_highest);

    pbest_x = X;
    Y = cal_Y();
    pbest_y = MatrixXd::Constant(pop, 1, inf);
    gbest_x = pbest_x.colwise().mean();
    gbest_y = inf;
}

MatrixXd PSO::cal_Y()
{
    Y = func(X);
    return Y;
}

void PSO::update_w(int iter)
{
    w = w0 - (w0 - w_min) * iter / (max_iter - 1);
}
void PSO::update_V()
{
    // (0,1)之间的随机系数
    MatrixXd r1 = MatrixXd::Random(pop, n_dim).array().abs();
    MatrixXd r2 = MatrixXd::Random(pop, n_dim).array().abs();
    //速度转移公式
    V = w * V + cp * (r1.cwiseProduct(pbest_x - X)) + cg * (r2.cwiseProduct(gbest_x.replicate(pop, 1) - X));
    // V = V.cwiseMin(MatrixXd::Constant(pop, n_dim, 2)).cwiseMax(MatrixXd::Constant(pop, n_dim, -2));
}
void PSO::update_X()
{
    X = X + V;
    X = X.cwiseMin(ub.replicate(pop, 1)).cwiseMax(lb.replicate(pop, 1));
}

void PSO::update_pbest()
{

    ArrayXXd Need_update = (pbest_y.array() > Y.array()).cast<double>();
    //未考虑约束问题
    for (int i = 0; i < n_dim; i++)
    {
        pbest_x.col(i) = (pbest_y.array() > Y.array()).select(X.col(i), pbest_x.col(i));
    };

    pbest_y = (pbest_y.array() > Y.array()).select(Y, pbest_y);
}
void PSO::update_gbest()
{
    MatrixXd::Index minRow, minCol;
    double present_best_y = pbest_y.minCoeff(&minRow, &minCol);
    if (gbest_y > present_best_y)
    {
        gbest_x = X.row(minRow);
        gbest_y = present_best_y;
    };
}

// pso优化执行函数 verbose默认为true
int PSO::run()
{   
    cout << "PSO process begin!\n"
         << endl;
    for (int iter_num = 0; iter_num < max_iter; iter_num++)
    {        
        // auto t1 = Clock::now(); //计时
        update_V();
        // auto t2 = Clock::now(); 
        update_w(iter_num);
        // auto t3 = Clock::now();
        update_X();
        // auto t4 = Clock::now();
        cal_Y();
        // auto t5 = Clock::now();
        update_pbest();
        // auto t6 = Clock::now();
        update_gbest();
        // auto t7 = Clock::now();
        if (verbose)
        {
            cout << "Step " << iter_num << " : Best(minimum) value : " << gbest_y << " at " << gbest_x << " \n";
        
        // std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()/1e+6 <<" ms in update_V();"<< '\n';
        // std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count()/1e+6 <<" ms in update_w();"<< '\n';
        // std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count()/1e+6 <<" ms in update_X();"<< '\n';
        // std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count()/1e+6 <<" ms in cal_Y();"<< '\n';
        // std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count()/1e+6 <<" ms in update_pbest();"<< '\n';
        // std::cout <<"it cost: "<< std::chrono::duration_cast<std::chrono::nanoseconds>(t7 - t6).count()/1e+6 <<" ms in update_gbest();"<< '\n';
        }
    }

    cout << "PSO process end!\n"
         << endl;
    return 0;
}
