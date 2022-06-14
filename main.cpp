#include "pso.h"
// #include <ctime>
using namespace std;
using namespace Eigen;

int main()
{
    // static default_random_engine e(time(0));
    // static normal_distribution<double> n(0, 10);
    // srand((unsigned)time(NULL)); //时间做随机种子
    // MatrixXd m = (MatrixXd::Random(2, 1)).array().abs() * 2;
    //   MatrixXd m=MatrixXd::Zero(2,2).unaryExpr([](double dummy){return n(e);});
    MatrixXd lb = MatrixXd::Constant(2,1,0);
    MatrixXd ub = MatrixXd::Constant(2,1,2);
    // cout << "Gaussian random matrix row mean:\n"
        //  << m.rowwise().mean() << endl;
    // cout << "Mean: " << lb.mean() << endl;
    // MatrixXd m2 = (m.array() - m.mean()) * (m.array() - m.mean());
    // cout << "std: " << sqrt(m2.sum() / (m2.size() - 1)) << endl;

    // MatrixXd (*func)(MatrixXd X);
    // func = testFunc::rosenBrock;
    // MatrixXd test = func(m);
    // cout << testFunc::rosenBrock(m) << endl;
    // cout << test << endl;

    PSO pso(testFunc::rosenBrock, 2, 5, 1, lb, ub, 0.8, 0.1, 0.6, 0.5, true);
    cout << "pso.X:\n"<<pso.X << endl;
    cout << "pso.Y:\n"<<pso.Y << endl;
    cout << "pso.V:\n"<<pso.V << endl;
    cout << "pso.gbest_y"<< pso.gbest_y<<endl;
    cout << "pso.gbest_x"<< pso.gbest_x<<endl;
    return 0;
}
