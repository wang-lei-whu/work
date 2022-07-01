#include "pso.h"
#include <ctime>

using namespace std;
using namespace Eigen;

int main()
{
     // srand((unsigned)time(NULL)); //时间做随机种子
     // MatrixXd m = (MatrixXd::Random(2, 1)).array().abs() * 2;

     //
     int dim = 2;
     MatrixXd lb = MatrixXd::Constant(1, dim, -2);
     // lb(1, 0) = -4;
     MatrixXd ub = MatrixXd::Constant(1, dim, 2);
     // ub(1, 0) = 4;     
     
     MatrixXd m = 1.5 * toolFunc::random(5, dim,lb, ub);
     // MatrixXd m2 = m.cwiseMin(ub.replicate(5,1)).cwiseMax(lb.replicate(5,1));
     // cout << m <<"\n"<< endl;
     // cout << m2 << endl;
     // MatrixXd m1 = MatrixXd::Zero(5,dim);
     //cout << 1*m.cwiseProduct(m1) << endl; //点乘
     // MatrixXd m = MatrixXd::Zero(10, dim);
     // m = toolFunc::random(10, dim,lb, ub);
     // cout << m << endl;

     // m = i;
     // cout << m << endl;
     //     cout << m << endl;
     // for (int i = 0; i < lb.cols() ; i++)
     // {
     //     normal_distribution<double> n(lb(i), ub(i));
     // 	m.row(i).unaryExpr([n,e](double dummy){return n(e);});
     //     cout<< n(e)<<endl;
     // }
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

     PSO pso(toolFunc::rosenBrock, dim, 5000, 3000, lb, ub, 0.8, 0.1, 0.6, 0.5, true);
     pso.run();
     cout << "pso.X:\n"
          << pso.X << endl;
     cout << "pso.Y:\n"
          << pso.Y << endl;
     cout << "pso.V:\n"
          << pso.V << endl;
     cout << "pso.gbest_y:\n"
          << pso.gbest_y << endl;
     cout << "pso.gbest_x:\n"
          << pso.gbest_x << endl;
     return 0;
}
