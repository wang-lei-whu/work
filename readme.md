本仓库用于学习和开发PSO算法的C++程序。

## 依赖项（带\*的必须安装）
- \* Eigen3：用于矩阵操作
    在debian/ubuntu等系统下可以使用apt安装，例如：
    ```
        sudo apt-get install libeigen3-dev
    ```
- cmake：用于编译
    ```
        sudo apt-get install cmake
    ```
## 编译可执行文件
本项目可以使用cmake(version＞2.8.12)编译，同时附带vscode设置文件用于gdb断点调试。


- cmake：示例如下
    ```bash
        # 项目文件夹下运行
        mkdir -p build
        cd build
        cmake ..
        make
        # 得到的执行文件main在bin文件夹下
    ```

- vscode:附带.vscode文件夹示范，用于vscode自动调试。
    - 配置好vscode-Ｃ++调试环境;
    - Ctrl+Shift+B 生成可执行文件;
    - F5启动断点调试。

- g++:示例如下
    ```bash
        g++ -g -I /usr/include/eigen3/ -I ./include src/*.cpp -o ./bin/main -std=c++11

        # 其中/usr/local/include/eigen3/为eigen3安装路径
        # ./bin/main 为可执行文件
    ```
## 修改记录
- 2022.06.13 配置环境并编写pso类定义;
- 2022.06.14 完成pso类初始化;
- 2022.6.30 pso串行版本完成
- 2022.7.2 应用cmake管理框架
- 2022.7.3 将lib和主程序分离
- 2022.7.8 cuda(cublas)测试
