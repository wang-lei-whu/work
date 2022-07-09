#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>

using namespace std;
#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))

#define IDX2C(i,j,ld) (((j)*(ld))+(i))
//ld是维度，i是行，j是列，cuBLAS使用的是列存储
//这个宏表示， 矩阵第i行第j列的元素位置在C语言中 数组存储的索引
//ld表示 矩阵的第一个维的元素个数，就是 矩阵的行数。
__global__
void show(float* ptr, int size)
{
    for (int i = 0; i < size; i++)
        printf("%f\n", ptr);
}


void print_matrix(int R, int C, float* A, const char* name)
{
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r)
    {
        printf("[");
        for (int c = 0; c < C; ++c)
        {
            printf("%10.6f", A[c * R + r]);
        }
        printf("] \n");
    }
}


void print_matrix_(int R, int C, float* A, const char* name)
{
    printf("%s = \n", name);
    for (int r = 0; r < R; ++r)
    {
        printf("[");
        for (int c = 0; c < C; ++c)
        {
            printf("%10.6f", A[r * C + c]);
        }
        printf("]\n");
    }
}

int main()
{
    int M = 6; //行数 矩阵A的行，结果矩阵C的行数.A=[3,9]
    int N = 4; //列数 矩阵A的列，矩阵B的列   B=[3,9]
    int B = 4; //行数 矩阵B的行
    int K = 3; //列数，结果矩阵C的列数  C=[3,3]

    //分配主机矩阵并初始化
    float* a, * b, * c;
    cudaHostAlloc((void**)&a, sizeof(float) * M * N, cudaHostAllocDefault);
    cudaHostAlloc((void**)&b, sizeof(float) * B * N, cudaHostAllocDefault);
    cudaHostAlloc((void**)&c, sizeof(float) * M * K, cudaHostAllocDefault);

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < M; i++) {
            a[IDX2C(i, j, M)] =  1.0;
        }
    }

    /*
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i * N + j] = 1;
        }
    }

   
    print_matrix_(M, N, a, "A");

   
    */

    //可视化矩阵
    print_matrix(M, N, a, "A");

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < B; i++) {
            b[IDX2C(i, j, B)] = (float)(i * N + j + 1);
        }
    }

    /*
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < N; ++j) {
            b[i * N + j] = i * N + j + 1;
        }
    }
    */
    //print_matrix(B, N, b, "B");

    print_matrix(B, N, b, "B");
    //分配设备的数据
    float* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, sizeof(float) * M * N);
    cudaMalloc(&d_b, sizeof(float) * B * N);
    cudaMalloc(&d_c, sizeof(float) * M * K);

    //Host->device
    cudaMemcpy(d_a, a, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * B * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, sizeof(float) * M * K, cudaMemcpyHostToDevice);
    //
    // print_matrix(B, N, d_b, "d_B");
    cublasHandle_t handle;
    cublasStatus_t ret;
    ret = cublasCreate(&handle);

    //矩阵分块计算
    float* a_array[9], * b_array[9];
    float* c_array[9];

    int r = 3;
    int l = 3;

    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < l; ++j) {
            a_array[i * l + j] = d_a + i * 9 + j * r;
            b_array[i * l + j] = d_b + i * 9 + j * r;
            c_array[i * l + j] = d_c + i * 9 + j * r;
            //printf("%d\n",*a_array[i * l + j]);
        }

    }
    //print_matrix_(r, l, *(a_array), "a_array");
    const float** d_Marray, ** d_Narray;
    float** d_Parray;
    cudaMalloc((void**)&d_Marray, N * sizeof(float*));
    cudaMalloc((void**)&d_Narray, N * sizeof(float*));
    cudaMalloc((void**)&d_Parray, N * sizeof(float*));
    cudaMemcpy(d_Marray, a_array, N * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Narray, b_array, N * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Parray, c_array, N * sizeof(float*), cudaMemcpyHostToDevice);


    const float alpha = 1.0f;
    const float beta = 0.0f;

    //需要的是A矩阵的一个1行3列的矩阵乘以矩阵b的三行一列，
    int m =3; //按列 m = 1
    int n = 1; //按列 n = 3
    int k = 1; //按列 k = 1

    int lda = 9;
    int ldb = 9;
    int ldc = 9;
    int batch = 9;
    //    矩阵OP(Ａ)的维度是ｍ×ｋ
    //    矩阵OP(B)的维度是ｋ×ｎ
    //    矩阵C的维度是ｍ×ｎ
    //    运算为C = alpha * A * B + beta * C
    ret = cublasSgemmBatched(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        m, n, k,
        &alpha,
        d_Narray, ldb,
        d_Marray, lda,
        &beta,
        d_Parray, ldc,
        batch);

    cublasDestroy(handle);
   
    if (ret == CUBLAS_STATUS_SUCCESS)
    {
        printf("sgemm success  %d, line(%d)\n", ret, __LINE__);
    }
   
    //show << <1, 1 >> > (c_array[0], 16);
    cudaMemcpy(c, d_c, sizeof(float) * M * K, cudaMemcpyDeviceToHost);

    print_matrix(M, K, c, "C = A x B");
    return 0;

}
