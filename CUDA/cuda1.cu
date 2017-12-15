/*
 * Almost the minimal CUDA C++ example.
 *
 * Compile with `nvcc -o cuda1 cuda1.cu`
 */

#include <iostream>

using namespace std;

/*
 * Square all elements of array
 */
static __global__
void square(float *x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    x[i] = x[i]*x[i];
}

int main(int argc, char **argv) {
    /*
     * From CUDA documentation:
     * "There is no explicit initialization function for the runtime;
     * it initializes the first time a runtime function is called"
     */

    /*
     * Create data on CPU
     */
    float *host_data = new float[256];

    for (int i = 0; i < 256; ++i) {
        host_data[i] = float(i);
    }

    /*
     * Copy data to GPU
     */
    float *device_data;
    cudaMalloc(&device_data, 256*sizeof(float));
    cudaMemcpy(device_data, host_data, 256*sizeof(float), cudaMemcpyHostToDevice);

    /*
     * Call `square` with array.
     * Using 2 blocks, each with 128 threads, to evaluate 256 elements.
     */
    square<<<2, 128>>>(device_data);

    /*
     * Return data to CPU
     */
    cudaMemcpy(host_data, device_data, 256*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 256; ++i) {
        cout << host_data[i] << ' ';
    }
    cout << endl;

    cudaFree(device_data);
    free(host_data);
}
