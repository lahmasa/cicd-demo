#include <mpi.h>
#include <H5Cpp.h>
#include <H5public.h>
#include <H5Ppublic.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

// Check CUDA operation for errors
void checkCuda(cudaError_t error, const char* operation,
               int rank, MPI_Comm comm) {

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error on rank "
                  << rank << " during "
                  << operation << ": "
                  << cudaGetErrorString(error)
                  << std::endl;

        MPI_Abort(comm, EXIT_FAILURE);
        std::exit(EXIT_FAILURE);
    }
}

// CUDA kernel for vector addition
__global__ void vectorAdd(const int* a,
                          const int* b,
                          int* c,
                          int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Info info = MPI_INFO_NULL;
    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Use filename from argv[1] or default
    std::string filename = "parallel_hello_gpu.h5";
    if (argc > 1) {
        filename = argv[1];
    }

    const std::string dataset_name = "rank_data";

    // Number of elements computed by each MPI rank
    const int N = 257;

    const hsize_t total_size =
        static_cast<hsize_t>(size) * N;

    const size_t bytes = N * sizeof(int);

    // Allocate CPU vectors
    std::vector<int> a(N);
    std::vector<int> b(N);
    std::vector<int> c(N);

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        int global_index = rank * N + i;

        a[i] = global_index;
        b[i] = 2 * global_index;
    }

    // Select GPU device
    checkCuda(cudaSetDevice(0),
              "cudaSetDevice", rank, comm);

    // Allocate GPU memory
    int *d_a, *d_b, *d_c;

    checkCuda(cudaMalloc(&d_a, bytes),
              "cudaMalloc d_a", rank, comm);

    checkCuda(cudaMalloc(&d_b, bytes),
              "cudaMalloc d_b", rank, comm);

    checkCuda(cudaMalloc(&d_c, bytes),
              "cudaMalloc d_c", rank, comm);

    // Copy input vectors from CPU to GPU
    checkCuda(
        cudaMemcpy(d_a, a.data(), bytes,
                   cudaMemcpyHostToDevice),
        "Copy a to GPU", rank, comm
    );

    checkCuda(
        cudaMemcpy(d_b, b.data(), bytes,
                   cudaMemcpyHostToDevice),
        "Copy b to GPU", rank, comm
    );

    // Configure CUDA kernel
    const int threads = 128;
    const int blocks = (N + threads - 1) / threads;

    // Perform vector addition on GPU
    vectorAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);

    checkCuda(cudaGetLastError(),
              "Kernel launch", rank, comm);

    checkCuda(cudaDeviceSynchronize(),
              "Kernel execution", rank, comm);

    // Copy result from GPU to CPU
    checkCuda(
        cudaMemcpy(c.data(), d_c, bytes,
                   cudaMemcpyDeviceToHost),
        "Copy c to CPU", rank, comm
    );

    // Free GPU memory
    checkCuda(cudaFree(d_a),
              "cudaFree d_a", rank, comm);

    checkCuda(cudaFree(d_b),
              "cudaFree d_b", rank, comm);

    checkCuda(cudaFree(d_c),
              "cudaFree d_c", rank, comm);

    try {
        // Set up MPI-IO property list using C API
        hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_fapl_mpio(plist_id, comm, info);

        // Create HDF5 file
        H5::H5File file(
            filename,
            H5F_ACC_TRUNC,
            H5::FileCreatPropList::DEFAULT,
            H5::FileAccPropList(plist_id)
        );

        H5Pclose(plist_id);

        // Create dataset
        hsize_t dims[1] = {total_size};

        H5::DataSpace dataspace(1, dims);

        H5::DataSet dataset = file.createDataSet(
            dataset_name,
            H5::PredType::NATIVE_INT,
            dataspace
        );

        // Each rank writes N values
        hsize_t start[1] = {
            static_cast<hsize_t>(rank) * N
        };

        hsize_t count[1] = {
            static_cast<hsize_t>(N)
        };

        H5::DataSpace memspace(1, count);

        dataspace.selectHyperslab(
            H5S_SELECT_SET, count, start
        );

        // Use collective I/O
        hid_t xfer_plist_id =
            H5Pcreate(H5P_DATASET_XFER);

        H5Pset_dxpl_mpio(
            xfer_plist_id,
            H5FD_MPIO_COLLECTIVE
        );

        dataset.write(
            c.data(),
            H5::PredType::NATIVE_INT,
            memspace,
            dataspace,
            H5::DSetMemXferPropList(xfer_plist_id)
        );

        H5Pclose(xfer_plist_id);

        std::cout << "Rank " << rank
                  << " computed " << N
                  << " values on GPU and wrote them to file '"
                  << filename << "'\n";

    } catch (H5::Exception& e) {
        std::cerr << "HDF5 Error on rank "
                  << rank << ": "
                  << e.getDetailMsg()
                  << std::endl;

        MPI_Abort(comm, EXIT_FAILURE);
        return EXIT_FAILURE;
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}