cmake_minimum_required(VERSION 3.20)
project(MPI_HDF5_Hello CXX)

# Optional CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)

# Required packages
find_package(MPI REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)

enable_testing()

# Executable: parallel HDF5 writer
add_executable(hello_mpi_hdf5 main.cpp)
target_include_directories(hello_mpi_hdf5 PRIVATE ${MPI_CXX_INCLUDE_PATH} ${HDF5_INCLUDE_DIRS})
target_link_libraries(hello_mpi_hdf5 PRIVATE ${MPI_CXX_LIBRARIES} ${HDF5_CXX_LIBRARIES})

# Executable: verification step
add_executable(verify_hdf5 verify.cpp)
target_include_directories(verify_hdf5 PRIVATE ${HDF5_INCLUDE_DIRS})
target_link_libraries(verify_hdf5 PRIVATE ${HDF5_CXX_LIBRARIES})

# Test: Run MPI program (creates HDF5 file)
add_test(NAME RunMPIProgram
         COMMAND ${MPIEXEC_EXECUTABLE}
                 ${MPIEXEC_NUMPROC_FLAG} 4
                 $<TARGET_FILE:hello_mpi_hdf5>
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Test: Verify output file after it is created
add_test(NAME VerifyHDF5Output
         COMMAND $<TARGET_FILE:verify_hdf5>
         WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

# Ensure verification runs only after MPI program
set_tests_properties(VerifyHDF5Output
                     PROPERTIES DEPENDS RunMPIProgram)

# -------------------------------
# Optional CUDA support
# -------------------------------

if(ENABLE_CUDA)

    enable_language(CUDA)

    # Executable: GPU parallel HDF5 writer
    add_executable(hello_mpi_hdf5_gpu main_gpu.cu)

    target_include_directories(hello_mpi_hdf5_gpu PRIVATE
        ${MPI_CXX_INCLUDE_PATH}
        ${HDF5_INCLUDE_DIRS})

    target_link_libraries(hello_mpi_hdf5_gpu PRIVATE
        ${MPI_CXX_LIBRARIES}
        ${HDF5_CXX_LIBRARIES})

    set_target_properties(hello_mpi_hdf5_gpu PROPERTIES
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        CUDA_RUNTIME_LIBRARY Shared)

    # Test: Run GPU MPI program
    add_test(NAME RunMPIGPUProgram
             COMMAND ${MPIEXEC_EXECUTABLE}
                     ${MPIEXEC_NUMPROC_FLAG} 4
                     $<TARGET_FILE:hello_mpi_hdf5_gpu>
                     parallel_hello_gpu.h5
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # Test: Verify GPU output file
    add_test(NAME VerifyGPUHDF5Output
             COMMAND $<TARGET_FILE:verify_hdf5>
                     parallel_hello_gpu.h5
             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

    # Ensure verification runs after GPU program
    set_tests_properties(VerifyGPUHDF5Output
                         PROPERTIES DEPENDS RunMPIGPUProgram)

    # Label GPU tests
    set_tests_properties(
        RunMPIGPUProgram
        VerifyGPUHDF5Output
        PROPERTIES LABELS GPU)

endif()

# -------------------------------
# Installation Instructions
# -------------------------------
include(GNUInstallDirs)

# Install CPU binaries
install(TARGETS hello_mpi_hdf5 verify_hdf5
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install GPU binary when enabled
if(ENABLE_CUDA)
    install(TARGETS hello_mpi_hdf5_gpu
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()