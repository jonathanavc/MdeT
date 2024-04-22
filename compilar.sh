#g++ ./openmp/TNF.cpp -lz -fopenmp -O3 -o omp_ex
#nvcc ./cuda/TNF.cu -lz -O3 -arch=native -o cuda_ex
#nvcc ./cuda/TNF2.cu -lz -O3 -arch=native -o cuda2_ex
#nvcc ./cuda/TNF3.cu -lz -std=c++17 -O3 -arch=native -Xcompiler="-pthread" -o cuda3_ex
#nvcc ./cuda/Cuda.cu -std=c++17 -lboost_program_options -lboost_filesystem -lboost_graph -lboost_serialization -lboost_system -O3 -arch=native -lineinfo -use_fast_math -Xcompiler="-march=native -pthread -fopenmp" -o metabatcuda
nvcc ./cuda/Cuda2.cu --default-stream per-thread -std=c++17 -lboost_program_options -lboost_filesystem -lboost_graph -lboost_serialization -lboost_system -O2 -arch=native -lineinfo -Xcompiler="-march=native -pthread -fopenmp" -o metabatcuda2
#nvcc ./cuda/metabatcuda2.cu --default-stream per-thread -std=c++17 -lboost_program_options -lboost_filesystem -lboost_graph -lboost_serialization -lboost_system -O3 -arch=native -lineinfo -use_fast_math -lz -Xcompiler="-march=native -pthread -fopenmp" -o metabatcuda2t
#g++ ./extra/comp.cpp -lz -fopenmp -O3 -o comp_ex