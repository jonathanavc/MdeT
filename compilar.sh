g++ ./openmp/TNF.cpp -lz -fopenmp -O3 -o omp_ex
nvcc ./cuda/TNF.cu -lz -O3 -arch=native -o cuda_ex
nvcc ./cuda/TNF2.cu -lz -O3 -arch=native -o cuda2_ex
nvcc ./cuda/TNF3.cu -lz --std=c++11 -O3 -arch=native -Xcompiler="-pthread" -o cuda3_ex
g++ ./extra/comp.cpp -lz -fopenmp -O3 -o comp_ex