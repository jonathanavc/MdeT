g++ ./openmp/TNF.cpp -lz -fopenmp -O3 -o omp_ex
nvcc ./cuda/TNF.cu -lz -O3 -o cuda_ex
nvcc ./cuda/TNF2.cu -lz -O3 -o cuda2_ex
nvcc ./cuda/TNF3.cu -lz -O3 -Xcompiler="-pthread" -o cuda3_ex
g++ ./extra/comp.cpp -lz -fopenmp -O3 -o comp_ex