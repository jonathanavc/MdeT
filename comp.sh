g++ ./openmp/TNF.cpp -lz -fopenmp -O3 -o omp
nvcc ./cuda/TNF.cu -lz -O3 -o cuda
nvcc ./cuda/TNF2.cu -lz -O3 -o cuda2
g++ ./extra/comp.cpp -lz -fopenmp -O3 -o comp

