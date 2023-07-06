// nvcc TNF.cu -lz
// ta bien
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../extra/metrictime2.hpp"

__device__ __constant__ unsigned char TNmap_d[256] = {
    2,   21,  31,  115, 101, 119, 67,  50,  135, 126, 69,  92,  116, 88,  8,   78,  47,  96,  3,   70,
    106, 38,  48,  83,  16,  22,  136, 114, 5,   54,  107, 120, 72,  41,  44,  26,  27,  23,  136, 53,
    12,  81,  136, 127, 30,  110, 136, 80,  132, 123, 71,  102, 79,  1,   35,  124, 29,  4,   136, 34,
    91,  17,  136, 52,  9,   77,  136, 117, 76,  93,  136, 65,  6,   73,  136, 68,  28,  94,  136, 113,
    121, 36,  136, 10,  103, 99,  136, 87,  129, 14,  136, 136, 98,  19,  136, 97,  15,  56,  136, 131,
    57,  46,  136, 136, 122, 60,  136, 136, 42,  62,  136, 136, 7,   130, 136, 51,  133, 20,  136, 134,
    89,  86,  136, 136, 104, 95,  136, 136, 49,  136, 136, 136, 105, 136, 136, 136, 33,  136, 136, 136,
    43,  136, 136, 136, 55,  136, 136, 136, 112, 136, 136, 136, 136, 136, 136, 136, 75,  136, 136, 136,
    32,  136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 100, 136, 136, 136,
    63,  136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 125, 108, 136, 136, 58,  24,  136, 136,
    84,  13,  136, 136, 25,  66,  136, 136, 18,  128, 136, 136, 74,  61,  136, 136, 85,  136, 136, 136,
    118, 40,  136, 136, 109, 90,  136, 136, 45,  136, 136, 136, 111, 136, 136, 136, 82,  136, 136, 136,
    59,  11,  136, 136, 64,  37,  136, 136, 0,   136, 136, 136, 39,  136, 136, 136};

__device__ __constant__ unsigned char BN[256] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 0, 4, 1, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

__device__ short get_tn(const char *contig, const size_t index) {
    unsigned char N;
    short tn = 0;
    for (short i = 0; i < 4; i++) {
        N = BN[contig[index + i]];
        if (N & 4) return 256;
        tn = (tn << 2) | N;
    }
    return tn;
}

// esto mejoró "bastante" el rendimento
__device__ short get_revComp_tn_d(short tn) {
    unsigned char rctn = 0;
    for (short i = 0; i < 4; i++) {
        rctn = (rctn << 2) | (((tn & 3) + 2) % 4);
        tn = tn >> 2;
    }
    return rctn;
}

__device__ const char *get_contig_d(int contig_index, const char *seqs_d, const size_t *seqs_d_index) {
    return seqs_d + seqs_d_index[contig_index];
}

__global__ void get_TNF(double *TNF_d, const char *seqs_d, const size_t *seqs_d_index, size_t nobs,
                        const size_t contigs_per_thread, const size_t seqs_d_index_size) {
    const size_t minContig = 2500;
    const size_t minContigByCorr = 1000;
    const size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = 0; i < contigs_per_thread; i++) {
        const size_t contig_index = (thead_id * contigs_per_thread) + i;
        const size_t tnf_index = contig_index * 136;
        if (contig_index >= nobs) break;
        for (int j = 0; j < 136; j++) TNF_d[tnf_index + j] = 0;
    }

    for (size_t i = 0; i < contigs_per_thread; i++) {
        const size_t contig_index = (thead_id * contigs_per_thread) + i;
        const size_t tnf_index = contig_index * 136;
        if (contig_index >= nobs) break;
        size_t contig_size = seqs_d_index[contig_index + seqs_d_index_size] - seqs_d_index[contig_index];
        // calcular independiente si es small contig o no
        // if (contig_size >= minContig || contig_size < minContigByCorr) {
        const char *contig = seqs_d + seqs_d_index[contig_index];
        for (size_t j = 0; j < contig_size - 3; ++j) {
            short tn = get_tn(contig, j);
            if (tn & 256) continue;
            if (TNmap_d[tn] == 136) {
                tn = get_revComp_tn_d(tn);
            }
            ++TNF_d[tnf_index + TNmap_d[tn]];
        }
        double rsum = 0;
        for (size_t c = 0; c < 136; ++c) {
            rsum += TNF_d[tnf_index + c] * TNF_d[tnf_index + c];
        }
        rsum = sqrt(rsum);
        for (size_t c = 0; c < 136; ++c) {
            TNF_d[tnf_index + c] /= rsum;  // OK
        }
        //}
    }
}

__global__ void get_TNF_local(double *TNF_d, const char *seqs_d, const size_t *seqs_d_index, size_t nobs,
                              const size_t contigs_per_thread, const size_t seqs_d_index_size) {
    const size_t minContig = 2500;
    const size_t minContigByCorr = 1000;
    const size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;
    // crea un tnf de forma local
    double TNF_temp[136];

    for (size_t i = 0; i < contigs_per_thread; i++) {
        for (int j = 0; j < 136; j++) {
            TNF_temp[j] = 0;
        }
        const size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs) break;
        size_t contig_size = seqs_d_index[contig_index + seqs_d_index_size] - seqs_d_index[contig_index];
        // calcular independiente si es small contig o no
        // if (contig_size >= minContig || contig_size < minContigByCorr) {
        const char *contig = get_contig_d(contig_index, seqs_d, seqs_d_index);
        for (size_t j = 0; j < contig_size - 3; ++j) {
            short tn = get_tn(contig, j);
            if (tn & 256) continue;
            if (TNmap_d[tn] == 136) {
                tn = get_revComp_tn_d(tn);
                continue;
            }
            ++TNF_temp[TNmap_d[tn]];
        }
        double rsum = 0;
        for (size_t c = 0; c < 136; ++c) {
            rsum += TNF_temp[c] * TNF_temp[c];
        }
        rsum = sqrt(rsum);
        for (size_t c = 0; c < 136; ++c) {
            TNF_temp[c] /= rsum;  // OK
        }
        //}
        // guardar en la memoria global
        for (size_t c = 0; c < 136; ++c) {
            TNF_d[contig_index * 136 + c] = TNF_temp[c];
        }
    }
}

void reader(int fpint, int id, size_t chunk, size_t _size, char *_mem) {
    size_t readSz = 0;
    while (readSz < _size) {
        size_t _bytesres = _size - readSz;
        readSz += pread(fpint, _mem + (id * chunk) + readSz, _bytesres, (id * chunk) + readSz);
    }
}

int n_BLOCKS = 512;
int n_THREADS = 16;
char *_mem;
size_t fsize;
std::vector<size_t> seqs_d_index_i;
std::vector<size_t> seqs_d_index_e;

std::vector<std::string_view> seqs;
std::vector<std::string_view> contig_names;
std::unordered_map<std::string_view, size_t> ignored;
std::unordered_map<std::string_view, size_t> lCtgIdx;
std::unordered_map<size_t, size_t> gCtgIdx;
std::unordered_set<int> smallCtgs;

static size_t minContig = 2500;        // minimum contig size for binning
static size_t minContigByCorr = 1000;  // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000;  // for graph generation purpose
size_t nobs;
size_t nresv;
double *TNF;

int main(int argc, char const *argv[]) {
    std::string inFile = "test.gz";
    if (argc > 2) {
        n_BLOCKS = atoi(argv[1]);
        n_THREADS = atoi(argv[2]);
        if (argc > 3) {
            inFile = argv[3];
        }
    }
    TIMERSTART(total);

    FILE *fp = fopen(inFile.c_str(), "r");
    if (fp == NULL) {
        std::cout << "Error opening file: " << inFile << std::endl;
        return 1;
    } else {
        TIMERSTART(load_file);
        int nth = std::thread::hardware_concurrency();  // obtener el numero de hilos maximo
        fseek(fp, 0L, SEEK_END);
        fsize = ftell(fp);  // obtener el tamaño del archivo
        fclose(fp);

        size_t chunk = fsize / nth;

        cudaMallocHost((void **)&_mem, fsize);

        int fpint = open(inFile.c_str(), O_RDWR | O_CREAT, S_IREAD | S_IWRITE | S_IRGRP | S_IROTH);
        std::thread readerThreads[nth];

        for (int i = 0; i < nth; i++) {  // leer el archivo en paralelo
            if (i == nth - 1)
                readerThreads[i] = std::thread(reader, fpint, i, chunk, fsize - (i * chunk), _mem);
            else
                readerThreads[i] = std::thread(reader, fpint, i, chunk, chunk, _mem);
        }

        for (int i = 0; i < nth; i++) {  // esperar a que terminen de leer
            readerThreads[i].join();
        }

        close(fpint);

        TIMERSTOP(load_file);

        TIMERSTART(read_file);

        size_t __min = std::min(minContigByCorr, minContigByCorrForGraph);
        nobs = 0;
        nresv = 0;
        size_t contig_name_i;
        size_t contig_name_e;
        size_t contig_i;
        size_t contig_e;
        size_t contig_size;
        seqs.reserve(fsize % __min);
        ignored.reserve(fsize % __min);
        contig_names.reserve(fsize % __min);
        lCtgIdx.reserve(fsize % __min);
        gCtgIdx.reserve(fsize % __min);
        seqs_d_index_i.reserve(fsize % __min);
        seqs_d_index_e.reserve(fsize % __min);
        for (size_t i = 0; i < fsize; i++) {  // leer el archivo caracter por caracter
            if (_mem[i] < 65) {
                contig_name_i = i;  // guardar el inicio del nombre del contig
                while (_mem[i] != 10) i++;
                contig_name_e = i;  // guardar el final del nombre del contig
                i++;
                contig_i = i;  // guardar el inicio del contig
                while (i < fsize && _mem[i] != 10) i++;
                contig_e = i;  // guardar el final del contig
                contig_size = contig_e - contig_i;
                if (contig_size >= __min) {
                    if (contig_size < minContig) {
                        if (contig_size >= minContigByCorr)
                            smallCtgs.insert(nobs);
                        else
                            nresv++;
                    }
                    seqs_d_index_i.emplace_back(contig_i);
                    seqs_d_index_e.emplace_back(contig_e);
                    lCtgIdx[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] = nobs;
                    gCtgIdx[nobs++] = seqs.size();
                } else {
                    ignored[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] =
                        seqs.size();
                }
                contig_names.emplace_back(
                    std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i));
                seqs.emplace_back(std::string_view(_mem + contig_i, contig_e - contig_i));
            }
        }
        seqs_d_index_i.shrink_to_fit();  // liberar memoria no usada
        seqs_d_index_e.shrink_to_fit();  // liberar memoria no usada
        seqs.shrink_to_fit();            // liberar memoria no usada
        contig_names.shrink_to_fit();    // liberar memoria no usada
        TIMERSTOP(read_file);
    }
    std::cout << seqs.size() << " contigs" << std::endl;
    std::cout << nobs << " contigs with size >= " << minContig << std::endl;

    // calcular matriz de tetranucleotidos
    TIMERSTART(tnf);
    if (1) {
        double *TNF_d;
        char *seqs_d;
        size_t *seqs_d_index;
        dim3 blkDim(n_THREADS, 1, 1);
        dim3 grdDim(n_BLOCKS, 1, 1);
        cudaMallocHost((void **)&TNF, nobs * 136 * sizeof(double));
        cudaMalloc(&TNF_d, nobs * 136 * sizeof(double));
        cudaMalloc(&seqs_d, fsize);
        cudaMalloc(&seqs_d_index, 2 * nobs * sizeof(size_t));

        cudaMemcpy(seqs_d, _mem, fsize, cudaMemcpyHostToDevice);
        cudaMemcpy(seqs_d_index, seqs_d_index_i.data(), nobs * sizeof(size_t), cudaMemcpyHostToDevice);
        cudaMemcpy(seqs_d_index + nobs, seqs_d_index_e.data(), nobs * sizeof(size_t), cudaMemcpyHostToDevice);

        size_t contigs_per_thread = (nobs + (n_THREADS * n_BLOCKS) - 1) / (n_THREADS * n_BLOCKS);
        get_TNF<<<grdDim, blkDim>>>(TNF_d, seqs_d, seqs_d_index, nobs, contigs_per_thread, nobs);

        cudaDeviceSynchronize();

        cudaMemcpy(TNF, TNF_d, nobs * 136 * sizeof(double), cudaMemcpyDeviceToHost);
    }
    TIMERSTOP(tnf);

    std::ofstream out("TNF.bin", ios::out | ios::binary);
    if (out) {
        out.write((char *)TNF, nobs * 136 * sizeof(double));
        out.close();
    } else {
        std::cout << "Error al guardar en TNF.bin" << std::endl;
    }
    out.close();

    cudaFreeHost(_mem);
    TIMERSTOP(total);
    return 0;
}
