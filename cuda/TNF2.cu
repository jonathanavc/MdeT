// nvcc TNF.cu -lz
// ta bien
#include "../extra/KseqReader.h"
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

__device__ __constant__ int n_TNF_d = 136;

__device__ __constant__ unsigned char TNmap_d[256] = {
    2,   21,  31,  115, 101, 119, 67,  50,  135, 126, 69,  92,  116, 88,  8,   78,  47,  96,  3,   70,  106, 38,
    48,  83,  16,  22,  136, 114, 5,   54,  107, 120, 72,  41,  44,  26,  27,  23,  136, 53,  12,  81,  136, 127,
    30,  110, 136, 80,  132, 123, 71,  102, 79,  1,   35,  124, 29,  4,   136, 34,  91,  17,  136, 52,  9,   77,
    136, 117, 76,  93,  136, 65,  6,   73,  136, 68,  28,  94,  136, 113, 121, 36,  136, 10,  103, 99,  136, 87,
    129, 14,  136, 136, 98,  19,  136, 97,  15,  56,  136, 131, 57,  46,  136, 136, 122, 60,  136, 136, 42,  62,
    136, 136, 7,   130, 136, 51,  133, 20,  136, 134, 89,  86,  136, 136, 104, 95,  136, 136, 49,  136, 136, 136,
    105, 136, 136, 136, 33,  136, 136, 136, 43,  136, 136, 136, 55,  136, 136, 136, 112, 136, 136, 136, 136, 136,
    136, 136, 75,  136, 136, 136, 32,  136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136,
    100, 136, 136, 136, 63,  136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 125, 108, 136, 136, 58,  24,
    136, 136, 84,  13,  136, 136, 25,  66,  136, 136, 18,  128, 136, 136, 74,  61,  136, 136, 85,  136, 136, 136,
    118, 40,  136, 136, 109, 90,  136, 136, 45,  136, 136, 136, 111, 136, 136, 136, 82,  136, 136, 136, 59,  11,
    136, 136, 64,  37,  136, 136, 0,   136, 136, 136, 39,  136, 136, 136};
__device__ __constant__ unsigned char TNPmap_d[256] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

__device__ const char *get_contig_d(int contig_index, const char *seqs_d, const size_t *seqs_d_index)
{
    size_t contig_beg = 0;
    if (contig_index != 0)
    {
        contig_beg = seqs_d_index[contig_index - 1];
    }
    return seqs_d + contig_beg;
}

__device__ __host__ unsigned char get_tn(const char *contig, size_t index)
{
    unsigned char tn = 0;
    for (int i = 0; i < 4; i++)
    {
        char N = contig[index + i];
        if (N == 'A')
            N = 0;
        else if (N == 'C')
            N = 1;
        else if (N == 'T')
            N = 2;
        else if (N == 'G')
            N = 3;
        else
            return 170; // no existe en TNmap[]

        tn = (tn << 2) + N;
    }
    return tn;
}

__device__ unsigned char get_revComp_tn_d(const char *contig, size_t index)
{
    unsigned char tn = 0;
    for (int i = 3; i >= 0; i--)
    {
        char N = contig[index + i];
        if (N == 'A')
            N = 2;
        else if (N == 'C')
            N = 3;
        else if (N == 'T')
            N = 0;
        else if (N == 'G')
            N = 1;
        else
            return 170; // no existe en TNmap[]
        tn = (tn << 2) + N;
    }
    return tn;
}

__global__ void get_TNF(double *TNF_d, const char *seqs_d, const size_t *seqs_d_index, size_t nobs,
                        const unsigned char *smallCtgs, size_t contigs_per_thread)
{
    size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = 0; i < contigs_per_thread; i++)
    {
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs)
            break;
        for (int j = 0; j < n_TNF_d; j++)
        {
            TNF_d[contig_index * n_TNF_d + j] = 0;
        }
    }

    for (size_t i = 0; i < contigs_per_thread; i++)
    {
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs)
            break;
        if (smallCtgs[contig_index] == 0)
        {
            const char *contig = get_contig_d(contig_index, seqs_d, seqs_d_index);
            size_t contig_size = seqs_d_index[contig_index];
            if (contig_index != 0)
            {
                contig_size -= seqs_d_index[contig_index - 1];
            }
            for (size_t j = 0; j < contig_size - 3; ++j)
            {
                unsigned char tn = get_tn(contig, j);
                // SI tn NO SE ENCUENTRA EN TNmap el complemento del palindromo sí estará
                if (TNmap_d[tn] != n_TNF_d)
                {
                    ++TNF_d[contig_index * n_TNF_d + TNmap_d[tn]];
                }

                tn = get_revComp_tn_d(contig, j);

                // SALTA EL PALINDROMO PARA NO INSERTARLO NUEVAMENTE
                if (TNPmap_d[tn] == 0)
                {
                    if (TNmap_d[tn] != n_TNF_d)
                    {
                        ++TNF_d[contig_index * n_TNF_d + TNmap_d[tn]];
                    }
                }
            }
            double rsum = 0;
            for (size_t c = 0; c < n_TNF_d; ++c)
            {
                rsum += TNF_d[contig_index * n_TNF_d + c] * TNF_d[contig_index * n_TNF_d + c];
            }
            rsum = sqrt(rsum);
            for (size_t c = 0; c < n_TNF_d; ++c)
            {
                TNF_d[contig_index * n_TNF_d + c] /= rsum; // OK
            }
        }
    }
}

static const std::string TN[] = {
    "GGTA", "AGCC", "AAAA", "ACAT", "AGTC", "ACGA", "CATA", "CGAA", "AAGT", "CAAA", "CCAG", "GGAC", "ATTA", "GATC",
    "CCTC", "CTAA", "ACTA", "AGGC", "GCAA", "CCGC", "CGCC", "AAAC", "ACTC", "ATCC", "GACC", "GAGA", "ATAG", "ATCA",
    "CAGA", "AGTA", "ATGA", "AAAT", "TTAA", "TATA", "AGTG", "AGCT", "CCAC", "GGCC", "ACCC", "GGGA", "GCGC", "ATAC",
    "CTGA", "TAGA", "ATAT", "GTCA", "CTCC", "ACAA", "ACCT", "TAAA", "AACG", "CGAG", "AGGG", "ATCG", "ACGC", "TCAA",
    "CTAC", "CTCA", "GACA", "GGAA", "CTTC", "GCCC", "CTGC", "TGCA", "GGCA", "CACG", "GAGC", "AACT", "CATG", "AATT",
    "ACAG", "AGAT", "ATAA", "CATC", "GCCA", "TCGA", "CACA", "CAAC", "AAGG", "AGCA", "ATGG", "ATTC", "GTGA", "ACCG",
    "GATA", "GCTA", "CGTC", "CCCG", "AAGC", "CGTA", "GTAC", "AGGA", "AATG", "CACC", "CAGC", "CGGC", "ACAC", "CCGG",
    "CCGA", "CCCC", "TGAA", "AACA", "AGAG", "CCCA", "CGGA", "TACA", "ACCA", "ACGT", "GAAC", "GTAA", "ATGC", "GTTA",
    "TCCA", "CAGG", "ACTG", "AAAG", "AAGA", "CAAG", "GCGA", "AACC", "ACGG", "CCAA", "CTTA", "AGAC", "AGCG", "GAAA",
    "AATC", "ATTG", "GCAC", "CCTA", "CGAC", "CTAG", "AGAA", "CGCA", "CGCG", "AATA"};

static const std::string TNP[] = {"ACGT", "AGCT", "TCGA", "TGCA", "CATG", "CTAG", "GATC", "GTAC",
                                  "ATAT", "TATA", "CGCG", "GCGC", "AATT", "TTAA", "CCGG", "GGCC"};
int n_STREAMS = 1;
int n_THREADS = 32;
int n_BLOCKS = 128;

std::vector<std::string> seqs;
std::unordered_map<size_t, size_t> gCtgIdx;
std::unordered_set<int> smallCtgs;

const int n_TNF = 136;
const int n_TNFP = 16;

unsigned char TNmap[256];
unsigned char TNPmap[256];

static size_t minContig = 2500;               // minimum contig size for binning
static size_t minContigByCorr = 1000;         // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000; // for graph generation purpose

std::thread *streams;
double **TNF_d;
char **seqs_d;
size_t **seqs_d_index;
unsigned char **smallCtgs_d;

size_t nobs_cont;
size_t kernel_cont;
std::string *seqs_kernel;
std::vector<double *> TNF;
size_t *seqs_kernel_index;
unsigned char *smallCtgs_kernel;

void kernel(dim3 blkDim, dim3 grdDim, int cont)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int index = cont % n_STREAMS;

    // std::cout << "kernel: " << kernel_cont<< std::endl;
    cudaMalloc(&seqs_d[index], seqs_kernel[index].size());
    cudaMemcpy(seqs_d[index], seqs_kernel[index].data(), seqs_kernel[index].size(), cudaMemcpyHostToDevice);
    cudaMemcpy(seqs_d_index[index], &seqs_kernel_index[index * n_THREADS * n_BLOCKS * sizeof(size_t)],
               n_BLOCKS * n_THREADS * sizeof(size_t), cudaMemcpyHostToDevice); // seqs_index
    cudaMemcpy(smallCtgs_d[index], &smallCtgs_kernel[index * n_THREADS * n_BLOCKS], n_BLOCKS * n_THREADS,
               cudaMemcpyHostToDevice);

    get_TNF<<<grdDim, blkDim, 0, stream>>>(TNF_d[index], seqs_d[index], seqs_d_index[index], nobs_cont,
                                           smallCtgs_d[index], 1);

    cudaStreamSynchronize(stream);
    cudaFree(seqs_d[index]);
    cudaMemcpy(TNF[cont], TNF_d[index], n_BLOCKS * n_THREADS * n_TNF * sizeof(double), cudaMemcpyDeviceToHost);

    cudaStreamDestroy(stream);

    seqs_kernel[index] = "";
}

int main(int argc, char const *argv[])
{
    if (argc > 3)
    {
        n_BLOCKS = atoi(argv[1]);
        n_THREADS = atoi(argv[2]);
        n_STREAMS = atoi(argv[3]);
    }
    // std::cout << "n°bloques: "<< n_BLOCKS <<", n°threads:"<< n_THREADS << std::endl;

    // se inicializan los mapas
    for (int i = 0; i < 256; i++)
    {
        TNmap[i] = n_TNF;
        TNPmap[i] = 0;
    }
    for (int i = 0; i < n_TNF; ++i)
    {
        unsigned char key = get_tn(TN[i].c_str(), 0);
        TNmap[key] = i;
    }

    for (size_t i = 0; i < n_TNFP; ++i)
    {
        unsigned char key = get_tn(TNP[i].c_str(), 0);
        TNPmap[key] = 1;
    }

    auto start_global = std::chrono::system_clock::now();
    auto start = std::chrono::system_clock::now();

    // crear streams
    streams = new std::thread[n_STREAMS];

    seqs_kernel = new std::string[n_STREAMS];

    TNF_d = new double *[n_STREAMS];
    seqs_d = new char *[n_STREAMS];
    seqs_d_index = new size_t *[n_STREAMS];
    smallCtgs_d = new unsigned char *[n_STREAMS];

    bool bool_thread[n_STREAMS];
    dim3 blkDim(n_THREADS, 1, 1);
    dim3 grdDim(n_BLOCKS, 1, 1);

    nobs_cont = 0;
    kernel_cont = 0;
    seqs_kernel_index = (size_t *)malloc(n_THREADS * n_BLOCKS * sizeof(size_t) * n_STREAMS);
    smallCtgs_kernel = (unsigned char *)malloc(n_THREADS * n_BLOCKS * n_STREAMS);

    for (int i = 0; i < n_STREAMS; i++)
    {
        seqs_kernel[i] = "";
        bool_thread[i] = 0;
        cudaMalloc(&TNF_d[i], n_BLOCKS * n_THREADS * n_TNF * sizeof(double));
        cudaMalloc(&seqs_d_index[i], n_BLOCKS * n_THREADS * sizeof(size_t));
        cudaMalloc(&smallCtgs_d[i], n_BLOCKS * n_THREADS);
    }

    size_t nobs = 0;

    int nresv = 0;
    std::string inFile = "test.gz";

    gzFile f = gzopen(inFile.c_str(), "r");
    if (f == NULL)
    {
        cerr << "[Error!] can't open the sequence fasta file " << inFile << endl;
        return 1;
    }
    else
    {
        kseq_t *kseq = kseq_init(f);
        int64_t len;
        while ((len = kseq_read(kseq)) > 0)
        {
            std::transform(kseq->seq.s, kseq->seq.s + len, kseq->seq.s, ::toupper);
            if (kseq->name.l > 0)
            {
                std::cout <<"."; 
                size_t index = (kernel_cont % n_STREAMS) * n_THREADS * n_BLOCKS;
                if (bool_thread[kernel_cont % n_STREAMS] && kernel_cont % n_STREAMS < kernel_cont)
                {
                    bool_thread[kernel_cont % n_STREAMS] = 0;
                    streams[kernel_cont % n_STREAMS].join();
                }
                std::cout <<","; 

                if (len >= (int)std::min(minContigByCorr, minContigByCorrForGraph))
                {
                    if (len < (int)minContig)
                    {
                        if (len >= (int)minContigByCorr)
                        {
                            smallCtgs.insert(1);
                            smallCtgs_kernel[index + nobs_cont] = 1;
                        }
                        else
                        {
                            smallCtgs_kernel[index + nobs_cont] = 0;
                            ++nresv;
                        }
                    }
                    else
                    {
                        smallCtgs_kernel[index + nobs_cont] = 0;
                    }
                    gCtgIdx[nobs++] = seqs.size();

                    seqs_kernel[kernel_cont % n_STREAMS] += kseq->seq.s;
                    seqs_kernel_index[index * sizeof(size_t) + nobs_cont] = seqs_kernel[kernel_cont % n_STREAMS].size();
                    nobs_cont++;
                }
                else
                {
                    // ignored[kseq->name.s] = seqs.size();
                }
                // contig_names.push_back(kseq->name.s);
                seqs.push_back(kseq->seq.s);

                if (nobs_cont == n_BLOCKS * n_THREADS)
                {
                    std::cout << "hola:"<< kernel_cont % n_STREAMS<< std::endl;
                    if (bool_thread[kernel_cont % n_STREAMS] && kernel_cont % n_STREAMS < kernel_cont){
                        bool_thread[kernel_cont % n_STREAMS] = 0;
                        streams[kernel_cont % n_STREAMS].join();
                    }   
                    TNF.emplace_back((double *)malloc(n_BLOCKS * n_THREADS * n_TNF * sizeof(double)));
                    streams[kernel_cont % n_STREAMS] = std::thread(kernel, blkDim, grdDim, kernel_cont);
                    std::cout << "adios"<< kernel_cont % n_STREAMS<< std::endl;
                    bool_thread[kernel_cont % n_STREAMS] = 1;
                    kernel_cont++;
                    nobs_cont = 0;
                }
            }
        }
        kseq_destroy(kseq);
        kseq = NULL;
        gzclose(f);
    }
    if (nobs_cont != 0)
    {
        if (bool_thread[kernel_cont % n_STREAMS] && kernel_cont % n_STREAMS < kernel_cont)
        {
            streams[kernel_cont % n_STREAMS].join();
        }
        TNF.emplace_back((double *)malloc(n_BLOCKS * n_THREADS * n_TNF * sizeof(double)));
        streams[kernel_cont % n_STREAMS] = std::thread(kernel, blkDim, grdDim, kernel_cont);
        kernel_cont++;
        nobs_cont = 0;
    }

    for (int i = 0; i < std::min((size_t)n_STREAMS, kernel_cont); i++)
    {
        bool_thread[i] = 0;
        streams[i].join();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    // std::cout <<"leer contigs + procesamiento "<< duration.count()/1000.f << "s " << std::endl;

    auto end_global = std::chrono::system_clock::now();
    duration = end_global - start_global;
    std::cout << duration.count() / 1000.f << std::endl;

    std::ofstream out("TNF.bin", ios::out | ios::binary);
    if (out)
    {
        for (size_t i = 0; i < TNF.size(); i++)
        {
            if (i < (TNF.size() - 1) || nobs % (n_BLOCKS * n_THREADS) == 0)
                out.write((char *)TNF[i], n_BLOCKS * n_THREADS * n_TNF * sizeof(double));
            else
                out.write((char *)TNF[i], (nobs % (n_BLOCKS * n_THREADS)) * n_TNF * sizeof(double));
        }
        // std::cout << "TNF guardado" << std::endl;
    }
    else
    {
        // std::cout << "Error al guardar" << std::endl;
    }
    out.close();

    for (int i = 0; i < TNF.size(); i++)
    {
        free(TNF[i]);
    }
    return 0;
}
