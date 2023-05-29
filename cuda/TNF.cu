// nvcc TNF.cu -lz
// ta bien
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "../extra/KseqReader.h"

__device__ __constant__ short n_TNF_d = 136;

__device__ __constant__ unsigned char TNmap[256] = {2, 21, 31, 115, 101, 119, 67, 50, 135, 126, 69, 92, 116, 88, 8, 78, 47, 96, 3, 70, 106, 38, 48, 83, 16, 22, 136, 114, 5, 54, 107, 120, 72, 41, 44, 26, 27, 23, 136, 53, 12, 81, 136, 127, 30, 110, 136, 80, 132, 123, 71, 102, 79, 1, 35, 124, 29, 4, 136, 34, 91, 17, 136, 52, 9, 77, 136, 117, 76, 93, 136, 65, 6, 73, 136, 68, 28, 94, 136, 113, 121, 36, 136, 10, 103, 99, 136, 87, 129, 14, 136, 136, 98, 19, 136, 97, 15, 56, 136, 131, 57, 46, 136, 136, 122, 60, 136, 136, 42, 62, 136, 136, 7, 130, 136, 51, 133, 20, 136, 134, 89, 86, 136, 136, 104, 95, 136, 136, 49, 136, 136, 136, 105, 136, 136, 136, 33, 136, 136, 136, 43, 136, 136, 136, 55, 136, 136, 136, 112, 136, 136, 136, 136, 136, 136, 136, 75, 136, 136, 136, 32, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 100, 136, 136, 136, 63, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 125, 108, 136, 136, 58, 24, 136, 136, 84, 13, 136, 136, 25, 66, 136, 136, 18, 128, 136, 136, 74, 61, 136, 136, 85, 136, 136, 136, 118, 40, 136, 136, 109, 90, 136, 136, 45, 136, 136, 136, 111, 136, 136, 136, 82, 136, 136, 136, 59, 11, 136, 136, 64, 37, 136, 136, 0, 136, 136, 136, 39, 136, 136, 136};
__device__ __constant__ unsigned char TNPmap[256] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

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
                        const unsigned char *TNmap_eliminar, const unsigned char *TNPmap_eliminar, const unsigned char *smallCtgs,
                        const size_t *gCtgIdx_d, size_t contigs_per_thread)
{

    size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;
    // crea un tnf de forma local
    double TNF_temp[136];
    // inicializar valores de vector en 0
    /*
    for(size_t i = 0; i < contigs_per_thread; i++){
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        for(int j = 0; j < n_TNF_d; j++){
            TNF_d[contig_index * n_TNF_d + j] = 0;
        }
    }
    */
    //__syncthreads();

    for (size_t i = 0; i < contigs_per_thread; i++)
    {
        for (int j = 0; j < n_TNF_d; j++)
        {
            TNF_temp[j] = 0;
        }
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs)
            break;
        if (smallCtgs[contig_index] == 0)
        {
            const char *contig = get_contig_d(gCtgIdx_d[contig_index], seqs_d, seqs_d_index);
            size_t contig_size = seqs_d_index[gCtgIdx_d[contig_index]];
            if (gCtgIdx_d[contig_index] != 0)
            {
                contig_size -= seqs_d_index[gCtgIdx_d[contig_index] - 1];
            }
            for (size_t j = 0; j < contig_size - 3; ++j)
            {
                unsigned char tn = get_tn(contig, j);
                // SI tn NO SE ENCUENTRA EN TNmap el complemento del palindromo sí estará
                if (TNmap[tn] != n_TNF_d)
                {
                    ++TNF_temp[TNmap[tn]];
                }

                tn = get_revComp_tn_d(contig, j);

                // SALTA EL PALINDROMO PARA NO INSERTARLO NUEVAMENTE
                if (TNPmap[tn] == 0)
                {
                    if (TNmap[tn] != n_TNF_d)
                    {
                        ++TNF_temp[TNmap[tn]];
                    }
                }
            }
            double rsum = 0;
            for (size_t c = 0; c < n_TNF_d; ++c)
            {
                rsum += TNF_temp[c] * TNF_temp[c];
            }
            rsum = sqrt(rsum);
            for (size_t c = 0; c < n_TNF_d; ++c)
            {
                TNF_temp[c] /= rsum; // OK
            }
        }
        // guardar en la memoria global
        for (size_t c = 0; c < n_TNF_d; ++c)
        {
            TNF_d[contig_index * n_TNF_d + c] = TNF_temp[c];
        }
    }
}

static const std::string TN[] = {"GGTA", "AGCC", "AAAA", "ACAT", "AGTC", "ACGA", "CATA", "CGAA", "AAGT", "CAAA",
                                 "CCAG", "GGAC", "ATTA", "GATC", "CCTC", "CTAA", "ACTA", "AGGC", "GCAA", "CCGC", "CGCC", "AAAC", "ACTC", "ATCC",
                                 "GACC", "GAGA", "ATAG", "ATCA", "CAGA", "AGTA", "ATGA", "AAAT", "TTAA", "TATA", "AGTG", "AGCT", "CCAC", "GGCC",
                                 "ACCC", "GGGA", "GCGC", "ATAC", "CTGA", "TAGA", "ATAT", "GTCA", "CTCC", "ACAA", "ACCT", "TAAA", "AACG", "CGAG",
                                 "AGGG", "ATCG", "ACGC", "TCAA", "CTAC", "CTCA", "GACA", "GGAA", "CTTC", "GCCC", "CTGC", "TGCA", "GGCA", "CACG",
                                 "GAGC", "AACT", "CATG", "AATT", "ACAG", "AGAT", "ATAA", "CATC", "GCCA", "TCGA", "CACA", "CAAC", "AAGG", "AGCA",
                                 "ATGG", "ATTC", "GTGA", "ACCG", "GATA", "GCTA", "CGTC", "CCCG", "AAGC", "CGTA", "GTAC", "AGGA", "AATG", "CACC",
                                 "CAGC", "CGGC", "ACAC", "CCGG", "CCGA", "CCCC", "TGAA", "AACA", "AGAG", "CCCA", "CGGA", "TACA", "ACCA", "ACGT",
                                 "GAAC", "GTAA", "ATGC", "GTTA", "TCCA", "CAGG", "ACTG", "AAAG", "AAGA", "CAAG", "GCGA", "AACC", "ACGG", "CCAA",
                                 "CTTA", "AGAC", "AGCG", "GAAA", "AATC", "ATTG", "GCAC", "CCTA", "CGAC", "CTAG", "AGAA", "CGCA", "CGCG", "AATA"};

static const std::string TNP[] = {"ACGT", "AGCT", "TCGA", "TGCA", "CATG", "CTAG", "GATC", "GTAC", "ATAT", "TATA", "CGCG",
                                  "GCGC", "AATT", "TTAA", "CCGG", "GGCC"};

int n_THREADS = 32;
int n_BLOCKS = 128;

std::vector<std::string> seqs;
std::vector<size_t> gCtgIdx;
std::vector<unsigned char> smallCtgs;

const int n_TNF = 136;
const int n_TNFP = 16;

unsigned char TNmap[256];
unsigned char TNPmap[256];

static size_t minContig = 2500;               // minimum contig size for binning
static size_t minContigByCorr = 1000;         // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000; // for graph generation purpose

double *TNF_d;
static char *seqs_d;
static size_t *seqs_d_index;
static unsigned char *TNmap_d;
static unsigned char *TNPmap_d;
static unsigned char *smallCtgs_d;
static size_t *gCtgIdx_d;

int main(int argc, char const *argv[])
{
    if (argc > 2)
    {
        n_BLOCKS = atoi(argv[1]);
        n_THREADS = atoi(argv[2]);
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

    size_t nobs = 0;
    int nresv = 0;
    std::string inFile = "test.gz";

    auto start_global = std::chrono::system_clock::now();
    auto start = std::chrono::system_clock::now();

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
                if (len >= (int)std::min(minContigByCorr, minContigByCorrForGraph))
                {
                    if (len < (int)minContig)
                    {
                        if (len >= (int)minContigByCorr)
                            // cambio para facilitar la transferencia de smallCtgs
                            smallCtgs.emplace_back(1);
                        // smallCtgs.insert(nobs);
                        else
                        {
                            // cambio para facilitar la transferencia de smallCtgs
                            smallCtgs.emplace_back(0);
                            ++nresv;
                        }
                    }
                    else
                        smallCtgs.emplace_back(0);
                    // lCtgIdx[kseq->name.s] = nobs;
                    /////// cambio para facilitar la transferencia de gCtgIdx
                    gCtgIdx.emplace_back(seqs.size());
                    nobs++;
                    ///////
                }
                else
                {
                    // ignored[kseq->name.s] = seqs.size();
                }
                // contig_names.push_back(kseq->name.s);
                seqs.push_back(kseq->seq.s);
            }
        }
        kseq_destroy(kseq);
        kseq = NULL;
        gzclose(f);
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> duration = end - start;
    // std::cout <<"leer contigs "<< duration.count()/1000.f << "s " << std::endl;

    // std::cout << "nobs: " << nobs << ", small: " << smallCtgs.size() << ", gctg:" << gCtgIdx.size() << std::endl;

    start = std::chrono::system_clock::now();

    std::string seqs_h;
    std::vector<size_t> seqs_h_index;
    for (std::string const &contig : seqs)
    {
        seqs_h += contig;
        seqs_h_index.emplace_back(seqs_h.size());
    }

    end = std::chrono::system_clock::now();
    duration = end - start;
    // std::cout <<"vector<string> -> string "<< duration.count()/1000.f << "s " << std::endl;

    start = std::chrono::system_clock::now();

    int err = cudaMalloc(&TNF_d, (nobs * n_TNF * sizeof(double))); // memoria para almacenar TNF

    err += cudaMalloc(&TNmap_d, 256);
    err += cudaMemcpy(TNmap_d, TNmap, 256, cudaMemcpyHostToDevice); // TNmap

    err += cudaMalloc(&TNPmap_d, 256);
    err += cudaMemcpy(TNPmap_d, TNPmap, 256, cudaMemcpyHostToDevice); // TNPmap

    err += cudaMalloc(&seqs_d, seqs_h.size());
    err += cudaMemcpy(seqs_d, seqs_h.data(), seqs_h.size(), cudaMemcpyHostToDevice);

    err += cudaMalloc(&seqs_d_index, seqs_h_index.size() * sizeof(size_t));
    err += cudaMemcpy(seqs_d_index, seqs_h_index.data(), seqs_h_index.size() * sizeof(size_t), cudaMemcpyHostToDevice); // seqs_index

    err += cudaMalloc(&gCtgIdx_d, nobs * sizeof(size_t));
    err += cudaMemcpy(gCtgIdx_d, gCtgIdx.data(), nobs * sizeof(size_t), cudaMemcpyHostToDevice); // gCtgIdx

    err += cudaMalloc(&smallCtgs_d, nobs);
    err += cudaMemcpy(smallCtgs_d, smallCtgs.data(), nobs, cudaMemcpyHostToDevice); // seqs

    end = std::chrono::system_clock::now();
    duration = end - start;
    // std::cout <<"cudaMemcpy -> deveice "<< duration.count()/1000.f << "s " << std::endl;

    start = std::chrono::system_clock::now();

    size_t contigs_per_thread = 1 + ((nobs - 1) / (n_THREADS * n_BLOCKS));
    dim3 blkDim(n_THREADS, 1, 1);
    dim3 grdDim(n_BLOCKS, 1, 1);

    get_TNF<<<grdDim, blkDim>>>(TNF_d, seqs_d, seqs_d_index, nobs, TNmap_d, TNPmap_d, smallCtgs_d, gCtgIdx_d, contigs_per_thread);

    cudaDeviceSynchronize();

    end = std::chrono::system_clock::now();
    duration = end - start;
    // std::cout <<"kernel "<< duration.count()/1000.f << "s " << std::endl;
    start = std::chrono::system_clock::now();

    double *TNF = (double *)malloc(nobs * n_TNF * sizeof(double));

    cudaMemcpy(TNF, TNF_d, nobs * n_TNF * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    end = std::chrono::system_clock::now();
    duration = end - start;
    // std::cout <<"cudaMemcpy -> host "<< duration.count()/1000.f << "s " << std::endl;

    auto end_global = std::chrono::system_clock::now();
    duration = end_global - start_global;
    std::cout << duration.count() / 1000.f << std::endl;

    std::ofstream out("TNF.bin", ios::out | ios::binary);
    if (out)
    {
        out.write((char *)TNF, nobs * n_TNF * sizeof(double));
        out.close();
        // std::cout << "TNF guardado" << std::endl;
    }
    else
    {
        // std::cout << "Error al guardar" << std::endl;
    }
    out.close();

    free(TNF);

    return 0;
}
