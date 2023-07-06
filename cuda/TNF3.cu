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
                        size_t contigs_per_thread)
{
    size_t minContig = 2500;
    size_t minContigByCorr = 1000;
    size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;

    for (size_t i = 0; i < contigs_per_thread; i++)
    {
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs)
            break;
        for (int j = 0; j < n_TNF_d; j++)
            TNF_d[contig_index * n_TNF_d + j] = 0;
    }

    for (size_t i = 0; i < contigs_per_thread; i++)
    {
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs)
            break;
        size_t contig_size = seqs_d_index[contig_index];
        if (contig_index != 0)
            contig_size -= seqs_d_index[contig_index - 1];
        //tengo dudas sobre esta parte ------------------------
        if (contig_size > minContig || contig_size < minContigByCorr)
        {
            const char *contig = get_contig_d(contig_index, seqs_d, seqs_d_index);
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

__global__ void get_TNF_local(double *TNF_d, const char *seqs_d, const size_t *seqs_d_index, size_t nobs,
                              size_t contigs_per_thread)
{
    size_t minContig = 2500;
    size_t minContigByCorr = 1000;     
    size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;
    // crea un tnf de forma local
    double TNF_temp[136];

    for (size_t i = 0; i < contigs_per_thread; i++)
    {
        for (int j = 0; j < n_TNF_d; j++)
        {
            TNF_temp[j] = 0;
        }
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if (contig_index >= nobs)
            break;
        size_t contig_size = seqs_d_index[contig_index];
        if (contig_index != 0)
            contig_size -= seqs_d_index[contig_index - 1];
        //tengo dudas sobre esta parte ------------------------
        if (contig_size < minContig || contig_size < minContigByCorr)
        {
            const char *contig = get_contig_d(contig_index, seqs_d, seqs_d_index);
            for (size_t j = 0; j < contig_size - 3; ++j)
            {
                unsigned char tn = get_tn(contig, j);
                // SI tn NO SE ENCUENTRA EN TNmap el complemento del palindromo sí estará
                if (TNmap_d[tn] != n_TNF_d)
                {
                    ++TNF_temp[TNmap_d[tn]];
                }

                tn = get_revComp_tn_d(contig, j);

                // SALTA EL PALINDROMO PARA NO INSERTARLO NUEVAMENTE
                if (TNPmap_d[tn] == 0)
                {
                    if (TNmap_d[tn] != n_TNF_d)
                    {
                        ++TNF_temp[TNmap_d[tn]];
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

int contig_per_thread = 1;
int n_THREADS = 32;
int n_BLOCKS = 128;

std::vector<size_t> seqs_index_i;
std::vector<size_t> seqs_index_e;
std::vector<std::string_view> seqs;
std::vector<std::string_view> contig_names;
std::unordered_map<std::string_view, size_t> ignored;
std::unordered_map<std::string_view, size_t> lCtgIdx;
std::unordered_map<size_t, size_t> gCtgIdx;
std::unordered_set<int> smallCtgs;

const int n_TNF = 136;
const int n_TNFP = 16;

unsigned char TNmap[256];
unsigned char TNPmap[256];

static size_t minContig = 2500;               // minimum contig size for binning
static size_t minContigByCorr = 1000;         // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000; // for graph generation purpose

size_t nobs_cont;
size_t kernel_cont;
std::vector<double *> TNF;
size_t *seqs_kernel_index[2];

void kernel(dim3 blkDim, dim3 grdDim, int SUBP_IND, int cont, int size)
{
    // std::cout << "kernel: " << kernel_cont<< std::endl;
    cudaStream_t _s;
    cudaStreamCreate(&_s);
    char *seqs_d;
    TNF[cont] = (double *)malloc(n_BLOCKS * n_THREADS * contig_per_thread * n_TNF * sizeof(double));
    cudaMallocAsync(&seqs_d, seqs_kernel[SUBP_IND].size(), _s);
    cudaMemcpyAsync(seqs_d, seqs_kernel[SUBP_IND].data(), seqs_kernel[SUBP_IND].size(), cudaMemcpyHostToDevice, _s);
    cudaMemcpyAsync(seqs_d_index[SUBP_IND], seqs_kernel_index[SUBP_IND],
                    n_BLOCKS * n_THREADS * contig_per_thread * sizeof(size_t), cudaMemcpyHostToDevice,
                    _s); // seqs_index
    get_TNF<<<grdDim, blkDim, 0, _s>>>(TNF_d[SUBP_IND], seqs_d, seqs_d_index[SUBP_IND], size, contig_per_thread);
    cudaFreeAsync(seqs_d, _s);
    cudaMemcpyAsync(TNF[cont], TNF_d[SUBP_IND], n_BLOCKS * n_THREADS * contig_per_thread * n_TNF * sizeof(double),
                    cudaMemcpyDeviceToHost, _s);
    cudaStreamSynchronize(_s);
    seqs_kernel[SUBP_IND] = "";
}

int main(int argc, char const *argv[])
{
    std::string inFile = "test.gz";
    if (argc > 2)
    {
        n_BLOCKS = atoi(argv[1]);
        n_THREADS = atoi(argv[2]);
        if (argc > 3)
        {
            inFile = argv[3];
        }
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
    

    global_contigs_target = n_BLOCKS * n_THREADS;

    int nresv = 0;
    kernel_cont = 0;
    char *_mem;
    size_t fsize;

    gzFile f = gzopen(inFile.c_str(), "r");
    if (f == NULL)
    {
        cerr << "[Error!] can't open the sequence fasta file " << inFile << endl;
        return 1;
    } else {
        auto _start = std::chrono::system_clock::now();

        int nth = std::thread::hardware_concurrency();
        int fpint = -1;

        FILE *fp = fopen(inFile.c_str(), "r");
        fseek(fp, 0L, SEEK_END);  // seek to the EOF
        fsize = ftell(fp);        // get the current position
        fclose(fp);
        size_t chunk = fsize / nth;
        cudaMallocHost((void **)&_mem, fsize);
        std::cout << "tamaño total:" << fsize << std::endl;
        std::cout << "threads:" << nth << std::endl;

        fpint = open(inFile.c_str(), O_RDWR | O_CREAT, S_IREAD | S_IWRITE | S_IRGRP | S_IROTH);
        thread readerThreads[nth];
        size_t total = 0;
        for (int i = 0; i < nth; i++) {
            size_t _size;
            if (i != nth - 1)
                _size = chunk;
            else
                _size = chunk + (fsize % nth);
            total += _size;
            std::cout << "tamaño chunk:" << _size << std::endl;
            readerThreads[i] = thread(reader, fpint, i, chunk, _size, _mem);
        }
        for (int i = 0; i < nth; i++) {
            readerThreads[i].join();
        }

        std::cout << "total:" << total << std::endl;

        close(fpint);

        auto _end = std::chrono::system_clock::now();
        std::chrono::duration<float, std::milli> _duration = _end - _start;
        std::cout << "cargar archivo descomprimido a ram(pinned):" << _duration.count() / 1000.f << std::endl;

        _start = std::chrono::system_clock::now();
        size_t __min = std::min(minContigByCorr, minContigByCorrForGraph);
        std::string contig_name;
        size_t contig_name_i;
        size_t contig_name_e;
        size_t contig_i;
        size_t contig_e;
        size_t contig_size;
        seqs.reserve(fsize % __min);
        seqs_index_i.reserve(fsize % __min);
        seqs_index_e.reserve(fsize % __min);
        ignored.reserve(fsize % __min);
        contig_names.reserve(fsize % __min);
        lCtgIdx.reserve(fsize % __min);
        gCtgIdx.reserve(fsize % __min);
        for (size_t i = 0; i < fsize; i++) {
            if (_mem[i] < 65) {
                contig_name_i = i;
                while (_mem[i] != 10) i++;
                contig_name_e = i;
                i++;

                contig_i = i;
                while (i < fsize && _mem[i] != 10) i++;
                contig_e = i;

                contig_size = contig_e - contig_i;

                if (contig_size >= __min) {
                    if (contig_size < minContig) {
                        if (contig_size >= minContigByCorr)
                            smallCtgs.insert(nobs);
                        else
                        {
                            ++nresv;
                        }
                    }
                } else {
                    ignored[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] =
                        seqs.size();
                }
                contig_names.emplace_back(_mem + contig_name_i, contig_name_e - contig_name_i);
                seqs.emplace_back(_mem + contig_i, contig_e - contig_i);
                seqs_index_i.emplace_back(contig_i);
                seqs_index_e.emplace_back(contig_e);
            }
        }
        seqs.shrink_to_fit();
        seqs_index_i.shrink_to_fit();
        seqs_index_e.shrink_to_fit();
        contig_names.shrink_to_fit();

        _end = std::chrono::system_clock::now();
        _duration = _end - _start;
        std::cout << "cargar estructuras TNF:" << _duration.count() / 1000.f << std::endl;
    }

    char * TNF_d;
    char * seqs_d;
    char * seqs_index_d;

    cudaMalloc(&TNF_d, nobs * n_TNF * sizeof(double));
    cudaMalloc(&seqs_d, fsize);
    cudaMemcpy(seqs_d, _mem, fsize, cudaMemcpyHostToDevice);
    cudaMalloc(&seqs_index_d, seqs.size() * 2 * sizeof(double));
    cudaMemcpy(seqs_index_d, seqs_index_i.data(), seqs.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(seqs_index_d + seqs.size() * sizeof(double), seqs_index_e.data(), seqs.size() * sizeof(double),
               cudaMemcpyHostToDevice);

    dim3 blkDim(n_THREADS, 1, 1);
    dim3 grdDim(n_BLOCKS, 1, 1);

    get_TNF<<<grdDim, blkDim>>>(TNF_d, seqs_d, seqs_index_d, nobs, ((nobs + global_contigs_target - 1)/global_contigs_target), global_contigs_target);

    _end = std::chrono::system_clock::now();
    _duration = _end - _start;
    std::cout << "calcular TNF:" << _duration.count() / 1000.f << std::endl;

    auto end_global = std::chrono::system_clock::now();
    duration = end_global - start_global;
    std::cout << duration.count() / 1000.f << std::endl;

    std::ofstream out("TNF.bin", ios::out | ios::binary);
    if (out)
    {
        for (size_t i = 0; i < TNF.size(); i++)
        {
            if (i < (TNF.size() - 1) || nobs % (n_BLOCKS * n_THREADS * contig_per_thread) == 0)
                out.write((char *)TNF[i], n_BLOCKS * n_THREADS * contig_per_thread * n_TNF * sizeof(double));
            else
                out.write((char *)TNF[i], (nobs % (n_BLOCKS * n_THREADS * contig_per_thread)) * n_TNF * sizeof(double));
        }
        // std::cout << "TNF guardado" << std::endl;
    } else {
        std::cout << "Error al guardar TNF.bin" << std::endl;
    }
    out.close();

    for (int i = 0; i < TNF.size(); i++) cudaFreeHost(TNF[i]);
    cudaFreeHost(_mem);
    return 0;
}