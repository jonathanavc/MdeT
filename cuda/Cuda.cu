// nvcc TNF.cu -lz
// ta bien
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
// #include <math.h>
// #include <math_functions.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
// #include <texture_fetch_functions.h>
#include <unistd.h>

#define BOOST_UBLAS_INLINE inline
#define BOOST_UBLAS_CHECK_ENABLE 0
#define BOOST_UBLAS_USE_FAST_SAME
#define BOOST_UBLAS_TYPE_CHECK 0

#include <math.h>

#include <algorithm>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <boost/graph/adj_list_serialize.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/undirected_graph.hpp>
#include <boost/math/distributions.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/program_options.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/serialization/vector.hpp>
#include <chrono>
#include <cstdarg>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../extra/metrictime2.hpp"
#include "ProgressTracker.h"
#include "igraph.h"

// #include "../extra/metrictime2.hpp"

namespace po = boost::program_options;

__device__ __constant__ double _log10 = 2.3025850929940456840179914546844;

__device__ __constant__ unsigned char TNmap_d[256] = {
    2,   21,  31,  115, 101, 119, 67,  50, 135, 126, 69,  92,  116, 88,  8,   78,  47,  96,  3,   70,  106, 38,  48,  83,  16,  22,
    8,   114, 5,   54,  107, 120, 72,  41, 44,  26,  27,  23,  71,  53,  12,  81,  31,  127, 30,  110, 3,   80,  132, 123, 71,  102,
    79,  1,   35,  124, 29,  4,   67,  34, 91,  17,  48,  52,  9,   77,  127, 117, 76,  93,  34,  65,  6,   73,  92,  68,  28,  94,
    114, 113, 121, 36,  80,  10,  103, 99, 52,  87,  129, 14,  78,  113, 98,  19,  120, 97,  15,  56,  26,  131, 57,  46,  102, 51,
    122, 60,  115, 117, 42,  62,  70,  10, 7,   130, 53,  51,  133, 20,  124, 134, 89,  86,  50,  65,  104, 95,  83,  87,  49,  111,
    12,  122, 105, 0,   29,  89,  33,  84, 135, 6,   43,  85,  16,  129, 55,  82,  30,  42,  112, 39,  91,  104, 43,  25,  116, 28,
    75,  118, 5,   98,  32,  109, 72,  15, 100, 59,  132, 7,   49,  125, 2,   9,   55,  18,  47,  121, 100, 45,  27,  57,  63,  64,
    79,  133, 105, 58,  101, 76,  112, 74, 106, 103, 125, 108, 81,  60,  58,  24,  4,   86,  84,  13,  126, 73,  25,  66,  22,  14,
    18,  128, 110, 62,  74,  61,  17,  95, 85,  66,  88,  94,  118, 40,  54,  19,  109, 90,  41,  56,  45,  11,  123, 130, 111, 108,
    21,  77,  82,  128, 96,  36,  59,  11, 23,  46,  64,  37,  1,   20,  0,   24,  119, 93,  39,  61,  38,  99};

__device__ __constant__ unsigned char BN[256] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

__device__ __constant__ double _b1[18] = {46349.1624324381,  -76092.3748553155, -639.918334183,  53873.3933743949, -156.6547554844,
                                          -21263.6010657275, 64.7719132839,     5003.2646455284, -8.5014386744,    -700.5825500292,
                                          0.3968284526,      54.037542743,      -1.7713972342,   474.0850141891,   -23.966597785,
                                          0.7800219061,      -0.0138723693,     0.0001027543};

__device__ __constant__ double _c1[18] = {-443565.465710869, 718862.10804858, 5114.1630934534,   -501588.206183097, 784.4442123743,
                                          194712.394138513,  -377.9645994741, -45088.7863182741, 50.5960513287,     6220.3310639927,
                                          -2.3670776453,     -473.269785487,  15.3213264134,     -3282.8510348085,  164.0438603974,
                                          -5.2778800755,     0.0929379305,    -0.0006826817};

__device__ __constant__ double _b2[17] = {6770.9351457442, -5933.7589419767, -2976.2879986855, 3279.7524685865, 1602.7544794819,
                                          -967.2906583423, -462.0149190219,  159.8317289682,   74.4884405822,   -14.0267151808,
                                          -6.3644917671,   0.5108811613,     0.2252455343,     0.965040193,     -0.0546309127,
                                          0.0012917084,    -1.14383e-05};

__device__ __constant__ double _c2[19] = {39406.5712626297,  -77863.1741143294, 9586.8761567725, 55360.1701572325, -5825.2491611377,
                                          -21887.8400068324, 1751.6803621934,   5158.3764225203, -290.1765894829,  -724.0348081819,
                                          25.364646181,      56.0522105105,     -0.9172073892,   -1.8470088417,    449.4660736502,
                                          -24.4141920625,    0.8465834103,      -0.0158943762,   0.0001235384};

//__device__ __constant__ double floor_prob = 0.1;
__device__ __constant__ double floor_preProb = 2.197224577336219564216435173875652253627777099609375;

__device__ double cal_tnf_dist_d(size_t r1, size_t r2, float *TNF1, float *TNF2) {
    double d = 0.0;
    float tn1, tn2, _diff;
    for (size_t i = 0; i < 136; ++i) {
        tn1 = TNF1[i];
        tn2 = TNF2[i];
        _diff = tn1 - tn2;
        d += _diff * _diff;
    }

    d = sqrt(d);

    double b, c;
    double ctg1 = min((double)r1, (double)500000);
    double ctg2 = min((double)r2, (double)500000);
    double lw[19];
    lw[0] = log10(min(ctg1, ctg2));
    lw[1] = log10(max(ctg1, ctg2));
    lw[2] = lw[0] * lw[0];
    lw[4] = lw[2] * lw[0];
    lw[6] = lw[4] * lw[0];
    lw[8] = lw[6] * lw[0];
    lw[10] = lw[8] * lw[0];
    lw[11] = lw[10] * lw[0];
    lw[3] = lw[1] * lw[1];
    lw[5] = lw[3] * lw[1];
    lw[7] = lw[5] * lw[1];
    lw[9] = lw[7] * lw[1];
    lw[12] = lw[0] * lw[1];
    lw[14] = lw[4] * lw[5];
    lw[15] = lw[6] * lw[7];
    lw[16] = lw[8] * lw[9];
    lw[13] = lw[2] * lw[3];
    lw[18] = lw[9] * lw[1];

    double prob;

    b = _b1[0] + _b1[1] * lw[0] + _b1[2] * lw[1] + _b1[3] * lw[2] + _b1[4] * lw[3] + _b1[5] * lw[4] + _b1[6] * lw[5] + _b1[7] * lw[6] +
        _b1[8] * lw[7] + _b1[9] * lw[8] + _b1[10] * lw[9] + _b1[11] * lw[10] + _b1[12] * lw[11] + _b1[13] * lw[12] + _b1[14] * lw[13] +
        _b1[15] * lw[14] + _b1[16] * lw[15] + _b1[17] * lw[16];

    c = _c1[0] + _c1[1] * lw[0] + _c1[2] * lw[1] + _c1[3] * lw[2] + _c1[4] * lw[3] + _c1[5] * lw[4] + _c1[6] * lw[5] + _c1[7] * lw[6] +
        _c1[8] * lw[7] + _c1[9] * lw[8] + _c1[10] * lw[9] + _c1[11] * lw[10] + _c1[12] * lw[11] + _c1[13] * lw[12] + _c1[14] * lw[13] +
        _c1[15] * lw[14] + _c1[16] * lw[15] + _c1[17] * lw[16];

    double preProb = -(b + c * d);
    prob = preProb <= floor_preProb ? 0.1 : 1.0 / (1 + exp(preProb));

    if (prob >= 0.1) {
        b = _b2[0] + _b2[1] * lw[0] + _b2[2] * lw[1] + _b2[3] * lw[2] + _b2[4] * lw[3] + _b2[5] * lw[4] + _b2[6] * lw[5] +
            _b2[7] * lw[6] + _b2[8] * lw[7] + _b2[9] * lw[8] + _b2[10] * lw[9] + _b2[11] * lw[10] + _b2[12] * lw[18] +
            _b2[13] * lw[13] + _b2[14] * lw[14] + _b2[15] * lw[15] + _b2[16] * lw[16];
        c = _c2[0] + _c2[1] * lw[0] + _c2[2] * lw[1] + _c2[3] * lw[2] + _c2[4] * lw[3] + _c2[5] * lw[4] + _c2[6] * lw[5] +
            _c2[7] * lw[6] + _c2[8] * lw[7] + _c2[9] * lw[8] + _c2[10] * lw[9] + _c2[11] * lw[10] + _c2[12] * lw[18] +
            _c2[13] * lw[11] + _c2[14] * lw[12] + _c2[15] * lw[13] + _c2[16] * lw[14] + _c2[17] * lw[15] + _c2[18] * lw[16];
        preProb = -(b + c * d);
        // de metabat2, será correcto? SÍ xd
        prob = preProb <= floor_preProb ? 1.0 / (1 + exp(preProb)) : 0.1;
        // prob = 1.0 / (1 + exp(-(b + c * d)));
        // prob = prob < floor_prob ? floor_prob : prob;
    }
    return prob;
}

__global__ void get_tnf_prob(double *__restrict__ tnf_dist, float *TNF, size_t *seqs_d_size, size_t _des,
                             const size_t contig_per_thread, const size_t limit) {
    size_t r1;
    size_t r2;
    float _TNF1[136];
    // float _TNF2[136];
    size_t tnf_dist_index = (threadIdx.x + blockIdx.x * blockDim.x) * contig_per_thread;
    size_t prob_index = _des + tnf_dist_index;
    size_t discriminante = 1 + 8 * prob_index;
    r1 = (1 + sqrt((double)discriminante)) / 2;
    r2 = prob_index - r1 * (r1 - 1) / 2;
    float *TNF_r2 = TNF + r2 * 136;
    float *TNF_r1 = TNF + r1 * 136;
    size_t _limit2 = min(tnf_dist_index + contig_per_thread, limit - _des);
    if (tnf_dist_index >= _limit2) return;
    while (tnf_dist_index != _limit2) {
        for (int i = 0; i < 136; i++) {
            _TNF1[i] = TNF_r1[i];
        }
        while (r2 < r1) {
            if (tnf_dist_index == _limit2) break;
            tnf_dist[tnf_dist_index] = cal_tnf_dist_d(seqs_d_size[r1], seqs_d_size[r2], _TNF1, TNF_r2);
            tnf_dist_index++;
            TNF_r2 += 136;
            r2++;
        }
        TNF_r2 = TNF;
        TNF_r1 += 136;
        r2 = 0;
        r1++;
    }
}

__device__ short get_tn(char *__restrict__ contig) {
    unsigned char N;
    short tn = 0;
    for (short i = 0; i < 4; i++) {
        N = BN[contig[i]];
        if (N & 4) return 256;
        tn = (tn << 2) | N;
    }
    return tn;
}

__device__ const char *get_contig_d(int contig_index, const char *seqs_d, const size_t *seqs_d_index) {
    return seqs_d + seqs_d_index[contig_index];
}

__device__ void next_contig(char *__restrict__ contig, char c) {
    for (int i = 0; i < 3; i++) {
        contig[i] = contig[i + 1];
    }
    contig[3] = c;
}

__global__ void get_TNF(float *__restrict__ TNF_d, const char *__restrict__ seqs_d, const size_t *__restrict__ seqs_d_index,
                        const size_t nobs, const size_t contigs_per_thread, const size_t seqs_d_index_size) {
    const size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;
    // char local_contig[4];
    size_t limit = min(thead_id * contigs_per_thread + contigs_per_thread, nobs);
    for (size_t contig_index = thead_id * contigs_per_thread; contig_index < limit; contig_index++) {
        char contig_temp[4] = {0};
        float TNF_temp[136] = {0};
        const size_t tnf_index = contig_index * 136;
        size_t contig_size = seqs_d_index[contig_index + seqs_d_index_size] - seqs_d_index[contig_index];
        const char *contig = seqs_d + seqs_d_index[contig_index];
        if (contig_size < 4) continue;
        for (size_t j = 0; j < 3; j++) next_contig(contig_temp, contig[j]);
        for (size_t j = 3; j < contig_size; ++j) {
            next_contig(contig_temp, contig[j]);
            short tn = get_tn(contig_temp);
            if (tn & 256) continue;
            TNF_temp[TNmap_d[tn]]++;
        }
        double rsum = 0;
        for (int c = 0; c < 136; ++c) {
            rsum += TNF_temp[c] * TNF_temp[c];
        }
        rsum = sqrt(rsum);
        for (int c = 0; c < 136; ++c) {
            TNF_d[tnf_index + c] = TNF_temp[c] / rsum;
        }
        /*
        float TNF_temp[136] = {0};
        const size_t tnf_index = contig_index * 136;
        size_t contig_size = seqs_d_index[contig_index + seqs_d_index_size] - seqs_d_index[contig_index];
        const char *contig = seqs_d + seqs_d_index[contig_index];
        for (size_t j = 0; j < contig_size - 3; ++j) {
            short tn = get_tn(contig + j);
            if (tn & 256) continue;
            TNF_temp[TNmap_d[tn]]++;
        }
        double rsum = 0;
        for (int c = 0; c < 136; ++c) {
            rsum += TNF_temp[c] * TNF_temp[c];
        }
        rsum = sqrt(rsum);
        for (int c = 0; c < 136; ++c) {
            TNF_d[tnf_index + c] = TNF_temp[c] / rsum;
        }
        */
    }
}

typedef double Distance;
typedef double Similarity;

typedef boost::math::normal_distribution<Distance> Normal;
typedef boost::math::poisson_distribution<Distance> Poisson;

typedef boost::property<boost::edge_weight_t, double> Weight;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, boost::no_property, Weight> UndirectedGraph;
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS> DirectedSimpleGraph;
typedef boost::graph_traits<UndirectedGraph>::edge_descriptor edge_descriptor;
typedef boost::graph_traits<UndirectedGraph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<UndirectedGraph>::vertex_descriptor vertex_descriptor;

static std::string version = "Metabat cuda 1.0";
static std::string DATE = "2023-07-14";
static bool verbose = false;
static bool debug = false;
static bool keep = false;
static bool noBinOut = false;
static size_t seedClsSize = 10000;
static size_t minClsSize = 200000;
static size_t minContig = 2500;                // minimum contig size for binning
static size_t minContigByCorr = 1000;          // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000;  // for graph generation purpose
static std::string inFile;
static std::string abdFile;
static bool cvExt;
static std::string pairFile;
static std::string outFile;
static Similarity p1 = 0;
static Similarity p2 = 0;
static Similarity p3 = 95;
static double pB = 50;
static Similarity minProb = 0;
static Similarity minBinned = 0;
static bool verysensitive = false;
static bool sensitive = false;
static bool specific = false;
static bool veryspecific = false;
static bool superspecific = false;
static bool onlyLabel = false;
static int numThreads = 0;
static int numBlocks = 0;
static int numThreads2 = 0;
static int n_STREAMS = 1;
static Distance minCV = 1;
static Distance minCVSum = 2;
static Distance minTimes = 10;
static Distance minCorr = 0;
static size_t minSamples = 10;  // minimum number of sample sizes for considering correlation based recruiting
static bool sumLowCV = false;
static bool fuzzy = false;
static bool useEB = true;  // Ensemble Binning
static Similarity minShared = 0;
static bool saveCls = false;
static bool outUnbinned = false;
static Distance maxVarRatio = 0.0;
static double LOG101 = log(101);

static const char line_delim = '\n';
static const char tab_delim = '\t';
static const char fasta_delim = '>';
static const std::size_t buf_size = 1024 * 1024;
static char os_buffer[buf_size];
static size_t commandline_hash;

#ifdef __APPLE__
vm_statistics_data_t vmStats;
mach_msg_type_number_t infoCount = HOST_VM_INFO_COUNT;
#else
struct sysinfo memInfo;
#endif
double totalPhysMem = 0.;

// Similarity *gprob;
static UndirectedGraph gprob;
static DirectedSimpleGraph paired;
// static DirectedSimpleGraph paired;
static boost::property_map<UndirectedGraph, boost::vertex_index_t>::type gIdx;
static boost::property_map<UndirectedGraph, boost::edge_weight_t>::type gWgt;

static std::unordered_map<std::string_view, size_t> lCtgIdx;  // map of sequence label => local index
static std::unordered_map<size_t, size_t> gCtgIdx;            // local index => global index of contig_names and seqs
static std::unordered_map<std::string_view, size_t> ignored;  // map of sequence label => index of contig_names and seqs
static std::vector<std::string_view> contig_names;
static std::vector<std::string_view> seqs;
static std::vector<size_t> seqs_h_index_i;
static std::vector<size_t> seqs_h_index_e;
static std::vector<size_t> contigs;
static float *TNF_d;

// char *seqs_d;
static size_t ncontigs;
static char *_mem;
static size_t fsize;
static double *tnf_prob[2];
static double *tnf_prob_d[2];
static size_t *seqs_d_size_d;
static size_t max_prob_per_kernel;
static char *seqs_d;
static size_t *seqs_d_index;

typedef std::vector<int> ContigVector;
typedef std::set<int> ClassIdType;  // ordered
typedef std::unordered_set<int> ContigSet;
typedef std::unordered_map<int, ContigVector> ClassMap;

static ContigSet smallCtgs;
static size_t nobs = 0;
static size_t nobs2;  // number of contigs used for binning

static boost::numeric::ublas::matrix<float> ABD;
static boost::numeric::ublas::matrix<float> ABD_VAR;
static boost::numeric::ublas::matrix<float> TNF;
static float *TNF_data;

// typedef boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<float> > MatrixRowType;

typedef std::pair<int, size_t> ClsSizePair;
typedef std::pair<int, Distance> DistancePair;
static std::list<DistancePair> rABD;   // sum of abundance sorted by abundance
static std::list<DistancePair> rABD2;  // backup queue
typedef std::pair<int, size_t> OutDegPair;
static std::list<OutDegPair> oDeg;  // out degree of all vertices

static int B = 0;
static size_t nABD = 0;
static unsigned long long seed = 0;
static std::chrono::steady_clock::time_point t1, t2;

/*
template <class T, class S>
struct pair_equal_to : std::binary_function<T, std::pair<T, S>, bool> {
    bool operator()(const T &y, const std::pair<T, S> &x) const { return x.first == y; }
};
*/

int parseLine(char *line) {
    int i = strlen(line);
    while (*line < '0' || *line > '9') line++;
    line[i - 3] = '\0';
    i = atoi(line);
    return i;
}

int getFreeMem() {
#ifdef __APPLE__
    kern_return_t kernReturn = host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmStats, &infoCount);
    if (kernReturn != KERN_SUCCESS) return 0;
    return (vm_page_size * vmStats.free_count) / 1024;
#else
    FILE *file = fopen("/proc/meminfo", "r");
    size_t result = 0;
    char line[128];

    while (fgets(line, 128, file) != NULL) {
        if (strncmp(line, "MemFree:", 6) == 0 || strncmp(line, "Buffers:", 6) == 0 || strncmp(line, "Cached:", 6) == 0 ||
            strncmp(line, "SwapFree:", 6) == 0) {
            result += parseLine(line);
        }
    }
    fclose(file);
    return result;  // Kb
#endif
}

double getTotalPhysMem() {
    if (totalPhysMem < 1) {
#ifdef __APPLE__
        kern_return_t kernReturn = host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmStats, &infoCount);
        if (kernReturn != KERN_SUCCESS) return 0;
        return (vm_page_size * (vmStats.wire_count + vmStats.active_count + vmStats.inactive_count + vmStats.free_count)) / 1024;
#else
        sysinfo(&memInfo);
        long long _totalPhysMem = memInfo.totalram;
        _totalPhysMem *= memInfo.mem_unit;
        totalPhysMem = (double)_totalPhysMem / 1024;  // kb
#endif
    }
    return totalPhysMem;
}

double getUsedPhysMem() { return (getTotalPhysMem() - getFreeMem()) / 1024. / 1024.; }

static void print_message(const char *format, ...) {
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stdout, format, argptr);
    std::cout.flush();
    va_end(argptr);
}

static void verbose_message(const char *format, ...) {
    if (verbose) {
        t2 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration duration = t2 - t1;
        int elapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(duration).count();  // seconds
        printf("[%02d:%02d:%02d] ", elapsed / 3600, (elapsed % 3600) / 60, elapsed % 60);
        va_list argptr;
        va_start(argptr, format);
        vfprintf(stdout, format, argptr);
        std::cout.flush();
        va_end(argptr);
    }
}

int igraph_community_label_propagation(igraph_t *graph, igraph_node_vector_t *membership, igraph_weight_vector_t *weights) {
    node_t no_of_nodes = igraph_vcount(graph);
    edge_t no_of_edges = igraph_ecount(graph);
    node_t no_of_not_fixed_nodes = no_of_nodes;
    igraph_bool_t running = 1;

    igraph_node_vector_t node_order;

    /* The implementation uses a trick to avoid negative array indexing:
     * elements of the membership vector are increased by 1 at the start
     * of the algorithm; this to allow us to denote unlabeled vertices
     * (if any) by zeroes. The membership vector is shifted back in the end
     */

    /* Do some initial checks */
    if (weights) {
        if (igraph_vector_size(weights) != no_of_edges) {
            std::cerr << "Invalid weight vector length" << std::endl;
            exit(1);
        } else if (igraph_vector_min(weights) < 0) {
            std::cerr << "Weights must be non-negative" << std::endl;
            exit(1);
        }
    }

    verbose_message("Running Ensemble Binning [%.1fGb / %.1fGb]\r", getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);

    RNG_BEGIN();

    /* Initialize node ordering vector with only the not fixed nodes */
    igraph_vector_init(&node_order, no_of_nodes);
    igraph_vector_init(membership, no_of_nodes);

#pragma omp parallel for
    for (node_t i = 0; i < no_of_nodes; ++i) {
        VECTOR(node_order)[i] = i;
        VECTOR(*membership)[i] = i + 1;
    }

    size_t iter = 0;
    running = 1;
    while (running) {
        running = 0;

        /* Shuffle the node ordering vector */
        igraph_vector_shuffle(&node_order);
/* In the prescribed order, loop over the vertices and reassign labels */
#pragma omp parallel for schedule(static, 1)  // non-reproducible
        for (node_t i = 0; i < no_of_not_fixed_nodes; i++) {
            node_t v1 = VECTOR(node_order)[i];

            /* Count the weights corresponding to different labels */
            double max_count = 0.0;

            std::vector<node_t> dominant_labels;
            std::unordered_map<node_t, float> label_counters;

            igraph_edge_vector_t *ineis = &graph->incs[v1];
            for (edge_t j = 0; j < igraph_vector_size(ineis); j++) {                         // # of neighbors
                node_t k = VECTOR(*membership)[IGRAPH_OTHER(graph, VECTOR(*ineis)[j], v1)];  // community membership of a neighbor
                if (k == 0) continue;                                                        /* skip if it has no label yet */
                if (label_counters.find(k) == label_counters.end()) label_counters[k] = 0.;
                label_counters[k] += VECTOR(*weights)[VECTOR(*ineis)[j]];  // sum of neighbors weights to cluster k
                if (max_count < label_counters[k]) {                       // found better community membership
                    max_count = label_counters[k];
                    dominant_labels.resize(1);
                    dominant_labels[0] = k;                   // new potential community membership
                } else if (max_count == label_counters[k]) {  // found equal contender
                    dominant_labels.push_back(k);
                }
            }

            if (dominant_labels.size() > 0) {
                /* Select randomly from the dominant labels */
                node_t k = dominant_labels[RNG_INTEGER(0, dominant_labels.size() - 1)];
                /* Check if the _current_ label of the node is also dominant */
                if (label_counters[VECTOR(*membership)[v1]] != max_count) {
                    /* Nope, we need at least one more iteration */
                    running = 1;
                }
                VECTOR(*membership)[v1] = k;
            }
        }
        verbose_message("Running Ensemble Binning %d [%.1fGb / %.1fGb]\r", ++iter, getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);
    }

    RNG_END();

    igraph_inclist_destroy(graph);
    igraph_vector_destroy(&node_order);

    /* Shift back the membership vector, permute labels in increasing order */

    igraph_vector_t<int_least32_t> label_counters2;
    igraph_vector_init(&label_counters2, no_of_nodes + 1);

    igraph_vector_fill<int_least32_t>(&label_counters2, -1);

    verbose_message("Running Ensemble Binning %d [%.1fGb / %.1fGb]\n", iter, getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);

    node_t j = 0;
    for (node_t i = 0; i < no_of_nodes; i++) {
        int_fast32_t k = (int_fast32_t)VECTOR(*membership)[i] - 1;
        if (k >= 0) {
            if (VECTOR(label_counters2)[k] == -1) {
                /* We have seen this label for the first time */
                VECTOR(label_counters2)[k] = j;
                k = j;
                j++;
            } else {
                k = VECTOR(label_counters2)[k];
            }
        } else {
            /* This is an unlabeled vertex */
        }
        VECTOR(*membership)[i] = k;
    }

    igraph_vector_destroy(&label_counters2);

    return 0;
}

inline float hsum_sse3(__m128 v) {
    __m128 shuf = _mm_movehdup_ps(v);  // broadcast elements 3,1 to 2,0
    __m128 maxs = _mm_add_ps(v, shuf);
    shuf = _mm_movehl_ps(shuf, maxs);  // high half -> low half
    maxs = _mm_add_ss(maxs, shuf);
    return _mm_cvtss_f32(maxs);
}

inline float hsum_avx(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);    // low 128
    __m128 hi = _mm256_extractf128_ps(v, 1);  // high 128
    lo = _mm_add_ps(lo, hi);                  // max the low 128
    return hsum_sse3(lo);                     // and inline the sse3 version
}

Distance cal_tnf_dist(size_t r1, size_t r2) {
    double d = 0;
    for (int i = 0; i < 136; i++) {
        d += (TNF(r1, i) - TNF(r2, i)) * (TNF(r1, i) - TNF(r2, i));  // euclidean distance
    }

    d = sqrt(d);
    Distance b, c;  // parameters
    size_t ctg1 = std::min(seqs[gCtgIdx[r1]].size(), (size_t)500000);
    size_t ctg2 = std::min(seqs[gCtgIdx[r2]].size(), (size_t)500000);
    Distance lw[19];
    lw[0] = std::log10(std::min(ctg1, ctg2));
    lw[1] = std::log10(std::max(ctg1, ctg2));
    lw[2] = lw[0] * lw[0];
    lw[4] = lw[2] * lw[0];
    lw[6] = lw[4] * lw[0];
    lw[8] = lw[6] * lw[0];
    lw[10] = lw[8] * lw[0];
    lw[11] = lw[10] * lw[0];
    lw[3] = lw[1] * lw[1];
    lw[5] = lw[3] * lw[1];
    lw[7] = lw[5] * lw[1];
    lw[9] = lw[7] * lw[1];
    lw[12] = lw[0] * lw[1];
    lw[14] = lw[4] * lw[5];
    lw[15] = lw[6] * lw[7];
    lw[16] = lw[8] * lw[9];
    lw[13] = lw[2] * lw[3];
    lw[18] = lw[9] * lw[1];
    Distance prob;

    b = 46349.1624324381 + -76092.3748553155 * lw[0] + -639.918334183 * lw[1] + 53873.3933743949 * lw[2] + -156.6547554844 * lw[3] +
        -21263.6010657275 * lw[4] + 64.7719132839 * lw[5] + 5003.2646455284 * lw[6] + -8.5014386744 * lw[7] + -700.5825500292 * lw[8] +
        0.3968284526 * lw[9] + 54.037542743 * lw[10] + -1.7713972342 * lw[11] + 474.0850141891 * lw[12] + -23.966597785 * lw[13] +
        0.7800219061 * lw[14] + -0.0138723693 * lw[15] + 0.0001027543 * lw[16];

    c = -443565.465710869 + 718862.10804858 * lw[0] + 5114.1630934534 * lw[1] + -501588.206183097 * lw[2] + 784.4442123743 * lw[3] +
        194712.394138513 * lw[4] + -377.9645994741 * lw[5] + -45088.7863182741 * lw[6] + 50.5960513287 * lw[7] +
        6220.3310639927 * lw[8] + -2.3670776453 * lw[9] + -473.269785487 * lw[10] + 15.3213264134 * lw[11] +
        -3282.8510348085 * lw[12] + 164.0438603974 * lw[13] + -5.2778800755 * lw[14] + 0.0929379305 * lw[15] + -0.0006826817 * lw[16];
    // logistic model
    double preProb = -(b + c * d);
    prob = preProb <= 2.197224577336219564216435173875652253627777099609375 ? 0.1 : 1.0 / (1 + exp(preProb));
    // prob = 1.0 / (1 + exp(-(b + c * d)));
    if (prob >= .1) {  // second logistic model
        b = 6770.9351457442 + -5933.7589419767 * lw[0] + -2976.2879986855 * lw[1] + 3279.7524685865 * lw[2] + 1602.7544794819 * lw[3] +
            -967.2906583423 * lw[4] + -462.0149190219 * lw[5] + 159.8317289682 * lw[6] + 74.4884405822 * lw[7] +
            -14.0267151808 * lw[8] + -6.3644917671 * lw[9] + 0.5108811613 * lw[10] + 0.2252455343 * lw[18] + 0.965040193 * lw[13] +
            -0.0546309127 * lw[14] + 0.0012917084 * lw[15] + -1.14383e-05 * lw[16];
        c = 39406.5712626297 + -77863.1741143294 * lw[0] + 9586.8761567725 * lw[1] + 55360.1701572325 * lw[2] +
            -5825.2491611377 * lw[3] + -21887.8400068324 * lw[4] + 1751.6803621934 * lw[5] + 5158.3764225203 * lw[6] +
            -290.1765894829 * lw[7] + -724.0348081819 * lw[8] + 25.364646181 * lw[9] + 56.0522105105 * lw[10] +
            -0.9172073892 * lw[18] + -1.8470088417 * lw[11] + 449.4660736502 * lw[12] + -24.4141920625 * lw[13] +
            0.8465834103 * lw[14] + -0.0158943762 * lw[15] + 0.0001235384 * lw[16];
        prob = 1.0 / (1 + exp(-(b + c * d)));
        prob = prob < .1 ? .1 : prob;
    }
    return prob;
}

Distance cal_abd_dist2(Normal &p1, Normal &p2) {
    Distance k1, k2, tmp, d = 0;
    Distance m1 = p1.mean();
    Distance m2 = p2.mean();
    Distance v1 = p1.standard_deviation();
    v1 = v1 * v1;
    Distance v2 = p2.standard_deviation();
    v2 = v2 * v2;

    // normal_distribution
    if (std::fabs(v2 - v1) < 1e-4) {
        k1 = k2 = (m1 + m2) / 2;
    } else {
        tmp = std::sqrt(v1 * v2 * ((m1 - m2) * (m1 - m2) - 2 * (v1 - v2) * std::log(std::sqrt(v2 / v1))));
        k1 = (tmp - m1 * v2 + m2 * v1) / (v1 - v2);
        k2 = (tmp + m1 * v2 - m2 * v1) / (v2 - v1);
    }

    if (k1 > k2) {
        tmp = k1;
        k1 = k2;
        k2 = tmp;
    }
    if (v1 > v2) {
        std::swap(p1, p2);
    }

    if (k1 == k2)
        d += std::log(std::fabs(boost::math::cdf(p1, k1) - boost::math::cdf(p2, k1)));
    else
        d += std::log(
            std::fabs(boost::math::cdf(p1, k2) - boost::math::cdf(p1, k1) + boost::math::cdf(p2, k1) - boost::math::cdf(p2, k2)));

    return d;
}

// for Poisson distributions
Distance cal_abd_dist2(Poisson &p1, Poisson &p2) {
    Distance k, m1, m2;
    m1 = p1.mean();
    m2 = p2.mean();
    k = (m1 - m2) / (log(m1) - log(m2));
    return log(fabs(boost::math::cdf(p1, k) - boost::math::cdf(p2, k)));
}

Distance cal_abd_dist(size_t r1, size_t r2, int &nnz) {
    Distance d = 0;
    int nns = 0;
    assert(r1 < nobs && r2 < nobs);
    Distance m1sum = 0, m2sum = 0;
    //	Distance v1sum = 0, v2sum = 0;
    for (size_t i = 0; i < nABD; ++i) {
        Distance m1 = ABD(r1, i);
        Distance m2 = ABD(r2, i);
        if (m1 > minCV || m2 > minCV) {  // compare only at least one >2
            ++nnz;
            m1 = std::max(m1, (Distance)1e-6);
            m2 = std::max(m2, (Distance)1e-6);
            if (m1 == m2) {
                ++nns;
                continue;
            }
            Distance v1 = ABD_VAR(r1, i) < 1 ? 1 : ABD_VAR(r1, i);
            Distance v2 = ABD_VAR(r2, i) < 1 ? 1 : ABD_VAR(r2, i);

            Normal p1(m1, sqrt(v1)), p2(m2, std::sqrt(v2));
            d += cal_abd_dist2(p1, p2);
        } else {
            m1sum += m1;
            m2sum += m2;
            //			v1sum += ABD_VAR(r1,i);
            //			v2sum += ABD_VAR(r2,i);
        }
    }

    if (sumLowCV && (m1sum > minCV || m2sum > minCV)) {
        if (std::fabs(m1sum - m2sum) > 1e-3) {
            // now include the sum of all samples that failed the minCV test
            m1sum = std::max(m1sum, (Distance)1e-6);
            m2sum = std::max(m2sum, (Distance)1e-6);
            Poisson p1(m1sum), p2(m2sum);
            // Normal p1(m1sum, SQRT(v1sum)), p2(m2sum, SQRT(v2sum));
            d += cal_abd_dist2(p1, p2);
        }  // else they are the same distribution, so d += 0
        ++nnz;
    } else if (nnz == 0) {
        // both samples are very low abundance, use TNF
        return 1;
    }
    if (nns == (int)nABD)  // the same
        return 0;
    else
        return std::pow(std::exp(d), 1.0 / nnz);
}

// maxDist: maximum distance for further calculation (to avoid unnecessary calculation)
Distance cal_dist(size_t r1, size_t r2, Distance maxDist, bool &passed) {
    assert(smallCtgs.find(r1) == smallCtgs.end());
    assert(smallCtgs.find(r2) == smallCtgs.end());
    Distance abd_dist = 0, tnf_dist = 0;
    int nnz = 0;
    if (r1 == r2) return 0;
    tnf_dist = cal_tnf_dist(r1, r2);
    if (!passed && tnf_dist > maxDist) {
        return 1;
    }
    if (abdFile.length() > 0) abd_dist = cal_abd_dist(r1, r2, nnz);
    passed = true;
    if (tnf_dist > 0.05) {  // minimum cutoff for considering abd
        return std::max(tnf_dist, abd_dist * 0.9);
    } else {
        Distance w = 0;
        if (nnz > 0) w = std::min(std::log(nnz + 1) / LOG101, 0.9);  // progressive weight depending on sample sizes
        return abd_dist * w + tnf_dist * (1 - w);
    }
}

Distance cal_dist(size_t r1, size_t r2) {
    Distance maxDist = 1;
    bool passed = true;
    return cal_dist(r1, r2, maxDist, passed);
}

Distance cal_dist2(size_t r1, size_t r2, Distance maxDist, bool &passed, Distance tnf_dist) {
    Distance abd_dist = 0;
    int nnz = 0;
    if (r1 == r2) return 0;

    if (!passed && tnf_dist > maxDist) {
        return 1;
    }
    if (abdFile.length() > 0) abd_dist = cal_abd_dist(r1, r2, nnz);
    passed = true;
    if (tnf_dist > 0.05) {  // minimum cutoff for considering abd
        return std::max(tnf_dist, abd_dist * 0.9);
    } else {
        Distance w = 0;
        if (nnz > 0) w = std::min(std::log(nnz + 1) / LOG101, 0.9);  // progressive weight depending on sample sizes
        return abd_dist * w + tnf_dist * (1 - w);
    }
}

static Similarity get_prob(size_t r1, size_t r2) {
    if (r1 == r2) return 1;
    edge_descriptor e;
    bool found;
    boost::tie(e, found) = boost::edge(r1, r2, gprob);
    return found ? boost::get(gWgt, e) : 1 - cal_dist(r1, r2);
}

static bool cmp_cls_size(const ClsSizePair &i, const ClsSizePair &j) {
    return j.second == i.second ? j.first > i.first : j.second > i.second;  // increasing
}

static bool cmp_abd(const DistancePair &i, const DistancePair &j) {
    return j.second < i.second;  // decreasing
}

Distance cal_abd_corr(size_t r1, size_t r2) {
    size_t i, ii;
    double sum_xsq = 0.0;
    double sum_ysq = 0.0;
    double sum_cross = 0.0;
    double ratio;
    double delta_x, delta_y;
    double mean_x = 0.0, mean_y = 0.0;
    double r = 0.0;
    size_t s = 0;  // skipped
    for (i = 0; i < nABD; ++i) {
        Distance m1 = ABD(r1, i);
        Distance m2 = ABD(r2, i);
        ii = i - s;
        if (ii == 0) {
            mean_x = m1;
            mean_y = m2;
            continue;
        }
        ratio = ii / (ii + 1.0);
        delta_x = m1 - mean_x;
        delta_y = m2 - mean_y;
        sum_xsq += delta_x * delta_x * ratio;
        sum_ysq += delta_y * delta_y * ratio;
        sum_cross += delta_x * delta_y * ratio;
        mean_x += delta_x / (ii + 1.0);
        mean_y += delta_y / (ii + 1.0);
    }
    r = sum_cross / (sqrt(sum_xsq) * sqrt(sum_ysq));
    if (nABD - s < minSamples) {
        return 0;
    }
    return r;
}

bool loadENSFromFile(igraph_t &g, igraph_weight_vector_t &weights) {
    if (true) return false;  // TODO need to handle g.incs

    std::string saveENSFile = "ens." + std::to_string(commandline_hash);

    std::ifstream is(saveENSFile);
    if (is.good()) {
        verbose_message("Loading ensemble intermediate file from %s\n", saveENSFile.c_str());
        try {
            boost::archive::binary_iarchive ia(is);

            edge_t num_edges;
            ia >> num_edges;

            igraph_vector_resize(&weights, num_edges);
            igraph_vector_resize(&g.from, num_edges);
            igraph_vector_resize(&g.to, num_edges);

            ia >> boost::serialization::make_array(weights.stor_begin, num_edges);
            ia >> boost::serialization::make_array(g.from.stor_begin, num_edges);
            ia >> boost::serialization::make_array(g.to.stor_begin, num_edges);

        } catch (...) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}

bool loadDistanceFromFile(std::string saveDistanceFile, Distance requiredMinP, size_t requiredMinContig) {
    if (saveDistanceFile.empty()) return false;
    std::ifstream is(saveDistanceFile.c_str());
    if (is.good()) {
        verbose_message("Loading saved graph from %s\n", saveDistanceFile.c_str());
        try {
            boost::archive::binary_iarchive ia(is);
            Distance loadedMinP;
            ia >> loadedMinP;
            if (loadedMinP > requiredMinP) {
                std::cerr << "[Warning!] Saved probability graph file has greater minP " << loadedMinP << " vs required "
                          << requiredMinP << ". Recalculating..." << std::endl;
                return false;
            }
            size_t loadedMinContig;
            ia >> loadedMinContig;
            if (loadedMinContig != requiredMinContig) {
                std::cerr << "[Warning!] Saved probability graph file has different minContig " << loadedMinContig << " vs required "
                          << requiredMinContig << ". Recalculating..." << std::endl;
                return false;
            }
            ia >> gprob;

            if (boost::num_vertices(gprob) != nobs) {
                std::cerr << "[Warning!] Saved probability graph file has different number of contigs " << boost::num_vertices(gprob)
                          << " vs required " << nobs << ". Recalculating..." << std::endl;
                return false;
            }
        } catch (...) {
            std::cerr << "[Warning!] A exception occurred. Saved graph file was possibly generated from different version of boost "
                         "library. Recalculating..."
                      << std::endl;
            return false;
        }
    } else {
        return false;
    }
    return true;
}

bool loadBootFromFile(boost::numeric::ublas::matrix<size_t> &boot) {
    std::string saveBootFile = "boot." + std::to_string(commandline_hash);

    std::ifstream is(saveBootFile);
    if (is.good()) {
        verbose_message("Loading bootstrap intermediate file from %s\n", saveBootFile.c_str());
        try {
            boost::archive::binary_iarchive ia(is);

            ia >> boot;

        } catch (...) {
            return false;
        }
    } else {
        return false;
    }

    return true;
}
/*
bool loadTNFFromFile(std::string saveTNFFile, size_t requiredMinContig) {
    if (saveTNFFile.empty()) return false;
    FILE *fp = fopen(saveTNFFile.c_str(), "r");
    if (fp == NULL) return false;
    fseek(fp, 0L, SEEK_END);
    size_t fsize = ftell(fp);  // obtener el tamaño del archivo
    fclose(fp);
    fsize = (fsize / sizeof(float)) - 2;  // el primer valor es el minContig
    if ((fsize / 136) != nobs) {
        std::cerr << "[Warning!] Saved TNF file was not generated from the same data. It should have " << nobs << " contigs, but have "
                  << (fsize / 136) << std::endl;
        return false;
    }
    size_t loadedMinContig = 0;
    int fpint = open(saveTNFFile.c_str(), O_RDWR | O_CREAT, S_IREAD | S_IWRITE | S_IRGRP | S_IROTH);
    size_t ok = pread(fpint, (void *)&loadedMinContig, 8, 0);
    if (ok != 8) {
        std::cerr << "[Warning!] A exception occurred."
                     "Recalculating..."
                  << std::endl;
        return false;
    }
    if (loadedMinContig != requiredMinContig) {
        std::cerr << "[Warning!] Saved TNF file has different minContig " << loadedMinContig << " vs required " << requiredMinContig
                  << ". Recalculating..." << std::endl;
        return false;
    }
    fsize *= sizeof(float);
    ok = pread(fpint, (void *)TNF, fsize, 8);
    if (ok != fsize) {
        std::cerr << "[Warning!] A exception occurred."
                     "Recalculating..."
                  << std::endl;
        return false;
    }
    close(fpint);
    return true;
}*/

bool loadTNFFromFile(std::string saveTNFFile, size_t requiredMinContig) {
    if (saveTNFFile.empty()) return false;
    assert(TNF.size1() == 0);

    std::ifstream is(saveTNFFile.c_str());
    if (is.good()) {
        verbose_message("Loading saved TNF from %s\n", saveTNFFile.c_str());
        try {
            boost::archive::binary_iarchive ia(is);

            size_t loadedMinContig;
            ia >> loadedMinContig;
            if (loadedMinContig != requiredMinContig) {
                std::cerr << "[Warning!] Saved TNF file has different minContig " << loadedMinContig << " vs required "
                          << requiredMinContig << ". Recalculating..." << std::endl;
                return false;
            }

            ia >> TNF;
            if (TNF.size1() != nobs) {
                std::cerr << "[Warning!] Saved TNF file was not generated from the same data. It should have " << nobs
                          << " contigs, but have " << TNF.size1() << std::endl;
                return false;
            }
        } catch (...) {
            std::cerr << "[Warning!] A exception occurred. Saved TNF file was possibly generated from different version of boost "
                         "library. Recalculating..."
                      << std::endl;
            return false;
        }
    } else {
        return false;
    }
    for (size_t i = 0; i < ncontigs; i++) {
        for (int j = 0; j < 136; j++) {
            TNF_data[i * 136 + j] = TNF(contigs[i], j);
        }
    }
    cudaMemcpy(TNF_d, TNF_data, ncontigs * 136 * sizeof(float), cudaMemcpyHostToDevice);
    return true;
}

/*
void saveTNFToFile(std::string saveTNFFile, size_t requiredMinContig) {
    if (saveTNFFile.empty()) return;
    std::ofstream out(saveTNFFile.c_str(), std::ios::out | std::ios::binary);
    if (out) {
        if (0) {  // quitar small contigs de TNF
            for (auto it = smallCtgs.begin(); it != smallCtgs.end(); it++) {
                for (size_t i = 0; i < 136; i++) {
                    TNF[*it * 136 + i] = 0;
                }
            }
        }
        out.write((char *)&minContig, sizeof(size_t));
        out.write((char *)TNF, nobs * 136 * sizeof(float));
        out.close();
    } else {
        std::cout << "Error al guardar en TNF.bin" << std::endl;
    }
    out.close();
}
*/

void saveTNFToFile(std::string saveTNFFile, size_t requiredMinContig) {
    if (saveTNFFile.empty()) return;
    std::ofstream os(saveTNFFile.c_str());
    if (os.good()) {
        verbose_message("Saving TNF file to %s                                    \n", saveTNFFile.c_str());
        boost::archive::binary_oarchive oa(os);
        oa << requiredMinContig;
        oa << TNF;
    }
}

void saveDistanceToFile(std::string saveDistanceFile, Distance requiredMinP, size_t requiredMinContig) {
    if (saveDistanceFile.empty()) return;

    std::ofstream os(saveDistanceFile.c_str());
    if (os.good()) {
        verbose_message("Saving probability graph to %s                                                     \n",
                        saveDistanceFile.c_str());
        boost::archive::binary_oarchive oa(os);
        oa << requiredMinP;
        oa << requiredMinContig;
        oa << gprob;
    }
}

void saveBootToFile(boost::numeric::ublas::matrix<size_t> &boot) {
    std::ofstream oos("boot." + std::to_string(commandline_hash));
    if (oos.good()) {
        boost::archive::binary_oarchive ooa(oos);
        ooa << boot;
        verbose_message("Saved bootstrap intermediate file (boot.%zu) for reuse in case of failure\n", commandline_hash);
    }
    oos.close();
}

void fish_objects(int m, ContigSet &mems, Similarity p1, Similarity p2, ContigVector &medoid_ids,
                  ContigSet &binned) {  // fish (assign) objects to medoid m.
    if (debug) {
        std::cout << "---------------------" << std::endl;
        std::cout << "medoid: " << medoid_ids[m] << " with non-zero friends: " << boost::out_degree(medoid_ids[m], gprob) << std::endl;
    }

    mems.insert(medoid_ids[m]);

    out_edge_iterator e, e_end;
    vertex_descriptor v = boost::vertex(medoid_ids[m], gprob);

    int maxFriends = boost::out_degree(v, gprob);
    if (maxFriends == 0) return;

    boost::tie(e, e_end) = boost::out_edges(v, gprob);

// find all friends of medoid >= p1
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < maxFriends; ++i) {
        out_edge_iterator ee = e + i;
        Similarity p = boost::get(gWgt, *ee);
        if (p >= p1) {
            int f = boost::get(gIdx, boost::target(*ee, gprob));
#pragma omp critical(FISH_OBJECTS_ADD_TO_CLUSTER)
            {
                if (binned.find(f) == binned.end()) {  // add only if it is fuzzy binning or f is still unbinned
                    mems.insert(f);
                }
            }
        }
    }

    ContigSet newbies;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < mems.size(); ++i) {
        ContigSet::iterator it = mems.begin();
        std::advance(it, i);
        if (*it == medoid_ids[m]) continue;
        vertex_descriptor v = boost::vertex(*it, gprob);
        out_edge_iterator e2, e_end2;
        for (boost::tie(e2, e_end2) = boost::out_edges(v, gprob); e2 != e_end2; ++e2) {
            Similarity pp = boost::get(gWgt, *e2);
            if (pp >= p2) {
                int ff = boost::get(gIdx, boost::target(*e2, gprob));
#pragma omp critical(FISH_OBJECTS_ADD_TO_CLUSTER_2)
                {
                    if (binned.find(ff) == binned.end()) {  // add only if it is fuzzy binning or ff is still unbinned
                        newbies.insert(ff);
                    }
                }
            }
        }
    }
    mems.insert(newbies.begin(), newbies.end());
    if (debug) {
        std::cout << "cls[m].size(): " << mems.size() << std::endl;
        for (ContigSet::iterator it = mems.begin(); it != mems.end(); ++it) {
            std::cout << *it << ", ";
        }
        std::cout << std::endl << "---------------------" << std::endl;
    }
    return;
}

void init_medoids_by_ABD(size_t k, ContigVector &medoid_ids, std::vector<double> &medoid_vals, ContigSet &binned) {
    std::list<DistancePair>::iterator it = rABD.begin();
    while (it != rABD.end()) {
        if (binned.find(it->first) == binned.end()) {
            medoid_ids[k] = it->first;
            medoid_vals[k] = it->second;
            if (debug)
                std::cout << "Selected medoid[" << k << "]: " << medoid_ids[k] << ", contig id: " << it->first << " with abundance "
                          << it->second << std::endl;
            break;  // no ++it;
        } else {
            if (useEB) {
                it->second = rand();
                rABD2.push_back(*it);
            }
            it = rABD.erase(it);
        }
    }
}

void pam_loop(int i, ContigVector &medoid_ids, std::vector<double> &medoid_vals, ContigSet &binned, ClassMap &cls) {
    init_medoids_by_ABD(i, medoid_ids, medoid_vals, binned);

    int updates = 0;
    bool updated = true;
    ContigSet medoid_prevs;
    ContigSet mems;

    while (updated) {
        updated = false;
        updates++;
        ContigSet _mems;
        fish_objects(i, _mems, p1, p2, medoid_ids, binned);

        std::vector<DistancePair> ssum(_mems.size());

#pragma omp parallel for schedule(dynamic)
        for (size_t j = 0; j < _mems.size(); ++j) {
            ContigSet::iterator it = _mems.begin();
            std::advance(it, j);
            DistancePair s(*it, 0);
            for (ContigSet::iterator it2 = _mems.begin(); it2 != _mems.end(); ++it2) {
                if (*it != *it2) s.second += get_prob(*it, *it2);
            }
            ssum[j] = s;
        }

        sort(ssum.begin(), ssum.end(), cmp_abd);

        mems = _mems;

        if (medoid_prevs.find(ssum[0].first) == medoid_prevs.end()) {  // preventing a loop!
            if (ssum[0].first != medoid_ids[i])                        // medoid is updated
                updated = true;

            medoid_ids[i] = ssum[0].first;
            medoid_prevs.insert(medoid_ids[i]);
        }
    }

    cls[i].insert(cls[i].end(), mems.begin(), mems.end());
    binned.insert(mems.begin(), mems.end());
    if (updates > 1)
        medoid_vals[i] =
            std::find_if(rABD.begin(), rABD.end(), [i = medoid_ids[i]](const DistancePair &dp) { return dp.first == i; })->second;

    if (debug)
        std::cout << "medoid[" << i << "]: " << medoid_ids[i] << " updates: " << updates << " size: " << cls[i].size() << std::endl;
}

int pam(ContigVector &medoid_ids, std::vector<double> &medoid_vals, ContigSet &binned, ClassMap &cls, ContigSet &leftovers,
        ClassIdType &good_class_ids) {
    ContigVector empty;
    int goodClusters = 0;

    ProgressTracker progress(nobs - binned.size());

    while (nobs != binned.size()) {
        medoid_ids.push_back(0);
        medoid_vals.push_back(0);

        size_t kk = medoid_ids.size() - 1;
        cls[kk] = empty;

        pam_loop(kk, medoid_ids, medoid_vals, binned, cls);

        size_t cls_size = 0;
        bool isGood = false;
        for (ContigVector::iterator it = cls[kk].begin(); it != cls[kk].end(); ++it) {
            cls_size += seqs[gCtgIdx[*it]].size();
            if (cls_size >= seedClsSize) {
                isGood = true;
                break;
            }
        }

        progress.setProgress(binned.size());
        if (!useEB && progress.isStepMarker()) {
            verbose_message("1st round binning %s\r", progress.getProgress());
        }

        if (isGood && cls[kk].size() > 2) {
            goodClusters++;
            good_class_ids.insert(kk);
        } else {
            if (cls[kk].size() > 1 || seqs[gCtgIdx[cls[kk][0]]].size() >=
                                          minContigByCorr)  // keep leftovers only if it is at least valid for corr recruiting
                leftovers.insert(cls[kk].begin(), cls[kk].end());
        }
    }

    progress.setProgress(binned.size());
    if (!useEB) verbose_message("1st round binning %s\n", progress.getProgress());

    return goodClusters;
}

static void trim_fasta_label(std::string &label) {
    size_t pos = label.find_first_of(" \t");
    if (pos != std::string::npos) label = label.substr(0, pos);
}

std::istream &safeGetline(std::istream &is, std::string &t) {
    t.clear();

    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.

    std::istream::sentry se(is, true);
    std::streambuf *sb = is.rdbuf();

    for (;;) {
        int c = sb->sbumpc();
        switch (c) {
            case '\n':
                return is;
            case '\r':
                if (sb->sgetc() == '\n') sb->sbumpc();
                return is;
            case EOF:
                // Also handle the case when the last line has no line ending
                if (t.empty()) is.setstate(std::ios::eofbit);
                return is;
            default:
                t += (char)c;
        }
    }
}

size_t countLines(const char *f) {
    size_t lines = 0;
    FILE *pFile;
    pFile = fopen(f, "r");
    if (pFile == NULL) {
        std::cerr << "[Error!] can't open input file " << f << std::endl;
        return 0;
    }
    while (EOF != fscanf(pFile, "%*[^\n]") && EOF != fscanf(pFile, "%*c")) ++lines;
    fclose(pFile);
    return lines;
}

size_t ncols(std::ifstream &is, int skip = 0) {
    size_t nc = 0;
    std::string firstLine;
    while (skip-- >= 0) std::getline(is, firstLine);
    std::stringstream ss(firstLine);
    std::string col;
    while (std::getline(ss, col, tab_delim)) ++nc;
    return nc;
}

size_t ncols(const char *f, int skip = 0) {
    std::ifstream is(f);
    if (!is.is_open()) {
        std::cerr << "[Error!] can't open input file " << f << std::endl;
        return 0;
    }

    return ncols(is, skip);
}

void gen_commandline_hash() {
    std::ostringstream oss;

    oss << B;
    oss << inFile << abdFile << cvExt << pairFile << p1 << p2 << p3 << minProb << minBinned;
    oss << minCorr << minSamples << minCV << minCVSum << minContig << minContigByCorr;
    oss << minShared << fuzzy << sumLowCV << maxVarRatio;

    std::string commandline = oss.str();
    std::hash<std::string> str_hash;
    commandline_hash = str_hash(commandline);
    // cout << commandline_hash << endl;
}

size_t fish_more_by_corr(ContigVector &medoid_ids, ClassMap &cls, ContigSet &leftovers, ClassIdType &good_class_ids) {
    double max_size = std::log10(100000);
    double min_size = std::log10(minContigByCorr);

    ContigVector leftovers2(leftovers.begin(), leftovers.end());
    std::sort(leftovers2.begin(), leftovers2.end());
    size_t fished = 0;

    ProgressTracker progress = ProgressTracker(leftovers2.size());

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < leftovers2.size(); ++i) {
        double max_corr = 0.;
        size_t which_max_corr = 0;

        for (ClassIdType::const_iterator it2 = good_class_ids.begin(); it2 != good_class_ids.end(); ++it2) {
            //			if (smallCtgs.find(*it) == smallCtgs.end() && cal_tnf_dist(medoid_ids[*it2], *it) > 0.2)
            //				continue;
            double corr = cal_abd_corr(medoid_ids[*it2], leftovers2[i]);
            if (corr > max_corr) {  // recruiting for large bins (>=20) cls[*it2].size() >= 200; corr >= (99. -
                                    // std::max(LOG10(cls[*it2].size()) - 1., 0.) * 5.)/100.
                max_corr = corr;
                which_max_corr = *it2;
            }
        }

        // 1000=>90, 100000=>99
        if (max_corr >= minCorr / 100.) {  // smallCtgs.find(*it) == smallCtgs.end() ? minCorr * 1.05 : minCorr
            double cutCorr = ((99. - minCorr) / (max_size - min_size) * std::log10(seqs[gCtgIdx[leftovers2[i]]].size()) +
                              (max_size * minCorr - min_size * 99.) / (max_size - min_size)) /
                             100.;
            if (max_corr >= std::min(cutCorr, .99)) {
#pragma omp critical(FISH_MORE_BY_CORR)
                {
                    ++fished;
                    cls[which_max_corr].push_back(leftovers2[i]);
                }
            }
        }

        if (!useEB) {
            progress.track();
            if (omp_get_thread_num() == 0 && progress.isStepMarker()) {
                verbose_message("fish_more_by_corr: %s\r", progress.getProgress());
            }
        }
    }

    return fished;
}

void fish_more(int m, ClassMap &cls, ContigSet &leftovers) {
    ContigSet newbies;

#pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < cls[m].size(); ++i) {
        out_edge_iterator e, e_end;
        vertex_descriptor v = boost::vertex(cls[m][i], gprob);
        for (boost::tie(e, e_end) = boost::out_edges(v, gprob); e != e_end; ++e) {
            if (boost::get(gWgt, *e) >= p3) {
                int ff = boost::get(gIdx, boost::target(*e, gprob));
                if (leftovers.find(ff) != leftovers.end()) {  // add only if it is fuzzy binning or fff is still unbinned
#pragma omp critical(FISH_MORE)
                    {
                        newbies.insert(ff);
                        //						std::cout << "new friends: " << v << " -> " << ff << " with "
                        //<< boost::get(gWgt, *e) << endl;
                    }
                }
            }
        }
    }

    for (ContigSet::iterator it = newbies.begin(); it != newbies.end(); ++it) {
        leftovers.erase(*it);
    }

    cls[m].insert(cls[m].end(), newbies.begin(), newbies.end());
}

void fish_more_by_friends_membership(ClassMap &cls, ContigSet &leftovers, ClassIdType &good_class_ids) {
    // profile distribution of friends and assign isolates to a bin using majority vote

    std::vector<int> clsMap(nobs, -1);

    ContigVector good_class_ids2(good_class_ids.begin(), good_class_ids.end());

#pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < good_class_ids2.size(); ++k) {
        for (size_t i = 0; i < cls[good_class_ids2[k]].size(); ++i) {
            assert(cls[good_class_ids2[k]][i] < (int)clsMap.size());
            clsMap[cls[good_class_ids2[k]][i]] = good_class_ids2[k];
        }
    }

    bool updated = true;

    while (updated) {
        updated = false;

        ContigVector newbies;
        ContigVector leftovers2(leftovers.begin(), leftovers.end());
        std::sort(leftovers2.begin(), leftovers2.end());

        for (size_t i = 0; i < leftovers2.size(); ++i) {
            int vid = leftovers2[i];
            out_edge_iterator e, e_end;
            vertex_descriptor v = boost::vertex(vid, gprob);

            boost::tie(e, e_end) = boost::out_edges(v, gprob);
            if (e == e_end) continue;

            std::unordered_map<int, int> summary;
            int _binned = 0;
            int maxFriends = 0;

            for (size_t j = 0; j < boost::out_degree(v, gprob); ++j) {
                out_edge_iterator ee = e + j;

                Similarity p = boost::get(gWgt, *ee);
                int f = boost::get(gIdx, boost::target(*ee, gprob));

                if (p >= minProb) {
                    ++maxFriends;
                    if (clsMap[f] >= 0) {  // count only binned contigs
                        ++_binned;
                        summary[clsMap[f]]++;
                    }
                }
            }

            for (std::unordered_map<int, int>::const_iterator it2 = summary.cbegin(); it2 != summary.cend(); ++it2) {
                if (_binned > maxFriends * minBinned && it2->second > _binned / 2) {  //   //majority
                    // cout << "Total Friends: " << maxFriends << ", Binned: " << total << ", Majority: " << it2->second << endl;
                    cls[it2->first].push_back(vid);
                    newbies.push_back(vid);
                    clsMap[vid] = it2->first;
                    updated = true;
                    break;
                }
            }
        }

        for (size_t i = 0; i < newbies.size(); ++i) {
            leftovers.erase(newbies[i]);
        }
    }
}

void fish_pairs(ContigSet &binned, ClassMap &cls, ClassIdType &good_class_ids) {
    binned.clear();

    for (ClassIdType::const_iterator it = good_class_ids.begin(); it != good_class_ids.end(); ++it) {
        ContigVector &clsV = cls[*it];

        // convert to global index
        for (size_t m = 0; m < clsV.size(); ++m) {
            clsV[m] = gCtgIdx[clsV[m]];
            binned.insert(clsV[m]);
        }
    }

    // for each cls
    // grab any reciprocal pairs
    boost::property_map<DirectedSimpleGraph, boost::vertex_index_t>::type gsIdx = boost::get(boost::vertex_index, paired);

    ContigVector good_class_ids2(good_class_ids.begin(), good_class_ids.end());

#pragma omp parallel for schedule(dynamic)
    for (size_t k = 0; k < good_class_ids2.size(); ++k) {
        ContigVector &clsV = cls[good_class_ids2[k]];
        ContigSet clsS(clsV.begin(), clsV.end());
        assert(clsV.size() == clsS.size());

        boost::graph_traits<DirectedSimpleGraph>::out_edge_iterator e, ee, e_end, ee_end;

        bool updated = true;

        while (updated) {
            updated = false;

            ContigSet newbies;

            // grab any reciprocal pairs
            for (size_t m = 0; m < clsV.size(); ++m) {
                size_t idx = clsV[m];
                // v = boost::vertex(clsV[m], paired);
                assert(boost::out_degree(idx, paired) <= 2);
                for (boost::tie(e, e_end) = boost::out_edges(idx, paired); e != e_end; ++e) {
                    int pp = boost::get(gsIdx, boost::target(*e, paired));
                    if (binned.find(pp) != binned.end())  // don't recruit already binned contigs
                        continue;
                    if (clsS.find(pp) == clsS.end()) {
                        // check if it is reciprocal pairs
                        assert(boost::out_degree(pp, paired) <= 2);
                        for (boost::tie(ee, ee_end) = boost::out_edges(pp, paired); ee != ee_end; ++ee) {
                            if (idx == boost::get(gsIdx, boost::target(*ee, paired))) {
                                newbies.insert(pp);
                                updated = true;
                            }
                        }
                    }
                }
            }

            if (debug && newbies.size() > 0)
                verbose_message("Bin %d recruited %d contigs by paired infomation\n", good_class_ids2[k], newbies.size());

            clsV.insert(clsV.end(), newbies.begin(), newbies.end());
            clsS.insert(newbies.begin(), newbies.end());
            assert(clsV.size() == clsS.size());
        }
    }

    binned.clear();

    good_class_ids2.clear();
    good_class_ids2.insert(good_class_ids2.begin(), good_class_ids.begin(), good_class_ids.end());

#pragma omp parallel for
    for (size_t j = 0; j < good_class_ids2.size(); ++j) {
        ContigVector &clsV = cls[good_class_ids2[j]];

        // convert to local index
        for (size_t m = 0; m < clsV.size(); ++m) {
            clsV[m] = lCtgIdx[contig_names[clsV[m]]];
            binned.insert(clsV[m]);
        }
    }
}

bool readPairFile() {
    std::ifstream is(pairFile.c_str());
    if (!is.is_open()) {
        std::cerr << "[Error!] can't open the paired read coverage file " << pairFile << std::endl;
        return false;
    }

    if (ncols(is, 1) != 3) {
        std::cerr << "[Error!] Number of columns in paired read coverage data file is not 3." << std::endl;
        return false;
    }

    paired.m_vertices.resize(seqs.size());

    int nRow = -1;
    bool isGood = true;
    size_t pastContigIdx = 0, contigIdx = 0;
    std::vector<DistancePair> contigPairs;

    for (std::string row; safeGetline(is, row) && is.good(); ++nRow) {
        if (nRow == -1)  // the first row is header
            continue;

        std::stringstream ss(row);
        int c = 0;
        size_t contigIdxMate;
        double AvgCoverage;

        for (std::string col; getline(ss, col, tab_delim); ++c) {
            if (col.empty()) break;

            if (c == 0)
                contigIdx = boost::lexical_cast<size_t>(col);
            else if (c == 1)
                contigIdxMate = boost::lexical_cast<size_t>(col);
            else if (c == 2)
                AvgCoverage = boost::lexical_cast<double>(col);
        }

        if (c != 3) {
            std::cerr << "[Error!] Number of columns in paired read coverage data file is not 3 in the row " << nRow + 1 << std::endl;
            isGood = false;
            break;
        }

        if (contigIdx >= seqs.size() || pastContigIdx >= seqs.size()) {
            std::cerr << "[Error!] Contig index " << contigIdx << " >= the number of total sequences " << seqs.size()
                      << " in assembly file " << inFile << std::endl;
            isGood = false;
            break;
        }

        if (contigIdx == pastContigIdx) {
            DistancePair tmp(contigIdxMate, AvgCoverage);
            contigPairs.push_back(tmp);
        } else {  // new index
            sort(contigPairs.begin(), contigPairs.end(), cmp_abd);

            if (contigPairs.size() == 2) {
                boost::add_edge(pastContigIdx, contigPairs[1].first, paired);
            } else if (contigPairs.size() == 3) {
                boost::add_edge(pastContigIdx, contigPairs[1].first, paired);
                boost::add_edge(pastContigIdx, contigPairs[2].first, paired);
            } else if (contigPairs.size() > 3) {
                if (contigPairs[1].second > contigPairs[3].second * minTimes)
                    boost::add_edge(pastContigIdx, contigPairs[1].first, paired);
                if (contigPairs[2].second > contigPairs[3].second * minTimes)
                    boost::add_edge(pastContigIdx, contigPairs[2].first, paired);
            }

            assert(boost::out_degree(pastContigIdx, paired) <= 2);

            contigPairs.clear();
            pastContigIdx = contigIdx;
        }
    }

    sort(contigPairs.begin(), contigPairs.end(), cmp_abd);

    if (contigPairs.size() == 2) {
        boost::add_edge(pastContigIdx, contigPairs[1].first, paired);
    } else if (contigPairs.size() == 3) {
        boost::add_edge(pastContigIdx, contigPairs[1].first, paired);
        boost::add_edge(pastContigIdx, contigPairs[2].first, paired);
    } else if (contigPairs.size() > 3) {
        if (contigPairs[1].second > contigPairs[3].second * minTimes) boost::add_edge(pastContigIdx, contigPairs[1].first, paired);
        if (contigPairs[2].second > contigPairs[3].second * minTimes) boost::add_edge(pastContigIdx, contigPairs[2].first, paired);
    }

    assert(boost::out_degree(contigIdx, paired) <= 2);

    if (contigIdx != seqs.size() - 1) {  // the last index doesn't cover
        std::cerr << "[Error!] The last index does not cover all sequences given " << contigIdx << " != " << seqs.size() - 1
                  << std::endl;
        isGood = false;
    }

    if (!isGood) paired.clear();

    return isGood;
}

void getError(std::string s = "") {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        if (s != "") std::cerr << s << "|";
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

void launch_tnf_kernel(size_t cobs, size_t _first, size_t global_des) {
    cudaStream_t streams[n_STREAMS];
    cudaMalloc((void **)&seqs_d, seqs_h_index_e[cobs - 1] * sizeof(char));
    cudaMalloc((void **)&seqs_d_index, 2 * cobs * sizeof(size_t));
    size_t contig_per_kernel = cobs / n_STREAMS;
    for (int i = 0; i < n_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        size_t contig_to_process = contig_per_kernel;
        size_t _des = contig_per_kernel * i;
        size_t TNF_des = _des * 136;
        if (i == n_STREAMS - 1) contig_to_process += (cobs % n_STREAMS);
        size_t contigs_per_thread = (contig_to_process + (numThreads2 * numBlocks) - 1) / (numThreads2 * numBlocks);
        cudaMemcpyAsync(seqs_d + seqs_h_index_i[_des], &seqs[gCtgIdx[contigs[_first]]][0] + seqs_h_index_i[_des],
                        seqs_h_index_e[_des + contig_to_process - 1] - seqs_h_index_i[_des], cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(seqs_d_index + _des, seqs_h_index_i.data() + _des, contig_to_process * sizeof(size_t), cudaMemcpyHostToDevice,
                        streams[i]);
        cudaMemcpyAsync(seqs_d_index + cobs + _des, seqs_h_index_e.data() + _des, contig_to_process * sizeof(size_t),
                        cudaMemcpyHostToDevice, streams[i]);
        get_TNF<<<numBlocks, numThreads2, 0, streams[i]>>>(TNF_d + 136 * global_des + TNF_des, seqs_d, seqs_d_index + _des,
                                                           contig_to_process, contigs_per_thread, cobs);
        cudaMemcpyAsync(TNF_data + 136 * global_des + TNF_des, TNF_d + 136 * global_des + TNF_des,
                        contig_to_process * 136 * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
    }
    for (int i = 0; i < n_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    getError("kernel");
    cudaFree(seqs_d);
    cudaFree(seqs_d_index);
}

void create_graph(size_t total_prob, size_t prob_des, Distance requiredMinP, int _index) {
    size_t _total = min(total_prob - prob_des, max_prob_per_kernel);
    size_t _prob_per_thread = (total_prob + numThreads - 1) / numThreads;
#pragma omp parallel for
    for (int i = 0; i < numThreads; i++) {
        size_t _limit = min(prob_des + _prob_per_thread * (i + 1), prob_des + _total);
        size_t prob_index = prob_des + _prob_per_thread * i;
        size_t discriminante = 1 + 8 * prob_index;
        size_t r1 = (1 + sqrt(discriminante)) / 2;
        size_t r2 = prob_index - r1 * (r1 - 1) / 2;
        while (prob_index < _limit) {
            while (r2 < r1) {
                if (prob_index == _limit) break;
                bool passed = true;
                Similarity s = 1. - cal_dist2(contigs[r1], contigs[r2], 1. - requiredMinP, passed,
                                              tnf_prob[_index][prob_index % max_prob_per_kernel]);
                if (passed && s >= requiredMinP) {
#pragma omp critical(ADD_EDGE_1)
                    {
                        // archivo << contigs[r1] << " " << contigs[r2] << " " << s << std::endl;
                        boost::add_edge(contigs[r1], contigs[r2], Weight(s), gprob);
                    }
                }
                prob_index++;
                r2++;
            }
            r2 = 0;
            r1++;
        }
    }
}

void launch_tnf_prob_kernel(size_t max_prob_per_kernel, size_t prob_des, size_t total_prob, Distance minProb, int _index) {
    cudaStream_t streams[n_STREAMS];
    size_t total_prob_kernel = min(max_prob_per_kernel, total_prob - prob_des);
    size_t prob_per_kernel = total_prob_kernel / n_STREAMS;
    for (int i = 0; i < n_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        size_t prob_to_process = prob_per_kernel;
        if (i == n_STREAMS - 1) prob_to_process += (total_prob_kernel % n_STREAMS);

        size_t prob_per_thread = (prob_to_process + (numThreads2 * numBlocks) - 1) / (numThreads2 * numBlocks);
        get_tnf_prob<<<numBlocks, numThreads2, 0, streams[i]>>>(tnf_prob_d[_index] + prob_per_kernel * i, TNF_d, seqs_d_size_d,
                                                                prob_des + prob_per_kernel * i, prob_per_thread,
                                                                prob_des + prob_per_kernel * i + prob_to_process);
        cudaMemcpyAsync(tnf_prob[_index] + prob_per_kernel * i, tnf_prob_d[_index] + prob_per_kernel * i,
                        prob_to_process * sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
    }
    for (int i = 0; i < n_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    getError("kernel");
    create_graph(total_prob, prob_des, minProb, _index);
}

void reader(int fpint, int id, size_t chunk, size_t _size, char *_mem) {
    size_t readSz = 0;
    while (readSz < _size) {
        size_t _bytesres = _size - readSz;
        readSz += pread(fpint, _mem + (id * chunk) + readSz, _bytesres, (id * chunk) + readSz);
        if (readSz == 0) {
            std::cerr << "Error reading file" << std::endl;
            exit(1);
        }
    }
}

int main(int argc, char const *argv[]) {
    std::string saveTNFFile, saveDistanceFile;
    po::options_description desc("Allowed options", 110, 110 / 2);
    desc.add_options()("help,h", "produce help message")("inFile,i", po::value<std::string>(&inFile),
                                                         "Contigs in (gzipped) fasta file format [Mandatory]")(
        "outFile,o", po::value<std::string>(&outFile),
        "Base file name for each bin. The default output is fasta format. Use -l option to output only contig names [Mandatory]")(
        "abdFile,a", po::value<std::string>(&abdFile),
        "A file having mean and variance of base coverage depth (tab delimited; the first column should be contig names, and the "
        "first row will be considered as the header and be skipped) [Optional]")(
        "cvExt", po::value<bool>(&cvExt)->zero_tokens(),
        "When a coverage file without variance (from third party tools) is used instead of abdFile from "
        "jgi_summarize_bam_contig_depths")(
        "pairFile,p", po::value<std::string>(&pairFile),
        "A file having paired reads mapping information. Use it to increase sensitivity. (tab delimited; should have 3 columns of "
        "contig index (ordered by), its mate contig index, and supporting mean read coverage. The first row will be considered as "
        "the "
        "header and be skipped) [Optional]")("p1", po::value<Similarity>(&p1)->default_value(0),
                                             "Probability cutoff for bin seeding. It mainly controls the number of potential bins "
                                             "and their specificity. The higher, the "
                                             "more (specific) bins would be. (Percentage; Should be between 0 and 100)")(
        "p2", po::value<Similarity>(&p2)->default_value(0),
        "Probability cutoff for secondary neighbors. It supports p1 and better be close to p1. (Percentage; Should be between 0 "
        "and "
        "100)")("minProb", po::value<Similarity>(&minProb)->default_value(0),
                "Minimum probability for binning consideration. It controls sensitivity. Usually it should be >= 75. (Percentage; "
                "Should be between 0 and 100)")("minBinned", po::value<Similarity>(&minBinned)->default_value(0),
                                                "Minimum proportion of already binned neighbors for one's membership inference. "
                                                "It contorls specificity. Usually it would be "
                                                "<= 50 (Percentage; Should be between 0 and 100)")(
        "verysensitive", po::value<bool>(&verysensitive)->zero_tokens(),
        "For greater sensitivity, especially in a simple community. It is the shortcut for --p1 90 --p2 85 --pB 20 --minProb 75 "
        "--minBinned 20 --minCorr 90")("sensitive", po::value<bool>(&sensitive)->zero_tokens(),
                                       "For better sensitivity [default]. It is the shortcut for --p1 90 --p2 90 --pB 20 "
                                       "--minProb 80 --minBinned 40 --minCorr 92")(
        "specific", po::value<bool>(&specific)->zero_tokens(),
        "For better specificity. Different from --sensitive when using correlation binning or ensemble binning. It is the "
        "shortcut "
        "for --p1 90 --p2 90 --pB 30 --minProb 80 --minBinned 40 --minCorr 96")(
        "veryspecific", po::value<bool>(&veryspecific)->zero_tokens(),
        "For greater specificity. No correlation binning for short contig recruiting. It is the shortcut for --p1 90 --p2 90 --pB "
        "40 "
        "--minProb 80 --minBinned 40")(
        "superspecific", po::value<bool>(&superspecific)->zero_tokens(),
        "For the best specificity. It is the shortcut for --p1 95 --p2 90 --pB 50 --minProb 80 --minBinned 20")(
        "minCorr", po::value<Distance>(&minCorr)->default_value(0),
        "Minimum pearson correlation coefficient for binning missed contigs to increase sensitivity (Helpful when there are many "
        "samples). Should be very high (>=90) to reduce contamination. (Percentage; Should be between 0 and 100; 0 disables)")(
        "minSamples", po::value<size_t>(&minSamples)->default_value(10),
        "Minimum number of sample sizes for considering correlation based recruiting")(
        "minCV,x", po::value<Distance>(&minCV)->default_value(1),
        "Minimum mean coverage of a contig to consider for abundance distance calculation in each library")(
        "minCVSum", po::value<Distance>(&minCVSum)->default_value(2),
        "Minimum total mean coverage of a contig (sum of all libraries) to consider for abundance distance calculation")(
        "minClsSize,s", po::value<size_t>(&minClsSize)->default_value(200000), "Minimum size of a bin to be considered as the output")(
        "minContig,m", po::value<size_t>(&minContig)->default_value(2500),
        "Minimum size of a contig to be considered for binning (should be >=1500; ideally >=2500). If # of samples >= minSamples, "
        "small contigs (>=1000) will be given a chance to be recruited to existing bins by default.")(
        "minContigByCorr", po::value<size_t>(&minContigByCorr)->default_value(1000),
        "Minimum size of a contig to be considered for recruiting by pearson correlation coefficients (activated only if # of "
        "samples "
        ">= minSamples; disabled when minContigByCorr > minContig)")("numThreads,t", po::value<int>(&numThreads)->default_value(0),
                                                                     "Number of threads to use (0: use all cores)")(
        "ct", po::value<int>(&numThreads2)->default_value(16), "Number of cuda threads")(
        "cb", po::value<int>(&numBlocks)->default_value(256), "Number of cuda blocks")(
        "cs", po::value<int>(&n_STREAMS)->default_value(1), "Number of cuda streams")(
        "minShared", po::value<Similarity>(&minShared)->default_value(50), "Percentage cutoff for merging fuzzy contigs")(
        "fuzzy", po::value<bool>(&fuzzy)->zero_tokens(),
        "Binning with fuzziness which assigns multiple memberships of a contig to bins (activated only with --pairFile at the "
        "moment)")("onlyLabel,l", po::value<bool>(&onlyLabel)->zero_tokens(),
                   "Output only sequence labels as a list in a column without sequences")(
        "sumLowCV,S", po::value<bool>(&sumLowCV)->zero_tokens(),
        "If set, then every sample that falls below the minCV will be used in an aggregate sample")(
        "maxVarRatio,V", po::value<Distance>(&maxVarRatio)->default_value(maxVarRatio),
        "Ignore any contigs where variance / mean exceeds this ratio (0 disables)")(
        "saveTNF", po::value<std::string>(&saveTNFFile), "File to save (or load if exists) TNF matrix for each contig in input")(
        "saveDistance", po::value<std::string>(&saveDistanceFile),
        "File to save (or load if exists) distance graph at lowest probability cutoff")(
        "saveCls", po::value<bool>(&saveCls)->zero_tokens(), "Save cluster memberships as a matrix format")(
        "unbinned", po::value<bool>(&outUnbinned)->zero_tokens(), "Generate [outFile].unbinned.fa file for unbinned contigs")(
        "noBinOut", po::value<bool>(&noBinOut)->zero_tokens(),
        "No bin output. Usually combined with --saveCls to check only contig memberships")(
        "B,B", po::value<int>(&B)->default_value(20), "Number of bootstrapping for ensemble binning (Recommended to be >=20)")(
        "pB", po::value<double>(&pB)->default_value(50),
        "Proportion of shared membership in bootstrapping. Major control for sensitivity/specificity. The higher, the specific. "
        "(Percentage; Should be between 0 and 100)")(
        "seed", po::value<unsigned long long>(&seed)->default_value(0),
        "For reproducibility in ensemble binning, though it might produce slightly different results. (0: use random seed)")(
        "keep", po::value<bool>(&keep)->zero_tokens(), "Keep the intermediate files for later usage")(
        "debug,d", po::value<bool>(&debug)->zero_tokens(), "Debug output")("verbose,v", po::value<bool>(&verbose)->zero_tokens(),
                                                                           "Verbose output");
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).allow_unregistered().run(), vm);
    po::notify(vm);

    if (vm.count("help") || inFile.length() == 0 || outFile.length() == 0) {
        std::cerr << "\nMetaBAT: Metagenome Binning based on Abundance and Tetranucleotide frequency (version 1:" << version << "; "
                  << DATE << ")" << std::endl;
        std::cerr << "by Don Kang (ddkang@lbl.gov), Jeff Froula, Rob Egan, and Zhong Wang (zhongwang@lbl.gov) \n" << std::endl;
        std::cerr << desc << std::endl << std::endl;

        if (!vm.count("help")) {
            if (inFile.empty()) {
                std::cerr << "[Error!] There was no --inFile specified" << std::endl;
            }
            if (outFile.empty()) {
                std::cerr << "[Error!] There was no --outFile specified" << std::endl;
            }
        }

        return vm.count("help") ? 0 : 1;
    }

    if (verbose) t1 = std::chrono::steady_clock::now();

    if (seed == 0) seed = time(0);
    srand(seed);

    if (p1 == 0 && p2 == 0) {
        int labeledOpts =
            (verysensitive ? 1 : 0) + (sensitive ? 1 : 0) + (specific ? 1 : 0) + (veryspecific ? 1 : 0) + (superspecific ? 1 : 0);
        if (labeledOpts > 1) {
            std::cerr << "[Error!] Please only specify one of the following options: " << std::endl
                      << "\t--verysensitive, --sensitive, --specific or --veryspecific or --superspecific" << std::endl;
            return 1;
        }
        if (labeledOpts == 0) sensitive = true;  // set the default, if none were specified

        if (verysensitive) {
            p1 = 90;
            p2 = 85;
            minProb = 75;
            minBinned = 30;
            minCorr = 90;
            p3 = 90;  // pB = pB ? pB : 20;
        } else if (sensitive) {
            p1 = 90;
            p2 = 90;
            minProb = 80;
            minBinned = 40;
            minCorr = 92;
        } else if (specific) {
            p1 = 90;
            p2 = 90;
            minProb = 80;
            minBinned = 40;
            minCorr = 96;
        } else if (veryspecific) {
            p1 = 90;
            p2 = 90;
            minProb = 80;
            minBinned = 40;
        } else if (superspecific) {
            p1 = 95;
            p2 = 90;
            minProb = 80;
            minBinned = 20;
        }
    }

    if (minContig < 1500) {
        std::cerr
            << "[Error!] Contig length < 1500 is not allowed to be used for binning, rather use smaller minContigByCorr value to "
               "achieve better sensitivity"
            << std::endl;
        return 1;
    }

    if (minContigByCorr > minContig) {  // disabling correlation based recruiting
        minCorr = 0;
    }

    if (minClsSize < seedClsSize) {
        std::cerr << "[Error!] minClsSize should be >= " << seedClsSize << std::endl;
        return 1;
    }

    if (p1 <= 0 || p1 >= 100) {
        std::cerr << "[Error!] p1 should be greater than 0 and less than 100" << std::endl;
        return 1;
    }

    if (p2 <= 0 || p2 >= 100) {
        std::cerr << "[Error!] p2 should be greater than 0 and less than 100" << std::endl;
        return 1;
    }

    if (p3 <= 0 || p3 >= 100) {
        std::cerr << "[Error!] p3 should be greater than 0 and less than 100" << std::endl;
        return 1;
    }

    if (pB < 0 || pB > 100) {
        std::cerr << "[Error!] pB should be >= 0 and <= 100" << std::endl;
        return 1;
    }

    if (minProb <= 0 || minProb >= 100) {
        std::cerr << "[Error!] minProb should be greater than 0 and less than 100" << std::endl;
        return 1;
    }

    if (minBinned <= 0 || minBinned >= 100) {
        std::cerr << "[Error!] minBinned should be greater than 0 and less than 100" << std::endl;
        return 1;
    }

    if (minShared < 0 || minShared > 100) {
        std::cerr << "[Error!] minShared should be >= 0 and <= 100" << std::endl;
        return 1;
    }

    if (minCV < 0) {
        std::cerr << "[Error!] minCV should be non-negative" << std::endl;
        return 1;
    }

    if (B <= 1) {
        B = 1;
        useEB = false;
    }

    if (useEB) {
        if (B < 10) std::cerr << "[Warning!] B < 10 may not be effective for ensemble binning. Consider B >= 20" << std::endl;
    }

    gen_commandline_hash();

    p1 /= 100.;
    p2 /= 100.;
    p3 /= 100.;
    pB /= 100.;
    minProb /= 100.;
    minBinned /= 100.;
    minShared /= 100.;

    boost::filesystem::path dir(outFile);
    if (dir.parent_path().string().length() > 0) {
        if (boost::filesystem::is_regular_file(dir.parent_path())) {
            std::cerr << "Cannot create directory: " << dir.parent_path().string() << ", which exists as a regular file." << std::endl;
            return 1;
        }
        boost::filesystem::create_directory(dir.parent_path());
    }
    if (verbose)
        print_message(
            "MetaBAT 1 CUDA (%s) using p1 %2.1f%%, p2 %2.1f%%, p3 %2.1f%%, minProb %2.1f%%, minBinned %2.0f%%, minCV %2.1f, "
            "minContig %d, minContigByCorr %d, minCorr %2.0f%%, paired %d, and %d bootstrapping\n",
            version.c_str(), p1 * 100, p2 * 100, p3 * 100, minProb * 100, minBinned * 100, minCV, minContig, minContigByCorr, minCorr,
            pairFile.length() > 0, useEB ? B : 0);

    if (numThreads == 0) numThreads = std::thread::hardware_concurrency();  // obtener el numero de hilos maximo
    omp_set_num_threads(numThreads);
    // print_message("Advanced setting. using CPU threads %d,GPU block (per grid) %d, GPU threads(per block) %d\n", numThreads,
    // numBlocks, numThreads2);

    TIMERSTART(total);
    nobs = 0;
    int nresv = 0;
    TIMERSTART(READ);
    FILE *fp = fopen(inFile.c_str(), "r");
    if (fp == NULL) {
        std::cout << "Error opening file: " << inFile << std::endl;
        return 1;
    } else {
        fseek(fp, 0L, SEEK_END);
        fsize = ftell(fp);  // obtener el tamaño del archivo
        fclose(fp);
        size_t chunk = fsize / numThreads;
        cudaError_t cudaStatus = cudaMallocHost((void **)&_mem, fsize);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMallocHost failed!");
            return 1;
        }
        int fpint = open(inFile.c_str(), O_RDWR | O_CREAT, S_IREAD | S_IWRITE | S_IRGRP | S_IROTH);
        if (fpint == -1) {
            std::cout << "Error opening file: " << inFile << std::endl;
            return 1;
        }
        std::thread readerThreads[numThreads];
        for (int i = 0; i < numThreads; i++) {
            size_t _size;
            if (i != numThreads - 1)
                _size = chunk;
            else
                _size = chunk + (fsize % numThreads);
            readerThreads[i] = std::thread(reader, fpint, i, chunk, _size, _mem);
        }
        for (int i = 0; i < numThreads; i++) {  // esperar a que terminen de leer
            readerThreads[i].join();
        }

        close(fpint);
        size_t __min = std::min(minContigByCorr, minContigByCorrForGraph);
        size_t contig_name_i;
        size_t contig_name_e;
        size_t contig_i;
        size_t contig_e;
        size_t contig_size;
        seqs.reserve(fsize / __min);
        ignored.reserve(fsize / __min);
        contig_names.reserve(fsize / __min);
        lCtgIdx.reserve(fsize / __min);
        gCtgIdx.reserve(fsize / __min);
        for (size_t i = 0; i < fsize; i++) {  // leer el archivo caracter por caracter
            if (_mem[i] == fasta_delim) {
                i++;
                contig_name_i = i;  // guardar el inicio del nombre del contig
                while (_mem[i] != line_delim) i++;
                contig_name_e = i;  // guardar el final del nombre del contig
                i++;
                contig_i = i;  // guardar el inicio del contig
                while (i < fsize && _mem[i] != line_delim) i++;
                contig_e = i;  // guardar el final del contig
                contig_size = contig_e - contig_i;
                if (contig_size >= __min) {
                    if (contig_size < minContig) {
                        if (contig_size >= minContigByCorr)
                            smallCtgs.insert(nobs);
                        else
                            nresv++;
                    }
                    lCtgIdx[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] = nobs;
                    gCtgIdx[nobs++] = seqs.size();
                } else {
                    ignored[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] = seqs.size();
                }
                contig_names.emplace_back(std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i));
                seqs.emplace_back(std::string_view(_mem + contig_i, contig_e - contig_i));
            }
        }
        seqs.shrink_to_fit();          // liberar memoria no usada
        contig_names.shrink_to_fit();  // liberar memoria no usada
    }
    TIMERSTOP(READ);

    nobs2 = ignored.size();

    verbose_message("Finished reading %d contigs. Number of target contigs >= %d are %d, and [%d and %d) are %d \n", nobs + nobs2,
                    minContig, nobs - smallCtgs.size() - nresv, minContigByCorr, minContig, smallCtgs.size());
    if (contig_names.size() != nobs + nobs2 || seqs.size() != nobs + nobs2) {  // error en la lectura del archivo
        std::cerr << "[Error!] Need to check whether there are duplicated sequence ids in the assembly file" << std::endl;
        return 1;
    }

    // cargar el archivo de abundancias
    const int nNonFeat = cvExt ? 1 : 3;  // number of non-feature columns
    if (abdFile.length() > 0) {
        smallCtgs.clear();
        std::unordered_map<std::string_view, size_t> lCtgIdx2;
        std::unordered_map<size_t, size_t> gCtgIdx2;

        nobs = std::min(nobs, countLines(abdFile.c_str()) - 1);  // la primera linea es el header
        if (nobs < 1) {
            std::cerr << "[Error!] There are no lines in the abundance depth file or fasta file!" << std::endl;
            exit(1);
        }
        nABD = ncols(abdFile.c_str(), 1) - nNonFeat;
        // num of features (excluding the first three columns which is the contigName,
        // contigLen, and totalAvgDepth);
        if (!cvExt) {
            if (nABD % 2 != 0) {
                std::cerr << "[Error!] Number of columns (excluding the first column) in abundance data file "
                             "is not even."
                          << std::endl;
                return 1;
            }
            nABD /= 2;
        }
        ABD.resize(nobs, nABD);
        ABD_VAR.resize(nobs, nABD);

        std::ifstream is(abdFile.c_str());
        if (!is.is_open()) {
            std::cerr << "[Error!] can't open the contig coverage depth file " << abdFile << std::endl;
            return 1;
        }

        int r = -1;
        int nskip = 0;

        for (std::string row; safeGetline(is, row) && is.good(); ++r) {  // leer el archivo linea por linea
            if (r == -1)                                                 // the first row is header
                continue;

            std::stringstream ss(row);  // convertir la linea en un stream
            int c = -nNonFeat;
            Distance mean, variance, meanSum = 0;
            std::string label;   // almacena el nombre del contig
            bool isGood = true;  // indica si el contig se encuentra en el las estucturas de datos
            DistancePair tmp(0, 0);

            for (std::string col; getline(ss, col, tab_delim); ++c) {  // leer la linea por columnas
                if (col.empty()) break;
                if (c == -3 || (cvExt && c == -1)) {  // contig name
                    trim_fasta_label(col);
                    label = col;
                    if (lCtgIdx.find(label) == lCtgIdx.end()) {  // no se encuentra el contig
                        if (ignored.find(label) == ignored.end()) {
                            verbose_message(
                                "[Warning!] Cannot find the contig (%s) in abundance file from the "
                                "assembly file\n",
                                label.c_str());
                        } else if (debug) {
                            verbose_message("[Info] Ignored a small contig (%s) having length %d < %d\n", label.c_str(),
                                            seqs[ignored[label]].size(), minContig);
                        }
                        isGood = false;  // cannot find the contig from fasta file. just skip it!
                        break;
                    }
                    continue;
                } else if (c == -2) {
                    continue;
                } else if (c == -1) {
                    meanSum = std::stod(col.c_str());
                    if (meanSum < minCVSum) {
                        if (debug)
                            verbose_message("[Info] Ignored a contig (%s) having mean coverage %2.2f < %2.2f \n", label.c_str(),
                                            meanSum, minCVSum);
                        isGood = false;  // cannot find the contig from fasta file. just skip it!
                        break;
                    }
                    continue;
                }

                assert(r - nskip >= 0 && r - nskip < (int)nobs);

                bool checkMean = false, checkVar = false;

                if (cvExt) {
                    mean = ABD(r - nskip, c) = std::stod(col.c_str());
                    meanSum += mean;
                    variance = ABD_VAR(r - nskip, c) = mean;
                    checkMean = true;
                } else {
                    if (c % 2 == 0) {
                        mean = ABD(r - nskip, c / 2) = std::stod(col.c_str());
                        checkMean = true;
                    } else {
                        variance = ABD_VAR(r - nskip, c / 2) = std::stod(col.c_str());
                        checkVar = true;
                    }
                }

                if (checkMean) {
                    if (mean > 1e+7) {
                        std::cerr << "[Error!] Need to check where the average depth is greater than 1e+7 for "
                                     "the contig "
                                  << label << ", column " << c + 1 << std::endl;
                        return 1;
                    }
                    if (mean < 0) {
                        std::cerr << "[Error!] Negative coverage depth is not allowed for the contig " << label << ", column " << c + 1
                                  << ": " << mean << std::endl;
                        return 1;
                    }
                }

                if (checkVar) {
                    if (variance > 1e+14) {
                        std::cerr << "[Error!] Need to check where the depth variance is greater than 1e+14 "
                                     "for the contig "
                                  << label << ", column " << c + 1 << std::endl;
                        return 1;
                    }
                    if (variance < 0) {
                        std::cerr << "[Error!] Negative variance is not allowed for the contig " << label << ", column " << c + 1
                                  << ": " << variance << std::endl;
                        return 1;
                    }
                    if (maxVarRatio > 0.0 && mean > 0 && variance / mean > maxVarRatio) {
                        std::cerr << "[Warning!] Skipping contig due to >maxVarRatio variance: " << variance << " / " << mean << " = "
                                  << variance / mean << ": " << label << std::endl;
                        isGood = false;
                        break;
                    }
                }

                if (c == (int)(nABD * (cvExt ? 1 : 2) - 1)) {
                    if (meanSum < minCVSum) {
                        if (debug)
                            verbose_message("[Info] Ignored a contig (%s) having mean coverage %2.2f < %2.2f \n", label.c_str(),
                                            meanSum, minCVSum);
                        isGood = false;  // cannot find the contig from fasta file. just skip it!
                        break;
                    }
                    tmp.second = meanSum;  // useEB ? rand() : meanSum
                }
            }

            if (isGood) {
                size_t _gidx = gCtgIdx[lCtgIdx[label]];
                if (seqs[_gidx].size() < minContig) {
                    smallCtgs.insert(r - nskip);
                    if (seqs[_gidx].size() < minContigByCorr) ++nresv;
                }
                lCtgIdx2[contig_names[_gidx]] = r - nskip;  // local index
                gCtgIdx2[r - nskip] = _gidx;                // global index
            } else {
                ++nskip;
                continue;
            }

            tmp.first = r - nskip;
            rABD.push_back(tmp);

            if ((int)nABD != (cvExt ? c : c / 2)) {
                std::cerr << "[Error!] Different number of variables for the object for the contig " << label << std::endl;
                return 1;
            }
        }
        is.close();

        verbose_message(
            "Finished reading %d contigs (using %d including %d short contigs) and %d coverages from "
            "%s\n",
            r, r - nskip - nresv, smallCtgs.size() - nresv, nABD, abdFile.c_str());

        if ((specific || veryspecific) && nABD < minSamples) {
            std::cerr << "[Warning!] Consider --superspecific for better specificity since both --specific "
                         "and --veryspecific would be the same as --sensitive when # of samples ("
                      << nABD << ") < minSamples (" << minSamples << ")" << std::endl;
        }

        if (nABD < minSamples) {
            std::cerr << "[Info] Correlation binning won't be applied since the number of samples (" << nABD << ") < minSamples ("
                      << minSamples << ")" << std::endl;
        }

        for (std::unordered_map<std::string_view, size_t>::const_iterator it = lCtgIdx.begin(); it != lCtgIdx.end(); ++it) {
            if (lCtgIdx2.find(it->first) == lCtgIdx2.end()) {  // given seq but missed depth info or skipped
                ignored[it->first] = gCtgIdx[it->second];
            }
        }

        lCtgIdx.clear();
        gCtgIdx.clear();

        lCtgIdx = lCtgIdx2;
        gCtgIdx = gCtgIdx2;

        assert(lCtgIdx.size() == gCtgIdx.size());
        assert(lCtgIdx.size() + ignored.size() == seqs.size());

        // nobs_aux = nobs;
        nobs = lCtgIdx.size();
        nobs2 = ignored.size();

        if (ABD.size1() != nobs) {
            ABD.resize(nobs, nABD, true);
            ABD_VAR.resize(nobs, nABD, true);
        }

        assert(rABD.size() == nobs);
    }
    size_t max_gpu_mem = 4000000000;  // 4gb
    // calcular matriz de tetranucleotidos
    contigs.reserve(nobs);
    for (int i = 0; i < nobs; i++) {
        if (smallCtgs.find(i) == smallCtgs.end()) {
            contigs.push_back(i);
        }
    }
    ncontigs = contigs.size();

    TIMERSTART(TNF_CAL);
    cudaMallocHost((void **)&TNF_data, ncontigs * 136 * sizeof(float));
    cudaMalloc((void **)&TNF_d, ncontigs * 136 * sizeof(float));
    seqs_h_index_i.reserve(ncontigs);
    seqs_h_index_e.reserve(ncontigs);
    if (!loadTNFFromFile(saveTNFFile, minContig)) {  // calcular TNF en paralelo en GPU de no estar guardado
        ProgressTracker progress(ncontigs);
        TNF.resize(nobs, 136);
        TNF.clear();
        size_t cobs = 0;  // current obs
        size_t _first = 0;
        for (size_t i = 0; i < ncontigs; i++) {
            if (seqs[gCtgIdx[contigs[i]]].data() - seqs[gCtgIdx[contigs[_first]]].data() + seqs[gCtgIdx[contigs[i]]].size() >
                max_gpu_mem) {
                launch_tnf_kernel(cobs, _first, i - cobs);
                seqs_h_index_i.clear();
                seqs_h_index_e.clear();
                progress.track(cobs);
                verbose_message("Calculating TNF %s\r", progress.getProgress());
                _first = i;
                cobs = 0;
            }
            seqs_h_index_i.emplace_back(seqs[gCtgIdx[contigs[i]]].data() - seqs[gCtgIdx[contigs[_first]]].data());
            seqs_h_index_e.emplace_back(seqs[gCtgIdx[contigs[i]]].data() - seqs[gCtgIdx[contigs[_first]]].data() +
                                        seqs[gCtgIdx[contigs[i]]].size());
            cobs++;
        }
        if (cobs != 0) {
            launch_tnf_kernel(cobs, _first, ncontigs - cobs);
            progress.track(cobs);
            verbose_message("Calculating TNF %s\r", progress.getProgress());
            seqs_h_index_i.clear();
            seqs_h_index_e.clear();
        }
        for (size_t i = 0; i < ncontigs; i++) {
            for (int j = 0; j < 136; j++) {
                if (TNF_data[i * 136 + j] != TNF_data[i * 136 + j]) {
                    std::cout << "ERROR:" << contigs[i] << " " << j << std::endl;
                }
                TNF(contigs[i], j) = TNF_data[i * 136 + j];
            }
        }
        saveTNFToFile(saveTNFFile, minContig);
    }
    cudaFreeHost(TNF_data);
    TIMERSTOP(TNF_CAL);
    verbose_message("Finished TNF calculation.                                  \n");

    if (rABD.size() == 0) {
        for (size_t i = 0; i < nobs; ++i) {
            rABD.push_back(std::make_pair(i, rand()));
        }
    }

    Distance requiredMinP = std::min(std::min(std::min(p1, p2), p3), minProb);
    if (requiredMinP > .75)  // allow every mode exploration without reforming graph.
        requiredMinP = .75;

    max_gpu_mem = min(max_gpu_mem, (size_t)500000000);

    TIMERSTART(probabilisticgraph);
    if (!loadDistanceFromFile(saveDistanceFile, requiredMinP, minContig)) {
        int _index = 0;
        gprob.m_vertices.resize(nobs);
        std::thread threads[2];
        size_t prob_des = 0;
        size_t total_prob = (ncontigs * (ncontigs - 1)) / 2;
        max_prob_per_kernel = max_gpu_mem / (sizeof(double) * 2);
        size_t cant_kernels = (total_prob + max_prob_per_kernel - 1) / max_prob_per_kernel;
        ProgressTracker progress = ProgressTracker(total_prob);
        for (size_t i = 0; i < ncontigs; i++) {
            seqs_h_index_i.emplace_back(seqs[gCtgIdx[contigs[i]]].length());
        }
        for (int i = 0; i < 2; i++) {
            cudaMallocHost((void **)&tnf_prob[i], max_prob_per_kernel * sizeof(double));
            cudaMalloc((void **)&tnf_prob_d[i], max_prob_per_kernel * sizeof(double));
        }
        cudaMalloc((void **)&seqs_d_size_d, ncontigs * sizeof(size_t));
        cudaMemcpy(seqs_d_size_d, seqs_h_index_i.data(), ncontigs * sizeof(size_t), cudaMemcpyHostToDevice);
        for (size_t i = 0; i < cant_kernels; i++) {
            if (threads[_index].joinable()) {
                threads[_index].join();
                if (_index == 1) {
                    progress.track(min(max_prob_per_kernel * 2, total_prob - prob_des));
                    verbose_message("Building a tnf graph: %s\r", progress.getProgress());
                }
            }
            threads[_index] = std::thread(launch_tnf_prob_kernel, max_prob_per_kernel, prob_des, total_prob, requiredMinP, _index);
            prob_des += max_prob_per_kernel;
            _index = (_index + 1) % 2;
        }
        for (int i = 0; i < 2; i++) {
            if (threads[i].joinable()) threads[i].join();
            progress.track(min(max_prob_per_kernel, total_prob - prob_des));
            verbose_message("Building a tnf graph: %s\r", progress.getProgress());
        }
        for (int i = 0; i < 2; i++) {
            cudaFree(tnf_prob_d[i]);
            cudaFreeHost(tnf_prob[i]);
        }
        cudaFree(seqs_d_size_d);
        seqs_h_index_i.clear();
        saveDistanceToFile(saveDistanceFile, requiredMinP, minContig);
    }
    cudaFree(TNF_d);
    TIMERSTOP(probabilisticgraph);

    verbose_message("Finished building a probabilistic graph. (%d vertices and %d edges)          \n", boost::num_vertices(gprob),
                    boost::num_edges(gprob));

    gIdx = boost::get(boost::vertex_index, gprob);
    gWgt = boost::get(boost::edge_weight, gprob);

    bool good_pair = pairFile.length() > 0 && readPairFile();

    boost::numeric::ublas::matrix<size_t> resES(nobs, B, 0);

    ClassMap cls;
    TIMERSTART(binning);
    if (!loadBootFromFile(resES)) {
        for (int b = 0; b < B; ++b) {
            ContigVector _medoid_ids;
            std::vector<double> medoid_vals;
            ContigSet binned;
            ContigSet leftovers;
            ClassIdType good_class_ids;
            cls.clear();

            if (b > 0) {
                if (rABD.size() > 0) {
                    for (std::list<DistancePair>::iterator it = rABD.begin(); it != rABD.end(); ++it) {
                        it->second = rand();
                        rABD2.push_back(*it);
                    }
                    rABD.clear();
                }
                rABD = rABD2;
                rABD2.clear();
            }
            rABD.sort(cmp_abd);

            pam(_medoid_ids, medoid_vals, binned, cls, leftovers, good_class_ids);

            if (!useEB)
                verbose_message("Leftover contigs before fish_more: %2.2f%% (%d out of %d)\n", (double)leftovers.size() / nobs * 100.,
                                leftovers.size(), nobs);

            bool leftout = true;
            int fished = 1;
            while (leftout) {
                leftout = false;

                fish_more_by_friends_membership(cls, leftovers, good_class_ids);
                if (!useEB)
                    verbose_message(
                        "Leftover contigs after fish_more_by_friends_membership (roughly): %2.2f%% (%d out of %d), %d bins   \r",
                        (double)leftovers.size() / nobs * 100., leftovers.size(), nobs, good_class_ids.size());

                ClassIdType good_class_ids2;
                for (ClassIdType::const_iterator it = good_class_ids.begin(); it != good_class_ids.end(); ++it) {
                    size_t s = 0;
                    size_t kk = *it;
                    for (ContigVector::iterator it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
                        s += seqs[gCtgIdx[*it2]].size();
                    }
                    if (s < std::min(seedClsSize * (size_t)std::pow(2, fished), minClsSize)) {
                        leftovers.insert(cls[kk].begin(), cls[kk].end());
                        leftout = true;
                    } else
                        good_class_ids2.insert(kk);
                }

                good_class_ids = good_class_ids2;

                fished++;
            }

            if (!useEB) std::cout << std::endl;

            for (ClassIdType::const_iterator it = good_class_ids.begin(); it != good_class_ids.end(); ++it) {
                fish_more(*it, cls, leftovers);
            }
            if (!useEB)
                verbose_message("Leftover contigs after fish_more (roughly): %2.2f%% (%d out of %d)\n",
                                (double)leftovers.size() / nobs * 100., leftovers.size(), nobs);

            if (minCorr > 0) {
                size_t fished = fish_more_by_corr(_medoid_ids, cls, leftovers, good_class_ids);
                if (!useEB)
                    verbose_message("Leftover contigs after fish_more_by_corr (roughly): %2.2f%% (%d out of %d)\n",
                                    (double)(leftovers.size() - fished) / nobs * 100., (leftovers.size() - fished), nobs);
            }

            if (good_pair) {
                fish_pairs(binned, cls, good_class_ids);

                if (!useEB) {
                    verbose_message("Number of clusters formed before merging: %d\n", good_class_ids.size());  // # of bins >= 2
                                                                                                               // members
                    verbose_message("Merging bins that share >= %2.2f%%\n", minShared * 100.);
                }
                // sort bin by # of contigs; for each bin; find the first bin that shared >= minShared and merge two bins; iterate
                size_t k = 0;

                // convert cls => cls bit set where each element represent each contig
                std::unordered_map<int, boost::dynamic_bitset<>> clsB;
                for (ClassIdType::const_iterator it = good_class_ids.begin(); it != good_class_ids.end(); ++it) {
                    boost::dynamic_bitset<> bs(seqs.size());
#pragma omp parallel for
                    for (size_t m = 0; m < cls[*it].size(); ++m) {
#pragma omp critical(FUZZY_1)
                        bs[cls[*it][m]] = 1;
                    }
                    assert(bs.count() == cls[*it].size());
                    clsB[*it] = bs;
                    assert(bs.count() == clsB[*it].count());
                }

                while (k < good_class_ids.size()) {
                    std::vector<ClsSizePair> cls_size;
                    for (ClassIdType::const_iterator it = good_class_ids.begin(); it != good_class_ids.end(); ++it) {
                        ClsSizePair csp(*it, cls[*it].size());
                        cls_size.push_back(csp);
                    }
                    sort(cls_size.begin(), cls_size.end(), cmp_cls_size);

                    int cls1 = cls_size[k].first;

                    bool isMerged = false;
                    std::vector<size_t> kk_hist(omp_get_max_threads(), cls_size.size());

#pragma omp parallel for schedule(static, 1)
                    for (size_t kk = k + 1; kk < cls_size.size(); ++kk) {
                        if (isMerged) continue;
                        int cls2 = cls_size[kk].first;

                        boost::dynamic_bitset<> tmp = clsB[cls2] & clsB[cls1];
                        double shared = (double)tmp.count() / cls_size[k].second;

                        if (debug && !useEB && omp_get_thread_num() == 0)
                            verbose_message("clsB[cls2]: %d, clsB[cls1]: %d, tmp: %d, cls_size[k].second: %d, shared: %2.2f\n",
                                            clsB[cls2].count(), clsB[cls1].count(), tmp.count(), cls_size[k].second, shared * 100);

                        if (shared >= minShared) {
                            if (!useEB && omp_get_thread_num() == 0)
                                verbose_message("Bin %d and %d were merged to %d (%2.2f%% shared)\n", cls1 + 1, cls2 + 1, cls2 + 1,
                                                shared * 100.);
                            kk_hist[omp_get_thread_num()] = kk;
                            isMerged = true;
                        }
                    }

                    if (isMerged) {
                        size_t kk = *std::min_element(kk_hist.begin(), kk_hist.end());
                        k = 0;  // reset whenever any bins are combined so that it start from the smallest again (inefficient but
                                // most thorough way)
                        size_t cls2 = cls_size[kk].first;
                        // combine cls1 and cls2 => make it as cls2
                        clsB[cls2] |= clsB[cls1];
                        clsB.erase(cls1);
                        ContigSet tmp;
                        tmp.insert(cls[cls2].begin(), cls[cls2].end());
                        tmp.insert(cls[cls1].begin(), cls[cls1].end());
                        cls[cls2].clear();
                        cls[cls2].insert(cls[cls2].end(), tmp.begin(), tmp.end());
                        cls[cls1].clear();
                        cls.erase(cls1);
                        good_class_ids.erase(cls1);
                    } else {  // k and kk were not merged
                        ++k;
                    }

                    if (debug) std::cout << "good_class_ids.size(): " << good_class_ids.size() << ", kk: " << k << std::endl;
                }
            }

            if (useEB) {
                for (ClassIdType::const_iterator it = good_class_ids.begin(); it != good_class_ids.end(); ++it) {
                    for (ContigVector::iterator it2 = cls[*it].begin(); it2 != cls[*it].end(); ++it2) {
                        resES(*it2, b) = *it;
                    }
                }
                verbose_message("Bootstrapping %d/%d [%.1fGb / %.1fGb]          \r", b + 1, B, getUsedPhysMem(),
                                getTotalPhysMem() / 1024 / 1024);
            }
        }

        if (useEB) {
            verbose_message("Bootstrapping %d/%d [%.1fGb / %.1fGb]             \n", B, B, getUsedPhysMem(),
                            getTotalPhysMem() / 1024 / 1024);
            saveBootToFile(resES);
        }
    }

    TNF.clear();
    // cudaFreeHost(ABD);
    // cudaFreeHost(ABD_VAR);
    gprob.clear();
    gprob.m_edges.resize(0);
    gprob.m_vertices.resize(0);
    gprob.m_vertices.shrink_to_fit();

    ABD.clear();
    ABD_VAR.clear();
    ABD.resize(0, 0, false);
    ABD_VAR.resize(0, 0, false);

    if (useEB) {
        igraph_t g;
        igraph_empty(&g, nobs, 0);

        igraph_weight_vector_t weights;
        igraph_vector_init(&weights, 0);

        g.incs = igraph_Calloc(g.n, igraph_edge_vector_t);
        for (node_t i = 0; i < g.n; i++) {
            igraph_vector_init(&g.incs[i], 0);
        }

        ProgressTracker progress = ProgressTracker(nobs * (nobs - 1) / 2, nobs / 100 + 1);

        if (!loadENSFromFile(g, weights)) {
            edge_t reserved = (edge_t)nobs * 1000;

            igraph_vector_reserve(&weights, reserved);
            igraph_vector_reserve(&g.from, reserved);
            igraph_vector_reserve(&g.to, reserved);

            size_t cutoff = (size_t)B * pB;
            std::vector<size_t> num_binned(nobs, 0);

#pragma omp parallel for
            for (size_t i = 0; i < nobs; ++i)
                for (int j = 0; j < B; ++j) num_binned[i] += resES(i, j) > 0;

#pragma omp parallel for schedule(dynamic, 100)
            for (node_t i = 0; i < nobs; ++i) {
                if (num_binned[i] >= cutoff) {
                    for (node_t j = i + 1; j < nobs; ++j) {
                        if (num_binned[j] < cutoff) continue;

                        size_t _scr = 0;
                        for (int h = 0; h < B; ++h)
                            if (resES(i, h) > 0 && resES(j, h) > 0) _scr += resES(i, h) == resES(j, h);

                        if (_scr >= cutoff) {
#pragma omp critical(ENSEMBLE_ADD_WEIGHT)
                            {
                                igraph_vector_push_back(&weights, (float)_scr / B);
                                igraph_vector_push_back(&g.from, (uint_least32_t)j);
                                igraph_vector_push_back(&g.to, (uint_least32_t)i);

                                igraph_vector_push_back(&g.incs[i], igraph_vector_size(&g.from) - 1);
                                igraph_vector_push_back(&g.incs[j], igraph_vector_size(&g.from) - 1);
                            }
                        }
                    }
                }
                if (verbose) {
                    progress.track(nobs - i - 1);
                    if (omp_get_thread_num() == 0 && progress.isStepMarker()) {
                        verbose_message("Building Ensemble Graph %s [%.1fGb / %.1fGb]\r", progress.getProgress(), getUsedPhysMem(),
                                        getTotalPhysMem() / 1024 / 1024);
                    }
                }
            }
            verbose_message("Building Ensemble Graph %s [%.1fGb / %.1fGb]\r", progress.getProgress(), getUsedPhysMem(),
                            getTotalPhysMem() / 1024 / 1024);

            igraph_vector_resize_min(&g.to);
            igraph_vector_resize_min(&g.from);
            igraph_vector_resize_min(&weights);

            // saveENSToFile(g, weights);
        }

        verbose_message("Finished Ensemble Graph (%lld vertices and %lld edges) [%.1fGb / %.1fGb]                          \n",
                        igraph_vcount(&g), igraph_ecount(&g), getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);

        igraph_node_vector_t membership;
        igraph_vector_init(&membership, 0);

        igraph_rng_seed(igraph_rng_default(), seed);

        verbose_message("Starting Ensemble Binning [%.1fGb / %.1fGb]\n", getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);
        igraph_community_label_propagation(&g, &membership, &weights);
        verbose_message("Finished Ensemble Binning [%.1fGb / %.1fGb]\n", getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);

        igraph_destroy(&g);
        igraph_vector_destroy(&weights);

        if (debug) {
            std::ofstream os(outFile.c_str());
            os.rdbuf()->pubsetbuf(os_buffer, buf_size);
            for (size_t i = 0; i < nobs; ++i) {
                os << contig_names[gCtgIdx[i]] << tab_delim;
                os << VECTOR(membership)[i] << line_delim;
            }
            for (std::unordered_map<std::string_view, size_t>::const_iterator it = ignored.begin(); it != ignored.end(); ++it) {
                os << contig_names[it->second] << tab_delim << 0 << line_delim;
            }
            os.close();
        }

        cls.clear();
        for (size_t i = 0; i < nobs; ++i) {
            cls[VECTOR(membership)[i]].push_back(i);
        }

        igraph_vector_destroy(&membership);
    }

    // if everything was fine, delete intermediate files
    if (!keep && useEB) {
        std::remove(("boot." + std::to_string(commandline_hash)).c_str());
        std::remove(("ens." + std::to_string(commandline_hash)).c_str());
        verbose_message("Cleaned up intermediate files\n");
    }

    Distance binnedSize = 0;

    // One of ways to make the bin ids deterministic... sort bins by their size
    std::unordered_map<size_t, size_t> cls_size;
    std::vector<DistancePair> cls_med_abd;
    for (ClassMap::const_iterator it = cls.begin(); it != cls.end(); ++it) {
        int kk = it->first;
        size_t s = 0;

        for (ContigVector::iterator it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
            s += seqs[gCtgIdx[*it2]].size();
        }
        binnedSize += s;
        cls_size[kk] = s;

        DistancePair dp(kk, s);
        cls_med_abd.push_back(dp);
    }
    sort(cls_med_abd.begin(), cls_med_abd.end(), cmp_abd);

    ContigSet binned;

    size_t bin_id = 1;
    for (size_t k = 0; k < cls_med_abd.size(); ++k) {
        size_t kk = cls_med_abd[k].first;

        if (!fuzzy) {
            int s = (int)cls_size[kk];
            binnedSize -= s;
            ContigSet unique;
            for (ContigVector::iterator it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
                if (binned.find(*it2) != binned.end()) {  // binned already
                    s -= (int)seqs[gCtgIdx[*it2]].size();
                } else {
                    binned.insert(*it2);
                    unique.insert(*it2);
                }
            }
            cls_size[kk] = s;
            if (cls_size[kk] < minClsSize) {
                continue;
            }
            binnedSize += cls_size[kk];
            cls[kk].clear();
            cls[kk].insert(cls[kk].end(), unique.begin(), unique.end());
        }

        if (!noBinOut) {
            std::string outFile_cls = outFile + ".";
            outFile_cls.append(boost::lexical_cast<std::string>(bin_id));
            if (!onlyLabel) outFile_cls.append(".fa");

            std::ofstream os(outFile_cls.c_str());
            os.rdbuf()->pubsetbuf(os_buffer, buf_size);

            for (ContigVector::iterator it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
                std::string_view &label = contig_names[gCtgIdx[*it2]];
                if (onlyLabel) {
                    os << label << line_delim;
                } else {
                    std::string_view &seq = seqs[gCtgIdx[*it2]];
                    os << fasta_delim << label << line_delim;
                    for (size_t s = 0; s < seq.length(); s += 60) {
                        os << seq.substr(s, 60) << line_delim;
                    }
                }
            }
            os.close();

            if (debug)
                std::cout << "Bin " << bin_id << " (" << cls_size[kk] << " bases in " << cls[kk].size()
                          << " contigs) was saved to: " << outFile_cls << std::endl;
        }

        bin_id++;
    }

    if (verbose) {
        unsigned long long totalSize = 0;
        for (std::vector<std::string_view>::iterator it = seqs.begin(); it != seqs.end(); ++it) totalSize += it->size();
        verbose_message("%2.2f%% (%lld out of %lld bases) was binned.\n", (double)binnedSize / totalSize * 100,
                        (unsigned long long)binnedSize, totalSize);
    }

    std::cout << "Number of clusters formed: " << bin_id - 1 << std::endl;

    if (saveCls || outUnbinned) {
#pragma omp parallel for
        for (size_t k = 0; k < cls_med_abd.size(); ++k) {
            ContigVector &clsV = cls[cls_med_abd[k].first];

            // convert to global index
            for (size_t m = 0; m < clsV.size(); ++m) {
                clsV[m] = gCtgIdx[clsV[m]];
            }
        }

        std::vector<size_t> clsMap(seqs.size(), 0);
#pragma omp parallel for
        for (size_t k = 0; k < cls_med_abd.size(); ++k) {
            size_t kk = cls_med_abd[k].first;
            for (size_t i = 0; i < cls[kk].size(); ++i) {
                assert(cls[kk][i] < (int)clsMap.size());
                clsMap[cls[kk][i]] = k + 1;
            }
        }

        if (saveCls) {
            if (!fuzzy) {
                std::ofstream os(outFile.c_str());
                os.rdbuf()->pubsetbuf(os_buffer, buf_size);

                for (size_t i = 0; i < clsMap.size(); ++i) {
                    os << contig_names[i];
                    os << tab_delim << clsMap[i] << line_delim;
                }
                os.flush();
                os.close();
            } else {
                // rows as contigs and columns as bins, so wanted to represent complete memberships.
            }
        }

        if (outUnbinned) {
            std::string outFile_cls = outFile + ".";
            outFile_cls.append("unbinned");
            if (!onlyLabel) outFile_cls.append(".fa");

            std::ofstream os(outFile_cls.c_str());
            os.rdbuf()->pubsetbuf(os_buffer, buf_size);

            for (size_t i = 0; i < clsMap.size(); ++i) {
                if (clsMap[i] == 0) {
                    if (onlyLabel) {
                        os << contig_names[i] << line_delim;
                    } else {
                        std::string_view &seq = seqs[i];
                        os << fasta_delim << contig_names[i] << line_delim;
                        for (size_t s = 0; s < seq.length(); s += 60) {
                            os << seq.substr(s, 60) << line_delim;
                        }
                    }
                }
            }
            os.flush();
            os.close();
        }
    }
    cudaFreeHost(_mem);
    TIMERSTOP(binning);
    TIMERSTOP(total);
    return 0;
}
