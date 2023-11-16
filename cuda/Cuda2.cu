#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cstdarg>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif

#include "../extra/metrictime2.hpp"
#include "ProgressTracker.h"
#include "cuckoohash_map.hh"
#include "ranker.h"
#include "tile.h"

// force BOOST ublas optimizations
#define BOOST_UBLAS_INLINE inline
#define BOOST_UBLAS_CHECK_ENABLE 0
#define BOOST_UBLAS_USE_FAST_SAME
#define BOOST_UBLAS_TYPE_CHECK 0

#include <boost/dynamic_bitset.hpp>
#include <boost/filesystem.hpp>
#include <boost/math/distributions.hpp>
#include <boost/program_options.hpp>
#include <boost/system/error_code.hpp>
#include <string>

#if (BOOST_VERSION / 100000 == 1) && (BOOST_VERSION / 100 % 1000 == 64)
#include <boost/serialization/array_wrapper.hpp>
#endif

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>

using std::cerr;
using std::cout;
using std::endl;
namespace po = boost::program_options;

typedef double Distance;
typedef double Similarity;
#define LOG log
#define LOG10 log10
#define SQRT sqrt
#define EXP exp
#define POW pow
#define FABS fabs

typedef boost::math::normal_distribution<Distance> Normal;

static std::string version = "Metabat 2 cuda 0.1";
static std::string DATE = "2023-09-25";
static bool verbose = false;
static bool debug = false;
static bool noBinOut = false;
static size_t minClsSize = 200000;
static size_t minContig = 2500;  // minimum contig size for binning
static std::string inFile;
static std::string abdFile;
static bool cvExt;
static std::string outFile;
static bool onlyLabel = false;
static bool noAdd = false;
static size_t numThreads = 0;
static Similarity maxP = 95;
static Similarity minS = 60;
static Similarity pTNF = 0;
static Distance minCV = 1;
static Distance minCVSum = 1;
static bool saveCls = false;
static bool outUnbinned = false;
static size_t minSample = 3;
static unsigned long long totalSize = 0, totalSize1 = 0;
static int numThreads2 = 32;

static size_t maxEdges = 200;
static const char line_delim = '\n';
static const char tab_delim = '\t';
static const char fasta_delim = '>';
static const std::size_t buf_size = 1024 * 1024;

static char* _mem;
static size_t fsize = 0;
static std::vector<std::string_view> contig_names;
static std::vector<std::string_view> small_contig_names;
static std::vector<std::string_view> seqs;
static std::vector<std::string_view> small_seqs;
static std::vector<Distance> logSizes;
static std::vector<size_t> sizes;
static std::vector<size_t> small_sizes;

typedef std::vector<int> ContigVector;
typedef std::unordered_set<int> ContigSet;
typedef std::unordered_map<int, ContigVector> ClassMap;

static size_t nobs = 0;   // # of large
static size_t nobs1 = 0;  // # of small

static boost::numeric::ublas::matrix<float> ABD;
static boost::numeric::ublas::matrix<float> ABD_VAR;
static boost::numeric::ublas::matrix<float> small_ABD;
// static boost::numeric::ublas::matrix<float> TNF;

// static float* TNF_data;
static float* TNF_d;
static char* seqs_d;
static size_t* seqs_d_index;
static double* contig_log;
std::vector<size_t> seqs_h_index_i;
std::vector<size_t> seqs_h_index_e;

typedef boost::numeric::ublas::matrix_row<boost::numeric::ublas::matrix<float>> MatrixRowType;
typedef boost::numeric::ublas::matrix_column<boost::numeric::ublas::matrix<float>> MatrixColumnType;

static size_t nABD = 0;
static const size_t nTNF = 136;
static unsigned long long seed = 0;

static std::chrono::steady_clock::time_point t1, t2;

static std::vector<int> TNLookup;  // lookup table 0 - 255 of raw 4-mer to tetramer index in TNF

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
//__device__ __constant__ double floor_preProb = 2.197224577336219564216435173875652253627777099609375;
// más preciso
__device__ __constant__ double floor_preProb = 2.1972245773362193827904904738450514092949811156454989034693886672;

__device__ double cal_tnf_pre_dist_d(double r1, double r2, const float* __restrict__ TNF1, const float* __restrict__ TNF2) {
    double d = 0.0;
    float _diff;
    for (size_t i = 0; i < 136; ++i) {
        _diff = TNF1[i] - TNF2[i];
        d += _diff * _diff;
    }

    d = sqrt(d);

    double b, c;

    double lw[19];
    lw[0] = min(r1, r2);
    lw[1] = max(r1, r2);
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

    double4 _b4, _b14, _lw4;

    b = _b1[0];
    for (size_t i = 0; i < 16; i += 4) {
        _b14 = *reinterpret_cast<double*>(_b1 + i + 1);
        _lw4 = *reinterpret_cast<double*>(lw + i);
        _b4 += _b14 * _lw4;
    }
    b += _b4.x + _b4.y + _b4.z + _b4.w;
    b += _b1[17] * lw[16];

    /*
    b = _b1[0] + _b1[1] * lw[0] + _b1[2] * lw[1] + _b1[3] * lw[2] + _b1[4] * lw[3] + _b1[5] * lw[4] + _b1[6] * lw[5] + _b1[7] * lw[6] +
        _b1[8] * lw[7] + _b1[9] * lw[8] + _b1[10] * lw[9] + _b1[11] * lw[10] + _b1[12] * lw[11] + _b1[13] * lw[12] + _b1[14] * lw[13] +
        _b1[15] * lw[14] + _b1[16] * lw[15] + _b1[17] * lw[16];
    */

    c = _c1[0];
    for (size_t i = 0; i < 17; i++) {
        c += _c1[i + 1] * lw[i];
    }
    /*
    c = _c1[0] + _c1[1] * lw[0] + _c1[2] * lw[1] + _c1[3] * lw[2] + _c1[4] * lw[3] + _c1[5] * lw[4] + _c1[6] * lw[5] + _c1[7] * lw[6] +
        _c1[8] * lw[7] + _c1[9] * lw[8] + _c1[10] * lw[9] + _c1[11] * lw[10] + _c1[12] * lw[11] + _c1[13] * lw[12] + _c1[14] * lw[13] +
        _c1[15] * lw[14] + _c1[16] * lw[15] + _c1[17] * lw[16];
    */
    double preProb = -(b + c * d);

    if (preProb <= floor_preProb) {
        b = _b2[0] + _b2[1] * lw[0] + _b2[2] * lw[1] + _b2[3] * lw[2] + _b2[4] * lw[3] + _b2[5] * lw[4] + _b2[6] * lw[5] +
            _b2[7] * lw[6] + _b2[8] * lw[7] + _b2[9] * lw[8] + _b2[10] * lw[9] + _b2[11] * lw[10] + _b2[12] * lw[18] +
            _b2[13] * lw[13] + _b2[14] * lw[14] + _b2[15] * lw[15] + _b2[16] * lw[16];
        c = _c2[0] + _c2[1] * lw[0] + _c2[2] * lw[1] + _c2[3] * lw[2] + _c2[4] * lw[3] + _c2[5] * lw[4] + _c2[6] * lw[5] +
            _c2[7] * lw[6] + _c2[8] * lw[7] + _c2[9] * lw[8] + _c2[10] * lw[9] + _c2[11] * lw[10] + _c2[12] * lw[18] +
            _c2[13] * lw[11] + _c2[14] * lw[12] + _c2[15] * lw[13] + _c2[16] * lw[14] + _c2[17] * lw[15] + _c2[18] * lw[16];
        preProb = -(b + c * d);
        if (preProb > floor_preProb) preProb = floor_preProb;
    }
    return preProb;
}

/*
__device__ double cal_tnf_dist_d(double r1, double r2, float* TNF1, float* TNF2) {
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

    double lw[19];
    lw[0] = min(r1, r2);
    lw[1] = max(r1, r2);
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

    if (prob == 0.1) {
        b = _b2[0] + _b2[1] * lw[0] + _b2[2] * lw[1] + _b2[3] * lw[2] + _b2[4] * lw[3] + _b2[5] * lw[4] + _b2[6] * lw[5] +
            _b2[7] * lw[6] + _b2[8] * lw[7] + _b2[9] * lw[8] + _b2[10] * lw[9] + _b2[11] * lw[10] + _b2[12] * lw[18] +
            _b2[13] * lw[13] + _b2[14] * lw[14] + _b2[15] * lw[15] + _b2[16] * lw[16];
        c = _c2[0] + _c2[1] * lw[0] + _c2[2] * lw[1] + _c2[3] * lw[2] + _c2[4] * lw[3] + _c2[5] * lw[4] + _c2[6] * lw[5] +
            _c2[7] * lw[6] + _c2[8] * lw[7] + _c2[9] * lw[8] + _c2[10] * lw[9] + _c2[11] * lw[10] + _c2[12] * lw[18] +
            _c2[13] * lw[11] + _c2[14] * lw[12] + _c2[15] * lw[13] + _c2[16] * lw[14] + _c2[17] * lw[15] + _c2[18] * lw[16];
        preProb = -(b + c * d);
        prob = preProb <= floor_preProb ? 1.0 / (1 + exp(preProb)) : 0.1;
    }
    return prob;
}
*/

__global__ void get_tnf_graph(double* graph, const float* __restrict__ TNF, const double* __restrict__ contig_log, size_t nc1,
                              size_t nc2, size_t off1, size_t off2, double floor_preProb_cutoff) {
    size_t prob_index = (threadIdx.x + blockIdx.x * blockDim.x);
    size_t r1 = prob_index / nc2;
    if (r1 >= nc1) return;
    size_t ct1 = off1 + r1;
    size_t ct2 = off2 + prob_index % nc2;
    if (ct1 == ct2) return;
    double preProb = cal_tnf_pre_dist_d(contig_log[ct1], contig_log[ct2], TNF + ct1 * 136, TNF + ct2 * 136);
    if (preProb > floor_preProb_cutoff)
        graph[prob_index] = 1.0 - (1.0 / (1 + exp(preProb)));
    else
        graph[prob_index] = 0.0;
}

__global__ void get_tnf_max_prob_sample3(double* max_dist, const float* __restrict__ TNF, double* size_log, size_t* contigs,
                                         size_t nobs, size_t _des, size_t limit) {
    extern __shared__ double shared_max[];
    size_t contig_idx = _des + blockIdx.x;
    if (contig_idx >= limit) return;
    double local_max = -1;
    float TNF1[136];
    for (int i = 0; i < 136; i++) {
        TNF1[i] = TNF[contigs[contig_idx] * 136 + i];
    }
    size_t dist_per_thread = (nobs + blockDim.x - 1) / blockDim.x;
    for (size_t i = dist_per_thread * threadIdx.x; i < min(dist_per_thread * threadIdx.x + dist_per_thread, nobs); i++) {
        if (i == contig_idx) continue;
        double dist = cal_tnf_pre_dist_d(size_log[contigs[contig_idx]], size_log[contigs[i]], TNF1, TNF + contigs[i] * 136);
        if (dist > local_max) {
            local_max = dist;
        }
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    // reduction max
    for (size_t i = blockDim.x / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            if (shared_max[threadIdx.x] < shared_max[threadIdx.x + i]) {
                shared_max[threadIdx.x] = shared_max[threadIdx.x + i];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        max_dist[contig_idx] = 1.0 - (1.0 / (1 + exp(shared_max[0])));
    }
}

/*
__global__ void get_tnf_prob_sample(double* __restrict__ tnf_dist, float* TNF, double* size_log, size_t* contigs, size_t nobs,
                                    size_t _des, const size_t contig_per_thread, const size_t limit) {
    size_t r1;
    size_t r2;
    float TNF1[136];
    size_t tnf_dist_index = (threadIdx.x + blockIdx.x * blockDim.x) * contig_per_thread;
    size_t prob_index = _des + tnf_dist_index;
    r1 = prob_index / nobs;
    r2 = prob_index % nobs;
    size_t _limit2 = min(tnf_dist_index + contig_per_thread, limit - _des);
    if (tnf_dist_index >= _limit2) return;
    while (tnf_dist_index != _limit2) {
        for (int i = 0; i < 136; i++) {
            TNF1[i] = TNF[contigs[r1] * 136 + i];
        }
        while (r2 < nobs) {
            if (tnf_dist_index == _limit2) break;
            tnf_dist[tnf_dist_index] =
                1. - cal_tnf_dist_d(size_log[contigs[r1]], size_log[contigs[r2]], TNF1, TNF + contigs[r2] * 136);
            tnf_dist_index++;
            r2++;
        }
        r2 = 0;
        r1++;
    }
}
*/

__device__ short get_tn(char* __restrict__ contig) {
    unsigned char N;
    short tn = 0;
    // if (contig[0] == 'X') return 256;
    for (short i = 0; i < 4; i++) {
        N = BN[contig[i]];
        if (N & 4) return 256;
        tn = (tn << 2) | N;
    }
    return tn;
}

__device__ void next_contig(char* __restrict__ contig, char c) {
    if (c == '\n') {
        contig[0] = 'X';
        return;
    }
    for (int i = 0; i < 3; i++) {
        contig[i] = contig[i + 1];
    }
    contig[3] = c;
}

__global__ void get_TNF(float* __restrict__ TNF_d, const char* __restrict__ seqs_d, const size_t* __restrict__ seqs_d_index,
                        const size_t nobs, const size_t contigs_per_thread, const size_t seqs_d_index_size) {
    const size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;
    size_t limit = min(thead_id * contigs_per_thread + contigs_per_thread, nobs);
    for (size_t contig_index = thead_id * contigs_per_thread; contig_index < limit; contig_index++) {
        char contig_temp[4] = {0};
        float TNF_temp[136] = {0};
        const size_t tnf_index = contig_index * 136;
        size_t contig_size = seqs_d_index[contig_index + seqs_d_index_size] - seqs_d_index[contig_index];
        const char* contig = seqs_d + seqs_d_index[contig_index];
        if (contig_size < 4) continue;
        for (size_t j = 0; j < 3; j++) next_contig(contig_temp, contig[j]);
        for (size_t j = 3; j < contig_size; ++j) {
            char c = contig[j];
            next_contig(contig_temp, c);
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
    }
}

void getError(std::string s = "") {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaDeviceReset();
        if (s != "") std::cerr << s << "|";
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
            exit(1);
        }
        exit(1);
    }
}

void launch_tnf_kernel(size_t cobs, size_t _first, size_t global_des) {
    size_t n_STREAMS = ((cobs + 9999) / 10000);
    cudaStream_t streams[n_STREAMS];
    cudaMalloc((void**)&seqs_d, seqs_h_index_e[cobs - 1] * sizeof(char));
    cudaMalloc((void**)&seqs_d_index, 2 * cobs * sizeof(size_t));
    size_t contig_per_kernel = cobs / n_STREAMS;
    for (int i = 0; i < n_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
        size_t contig_to_process = contig_per_kernel;
        size_t _des = contig_per_kernel * i;
        size_t TNF_des = _des * 136;
        if (i == n_STREAMS - 1) contig_to_process += (cobs % n_STREAMS);
        size_t bloqs = (contig_to_process + numThreads2 - 1) / numThreads2;
        size_t contigs_per_thread = 1;
        cudaMemcpyAsync(seqs_d + seqs_h_index_i[_des], seqs[_first].data() + seqs_h_index_i[_des],
                        seqs_h_index_e[_des + contig_to_process - 1] - seqs_h_index_i[_des], cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(seqs_d_index + _des, seqs_h_index_i.data() + _des, contig_to_process * sizeof(size_t), cudaMemcpyHostToDevice,
                        streams[i]);
        cudaMemcpyAsync(seqs_d_index + cobs + _des, seqs_h_index_e.data() + _des, contig_to_process * sizeof(size_t),
                        cudaMemcpyHostToDevice, streams[i]);
        get_TNF<<<bloqs, numThreads2, 0, streams[i]>>>(TNF_d + 136 * global_des + TNF_des, seqs_d, seqs_d_index + _des,
                                                       contig_to_process, contigs_per_thread, cobs);
        /*
        cudaMemcpyAsync(TNF_data + 136 * global_des + TNF_des, TNF_d + 136 * global_des + TNF_des,
                        contig_to_process * 136 * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        */
    }
    for (int i = 0; i < n_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    getError("kernel");
    cudaFree(seqs_d);
    cudaFree(seqs_d_index);
}

void launch_tnf_max_prob_sample_kernel(std::vector<size_t> idx, double* max_dist_d, double* max_dist_h, size_t _nobs) {
    size_t* contigs_d;
    cudaMalloc((void**)&contigs_d, idx.size() * sizeof(size_t));
    cudaMemcpy(contigs_d, idx.data(), idx.size() * sizeof(size_t), cudaMemcpyHostToDevice);
    get_tnf_max_prob_sample3<<<_nobs, numThreads2, numThreads2 * sizeof(double)>>>(max_dist_d, TNF_d, contig_log, contigs_d, nobs, 0,
                                                                                   _nobs);
    cudaDeviceSynchronize();
    cudaMemcpy(max_dist_h, max_dist_d, _nobs * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(contigs_d);
    getError("kernel");
}

void reader(int fpint, int id, size_t chunk, size_t _size, char* _mem) {
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

static void print_message(const char* format, ...) {
    va_list argptr;
    va_start(argptr, format);
    vfprintf(stdout, format, argptr);
    cout.flush();
    va_end(argptr);
}

static void verbose_message(const char* format, ...) {
    if (verbose) {
        t2 = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration duration = t2 - t1;
        int elapsed = (int)std::chrono::duration_cast<std::chrono::seconds>(duration).count();  // seconds
        printf("[%02d:%02d:%02d] ", elapsed / 3600, (elapsed % 3600) / 60, elapsed % 60);
        va_list argptr;
        va_start(argptr, format);
        vfprintf(stdout, format, argptr);
        cout.flush();
        va_end(argptr);
    }
}

class Graph {
   public:
    size_t n;
    std::vector<size_t> from;
    std::vector<size_t> to;
    std::vector<std::vector<size_t>> incs;  // incidence list which has edge id instead of node id (compared to adjacent list)
    std::vector<double> sTNF;
    std::vector<double> sSCR;  // composite score (weight) of sTNF and sABD
    ContigSet connected_nodes;
    bool hasEdges;
    Graph(size_t num_nodes, bool hasEdges = false) : n(num_nodes), hasEdges(hasEdges) {
        if (hasEdges) {
            incs.resize(num_nodes);
        }
    }
    ~Graph() {}
    size_t getNodeCount() { return n; }
    size_t getEdgeCount() { return from.size(); }
    size_t getOtherNode(size_t e, size_t v) {
        assert(e < from.size() && e < to.size());
        return from[e] == v ? to[e] : from[e];
    }
};

static void trim_fasta_label(std::string& label) {
    size_t pos = label.find_first_of(" \t");
    if (pos != std::string::npos) label = label.substr(0, pos);
}

std::ostream& printFasta(std::ostream& os, std::string_view label, std::string_view seq) {
    int64_t len = seq.size();
    if (len == 0) {
        cerr << "Warning attempt to print an empty fasta!" << endl;
        return os;
    }
    os << fasta_delim << label << line_delim;
    const char* _seq = seq.begin();
    const int maxWidth = 60;
    for (size_t s = 0; s < len; s += maxWidth) {
        int bytes = s + maxWidth < len ? maxWidth : len - s;
        os.write(_seq + s, bytes);
        os << line_delim;
    }
    return os;
}

// Fisher-Yates shuffle
// http://stackoverflow.com/questions/9345087/choose-m-elements-randomly-from-a-vector-containing-n-elements
template <class bidiiter>
bidiiter random_unique(bidiiter begin, bidiiter end, size_t num_random) {
    size_t left = std::distance(begin, end);
    while (num_random--) {
        bidiiter r = begin;
        std::advance(r, rand() % left);
        std::swap(*begin, *r);
        ++begin;
        --left;
    }
    return begin;
}

#ifdef __APPLE__
vm_statistics_data_t vmStats;
mach_msg_type_number_t infoCount = HOST_VM_INFO_COUNT;
#else
struct sysinfo memInfo;
#endif
double totalPhysMem = 0.;

int parseLine(char* line) {
    int i = strlen(line);
    while (*line < '0' || *line > '9') line++;
    line[i - 3] = '\0';
    i = atoi(line);
    return i;
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

// http://blog.csdn.net/hengshan/article/details/9201929
int getFreeMem() {
#ifdef __APPLE__
    kern_return_t kernReturn = host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmStats, &infoCount);
    if (kernReturn != KERN_SUCCESS) return 0;
    return (vm_page_size * vmStats.free_count) / 1024;
#else
    FILE* file = fopen("/proc/meminfo", "r");
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

double getUsedPhysMem() { return (getTotalPhysMem() - getFreeMem()) / 1024. / 1024.; }

int label_propagation(Graph& g, std::vector<size_t>& membership, std::vector<size_t>& node_order) {
    size_t no_of_nodes = g.getNodeCount();
    size_t no_of_edges = g.getEdgeCount();

    if (no_of_nodes == 0 || no_of_edges == 0) {
        cerr << "There were " << no_of_nodes << " nodes and " << no_of_edges << " edges -- skipping label_propagation" << endl;
        return 0;
    }

    if (g.sSCR.size() != no_of_edges) {
        cerr << "sSCR != no_of_edges" << endl;
        exit(1);
    }

    if (membership.size() != no_of_nodes) {
        membership.resize(no_of_nodes);
        std::iota(membership.begin(), membership.end(), 0);
    }
    /* Do some initial checks */
    if (*std::min_element(g.sSCR.begin(), g.sSCR.end()) < 0) {
        cerr << "sSCR must be non-negative" << endl;
        exit(1);
    }
    std::unordered_map<size_t, std::unordered_set<size_t>> visited;
    std::unordered_set<size_t> blacklist;
    size_t nLeftMin = INT_MAX;
    size_t attempt = 0;
    bool running = true;
    while (running) {
        running = false;
        size_t nLeft = 0;
        /* In the prescribed order, loop over the vertices and reassign labels */
        for (size_t i = 0; i < node_order.size();
             i++) {  // we reconsider all nodes regardless of its previous status, but is it better?
            size_t v1 = node_order[i];
            std::unordered_map<size_t, double> neighbor_scores;  // sum of neighbors scores to cluster k
            std::unordered_map<size_t, size_t> neighbor_counts;  // keep number of neighbors
            std::vector<size_t>& ineis = g.incs[v1];
            for (size_t j = 0; j < ineis.size(); j++) {  // # of neighbors (edges connected to v1)
                size_t edgeID = ineis[j];
                int_fast32_t k = membership[g.getOtherNode(edgeID, v1)];  // community membership of a neighbor (connected by j)
                if (neighbor_scores.find(k) == neighbor_scores.end()) {
                    neighbor_scores[k] = 0.;
                    neighbor_counts[k] = 0;
                }
                neighbor_scores[k] += LOG(1. - g.sSCR[edgeID]);  // as p-value
                neighbor_counts[k]++;
            }

            if (neighbor_scores.size() > 0) {
                for (auto& kv : neighbor_scores) {
                    // Fisher's method to compare significance of different number of probs.
                    boost::math::chi_squared chi_sqr_dist(2 * neighbor_counts[kv.first]);
                    kv.second = boost::math::cdf(chi_sqr_dist, -2.0 * kv.second);
                }
                auto best_neighbor = std::max_element(
                    neighbor_scores.begin(), neighbor_scores.end(),
                    [](const std::pair<size_t, double>& p1, const std::pair<size_t, double>& p2) { return p1.second < p2.second; });
                // however, if there was a clique (loop) out of >2 nodes
                int kPrev = membership[v1];
                if (kPrev != (int)best_neighbor->first && blacklist.find(v1) == blacklist.end()) {
                    membership[v1] = best_neighbor->first;
                    int kNext = membership[v1];
                    if (visited.find(v1) == visited.end() || visited[v1].find(kNext) == visited[v1].end()) {
                        // not have been assigned to the cls before
                        nLeft++;  // # of confirmation (that this choice is optimal) left
                        running = true;
                    } else {
                        blacklist.insert(v1);  // blacklist represents nodes that change cls in a circular form
                    }
                    visited[v1].insert(kNext);
                }
            }
        }

        if (nLeft < nLeftMin) {
            nLeftMin = nLeft;
            attempt = 0;
        } else {
            attempt++;
            if (attempt >= 10) {
                break;
            }
        }
        // cout << "nLeft: " << nLeft << " & attempt: " << attempt << endl;
    }

    return 0;
}

float get_element(boost::numeric::ublas::matrix<float> const& m, int i, int j) { return m(i, j); }

struct CompareEdge {
    constexpr bool operator()(std::pair<size_t, Similarity> const& a, std::pair<size_t, Similarity> const& b) const noexcept {
        return a.second > b.second;
    }
};

// for normal distributions
Distance cal_abd_dist2(Normal& p1, Normal& p2) {
    Distance k1, k2, tmp, d = 0;
    Distance m1 = p1.mean();
    Distance m2 = p2.mean();
    Distance v1 = p1.standard_deviation();
    v1 = v1 * v1;
    Distance v2 = p2.standard_deviation();
    v2 = v2 * v2;

    // normal_distribution
    if (FABS(v2 - v1) < 1e-4) {
        k1 = k2 = (m1 + m2) / 2;
    } else {
        tmp = SQRT(v1 * v2 * ((m1 - m2) * (m1 - m2) - 2 * (v1 - v2) * LOG(SQRT(v2 / v1))));
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
        d = (FABS(boost::math::cdf(p1, k1) - boost::math::cdf(p2, k1)));
    else
        d = (FABS(boost::math::cdf(p1, k2) - boost::math::cdf(p1, k1) + boost::math::cdf(p2, k1) - boost::math::cdf(p2, k2)));

    return d;
}

Distance cal_abd_dist(size_t r1, size_t r2, size_t i, bool& nz) {
    Distance d = 0;
    Distance m1 = ABD(r1, i);
    Distance m2 = ABD(r2, i);
    if (m1 > minCV || m2 > minCV) {
        nz = true;
        m1 = std::max(m1, (Distance)1e-6);
        m2 = std::max(m2, (Distance)1e-6);
        if (m1 != m2) {
            Distance v1 = ABD_VAR(r1, i) < 1 ? 1 : ABD_VAR(r1, i);
            Distance v2 = ABD_VAR(r2, i) < 1 ? 1 : ABD_VAR(r2, i);

            Normal p1(m1, SQRT(v1)), p2(m2, SQRT(v2));
            d = cal_abd_dist2(p1, p2);
        }
    }
    return std::min(std::max(d, 1e-6), 1. - 1e-6);
}

/*
Distance cal_tnf_dist(size_t r1, size_t r2) {
    // EXP(preProb) <= 9 yields prob >= 0.1, so preProb <= LOG(9.0);
    const Distance floor_prob = 0.1;
    const Distance floor_preProb = LOG((1.0 / floor_prob) - 1.0);
    Distance d = 0;
    for (size_t i = 0; i < nTNF; ++i) {
        d += (TNF(r1, i) - TNF(r2, i)) * (TNF(r1, i) - TNF(r2, i));  // euclidean distance
    }
    d = SQRT(d);
    Distance b, c;  // parameters
    Distance ctg1 = logSizes[r1];
    Distance ctg2 = logSizes[r2];
    Distance lw11 = std::min(ctg1, ctg2);
    Distance lw21 = std::max(ctg1, ctg2);
    Distance lw12 = lw11 * lw11;
    Distance lw13 = lw12 * lw11;
    Distance lw14 = lw13 * lw11;
    Distance lw15 = lw14 * lw11;
    Distance lw16 = lw15 * lw11;
    Distance lw17 = lw16 * lw11;
    Distance lw22 = lw21 * lw21;
    Distance lw23 = lw22 * lw21;
    Distance lw24 = lw23 * lw21;
    Distance lw25 = lw24 * lw21;
    Distance lw26 = lw25 * lw21;
    Distance prob;
    b = 46349.1624324381 + -76092.3748553155 * lw11 + -639.918334183 * lw21 + 53873.3933743949 * lw12 + -156.6547554844 * lw22 +
        -21263.6010657275 * lw13 + 64.7719132839 * lw23 + 5003.2646455284 * lw14 + -8.5014386744 * lw24 + -700.5825500292 * lw15 +
        0.3968284526 * lw25 + 54.037542743 * lw16 + -1.7713972342 * lw17 + 474.0850141891 * lw11 * lw21 + -23.966597785 * lw12 * lw22 +
        0.7800219061 * lw13 * lw23 + -0.0138723693 * lw14 * lw24 + 0.0001027543 * lw15 * lw25;
    c = -443565.465710869 + 718862.10804858 * lw11 + 5114.1630934534 * lw21 + -501588.206183097 * lw12 + 784.4442123743 * lw22 +
        194712.394138513 * lw13 + -377.9645994741 * lw23 + -45088.7863182741 * lw14 + 50.5960513287 * lw24 + 6220.3310639927 * lw15 +
        -2.3670776453 * lw25 + -473.269785487 * lw16 + 15.3213264134 * lw17 + -3282.8510348085 * lw11 * lw21 +
        164.0438603974 * lw12 * lw22 + -5.2778800755 * lw13 * lw23 + 0.0929379305 * lw14 * lw24 + -0.0006826817 * lw15 * lw25;

    // logistic model
    //  prob = 1.0 / (1 + EXP(-(b + c * d)));
    //  if (prob >= .1)  //second logistic model
    Distance preProb = -(b + c * d);
    // preProb <= LOG(9.0) yields prob > 0.1, so use second logistic model
    prob = preProb <= floor_preProb ? floor_prob : 1.0 / (1 + EXP(preProb));

    if (prob >= floor_prob) {  // second logistic model
        b = 6770.9351457442 + -5933.7589419767 * lw11 + -2976.2879986855 * lw21 + 3279.7524685865 * lw12 + 1602.7544794819 * lw22 +
            -967.2906583423 * lw13 + -462.0149190219 * lw23 + 159.8317289682 * lw14 + 74.4884405822 * lw24 + -14.0267151808 * lw15 +
            -6.3644917671 * lw25 + 0.5108811613 * lw16 + 0.2252455343 * lw26 + 0.965040193 * lw12 * lw22 +
            -0.0546309127 * lw13 * lw23 + 0.0012917084 * lw14 * lw24 + -1.14383e-05 * lw15 * lw25;
        c = 39406.5712626297 + -77863.1741143294 * lw11 + 9586.8761567725 * lw21 + 55360.1701572325 * lw12 + -5825.2491611377 * lw22 +
            -21887.8400068324 * lw13 + 1751.6803621934 * lw23 + 5158.3764225203 * lw14 + -290.1765894829 * lw24 +
            -724.0348081819 * lw15 + 25.364646181 * lw25 + 56.0522105105 * lw16 + -0.9172073892 * lw26 + -1.8470088417 * lw17 +
            449.4660736502 * lw11 * lw21 + -24.4141920625 * lw12 * lw22 + 0.8465834103 * lw13 * lw23 + -0.0158943762 * lw14 * lw24 +
            0.0001235384 * lw15 * lw25;
        // prob = 1.0 / (1 + EXP(-(b + c * d)));
        //  prob = prob < .1 ? .1 : prob;
        preProb = -(b + c * d);  // EXP(preProb) <= 9 yields prob >= 0.1, so preProb <= LOG(9.0) to calculate, otherwise use the floor
        prob = preProb <= floor_preProb ? 1.0 / (1 + EXP(preProb)) : floor_prob;
    }

    return prob;
}
*/

size_t countLines(const char* f) {
    size_t lines = 0;
    FILE* pFile;
    pFile = fopen(f, "r");
    if (pFile == NULL) {
        cerr << "[Error!] can't open input file " << f << endl;
        return 0;
    }
    while (EOF != fscanf(pFile, "%*[^\n]") && EOF != fscanf(pFile, "%*c")) ++lines;
    fclose(pFile);
    return lines;
}

size_t ncols(std::ifstream& is, int skip = 0) {
    size_t nc = 0;
    std::string firstLine;
    while (skip-- >= 0) std::getline(is, firstLine);
    std::stringstream ss(firstLine);
    std::string col;
    while (std::getline(ss, col, tab_delim)) {
        ++nc;
    }
    return nc;
}

size_t ncols(const char* f, int skip = 0) {
    std::ifstream is(f);
    if (!is.is_open()) {
        cerr << "[Error!] can't open input file " << f << endl;
        return 0;
    }

    return ncols(is, skip);
}

// refer to http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
std::istream& safeGetline(std::istream& is, std::string& t) {
    t.clear();
    // The characters in the stream are read one-by-one using a std::streambuf.
    // That is faster than reading them one-by-one using the std::istream.
    // Code that uses streambuf this way must be guarded by a sentry object.
    // The sentry object performs various tasks,
    // such as thread synchronization and updating the stream state.
    std::istream::sentry se(is, true);
    std::streambuf* sb = is.rdbuf();

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

bool is_nz(size_t r1, size_t r2) {
    if (abdFile.empty()) return true;
    Distance _minCV = 1;
    for (size_t i = 0; i < nABD; ++i) {
        Distance m1 = ABD(r1, i);
        Distance m2 = ABD(r2, i);
        if (m1 > _minCV || m2 > _minCV) {  // compare only at least one >2
            return true;
        }
    }
    return false;
}

#pragma omp declare reduction(merge_size_t : std::vector<size_t> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))
#pragma omp declare reduction(merge_double : std::vector<double> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

void gen_tnf_graph(Graph& g, Similarity cutoff) {
    ProgressTracker progress = ProgressTracker(nobs);
    std::vector<size_t>& from = g.from;
    std::vector<size_t>& to = g.to;
    std::vector<double>& sTNF = g.sTNF;
    size_t TILE = 10;
    try {
        TILE = std::max(
            (size_t)((CacheSize() * 1024.) / (2 * sizeof(float) * nTNF + maxEdges * (2 * sizeof(size_t) + 1 * sizeof(double)))),
            (size_t)10);
    } catch (...) {
    }
    // printf("TILE: %d\n", TILE);

    const double floor_preProb_cutoff = log((1.0 / (1. - cutoff)) - 1.0);
    // #pragma omp parallel for schedule(dynamic, 1) proc_bind(spread) reduction(merge_size_t: from) reduction(merge_size_t: to)
    // reduction(merge_double: sTNF)

#pragma omp parallel for schedule(dynamic, 1) reduction(merge_size_t : from) reduction(merge_size_t : to) \
    reduction(merge_double : sTNF)
    for (size_t ii = 0; ii < nobs; ii += TILE) {
        double *graph_d, *graph_h;
        cudaMalloc((void**)&graph_d, TILE * TILE * sizeof(double));
        cudaMallocHost((void**)&graph_h, TILE * TILE * sizeof(double));
        std::vector<std::priority_queue<std::pair<size_t, double>, std::vector<std::pair<size_t, double>>, CompareEdge>> edges(TILE);
        size_t matrix_y = min(TILE, (nobs - ii));
        for (size_t jj = 0; jj < nobs; jj += TILE) {
            size_t matrix_x = min(TILE, (nobs - jj));
            TIMERSTART(1);
            if (jj == 0) {
                size_t bloqs = ((matrix_x * matrix_y) + numThreads2 - 1) / numThreads2;
                get_tnf_graph<<<bloqs, numThreads2>>>(graph_d, TNF_d, contig_log, matrix_y, matrix_x, ii, jj, floor_preProb_cutoff);

                /*
                get_tnf_graph2<<<matrix_y, numThreads2>>>(graph_d, TNF_d, contig_log, matrix_y, matrix_x, ii, jj,
                                                          floor_preProb_cutoff);
                */
            }
            cudaDeviceSynchronize();
            TIMERSTOP(1);
            exit(1);

            cudaMemcpy(graph_h, graph_d, TILE * matrix_x * sizeof(double), cudaMemcpyDeviceToHost);
            if (jj + TILE <= nobs) {
                size_t matrix_next_x = min(TILE, (nobs - jj - TILE));
                size_t bloqs = ((matrix_next_x * matrix_y) + numThreads2 - 1) / numThreads2;
                get_tnf_graph<<<bloqs, numThreads2>>>(graph_d, TNF_d, contig_log, matrix_y, matrix_next_x, ii, jj + TILE,
                                                      floor_preProb_cutoff);
                /*
                get_tnf_graph2<<<matrix_y, numThreads2>>>(graph_d, TNF_d, contig_log, matrix_y, matrix_next_x, ii, jj + TILE,
                                                          floor_preProb_cutoff);
                */
            }
            for (size_t i = ii; i < ii + TILE && i < nobs; ++i) {
                size_t que_index = i - ii;
                size_t graph_des = que_index * matrix_x;
                for (size_t j = jj; j < jj + TILE && j < nobs; ++j) {
                    if (i == j || !is_nz(i, j)) continue;
                    double sTNF = graph_h[graph_des + (j - jj)];
                    if (sTNF && (edges[que_index].size() < maxEdges ||
                                 (edges[que_index].size() == maxEdges && sTNF > edges[que_index].top().second))) {
                        if (edges[que_index].size() == maxEdges) edges[que_index].pop();
                        edges[que_index].push(std::make_pair(j, sTNF));
                    }
                }
            }
        }
        for (size_t k = 0; k < TILE; ++k) {
            while (!edges[k].empty()) {
                std::pair<size_t, double> edge = edges[k].top();
                if ((ii + k) < edge.first) {
                    sTNF.push_back(edge.second);
                    from.push_back((ii + k));
                    to.push_back(edge.first);
                }
                edges[k].pop();
            }
        }
        if (verbose) {
            progress.track(TILE);
            if (omp_get_thread_num() == 0 && progress.isStepMarker()) {
                verbose_message("Building TNF Graph %s [%.1fGb / %.1fGb]                           \r", progress.getProgress(),
                                getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);
            }
        }
        cudaFree(graph_d);
        cudaFreeHost(graph_h);
    }

    verbose_message("Finished Building TNF Graph (%d edges) [%.1fGb / %.1fGb]                                          \n",
                    g.getEdgeCount(), getUsedPhysMem(), getTotalPhysMem() / 1024 / 1024);

    g.sTNF.shrink_to_fit();
    g.to.shrink_to_fit();
    g.from.shrink_to_fit();
}

size_t gen_tnf_graph_sample(double coverage = 1., bool full = false) {
    size_t _nobs = full ? nobs : std::min(nobs, (size_t)2500);

    // std::vector<unsigned char> connected_nodes;
    // connected_nodes.resize(_nobs);

    std::vector<size_t> idx(nobs);
    std::iota(idx.begin(), idx.end(), 0);
    random_unique(idx.begin(), idx.end(), _nobs);

    double *max_nobs_d, *max_nobs_h;
    cudaMallocHost((void**)&max_nobs_h, _nobs * sizeof(double));
    cudaMalloc((void**)&max_nobs_d, _nobs * sizeof(double));
    launch_tnf_max_prob_sample_kernel(idx, max_nobs_d, max_nobs_h, _nobs);
    getError("launch_tnf_max_prob_sample_kernel");
    cudaFree(max_nobs_d);

    std::sort(max_nobs_h, max_nobs_h + _nobs, std::greater<double>());

    size_t p = 999, pp = 1000;
    double cov = 0, pcov = 0;
    int round = 0, counton = 0;

    for (; p > 700;) {
        round++;

        double cutoff = (double)p / 1000.;

        while (max_nobs_h[counton] >= cutoff && counton < _nobs) counton++;

        cov = (double)counton / _nobs;

        if (cov >= coverage) {
            // previous cov is closer to coverage then choose prev p instead current p
            if (cov - coverage > coverage - pcov) {
                p = pp;
                cov = pcov;
            }

            break;
        } else
            verbose_message("Preparing TNF Graph Building [pTNF = %2.1f; %d / %d (P = %2.2f%%) round %d]               \r",
                            (double)p / 10., counton, _nobs, cov * 100, round);
        pp = p;
        pcov = cov;

        if (p > 990)              // 99.9, 99.6, 99.3, 99.0
            p -= rand() % 3 + 1;  // choose from 1,2,3
        else if (p > 900)         // 98.5, 98, 97.5, ... 90.0
            p -= rand() % 3 + 3;  // choose from 3,4,5
        else                      // 89, 88, 87, ..., 70
            p -= rand() % 3 + 9;  // choose from 9,10,11
    }

    cudaFreeHost(max_nobs_h);
    return p;
}

void rescue_singletons(ClassMap& cls) {
    // handle singleton bins that are of cluster size themselves
    verbose_message("There are %d bins already\n", cls.size());
    std::unordered_set<size_t> large_unbinned;
    for (auto i = 0; i < nobs; i++) {
        if (sizes[i] >= minClsSize) large_unbinned.insert(i);
        /*
        if (seqs[i].size() >= minClsSize) {
            large_unbinned.insert(i);
        }
        */
    }
    for (auto it = cls.begin(); it != cls.end(); ++it) {
        size_t kk = it->first;
        size_t s = 0, s1 = 0;

        for (auto it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
            if (*it2 < (int)nobs) {
                if (sizes[*it2] >= minClsSize) large_unbinned.erase(*it2);  // it was binned!
                // if (seqs[*it2].size() >= minClsSize) large_unbinned.erase(*it2);  // it was binned!
            }
        }
    }
    if (verbose && large_unbinned.size() > 0)
        verbose_message("Rescued %d large contig(s) into singleton bin(s)\n", large_unbinned.size());
    for (auto id : large_unbinned) {
        assert(cls.find(id) == cls.end());
        cls[id].push_back(id);
    }
}

void output_bins(ClassMap& cls) {
#pragma omp parallel
    {
#pragma omp single
        {
            Distance binnedSize = 0, binnedSize1 = 0;
            std::vector<size_t> clsMap(nobs + nobs1, 0);

            size_t bin_id = 1;  // start with bin #1
            for (auto it = cls.begin(); it != cls.end(); ++it) {
                size_t kk = it->first;
                assert(kk >= 0);
                size_t s = 0, s1 = 0;
                {
                    const auto& cluster = it->second;  // in new block for compatiblity with old OpenMP standard that does not support
                                                       // references in private vars

                    for (auto it2 = cluster.begin(); it2 != cluster.end(); ++it2) {
                        if (*it2 < (int)nobs) {
                            // s += seqs[*it2].size();
                            s += sizes[*it2];
                        } else {
                            // s1 += small_seqs[*it2 - nobs].size();
                            s1 += small_sizes[*it2 - nobs];
                        }
                    }

                    if (s + s1 < minClsSize) {
                        continue;
                    }

                    for (size_t i = 0; i < cluster.size(); ++i) {
                        assert(cluster[i] < (int)clsMap.size());
                        clsMap[cluster[i]] = kk + 1;
                    }
                }

                binnedSize += s;
                binnedSize1 += s1;

#pragma omp task
                if (!noBinOut) {
                    auto& cluster = it->second;  // in new block for compatiblity with old OpenMP standard that does not support
                                                 // references in private vars
                    std::string outFile_cls = outFile + ".";
                    outFile_cls.append(boost::lexical_cast<std::string>(bin_id));
                    if (!onlyLabel) outFile_cls.append(".fa");
                    std::sort(cluster.begin(), cluster.end());  // deterministic ordering of contigs within bins

                    size_t bases = 0;
                    std::ofstream os(outFile_cls.c_str());
                    if (!os) {
                        cerr << "[Error!] Could not write to " << outFile_cls << endl;
                        exit(1);
                    }
                    char os_buffer[buf_size];
                    os.rdbuf()->pubsetbuf(os_buffer, buf_size);
                    for (auto it2 = cluster.begin(); it2 != cluster.end(); ++it2) {
                        std::string_view& label = (*it2 < (int)nobs) ? contig_names[*it2] : small_contig_names[*it2 - nobs];
                        if (onlyLabel) {
                            os << label << line_delim;
                        } else {
                            std::string_view& seq = (*it2 < (int)nobs) ? seqs[*it2] : small_seqs[*it2 - nobs];
                            printFasta(os, label, seq);
                            bases += seq.size();
                        }
                    }
                    os.close();
                    if (!os) {
                        cerr << "[Error!] Failed to write to " << outFile_cls << endl;
                        exit(1);
                    }

                    if (debug)
                        cout << "Bin " << bin_id << " (" << bases << " bases in " << cluster.size()
                             << " contigs) was saved to: " << outFile_cls << endl;
                }

                bin_id++;
            }

            if (saveCls) {
#pragma omp task
                {
                    if (verbose) verbose_message("Saving cluster membership matrix to %s\n", outFile.c_str());

                    std::ofstream os(outFile.c_str());
                    if (!os) {
                        cerr << "[Error!] Could not write cluster membership to " << outFile << endl;
                        exit(1);
                    }
                    char os_buffer[buf_size];
                    os.rdbuf()->pubsetbuf(os_buffer, buf_size);

                    for (size_t i = 0; i < nobs; ++i) {
                        os << contig_names[i] << tab_delim << clsMap[i] << line_delim;
                    }
                    for (size_t i = nobs; i < nobs + nobs1; ++i) {
                        os << small_contig_names[i - nobs] << tab_delim << clsMap[i] << line_delim;
                    }

                    os.flush();
                    os.close();
                    if (!os) {
                        cerr << "[Error!] Failed to write cluster membership to " << outFile << endl;
                        exit(1);
                    }
                }
            }

            if (outUnbinned) {
#pragma omp task
                {
                    std::string outFile_cls = outFile + ".";
                    outFile_cls.append("unbinned");
                    if (!onlyLabel) outFile_cls.append(".fa");

                    if (verbose) verbose_message("Saving unbinned contigs to %s\n", outFile_cls.c_str());

                    std::ofstream os(outFile_cls.c_str());
                    if (!os) {
                        cerr << "[Error!] Could not to write unbinned contigs to " << outFile_cls << endl;
                        exit(1);
                    }
                    char os_buffer[buf_size];
                    os.rdbuf()->pubsetbuf(os_buffer, buf_size);

                    for (size_t i = 0; i < clsMap.size(); ++i) {
                        if (clsMap[i] == 0) {
                            std::string_view& label = ((i < nobs) ? contig_names[i] : small_contig_names[i - nobs]);
                            if (onlyLabel) {
                                os << label << line_delim;
                            } else {
                                std::string_view& seq = (i < nobs) ? seqs[i] : small_seqs[i - nobs];
                                printFasta(os, label, seq);
                            }
                        }
                    }
                    os.flush();
                    os.close();
                    if (!os) {
                        cerr << "[Error!] Failed to write unbinned contigs to " << outFile_cls << endl;
                        exit(1);
                    }
                }
            }

#pragma omp taskwait
            if (verbose) {
                verbose_message("%2.2f%% (%lld bases) of large (>=%d) and %2.2f%% (%lld bases) of small (<%d) contigs were binned.\n",
                                (double)binnedSize / totalSize * 100, (unsigned long long)binnedSize, minContig,
                                binnedSize1 == 0 ? 0 : (double)binnedSize1 / totalSize1 * 100, (unsigned long long)binnedSize1,
                                minContig);
            }
            cout.precision(20);
            cout << bin_id - 1 << " bins (" << binnedSize + binnedSize1 << " bases in total) formed." << std::endl;

        }  // omp single
    }      // omp parallel
}

Distance cal_abd_corr(size_t r1, size_t r2, bool is_small = false) {
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
        Distance m2 = is_small ? small_ABD(r2, i) : ABD(r2, i);
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
    return r;
}

int main(int ac, char* av[]) {
    po::options_description desc("Allowed options", 110, 110 / 2);
    desc.add_options()("help,h", "produce help message")("inFile,i", po::value<std::string>(&inFile),
                                                         "Contigs in (gzipped) fasta file format [Mandatory]")(
        "outFile,o", po::value<std::string>(&outFile),
        "Base file name and path for each bin. The default output is fasta format. Use -l option to output only contig names "
        "[Mandatory].")("abdFile,a", po::value<std::string>(&abdFile),
                        "A file having mean and variance of base coverage depth (tab delimited; the first column should be contig "
                        "names, and the first row will be considered as the header and be skipped) [Optional].")(
        "minContig,m", po::value<size_t>(&minContig)->default_value(2500), "Minimum size of a contig for binning (should be >=1500).")(
        "maxP", po::value<Similarity>(&maxP)->default_value(95),
        "Percentage of 'good' contigs considered for binning decided by connection among contigs. The greater, the more sensitive.")(
        "minS", po::value<Similarity>(&minS)->default_value(60),
        "Minimum score of a edge for binning (should be between 1 and 99). The greater, the more specific.")(
        "maxEdges", po::value<size_t>(&maxEdges)->default_value(200),
        "Maximum number of edges per node. The greater, the more sensitive.")(
        "pTNF", po::value<Similarity>(&pTNF)->default_value(0),
        "TNF probability cutoff for building TNF graph. Use it to skip the preparation step. (0: auto).")(
        "noAdd", po::value<bool>(&noAdd)->zero_tokens(), "Turning off additional binning for lost or small contigs.")(
        "cvExt", po::value<bool>(&cvExt)->zero_tokens(),
        "When a coverage file without variance (from third party tools) is used instead of abdFile from "
        "jgi_summarize_bam_contig_depths.")("minCV,x", po::value<Distance>(&minCV)->default_value(1),
                                            "Minimum mean coverage of a contig in each library for binning.")(
        "minCVSum", po::value<Distance>(&minCVSum)->default_value(1),
        "Minimum total effective mean coverage of a contig (sum of depth over minCV) for binning.")(
        "minClsSize,s", po::value<size_t>(&minClsSize)->default_value(200000), "Minimum size of a bin as the output.")(
        "numThreads,t", po::value<size_t>(&numThreads)->default_value(0), "Number of threads to use (0: use all cores).")(
        "onlyLabel,l", po::value<bool>(&onlyLabel)->zero_tokens(),
        "Output only sequence labels as a list in a column without sequences.")("saveCls", po::value<bool>(&saveCls)->zero_tokens(),
                                                                                "Save cluster memberships as a matrix format")(
        "unbinned", po::value<bool>(&outUnbinned)->zero_tokens(), "Generate [outFile].unbinned.fa file for unbinned contigs")(
        "noBinOut", po::value<bool>(&noBinOut)->zero_tokens(),
        "No bin output. Usually combined with --saveCls to check only contig memberships")(
        "seed", po::value<unsigned long long>(&seed)->default_value(0), "For exact reproducibility. (0: use random seed)")(
        "debug,d", po::value<bool>(&debug)->zero_tokens(), "Debug output")(
        "verbose,v", po::value<bool>(&verbose)->zero_tokens(), "Verbose output")("ct", po::value<int>(&numThreads2)->default_value(16),
                                                                                 "Number of cuda threads");

    po::variables_map vm;
    po::store(po::command_line_parser(ac, av).options(desc).positional({}).run(), vm);
    po::notify(vm);

    if (vm.count("help") || inFile.length() == 0 || outFile.length() == 0) {
        cerr << "\nMetaBAT: Metagenome Binning based on Abundance and Tetranucleotide frequency (version 2:" << version << "; " << DATE
             << ")" << endl;
        cerr << "by Don Kang (ddkang@lbl.gov), Feng Li, Jeff Froula, Rob Egan, and Zhong Wang (zhongwang@lbl.gov) \n" << endl;
        cerr << desc << endl << endl;

        if (!vm.count("help")) {
            if (inFile.empty()) {
                cerr << "[Error!] There was no --inFile specified" << endl;
            }
            if (outFile.empty()) {
                cerr << "[Error!] There was no --outFile specified" << endl;
            }
        }

        return vm.count("help") ? 0 : 1;
    }

    if (verbose) t1 = std::chrono::steady_clock::now();

    if (seed == 0) seed = time(0);
    srand(seed);

    if (maxP <= 0 || maxP >= 100) {
        cerr << "[Error!] maxP should be greater than 0 and less than 100" << endl;
        return 1;
    }

    if (minS <= 1 || minS >= 100) {
        cerr << "[Error!] minS should be greater than 1 and less than 100" << endl;
        return 1;
    }

    if (pTNF < 0 || pTNF >= 100) {
        cerr << "[Error!] pTNF should be >= 0 and < 100" << endl;
        return 1;
    }

    if (minContig < 1500) {
        cerr << "[Error!] Contig length < 1500 is not allowed to be used for binning." << endl;
        return 1;
    }

    if (minCV < 0) {
        cerr << "[Error!] minCV should be non-negative" << endl;
        return 1;
    }
    minCVSum = std::max(minCV, minCVSum);

    boost::filesystem::path dir(outFile);
    boost::system::error_code ec;
    if (dir.parent_path().string().length() > 0) {
        if (boost::filesystem::is_regular_file(dir.parent_path())) {
            cerr << "Cannot create directory: " << dir.parent_path().string() << ", which exists as a regular file." << endl;
            return 1;
        }
        if (!boost::filesystem::is_directory(dir.parent_path()) && !boost::filesystem::create_directory(dir.parent_path(), ec)) {
            cerr << "Cannot create directory: " << dir.parent_path().string() << ": " << ec << endl;
            return 1;
        }
    }

    print_message(
        "MetaBAT 2 (%s) using minContig %d, minCV %2.1f, minCVSum %2.1f, maxP %2.0f%%, minS %2.0f, maxEdges %d and minClsSize %d. "
        "with random seed=%lld\n",
        version.c_str(), minContig, minCV, minCVSum, maxP, minS, maxEdges, minClsSize, seed);

    maxP /= 100., minS /= 100.;

    if (numThreads == 0)
        numThreads = omp_get_max_threads();
    else
        numThreads = std::min(numThreads, (size_t)omp_get_max_threads());

    if (numThreads2 % 32 != 0) {
        cerr << "numThreads2 must be a multiple of 32" << endl;
        return 1;
    }

    omp_set_num_threads(numThreads);
    verbose_message("Executing with %d CPU threads and %d CUDA threads\n", numThreads, numThreads2);

    nobs = 0, nobs1 = 0;

    std::unordered_map<std::string_view, size_t> contigs;
    std::unordered_map<std::string_view, size_t> small_contigs;

    const int nNonFeat = cvExt ? 1 : 3;  // number of non features
    bool hasABD = abdFile.length() > 0;

    // validate fasta and depths file (abd) have same set of sequence identifiers (in same ordering)
    // todo read fasta first, then read depths, then validate
    {
        // need to handle the case where more data in assembly.fa than depth.txt (but contigs should be in order)
        std::unordered_set<std::string> inDepth;

        // todo refactor into validate depths file method
        if (hasABD) {
            verbose_message("Parsing abundance file\n");
            if (countLines(abdFile.c_str()) < 2) {  // the first row is header
                cerr << "[Error!] There are no lines in the abundance depth file or fasta file!" << endl;
                exit(1);
            }
            nABD = ncols(abdFile.c_str(), 1) - nNonFeat;  // num of features (excluding the first three columns which is the
                                                          // contigName, contigLen, and totalAvgDepth);
            if (!cvExt) {
                if (nABD % 2 != 0) {
                    cerr << "[Error!] Number of columns (excluding the first column) in abundance data file is not even." << endl;
                    exit(1);
                }
                nABD /= 2;
            }

            std::ifstream is(abdFile.c_str());
            if (!is.is_open()) {
                cerr << "[Error!] can't open the contig coverage depth file " << abdFile << endl;
                return 1;
            }

            int r = -1;
            for (std::string row; safeGetline(is, row) && is.good(); ++r) {
                if (r == -1)  // the first row is header
                    continue;
                std::stringstream ss(row);
                int c = -nNonFeat;
                for (std::string col; getline(ss, col, tab_delim); ++c) {
                    if (c == -3 || (cvExt && c == -1)) {  // contig name
                        if (col.empty()) break;
                        trim_fasta_label(col);
                        inDepth.insert(col);
                        break;
                    }
                }
            }
            is.close();
        }

        verbose_message("Parsing assembly file\n");

        FILE* fp = fopen(inFile.c_str(), "r");
        if (fp == NULL) {
            cerr << "[Error!] can't open the sequence fasta file " << inFile << endl;
            return 1;
        } else {
            std::ofstream* os = NULL;
            char os_buffer[buf_size];
            std::string filteredFile_cls;
            if (outUnbinned) {
                filteredFile_cls = outFile + ".";
                filteredFile_cls.append("tooShort");
                if (!onlyLabel) {
                    filteredFile_cls.append(".fa");
                }
                os = new std::ofstream(filteredFile_cls.c_str());
                if (!os->is_open() || os->fail() || !*os) {
                    cerr << "[Error!] can't open the output bin file: " << filteredFile_cls << endl;
                    return 1;
                }
                os->rdbuf()->pubsetbuf(os_buffer, buf_size);
                if (verbose) verbose_message("Outputting contigs that are too short to %s\n", filteredFile_cls.c_str());
            }
            fseek(fp, 0L, SEEK_END);
            fsize = ftell(fp);  // obtener el tamaño del archivo
            fclose(fp);
            size_t chunk = fsize / numThreads;
            cudaError_t cudaStatus = cudaMallocHost((void**)&_mem, fsize);
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cuda_Mallo_cHost failed!");
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

            size_t char_per_thread = (fsize + numThreads - 1) / numThreads;
#pragma omp parallel shared(os)
            {
                int t = omp_get_thread_num();
                size_t nobs_l = 0;
                size_t nobs1_l = 0;
                std::vector<Distance> logSizes_l;
                std::vector<size_t> sizes_l;
                std::vector<size_t> small_sizes_l;
                std::unordered_map<std::string_view, size_t> contigs_l;
                std::unordered_map<std::string_view, size_t> small_contigs_l;
                std::vector<std::string_view> contig_names_l;
                std::vector<std::string_view> small_contig_names_l;
                std::vector<std::string_view> seqs_l;
                std::vector<std::string_view> small_seqs_l;
                size_t contig_name_i;
                size_t contig_i;
                size_t contig_size;
                for (size_t i = t * char_per_thread; i < min(fsize, (t + 1) * char_per_thread); i++) {
                    if (_mem[i] == fasta_delim) {
                        size_t cont_lines = 0;
                        i++;
                        contig_name_i = i;  // guardar el inicio del nombre del contig
                        while (_mem[i] != line_delim) i++;
                        std::string_view name(_mem + contig_name_i, i - contig_name_i);
                        i++;
                        contig_i = i;  // guardar el inicio del contig
                        while (i < fsize && _mem[i] != fasta_delim) {
                            if (_mem[i] == line_delim) cont_lines++;
                            i++;
                        }
                        std::string_view seq(_mem + contig_i, i - contig_i);
                        contig_size = seq.length() - cont_lines;
                        if (contig_size >= (int)minContig) {
                            contigs_l[name] = nobs_l++;
                            contig_names_l.push_back(name);
                            seqs_l.push_back(seq);
                            logSizes_l.push_back(LOG10(std::min(contig_size, (size_t)500000)));
                            sizes_l.push_back(contig_size);
                        } else if (contig_size >= (int)1000) {
                            small_contigs_l[name] = nobs1_l++;
                            small_contig_names_l.push_back(name);
                            small_seqs_l.push_back(seq);
                            small_sizes_l.push_back(contig_size);
                        } else if (os) {
                            if (onlyLabel) {
                                *os << name << line_delim;
                            } else {
                                printFasta(*os, name, seq);
                            }
                        }
                        i--;
                    }
                }
                for (int i = 0; i < numThreads; i++) {
#pragma omp barrier
                    if (i == t) {
                        {
                            for (auto& pair : contigs_l) pair.second += nobs;
                            for (auto& pair : small_contigs_l) pair.second += nobs1;
                            contigs.merge(contigs_l);
                            small_contigs.merge(small_contigs_l);
                            contig_names.insert(contig_names.end(), contig_names_l.begin(), contig_names_l.end());
                            small_contig_names.insert(small_contig_names.end(), small_contig_names_l.begin(),
                                                      small_contig_names_l.end());
                            seqs.insert(seqs.end(), seqs_l.begin(), seqs_l.end());
                            small_seqs.insert(small_seqs.end(), small_seqs_l.begin(), small_seqs_l.end());
                            logSizes.insert(logSizes.end(), logSizes_l.begin(), logSizes_l.end());
                            sizes.insert(sizes.end(), sizes_l.begin(), sizes_l.end());
                            small_sizes.insert(small_sizes.end(), small_sizes_l.begin(), small_sizes_l.end());
                            nobs += nobs_l;
                            nobs1 += nobs1_l;
                        }
                    }
                }
            }
        }
    }

    if (contig_names.size() != contigs.size() || small_contig_names.size() != small_contigs.size()) {
        printf("contig_names: %lu, contigs: %lu, small_contig_names: %lu, small_contigs: %lu\n", contig_names.size(), contigs.size(),
               small_contig_names.size(), small_contigs.size());
        cerr << "[Error!] Need to check whether there are duplicated sequence ids in the assembly file" << endl;
        return 1;
    }
    verbose_message("Number of large contigs >= %d are %d. \n", minContig, nobs);

    if (hasABD) {
        ABD.resize(nobs, nABD);
        ABD_VAR.resize(nobs, nABD);
        small_ABD.resize(nobs1, nABD);

        verbose_message("Reading abundance file\n");
        std::ifstream is(abdFile.c_str());
        if (!is.is_open()) {
            cerr << "[Error!] can't open the contig coverage depth file " << abdFile << endl;
            return 1;
        }

        int r = -1;
        size_t num = 0, num1 = 0, nskip = 0, nskip1 = 0;

        std::ofstream* os = NULL;
        char os_buffer[buf_size];
        std::string filteredFile_cls;
        if (outUnbinned) {
            filteredFile_cls = outFile + ".";
            filteredFile_cls.append("lowDepth");
            if (!onlyLabel) {
                filteredFile_cls.append(".fa");
            }
            os = new std::ofstream(filteredFile_cls.c_str());
            if (!os->is_open() || os->fail() || !*os) {
                cerr << "[Error!] Failed to open to " << filteredFile_cls << endl;
                return 1;
            }
            os->rdbuf()->pubsetbuf(os_buffer, buf_size);
            if (verbose) verbose_message("Outputting contigs that are too low depth to %s\n", filteredFile_cls.c_str());
        }
        // todo refactor reading file
        for (std::string row; safeGetline(is, row) && is.good(); ++r) {
            if (r == -1)  // the first row is header
                continue;

            std::stringstream ss(row);
            int c = -nNonFeat;
            Distance mean = 0, variance, meanSum = 0;
            std::string label;
            bool isLarge = false, isGood = false, isSmall = false;
            for (std::string col; getline(ss, col, tab_delim); ++c) {
                if (c == -3 || (cvExt && c == -1)) {  // contig name
                    if (col.empty()) break;
                    trim_fasta_label(col);
                    label = col;

                    if (contigs.find(label) == contigs.end()) {  // small or additional contigs
                        if (small_contigs.find(label) == small_contigs.end())
                            break;
                        else
                            isSmall = true;
                    } else
                        isLarge = true;

                    if ((isSmall && small_contigs[label] != num1) || (isLarge && contigs[label] != num)) {
                        cerr << "[Error!] the order of contigs in abundance file is not the same as the assembly file: " << label
                             << endl;
                        exit(1);
                    }
                    isGood = true;
                    continue;
                } else if (c == -2) {  // contig length
                    continue;
                } else if (c == -1) {  // mean coverage
                    continue;
                }

                bool checkMean = false, checkVar = false;
                if (cvExt) {  // abd file from 3rd party contains only mean coverage, so assuming variance = mean
                    mean = boost::lexical_cast<Distance>(col.c_str());
                    if (mean >= minCV) {  // FIXME? Issue #68
                        meanSum += mean;
                    }
                    variance = mean;
                    checkMean = true;
                    if (isLarge) {
                        ABD(num - nskip, c) = mean;
                        ABD_VAR(num - nskip, c) = mean;
                    } else {
                        small_ABD(num1 - nskip1, c) = mean;
                    }
                } else {
                    if (c % 2 == 0) {
                        checkMean = true;
                        if (isLarge)
                            mean = ABD(num - nskip, c / 2) = boost::lexical_cast<Distance>(col.c_str());
                        else
                            mean = small_ABD(num1 - nskip1, c / 2) = boost::lexical_cast<Distance>(col.c_str());
                        if (mean >= minCV) {  // FIXME? Issue #68
                            meanSum += mean;
                        }
                    } else {
                        checkVar = true;
                        if (isLarge)
                            variance = ABD_VAR(num - nskip, c / 2) = boost::lexical_cast<Distance>(col.c_str());
                        else
                            variance = boost::lexical_cast<Distance>(col.c_str());
                    }
                }

                if (checkMean) {
                    if (mean > 1e+7) {
                        cerr << "[Warning!] Need to check where the average depth is greater than 1e+7 for the contig " << label
                             << ", column " << c + 1 << endl;
                        isGood = false;
                    }
                    if (mean < 0) {
                        cerr << "[Warning!] Negative coverage depth is not allowed for the contig " << label << ", column " << c + 1
                             << ": " << mean << endl;
                        isGood = false;
                    }
                }

                if (checkVar) {
                    if (variance > 1e+14) {
                        cerr << "[Warning!] Need to check where the depth variance is greater than 1e+14 for the contig " << label
                             << ", column " << c + 1 << endl;
                        isGood = false;
                    }
                    if (variance < 0) {
                        cerr << "[Warning!] Negative variance is not allowed for the contig " << label << ", column " << c + 1 << ": "
                             << variance << endl;
                        isGood = false;
                    }
                }

                if (c == (int)(nABD * (cvExt ? 1 : 2) - 1)) {  // last data of the line, check mean coverage
                    if (meanSum < minCVSum) {
                        if (debug)
                            verbose_message("[Info] Ignored a contig (%s) having effective mean coverage %2.2f < %2.2f \n",
                                            label.c_str(), meanSum, minCVSum);
                        isGood = false;
                    }
                }

                if (!isGood) break;
            }

            if (isGood && (int)nABD != (cvExt ? c : c / 2)) {
                cerr << "[Warning!] Different number of variables for the object for the contig " << label << endl;
                isGood = false;
            }

            if (isLarge) {
                if (!isGood) {
                    ++nskip;
                    if (os) {
                        if (onlyLabel) {
                            *os << contig_names[num] << line_delim;
                        } else {
                            printFasta(*os, contig_names[num], seqs[num]);
                        }
                    }
                    contig_names[num] = "";
                    seqs[num] = "";
                    logSizes[num] = -1;
                    sizes[num] = 0;
                }
                ++num;
            } else if (isSmall) {
                if (!isGood) {
                    ++nskip1;
                    if (os) {
                        if (onlyLabel) {
                            *os << small_contig_names[num1] << line_delim;
                        } else {
                            printFasta(*os, small_contig_names[num1], small_seqs[num1]);
                        }
                    }
                    small_contig_names[num1] = "";
                    small_seqs[num1] = "";
                    small_sizes[num1] = 0;
                }
                ++num1;
            }
        }
        is.close();
        if (os) {
            os->close();
            if (!*os) {
                cerr << "[Error!] Failed to write to " << filteredFile_cls << endl;
                return 1;
            }
            delete os;
        }

        assert(nobs == num && nobs1 == num1);

        nobs = num - nskip;
        nobs1 = num1 - nskip1;

        assert(contigs.size() == nobs + nskip && small_contigs.size() == nobs1 + nskip1);

        contigs.clear();
        small_contigs.clear();

        if (debug) {
            verbose_message("nobs = %d\n", nobs);
            verbose_message("r = %d (num = %d), (nskip = %d) \n", r, num, nskip);
        }
        verbose_message("Finished reading %d contigs and %d coverages from %s\n", r, nABD, abdFile.c_str());

        seqs.erase(std::remove(seqs.begin(), seqs.end(), ""), seqs.end());
        assert(nobs == seqs.size());
        small_seqs.erase(std::remove(small_seqs.begin(), small_seqs.end(), ""), small_seqs.end());
        assert(nobs1 == small_seqs.size());
        contig_names.erase(std::remove(contig_names.begin(), contig_names.end(), ""), contig_names.end());
        assert(nobs == contig_names.size());
        small_contig_names.erase(std::remove(small_contig_names.begin(), small_contig_names.end(), ""), small_contig_names.end());
        assert(nobs1 == small_contig_names.size());
        logSizes.erase(std::remove(logSizes.begin(), logSizes.end(), -1), logSizes.end());
        assert(nobs == logSizes.size());
        sizes.erase(std::remove(sizes.begin(), sizes.end(), 0), sizes.end());
        assert(nobs == sizes.size());
        small_sizes.erase(std::remove(small_sizes.begin(), small_sizes.end(), 0), small_sizes.end());
        assert(nobs1 == small_sizes.size());
        if (debug) {
            verbose_message("seqs.size = %d, contig_names.size = %d\n", seqs.size(), contig_names.size());
        }

        if (ABD.size1() != nobs) {
            ABD.resize(nobs, nABD, true);
            ABD_VAR.resize(nobs, nABD, true);
        }
        if (small_ABD.size1() != nobs1) {
            small_ABD.resize(nobs1, nABD, true);
        }
    }
    verbose_message("Number of target contigs: %d of large (>= %d) and %d of small ones (>=%d & <%d). \n", nobs, minContig, nobs1,
                    1000, minContig);

    // prepare logsizes
    /*
    logSizes.resize(nobs);
#pragma omp parallel for
    for (size_t r = 0; r < nobs; ++r) {
        logSizes[r] = LOG10(std::min(seqs[r].size(), (size_t)500000));
    }
    */

    cudaMalloc((void**)&contig_log, nobs * sizeof(double));
    cudaMemcpy(contig_log, logSizes.data(), nobs * sizeof(double), cudaMemcpyHostToDevice);

    verbose_message("Start TNF calculation. nobs = %zd\n", nobs);

    size_t max_gpu_mem = 4000000000;  // 4gb
    TIMERSTART(TNF_CAL);

    cudaMalloc((void**)&TNF_d, nobs * 136 * sizeof(float));
    {
        ProgressTracker progress(nobs);

        seqs_h_index_i.reserve(nobs);
        seqs_h_index_e.reserve(nobs);

        // TNF.resize(nobs, nTNF);
        // TNF.clear();
        size_t cobs = 0;  // current obs
        size_t _first = 0;
        for (size_t i = 0; i < nobs; i++) {
            if (seqs[i].data() - seqs[_first].data() + seqs[i].size() > max_gpu_mem) {
                launch_tnf_kernel(cobs, _first, i - cobs);
                seqs_h_index_i.clear();
                seqs_h_index_e.clear();
                progress.track(cobs);
                verbose_message("Calculating TNF %s\r", progress.getProgress());
                _first = i;
                cobs = 0;
            }
            seqs_h_index_i.emplace_back(seqs[i].data() - seqs[_first].data());
            seqs_h_index_e.emplace_back(seqs[i].data() - seqs[_first].data() + seqs[i].size());
            cobs++;
        }
        if (cobs != 0) {
            launch_tnf_kernel(cobs, _first, nobs - cobs);
            progress.track(cobs);
            verbose_message("Calculating TNF %s\r", progress.getProgress());
            seqs_h_index_i.clear();
            seqs_h_index_e.clear();
        }
    }
    TIMERSTOP(TNF_CAL);

    verbose_message("Finished TNF calculation.                                  \n");

    ClassMap cls;
    do {
        std::vector<size_t> mems;
        {
            Graph g(nobs);
            TIMERSTART(GRAPH_SAMPLE)
            // 1. sampling graph to find minp
            if (pTNF < 1.) {
                if (nobs <= 25000) {
                    pTNF = gen_tnf_graph_sample(maxP, true);
                } else {
                    for (size_t i = 0; i < 10; ++i) {
                        verbose_message("Attempt %d of 10 to gen_tnf_graph_sample\n", i);
                        double _minp = gen_tnf_graph_sample(maxP);
                        verbose_message("\n");
                        if (_minp < 701) _minp = 700.;
                        pTNF += _minp;
                        if (i == 1 && pTNF / 2 < 701) {
                            pTNF = 700.;
                            break;
                        }
                        if (i == 9) pTNF /= 10.;
                    }
                }
            } else {
                pTNF *= 10;
            }
            TIMERSTOP(GRAPH_SAMPLE)
            verbose_message("Finished Preparing TNF Graph Building [pTNF = %2.2f]                                             \n",
                            pTNF / 10.);

            // 2. build tnf graph
            TIMERSTART(TNF_GRAPH);
            gen_tnf_graph(g, pTNF / 1000.);
            TIMERSTOP(TNF_GRAPH);

            size_t nEdges = g.sTNF.size();

            if (nEdges == 0) {
                cout << "No edges were formed by TNF." << endl;
                break;
            }

            // 3. convert sTNF to sSCR
            if (!abdFile.empty()) {
                verbose_message("Applying coverage correlations to TNF graph with %d edges\n", nEdges);
                g.sSCR.resize(nEdges, 0);
                std::vector<double> abd_distr(nEdges);
                std::vector<int> nnz(nEdges, 0);

#pragma omp parallel for
                for (size_t i = 0; i < nEdges; ++i) {
                    for (size_t j = 0; j < nABD; ++j) {
                        bool nz = false;
                        Similarity abd = 1. - cal_abd_dist(g.to[i], g.from[i], j, nz);
                        if (nz) {
                            g.sSCR[i] += abd;
                            ++nnz[i];
                        }
                    }
                    g.sSCR[i] /= nnz[i];
                }

                std::partial_sort_copy(g.sSCR.begin(), g.sSCR.end(), abd_distr.begin(), abd_distr.end());

                rank(g.sTNF, g.sTNF, "min");

                std::vector<double> sCOR;
                if (nABD >= minSample) {
                    sCOR.resize(nEdges);
#pragma omp parallel for
                    for (size_t i = 0; i < nEdges; ++i) {
                        size_t r1 = g.to[i], r2 = g.from[i];
                        sCOR[i] = cal_abd_corr(r1, r2);
                    }
                    rank(sCOR, sCOR, "max");
                }

#pragma omp parallel for
                for (size_t i = 0; i < nEdges; ++i) {
                    g.sTNF[i] = abd_distr[round(g.sTNF[i]) - 1];  // fit tnf to abd (consider abd as reference distribution)

                    if (nABD >= minSample) sCOR[i] = abd_distr[round(sCOR[i]) - 1];

                    double wTNF = 1. / (1 + nnz[i]);

                    if (nABD >= minSample)
                        g.sSCR[i] = POW(POW(g.sSCR[i], 1. - wTNF) * POW(g.sTNF[i], wTNF) * sCOR[i], 1. / 2.);  // geometric mean
                    else
                        g.sSCR[i] = POW(g.sSCR[i], 1. - wTNF) * POW(g.sTNF[i], wTNF);
                }
            } else {
                g.sSCR = g.sTNF;
            }

            std::vector<double>().swap(g.sTNF);
            ABD_VAR.clear();
            ABD_VAR.resize(0, 0, false);

            if (debug)
                cout << *std::min_element(g.sSCR.begin(), g.sSCR.end()) << " : " << *std::max_element(g.sSCR.begin(), g.sSCR.end())
                     << endl;

            // 4. build sequential graph covering x % nodes and do clustering and add more edges and do clustering again
            std::vector<size_t> oSCR;
            orderhigh(g.sSCR, oSCR);

            std::vector<size_t> node_order;

            std::vector<Similarity> p_schedule2;
            for (size_t i = 1; i <= 10; ++i) p_schedule2.push_back(maxP / 10 * i);

            mems.resize(nobs);
            std::iota(mems.begin(), mems.end(), 0);  // each is a singleton to start
            verbose_message("Traversing graph with %d nodes and %d edges\n", nobs, nEdges);
            Graph g2(nobs, true);
            size_t which_p = 0;
            for (size_t i = 0; i < nEdges; ++i) {
                size_t ii = g.to[oSCR[i]], jj = g.from[oSCR[i]];

                // 1. check if they are binned to the same cluster. if then skip generating additional edges.
                if (mems[ii] != mems[jj]) {  // || which_p < 5 allow all edges from first 5 schedule
                    if (g2.connected_nodes.find(ii) == g2.connected_nodes.end()) {
                        node_order.push_back(ii);
                        g2.connected_nodes.insert(ii);
                    }
                    if (g2.connected_nodes.find(jj) == g2.connected_nodes.end()) {
                        node_order.push_back(jj);
                        g2.connected_nodes.insert(jj);
                    }

                    Similarity scr = g.sSCR[oSCR[i]];
                    if (scr > minS) {
                        g2.sSCR.push_back(scr);
                        g2.from.push_back(jj);
                        g2.to.push_back(ii);
                        g2.incs[ii].push_back(g2.from.size() - 1);
                        g2.incs[jj].push_back(g2.from.size() - 1);
                    } else {
                        i = nEdges - 1;  // early stopping
                    }
                }

                if (g2.getEdgeCount() > 0 &&
                    ((Similarity)g2.connected_nodes.size() / g2.n >= p_schedule2[which_p] || i == nEdges - 1)) {
                    // cout << "g2.sSCR.back(): " << g2.sSCR.back() << endl;
                    label_propagation(g2, mems, node_order);
                    verbose_message(
                        "Building SCR Graph and Binning (%d vertices and %d edges) [P = %2.2f%%; %.1fGb / %.1fGb]                 "
                        "    "
                        "      \n",
                        g2.connected_nodes.size(), g2.getEdgeCount(), p_schedule2[which_p] * 100, getUsedPhysMem(),
                        getTotalPhysMem() / 1024 / 1024);

                    if (debug) {
                        std::string osfileName("cluster.log." + boost::lexical_cast<std::string>(which_p));
                        std::ofstream os(osfileName);
                        if (!os) {
                            cerr << "[Error!] Failed to write to " << osfileName << endl;
                            return 1;
                        }
                        ClassMap _cls;
                        for (size_t i = 0; i < nobs; ++i) {
                            _cls[mems[i]].push_back(i);
                        }
                        for (size_t kk = 0; kk < nobs; ++kk) {
                            if (_cls[kk].size() > 1) {
                                os << kk << " : ";
                                ContigVector& vec = _cls[kk];
                                std::sort(vec.begin(), vec.end());
                                for (auto it2 = vec.begin(); it2 != vec.end(); ++it2) {
                                    os << *it2 << ",";
                                }
                                os << endl;
                            }
                        }
                        os.close();
                        if (!os) {
                            cerr << "[Error!] Failed to write to " << osfileName << endl;
                            return 1;
                        }
                    }

                    if (++which_p == p_schedule2.size()) break;
                }
            }
        }

        for (size_t i = 0; i < nobs; ++i) {
            cls[mems[i]].push_back(i);
        }

        if (verbose) {
#pragma omp parallel for reduction(+ : totalSize)
            for (size_t i = 0; i < nobs; ++i) {
                totalSize += sizes[i];
            }
// for ƒ(auto it = seqs.begin(); it != seqs.end(); ++it) totalSize += it->size();
#pragma omp parallel for reduction(+ : totalSize1)
            for (size_t i = 0; i < nobs1; ++i) {
                totalSize1 += small_sizes[i];
            }
            // for (auto it = small_seqs.begin(); it != small_seqs.end(); ++it) totalSize1 += it->size();
        }

        // dissolve all small bins and give them another chance to be binned with large ones.
        std::vector<size_t> leftovers, toBeErased;
        for (auto it = cls.begin(); it != cls.end(); ++it) {
            size_t kk = it->first;

            size_t s = 0;

            for (auto it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
                // s += seqs[*it2].size();
                s += sizes[*it2];
            }

            if (s < minClsSize) {
                leftovers.insert(leftovers.end(), cls[kk].begin(), cls[kk].end());
                std::vector<int>().swap(cls[kk]);
                toBeErased.push_back(kk);
            }
        }
        for (auto x : toBeErased) cls.erase(x);

        leftovers.shrink_to_fit();
        std::vector<size_t>().swap(toBeErased);

        // additional binning with small contigs
        if (!noAdd && nABD >= minSample && (leftovers.size() > 0 || nobs1 > 0)) {
            size_t minCS = 10;  // minimum class size for additional recruiting

            std::vector<float> rowMat(ABD.size2());
            for (size_t r = 0; r < ABD.size1(); ++r) {
                MatrixRowType rRow(ABD, r);
                std::copy(rRow.begin(), rRow.end(), rowMat.begin());
                rank(rowMat, rowMat);
                std::copy(rowMat.begin(), rowMat.end(), rRow.begin());
            }

            unsigned long long binned_size = 0;
            for (auto it = cls.begin(); it != cls.end(); ++it) {
                size_t kk = it->first;
                for (auto it2 = cls[kk].begin(); it2 != cls[kk].end(); ++it2) {
                    binned_size += seqs[*it2].size();
                }
            }

            // 1. calculate mean corr within bins
            // 2. cal mean corr from a contig to  a bin greater than mean and assign it to the best bin over the threshold
            std::unordered_map<size_t, double> cls_corr;
#pragma omp parallel
#pragma omp single
            for (auto it = cls.begin(); it != cls.end(); ++it) {
                size_t kk = it->first;
                size_t cs = it->second.size();
                if (cs >= minCS) {
#pragma omp task
                    {
                        double corr = 0;
                        const auto& c = it->second;
                        for (size_t i = 0; i < cs; ++i) {
                            for (size_t j = i + 1; j < cs; ++j) {
                                corr += cal_abd_corr(c[i], c[j]);
                            }
                        }

                        double x = corr / (cs * (cs - 1) / 2);
#pragma omp critical(CALC_MEAN_CORR)
                        cls_corr[kk] = x;
                    }
                }
            }

            verbose_message("Binning lost contigs...          \n");
            ClassMap cls_leftovers;
#pragma omp parallel for schedule(dynamic, 1)
            for (size_t l = 0; l < leftovers.size(); ++l) {
                int best_cls = -1;
                for (auto it = cls.begin(); it != cls.end(); ++it) {
                    size_t kk = it->first;
                    const auto& c = it->second;
                    size_t cs = c.size();
                    if (cs >= minCS) {
                        double corr = 0;
                        size_t i = 0;

                        // subset
                        for (; i < minCS; ++i) corr += cal_abd_corr(c[i], leftovers[l]);

                        // early stop
                        if (corr / minCS < cls_corr[kk]) continue;

                        for (; i < cs; ++i) corr += cal_abd_corr(c[i], leftovers[l]);

                        corr /= cs;
                        if (corr >= cls_corr[kk]) {
                            if (best_cls > -1) {  // only allow unique assignment.
                                best_cls = -1;
                                break;
                            }
                            best_cls = kk;
                        }
                    }
                }
                if (best_cls > -1) {
#pragma omp critical(ADD_LEFTOVER_CONTIGS)
                    cls_leftovers[best_cls].push_back(leftovers[l]);
                }
            }

            verbose_message("Binning small contigs...          \n");
            ClassMap cls_small;
            if (nobs1 > 0) {
                // Spearman corr
                for (size_t r = 0; r < small_ABD.size1(); ++r) {
                    MatrixRowType rRow(small_ABD, r);
                    std::copy(rRow.begin(), rRow.end(), rowMat.begin());
                    rank(rowMat, rowMat);
                    std::copy(rowMat.begin(), rowMat.end(), rRow.begin());
                }

#pragma omp parallel for schedule(dynamic)
                for (size_t s = 0; s < nobs1; ++s) {
                    int best_cls = -1;
                    for (auto it = cls.begin(); it != cls.end(); ++it) {
                        size_t kk = it->first;
                        const auto& c = it->second;
                        size_t cs = c.size();
                        if (cs >= minCS) {
                            double corr = 0;
                            size_t i = 0;
                            // subset
                            for (; i < minCS; ++i) corr += cal_abd_corr(c[i], s, true);

                            // early stop
                            if (corr / minCS < cls_corr[kk]) continue;

                            for (; i < cs; ++i) corr += cal_abd_corr(c[i], s, true);

                            corr /= cs;
                            if (corr >= cls_corr[kk]) {
                                if (best_cls > -1) {  // only allow unique assignment.
                                    best_cls = -1;
                                    break;
                                }
                                best_cls = kk;
                            }
                        }
                    }
                    if (best_cls > -1) {
#pragma omp critical(ADD_SMALL_CONTIGS)
                        cls_small[best_cls].push_back(s + nobs);
                    }
                }
            }

            unsigned long long added_sum = 0;
            for (auto it = cls_leftovers.begin(); it != cls_leftovers.end(); ++it) {
                size_t kk = it->first;
                for (auto it2 = cls_leftovers[kk].begin(); it2 != cls_leftovers[kk].end(); ++it2) {
                    added_sum += sizes[*it2];
                    // added_sum += seqs[*it2].size();
                }
            }

            if (added_sum > 0) {
                if ((double)added_sum / binned_size < .10) {  // allow only at most 10% recruiting
                    for (auto it = cls_leftovers.begin(); it != cls_leftovers.end(); ++it) {
                        size_t kk = it->first;
                        cls[kk].insert(cls[kk].end(), cls_leftovers[kk].begin(), cls_leftovers[kk].end());
                    }
                    verbose_message("%2.2f%% (%lld bases) of large (>=%d) contigs were re-binned out of small bins (<%d).\n",
                                    (double)added_sum / binned_size * 100, added_sum, minContig, minClsSize);
                } else {
                    verbose_message(
                        "[Info] Additional binning of lost contigs was ignored since it was too excessive [%2.2f%% (%lld bases) "
                        "of "
                        "large (>=%d) contigs is > %2.0f%%].\n",
                        (double)added_sum / binned_size * 100, added_sum, minContig, .10 * 100);
                }
            }

            added_sum = 0;
            for (auto it = cls_small.begin(); it != cls_small.end(); ++it) {
                size_t kk = it->first;
                for (auto it2 = cls_small[kk].begin(); it2 != cls_small[kk].end(); ++it2) {
                    added_sum += small_sizes[*it2 - nobs];
                    // added_sum += small_seqs[*it2 - nobs].size();
                }
            }

            if (added_sum > 0) {
                if ((double)added_sum / totalSize1 < .15) {  // allow only at most 15% recruiting
                    for (auto it = cls_small.begin(); it != cls_small.end(); ++it) {
                        size_t kk = it->first;
                        cls[kk].insert(cls[kk].end(), cls_small[kk].begin(), cls_small[kk].end());
                    }
                } else {
                    verbose_message(
                        "[Info] Additional binning of small contigs was ignored since it was too excessive [%2.2f%% (%lld bases) "
                        "of "
                        "small (<%d) contigs is > %2.0f%%].\n",
                        (double)added_sum / totalSize1 * 100, added_sum, minContig, .10 * 100);
                }
            }
        }

        ABD.clear();
        ABD.resize(0, 0, false);
        small_ABD.clear();
        small_ABD.resize(0, 0, false);
    } while (false);

    verbose_message("Rescuing singleton large contigs\n");
    rescue_singletons(cls);

    verbose_message("Outputting bins\n");
    output_bins(cls);
    verbose_message("Finished\n");
    cudaFreeHost(_mem);
    cudaFree(TNF_d);
    cudaFree(contig_log);
    return 0;
}