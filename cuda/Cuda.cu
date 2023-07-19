// nvcc TNF.cu -lz
// ta bien
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
// #include <boost/numeric/ublas/matrix.hpp>
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

// #include "../extra/metrictime2.hpp"

namespace po = boost::program_options;

__device__ __constant__ unsigned char TNmap_d[256] = {
    2,   21,  31,  115, 101, 119, 67,  50,  135, 126, 69,  92,  116, 88,  8,   78,  47,  96,  3,   70,  106, 38,  48,  83,  16,  22,
    136, 114, 5,   54,  107, 120, 72,  41,  44,  26,  27,  23,  136, 53,  12,  81,  136, 127, 30,  110, 136, 80,  132, 123, 71,  102,
    79,  1,   35,  124, 29,  4,   136, 34,  91,  17,  136, 52,  9,   77,  136, 117, 76,  93,  136, 65,  6,   73,  136, 68,  28,  94,
    136, 113, 121, 36,  136, 10,  103, 99,  136, 87,  129, 14,  136, 136, 98,  19,  136, 97,  15,  56,  136, 131, 57,  46,  136, 136,
    122, 60,  136, 136, 42,  62,  136, 136, 7,   130, 136, 51,  133, 20,  136, 134, 89,  86,  136, 136, 104, 95,  136, 136, 49,  136,
    136, 136, 105, 136, 136, 136, 33,  136, 136, 136, 43,  136, 136, 136, 55,  136, 136, 136, 112, 136, 136, 136, 136, 136, 136, 136,
    75,  136, 136, 136, 32,  136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 100, 136, 136, 136, 63,  136,
    136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 125, 108, 136, 136, 58,  24,  136, 136, 84,  13,  136, 136, 25,  66,  136, 136,
    18,  128, 136, 136, 74,  61,  136, 136, 85,  136, 136, 136, 118, 40,  136, 136, 109, 90,  136, 136, 45,  136, 136, 136, 111, 136,
    136, 136, 82,  136, 136, 136, 59,  11,  136, 136, 64,  37,  136, 136, 0,   136, 136, 136, 39,  136, 136, 136};

__device__ __constant__ unsigned char BN[256] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

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

__global__ void get_TNF(double *TNF_d, const char *seqs_d, const size_t *seqs_d_index, size_t nobs, const size_t contigs_per_thread,
                        const size_t seqs_d_index_size) {
    // const size_t minContig = 2500;
    // const size_t minContigByCorr = 1000;
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
    // const size_t minContig = 2500;
    // const size_t minContigByCorr = 1000;
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

typedef double Distance;
typedef double Similarity;

static std::string version = "Metabat cuda 0.1";
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

// static UndirectedGraph gprob;
// static DirectedSimpleGraph paired;
//  static boost::property_map<UndirectedGraph, boost::vertex_index_t>::type gIdx;
//  static boost::property_map<UndirectedGraph, boost::edge_weight_t>::type gWgt;

static std::unordered_map<std::string_view, size_t> lCtgIdx;  // map of sequence label => local index
static std::unordered_map<size_t, size_t> gCtgIdx;            // local index => global index of contig_names and seqs
static std::unordered_map<std::string_view, size_t> ignored;  // map of sequence label => index of contig_names and seqs
static std::vector<std::string_view> contig_names;
static std::vector<std::string_view> seqs;
static std::vector<size_t> seqs_h_index_i;
static std::vector<size_t> seqs_h_index_e;
static char *_mem;
static size_t fsize;

typedef std::vector<int> ContigVector;
typedef std::set<int> ClassIdType;  // ordered
typedef std::unordered_set<int> ContigSet;
typedef std::unordered_map<int, ContigVector> ClassMap;

static ContigSet smallCtgs;
static size_t nobs = 0;
static size_t nobs2;  // number of contigs used for binning

static float *ABD;
static float *ABD_VAR;
static double *TNF;

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

void reader(int fpint, int id, size_t chunk, size_t _size, char *_mem) {
    size_t readSz = 0;
    while (readSz < _size) {
        size_t _bytesres = _size - readSz;
        readSz += pread(fpint, _mem + (id * chunk) + readSz, _bytesres, (id * chunk) + readSz);
    }
}

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

bool loadTNFFromFile(std::string saveTNFFile, size_t requiredMinContig) {
    if (saveTNFFile.empty()) return false;
    FILE *fp = fopen(saveTNFFile.c_str(), "r");
    if (fp == NULL) return false;
    fseek(fp, 0L, SEEK_END);
    size_t fsize = ftell(fp);  // obtener el tamaño del archivo
    fclose(fp);
    fsize = (fsize / sizeof(double)) - 1;  // el primer valor es el minContig
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
    fsize *= sizeof(double);
    ok = pread(fpint, (void *)TNF, fsize * sizeof(double) * 136, 8);
    if (ok != fsize) {
        std::cerr << "[Warning!] A exception occurred."
                     "Recalculating..."
                  << std::endl;
        return false;
    }
    close(fpint);
    return true;
}

void saveTNFToFile(std::string saveTNFFile, size_t requiredMinContig) {
    if (saveTNFFile.empty()) return;
    std::ofstream out(saveTNFFile.c_str(), std::ios::out | std::ios::binary);
    if (out) {
        if (1) {  // quitar small contigs de TNF
            for (auto it = smallCtgs.begin(); it != smallCtgs.end(); it++) {
                for (size_t i = 0; i < 136; i++) {
                    TNF[*it * 136 + i] = 0;
                }
            }
        }
        out.write((char *)&minContig, sizeof(size_t));
        out.write((char *)TNF, nobs * 136 * sizeof(double));
        out.close();
    } else {
        std::cout << "Error al guardar en TNF.bin" << std::endl;
    }
    out.close();
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

int main(int argc, char const *argv[]) {
    /*
    if (argc > 2) {
        n_BLOCKS = atoi(argv[1]);
        n_THREADS = atoi(argv[2]);
        if (argc > 3) {
            inFile = argv[3];
        }
    }
    */
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
        "contig index (ordered by), its mate contig index, and supporting mean read coverage. The first row will be considered as the "
        "header and be skipped) [Optional]")(
        "p1", po::value<Similarity>(&p1)->default_value(0),
        "Probability cutoff for bin seeding. It mainly controls the number of potential bins and their specificity. The higher, the "
        "more (specific) bins would be. (Percentage; Should be between 0 and 100)")(
        "p2", po::value<Similarity>(&p2)->default_value(0),
        "Probability cutoff for secondary neighbors. It supports p1 and better be close to p1. (Percentage; Should be between 0 and "
        "100)")("minProb", po::value<Similarity>(&minProb)->default_value(0),
                "Minimum probability for binning consideration. It controls sensitivity. Usually it should be >= 75. (Percentage; "
                "Should be between 0 and 100)")(
        "minBinned", po::value<Similarity>(&minBinned)->default_value(0),
        "Minimum proportion of already binned neighbors for one's membership inference. It contorls specificity. Usually it would be "
        "<= 50 (Percentage; Should be between 0 and 100)")(
        "verysensitive", po::value<bool>(&verysensitive)->zero_tokens(),
        "For greater sensitivity, especially in a simple community. It is the shortcut for --p1 90 --p2 85 --pB 20 --minProb 75 "
        "--minBinned 20 --minCorr 90")(
        "sensitive", po::value<bool>(&sensitive)->zero_tokens(),
        "For better sensitivity [default]. It is the shortcut for --p1 90 --p2 90 --pB 20 --minProb 80 --minBinned 40 --minCorr 92")(
        "specific", po::value<bool>(&specific)->zero_tokens(),
        "For better specificity. Different from --sensitive when using correlation binning or ensemble binning. It is the shortcut "
        "for --p1 90 --p2 90 --pB 30 --minProb 80 --minBinned 40 --minCorr 96")(
        "veryspecific", po::value<bool>(&veryspecific)->zero_tokens(),
        "For greater specificity. No correlation binning for short contig recruiting. It is the shortcut for --p1 90 --p2 90 --pB 40 "
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
        "Minimum size of a contig to be considered for recruiting by pearson correlation coefficients (activated only if # of samples "
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

    print_message(
        "MetaBAT 1 CUDA (%s) using p1 %2.1f%%, p2 %2.1f%%, p3 %2.1f%%, minProb %2.1f%%, minBinned %2.0f%%, minCV %2.1f, "
        "minContig %d, minContigByCorr %d, minCorr %2.0f%%, paired %d, and %d bootstrapping\n",
        version.c_str(), p1 * 100, p2 * 100, p3 * 100, minProb * 100, minBinned * 100, minCV, minContig, minContigByCorr, minCorr,
        pairFile.length() > 0, useEB ? B : 0);

    print_message("Advanced setting CPU threads %s, GPU threads(per block) %s, GPU block (per grid) %s\n", numThreads, numThreads2,
                  numBlocks);

        if (numThreads == 0) numThreads = std::thread::hardware_concurrency();  // obtener el numero de hilos maximo

    // TIMERSTART(total);
    nobs = 0;
    int nresv = 0;

    FILE *fp = fopen(inFile.c_str(), "r");
    if (fp == NULL) {
        std::cout << "Error opening file: " << inFile << std::endl;
        return 1;
    } else {
        // TIMERSTART(load_file);
        fseek(fp, 0L, SEEK_END);
        fsize = ftell(fp);  // obtener el tamaño del archivo
        fclose(fp);
        size_t chunk = fsize / numThreads;
        cudaMallocHost((void **)&_mem, fsize);
        int fpint = open(inFile.c_str(), O_RDWR | O_CREAT, S_IREAD | S_IWRITE | S_IRGRP | S_IROTH);
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
        // TIMERSTOP(load_file);

        // TIMERSTART(read_file);
        size_t __min = std::min(minContigByCorr, minContigByCorrForGraph);
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
        for (size_t i = 0; i < fsize; i++) {  // leer el archivo caracter por caracter
            if (_mem[i] == fasta_delim) {
                i++;
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
        // TIMERSTOP(read_file);
    }
    // std::cout << contig_names[0] << std::endl;

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
        cudaMallocHost((void **)&ABD, nobs * nABD * sizeof(float));
        cudaMallocHost((void **)&ABD_VAR, nobs * nABD * sizeof(float));

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
                    mean = ABD[(r - nskip) * nABD + c] = std::stod(col.c_str());
                    meanSum += mean;
                    variance = ABD_VAR[(r - nskip) * nABD + c] = mean;
                    checkMean = true;
                } else {
                    if (c % 2 == 0) {
                        mean = ABD[(r - nskip) * nABD + (c / 2)] = std::stod(col.c_str());
                        checkMean = true;
                    } else {
                        variance = ABD_VAR[(r - nskip) * nABD + (c / 2)] = std::stod(col.c_str());
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

        /*
        if (ABD.size1() != nobs) {
            ABD.resize(nobs, nABD, true);
            ABD_VAR.resize(nobs, nABD, true);
        }
        */

        assert(rABD.size() == nobs);
    }

    // calcular matriz de tetranucleotidos
    // TIMERSTART(tnf);
    std::cout << nobs << std::endl;
    cudaMallocHost((void **)&TNF, nobs * 136 * sizeof(double));
    if (!loadTNFFromFile(saveTNFFile, minContig)) {  // calcular TNF en paralelo en GPU de no estar guardado
        double *TNF_d;
        char *seqs_d;
        size_t *seqs_d_index;
        dim3 blkDim(numThreads2, 1, 1);
        dim3 grdDim(numBlocks, 1, 1);
        cudaMalloc(&TNF_d, nobs * 136 * sizeof(double));
        cudaMalloc(&seqs_d, fsize);
        cudaMalloc(&seqs_d_index, 2 * nobs * sizeof(size_t));
        cudaStream_t streams[n_STREAMS];
        cudaMemcpyAsync(seqs_d, _mem, fsize, cudaMemcpyHostToDevice);
        size_t contig_per_kernel = nobs / n_STREAMS;
        // std::cout << "contig_per_kernel: " << contig_per_kernel << std::endl;
        for (int i = 0; i < n_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);
            size_t contig_to_process = contig_per_kernel;
            size_t _des = contig_per_kernel * i;
            size_t TNF_des = _des * 136;

            if (i == n_STREAMS - 1) contig_to_process += (nobs % n_STREAMS);
            size_t contigs_per_thread = (contig_to_process + (numThreads2 * numBlocks) - 1) / (numThreads2 * numBlocks);

            for (size_t j = 0; j < contig_to_process; j++) {
                seqs_h_index_i.emplace_back(&seqs[gCtgIdx[_des + j]][0] - _mem);
                seqs_h_index_e.emplace_back(&seqs[gCtgIdx[_des + j]][0] - _mem + seqs[gCtgIdx[_des + j]].size());
            }

            cudaMemcpyAsync(seqs_d_index + _des, seqs_h_index_i.data() + _des, contig_to_process * sizeof(size_t),
                            cudaMemcpyHostToDevice, streams[i]);
            cudaMemcpyAsync(seqs_d_index + nobs + _des, seqs_h_index_e.data() + _des, contig_to_process * sizeof(size_t),
                            cudaMemcpyHostToDevice, streams[i]);
            get_TNF<<<grdDim, blkDim, 0, streams[i]>>>(TNF_d + TNF_des, seqs_d, seqs_d_index + _des, contig_to_process,
                                                       contigs_per_thread, nobs);
            cudaMemcpyAsync(TNF + TNF_des, TNF_d + TNF_des, contig_to_process * 136 * sizeof(double), cudaMemcpyDeviceToHost,
                            streams[i]);
        }
        for (int i = 0; i < n_STREAMS; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
        seqs_h_index_i.clear();
        seqs_h_index_e.clear();
        cudaFree(TNF_d);
        cudaFree(seqs_d);
        cudaFree(seqs_d_index);
        saveTNFToFile(saveTNFFile, minContig);
    }
    verbose_message("Finished TNF calculation.                                  \n");
    // TIMERSTOP(tnf);

    cudaFreeHost(_mem);
    // TIMERSTOP(total);
    return 0;
}
