// nvcc TNF.cu -lz
// ta bien
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/program_options.hpp>
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

typedef double Distance;
typedef std::pair<int, Distance> DistancePair;

static const char tab_delim = '\t';

std::string inFile;
std::string abdFile;
int numThreads;
int n_BLOCKS;
int n_THREADS;
char *_mem;
size_t fsize;
std::vector<size_t> seqs_h_index_i;
std::vector<size_t> seqs_h_index_e;

std::vector<std::string_view> seqs;
std::vector<std::string_view> contig_names;
std::unordered_map<std::string_view, size_t> ignored;
std::unordered_map<std::string_view, size_t> lCtgIdx;
std::unordered_map<size_t, size_t> gCtgIdx;
std::unordered_set<int> smallCtgs;
boost::numeric::ublas::matrix<float> ABD;
boost::numeric::ublas::matrix<float> ABD_VAR;

static size_t minContig = 2500;                // minimum contig size for binning
static size_t minContigByCorr = 1000;          // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000;  // for graph generation purpose
size_t nobs;
size_t nresv;
double *TNF;

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

    while (EOF != fscanf(pFile, "%*[^\n]") && EOF != fscanf(pFile, "%*c")) {
        ++lines;
    }

    fclose(pFile);

    return lines;
}

size_t ncols(std::ifstream &is, int skip = 0) {
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

size_t ncols(const char *f, int skip = 0) {
    std::ifstream is(f);
    if (!is.is_open()) {
        std::cerr << "[Error!] can't open input file " << f << std::endl;
        return 0;
    }

    return ncols(is, skip);
}

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

void reader(int fpint, int id, size_t chunk, size_t _size, char *_mem) {
    size_t readSz = 0;
    while (readSz < _size) {
        size_t _bytesres = _size - readSz;
        readSz += pread(fpint, _mem + (id * chunk) + readSz, _bytesres, (id * chunk) + readSz);
    }
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

    boost::po::options_description desc("Allowed options", 110, 110 / 2);
    desc.add_options().("help,h", "produce help message");
    desc.add_options().("inFile,i", po::value<std::string>(&inFile), "Contigs in fasta file format [Mandatory]");
    desc.add_options().("abdFile,a", po::value<std::string>(&abdFile),
                        "A file having mean and variance of base coverage depth (tab delimited; the first column should be "
                        "contig names, and the first row will be considered as the header and be skipped) [Optional]");
    desc.add_options().("numThreads,t", po::value<size_t>(&numThreads)->default_value(0),
                        "Number of threads to use (0: use all cores)");
    desc.add_options().("cb", po::value<int>(&n_BLOCKS)->default_value(512), "Number of blocks");
    desc.add_options().("ct", po::value<int>(&n_THREADS)->default_value(16), "Number of threads");

    if (numThreads == 0) numThreads = std::thread::hardware_concurrency();  // obtener el numero de hilos maximo

    TIMERSTART(total);

    FILE *fp = fopen(inFile.c_str(), "r");
    if (fp == NULL) {
        std::cout << "Error opening file: " << inFile << std::endl;
        return 1;
    } else {
        TIMERSTART(load_file);
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
        seqs_h_index_i.reserve(fsize % __min);
        seqs_h_index_e.reserve(fsize % __min);
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
                    seqs_h_index_i.emplace_back(contig_i);
                    seqs_h_index_e.emplace_back(contig_e);
                    lCtgIdx[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] = nobs;
                    gCtgIdx[nobs++] = seqs.size();
                } else {
                    ignored[std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i)] = seqs.size();
                }
                contig_names.emplace_back(std::string_view(_mem + contig_name_i, contig_name_e - contig_name_i));
                seqs.emplace_back(std::string_view(_mem + contig_i, contig_e - contig_i));
            }
        }
        seqs_h_index_i.shrink_to_fit();  // liberar memoria no usada
        seqs_h_index_e.shrink_to_fit();  // liberar memoria no usada
        seqs.shrink_to_fit();            // liberar memoria no usada
        contig_names.shrink_to_fit();    // liberar memoria no usada
        TIMERSTOP(read_file);
    }
    std::cout << seqs.size() << " contigs" << std::endl;
    std::cout << nobs << " contigs with size >= " << minContig << std::endl;

    // cargar el archivo de abundancias
    if (1) {
        size_t nABD = 0;
        const int nNonFeat = cvExt ? 1 : 3;  // number of non-feature columns
        if (abdFile.length() > 0) {
            smallCtgs.clear();
            std::unordered_map<std::string_view, size_t> lCtgIdx2;
            std::unordered_map<size_t, size_t> gCtgIdx2;

            nobs = std::min(nobs, countLines(abdFile.c_str()) - 1);  // la primera linea es el header
            if (nobs < 1) {
                cerr << "[Error!] There are no lines in the abundance depth file or fasta file!" << endl;
                exit(1);
            }
            nABD = ncols(abdFile.c_str(), 1) - nNonFeat;
            // num of features (excluding the first three columns which is the contigName,
            // contigLen, and totalAvgDepth);
            if (!cvExt) {
                if (nABD % 2 != 0) {
                    cerr << "[Error!] Number of columns (excluding the first column) in abundance data file "
                            "is not even."
                         << endl;
                    return 1;
                }
                nABD /= 2;
            }
            ABD.resize(nobs, nABD);
            ABD_VAR.resize(nobs, nABD);

            std::ifstream is(abdFile.c_str());
            if (!is.is_open()) {
                cerr << "[Error!] can't open the contig coverage depth file " << abdFile << endl;
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
                                /*
                                verbose_message("[Warning!] Cannot find the contig (%s) in abundance file from the "
                                    "assembly file\n",
                                    label.c_str());
                                    */
                            } else if (debug) {
                                /*
                                verbose_message("[Info] Ignored a small contig (%s) having length %d < %d\n", label.c_str(),
                                                seqs[ignored[label]].size(), minContig);
                                                */
                            }
                            isGood = false;  // cannot find the contig from fasta file. just skip it!
                            break;
                        }
                        continue;
                    } else if (c == -2) {
                        continue;
                    } else if (c == -1) {
                        meanSum = boost::lexical_cast<Distance>(col.c_str());
                        if (meanSum < minCVSum) {
                            if (debug) {
                                /*
                                    verbose_message("[Info] Ignored a contig (%s) having mean coverage %2.2f < %2.2f \n",
                                   label.c_str(), meanSum, minCVSum);
                                                    */
                            }

                            isGood = false;  // cannot find the contig from fasta file. just skip it!
                            break;
                        }
                        continue;
                    }

                    assert(r - nskip >= 0 && r - nskip < (int)nobs);

                    bool checkMean = false, checkVar = false;

                    if (cvExt) {
                        mean = ABD(r - nskip, c) = boost::lexical_cast<Distance>(col.c_str());
                        meanSum += mean;
                        variance = ABD_VAR(r - nskip, c) = mean;
                        checkMean = true;
                    } else {
                        if (c % 2 == 0) {
                            mean = ABD(r - nskip, c / 2) = boost::lexical_cast<Distance>(col.c_str());
                            checkMean = true;
                        } else {
                            variance = ABD_VAR(r - nskip, c / 2) = boost::lexical_cast<Distance>(col.c_str());
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
                            std::cerr << "[Error!] Negative coverage depth is not allowed for the contig " << label << ", column "
                                      << c + 1 << ": " << mean << std::endl;
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
                            std::cerr << "[Error!] Negative variance is not allowed for the contig " << std::label << ", column "
                                      << c + 1 << ": " << variance << std::endl;
                            return 1;
                        }
                        if (maxVarRatio > 0.0 && mean > 0 && variance / mean > maxVarRatio) {
                            std::cerr << "[Warning!] Skipping contig due to >maxVarRatio variance: " << std::variance << " / " << mean
                                      << " = " << variance / mean << ": " << label << std::endl;
                            isGood = false;
                            break;
                        }
                    }

                    if (c == (int)(nABD * (cvExt ? 1 : 2) - 1)) {
                        if (meanSum < minCVSum) {
                            if (debug) {
                                /*
                                verbose_message("[Info] Ignored a contig (%s) having mean coverage %2.2f < %2.2f \n", label.c_str(),
                                                meanSum, minCVSum);
                                                */
                            }

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
                    lCtgIdx2[label] = r - nskip;  // local index
                    gCtgIdx2[r - nskip] = _gidx;  // global index
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
            /*
            verbose_message(
                "Finished reading %d contigs (using %d including %d short contigs) and %d coverages from "
                "%s\n",
                r, r - nskip - nresv, smallCtgs.size() - nresv, nABD, abdFile.c_str());
                */

            if ((specific || veryspecific) && nABD < minSamples) {
                std::cerr << "[Warning!] Consider --superspecific for better specificity since both --specific "
                             "and --veryspecific would be the same as --sensitive when # of samples ("
                          << nABD << ") < minSamples (" << minSamples << ")" << std::endl;
            }

            if (nABD < minSamples) {
                std::cerr << "[Info] Correlation binning won't be applied since the number of samples (" << nABD << ") < minSamples ("
                          << minSamples << ")" << std::endl;
            }

            for (std::unordered_map<std::string, size_t>::const_iterator it = lCtgIdx.begin(); it != lCtgIdx.end(); ++it) {
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

            nobs = lCtgIdx.size();
            nobs2 = ignored.size();

            if (ABD.size1() != nobs) {
                ABD.resize(nobs, nABD, true);
                ABD_VAR.resize(nobs, nABD, true);
            }

            assert(rABD.size() == nobs);
        }
    }

    // calcular matriz de tetranucleotidos
    TIMERSTART(tnf);
    if (1) {  // calcular TNF en paralelo en GPU
        double *TNF_d;
        char *seqs_d;
        size_t *seqs_d_index;
        dim3 blkDim(n_THREADS, 1, 1);
        dim3 grdDim(n_BLOCKS, 1, 1);
        cudaMallocHost((void **)&TNF, nobs * 136 * sizeof(double));
        cudaMalloc(&TNF_d, nobs * 136 * sizeof(double));
        cudaMalloc(&seqs_d, fsize);
        cudaMalloc(&seqs_d_index, 2 * nobs * sizeof(size_t));

        int n_STREAMS = 5;
        cudaStream_t streams[n_STREAMS];

        size_t contig_per_kernel = nobs / n_STREAMS;
        std::cout << "contig_per_kernel: " << contig_per_kernel << std::endl;

        for (int i = 0; i < n_STREAMS; i++) {
            cudaStreamCreate(&streams[i]);

            size_t contig_to_process = contig_per_kernel;
            size_t contigs_per_thread = (contig_to_process + (n_THREADS * n_BLOCKS) - 1) / (n_THREADS * n_BLOCKS);
            size_t _des = contig_per_kernel * i;

            std::cout << "stream: " << i << ", contig_to_process: " << contig_to_process
                      << ", contigs_per_thread: " << contigs_per_thread << std::endl;

            if (i == n_STREAMS - 1) contig_to_process += (nobs % n_STREAMS);
            size_t _mem_i = seqs_h_index_i[_des];  // puntero al inicio del primer contig a procesar
            size_t _mem_size = seqs_h_index_e[_des + contig_to_process - 1] - seqs_h_index_i[_des];  // tamaño de la memoria a copiar
            size_t TNF_des = _des * 136;

            cudaMemcpyAsync(seqs_d + _mem_i, _mem + _mem_i, _mem_size, cudaMemcpyHostToDevice, streams[i]);
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
        cudaFree(TNF_d);
        cudaFree(seqs_d);
        cudaFree(seqs_d_index);
    }
    TIMERSTOP(tnf);

    std::ofstream out("TNF.bin", std::ios::out | std::ios::binary);
    if (out) {
        if (1) {  // para verificar con el codigo de secuencial
            for (auto it = smallCtgs.begin(); it != smallCtgs.end(); it++) {
                for (size_t i = 0; i < 136; i++) {
                    TNF[*it * 136 + i] = 0;
                }
            }
        }
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
