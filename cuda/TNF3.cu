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
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../extra/KseqReader.h"

__device__ __constant__ unsigned char TNmap_d[256] = {
    2,   21,  31,  115, 101, 119, 67,  50,  135, 126, 69,  92,  116, 88,  8,
    78,  47,  96,  3,   70,  106, 38,  48,  83,  16,  22,  136, 114, 5,   54,
    107, 120, 72,  41,  44,  26,  27,  23,  136, 53,  12,  81,  136, 127, 30,
    110, 136, 80,  132, 123, 71,  102, 79,  1,   35,  124, 29,  4,   136, 34,
    91,  17,  136, 52,  9,   77,  136, 117, 76,  93,  136, 65,  6,   73,  136,
    68,  28,  94,  136, 113, 121, 36,  136, 10,  103, 99,  136, 87,  129, 14,
    136, 136, 98,  19,  136, 97,  15,  56,  136, 131, 57,  46,  136, 136, 122,
    60,  136, 136, 42,  62,  136, 136, 7,   130, 136, 51,  133, 20,  136, 134,
    89,  86,  136, 136, 104, 95,  136, 136, 49,  136, 136, 136, 105, 136, 136,
    136, 33,  136, 136, 136, 43,  136, 136, 136, 55,  136, 136, 136, 112, 136,
    136, 136, 136, 136, 136, 136, 75,  136, 136, 136, 32,  136, 136, 136, 136,
    136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 100, 136, 136, 136,
    63,  136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 136, 125, 108, 136,
    136, 58,  24,  136, 136, 84,  13,  136, 136, 25,  66,  136, 136, 18,  128,
    136, 136, 74,  61,  136, 136, 85,  136, 136, 136, 118, 40,  136, 136, 109,
    90,  136, 136, 45,  136, 136, 136, 111, 136, 136, 136, 82,  136, 136, 136,
    59,  11,  136, 136, 64,  37,  136, 136, 0,   136, 136, 136, 39,  136, 136,
    136};

__device__ __constant__ unsigned char BN[256] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 1, 4, 4, 4, 3,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4};

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

__device__ const char *get_contig_d(int contig_index, const char *seqs_d,
                                    const size_t *seqs_d_index) {
  size_t contig_beg = 0;
  if (contig_index != 0) {
    contig_beg = seqs_d_index[contig_index - 1];
  }
  return seqs_d + contig_beg;
}

__global__ void get_TNF(double *TNF_d, const char *seqs_d,
                        const size_t *seqs_d_index, size_t nobs,
                        size_t contigs_per_thread) {
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
    size_t contig_size = seqs_d_index[contig_index];
    if (contig_index != 0) contig_size -= seqs_d_index[contig_index - 1];
    // tengo dudas sobre esta parte ------------------------
    if (contig_size >= minContig || contig_size < minContigByCorr) {
      const char *contig = get_contig_d(contig_index, seqs_d, seqs_d_index);
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
    }
  }
}

__global__ void get_TNF_local(double *TNF_d, const char *seqs_d,
                              const size_t *seqs_d_index, size_t nobs,
                              const size_t contigs_per_thread) {
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
    size_t contig_size = seqs_d_index[contig_index];
    if (contig_index != 0) contig_size -= seqs_d_index[contig_index - 1];
    // tengo dudas sobre esta parte ------------------------
    if (contig_size >= minContig || contig_size < minContigByCorr) {
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
    }
    // guardar en la memoria global
    for (size_t c = 0; c < 136; ++c) {
      TNF_d[contig_index * 136 + c] = TNF_temp[c];
    }
  }
}

static const std::string TN[] = {
    "GGTA", "AGCC", "AAAA", "ACAT", "AGTC", "ACGA", "CATA", "CGAA", "AAGT",
    "CAAA", "CCAG", "GGAC", "ATTA", "GATC", "CCTC", "CTAA", "ACTA", "AGGC",
    "GCAA", "CCGC", "CGCC", "AAAC", "ACTC", "ATCC", "GACC", "GAGA", "ATAG",
    "ATCA", "CAGA", "AGTA", "ATGA", "AAAT", "TTAA", "TATA", "AGTG", "AGCT",
    "CCAC", "GGCC", "ACCC", "GGGA", "GCGC", "ATAC", "CTGA", "TAGA", "ATAT",
    "GTCA", "CTCC", "ACAA", "ACCT", "TAAA", "AACG", "CGAG", "AGGG", "ATCG",
    "ACGC", "TCAA", "CTAC", "CTCA", "GACA", "GGAA", "CTTC", "GCCC", "CTGC",
    "TGCA", "GGCA", "CACG", "GAGC", "AACT", "CATG", "AATT", "ACAG", "AGAT",
    "ATAA", "CATC", "GCCA", "TCGA", "CACA", "CAAC", "AAGG", "AGCA", "ATGG",
    "ATTC", "GTGA", "ACCG", "GATA", "GCTA", "CGTC", "CCCG", "AAGC", "CGTA",
    "GTAC", "AGGA", "AATG", "CACC", "CAGC", "CGGC", "ACAC", "CCGG", "CCGA",
    "CCCC", "TGAA", "AACA", "AGAG", "CCCA", "CGGA", "TACA", "ACCA", "ACGT",
    "GAAC", "GTAA", "ATGC", "GTTA", "TCCA", "CAGG", "ACTG", "AAAG", "AAGA",
    "CAAG", "GCGA", "AACC", "ACGG", "CCAA", "CTTA", "AGAC", "AGCG", "GAAA",
    "AATC", "ATTG", "GCAC", "CCTA", "CGAC", "CTAG", "AGAA", "CGCA", "CGCG",
    "AATA"};

static const std::string TNP[] = {
    "ACGT", "AGCT", "TCGA", "TGCA", "CATG", "CTAG", "GATC", "GTAC",
    "ATAT", "TATA", "CGCG", "GCGC", "AATT", "TTAA", "CCGG", "GGCC"};

int contig_per_thread = 1;
int n_THREADS = 32;
int n_BLOCKS = 128;

std::vector<std::string> seqs;
std::unordered_map<size_t, size_t> gCtgIdx;
std::unordered_set<int> smallCtgs;

const int n_TNF = 136;
const int n_TNFP = 16;

unsigned char TNmap[256];
unsigned char TNPmap[256];

static size_t minContig = 2500;  // minimum contig size for binning
static size_t minContigByCorr =
    1000;  // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000;  // for graph generation purpose

double *TNF_d[2];
static size_t *seqs_d_index[2];

size_t nobs_cont;
size_t kernel_cont;
static size_t global_contigs_target;
std::vector<double *> TNF;
cudaStream_t _s[2][2];
std::string seqs_kernel[2];
size_t *seqs_kernel_index[2];

void kernel(dim3 blkDim, dim3 grdDim, int SUBP_IND, int cont, int size) {
  char *seqs_d;
  cudaMallocHost((void **)&TNF[cont],
                 global_contigs_target * n_TNF * sizeof(double));
  // TNF[cont] = (double *)malloc(n_BLOCKS * n_THREADS * contig_per_thread *
  // n_TNF * sizeof(double));
  cudaMallocAsync(&seqs_d, seqs_kernel[SUBP_IND].size(), _s[SUBP_IND][0]);
  cudaMemcpyAsync(seqs_d_index[SUBP_IND], seqs_kernel_index[SUBP_IND],
                  global_contigs_target * sizeof(size_t),
                  cudaMemcpyHostToDevice,
                  _s[SUBP_IND][1]);  // seqs_index
  cudaMemcpyAsync(seqs_d, seqs_kernel[SUBP_IND].data(),
                  seqs_kernel[SUBP_IND].size(), cudaMemcpyHostToDevice,
                  _s[SUBP_IND][0]);
  for (int i = 0; i < 2; i++) cudaStreamSynchronize(_s[SUBP_IND][i]);

  get_TNF<<<grdDim, blkDim, 0, _s[SUBP_IND][0]>>>(
      TNF_d[SUBP_IND], seqs_d, seqs_d_index[SUBP_IND], size, contig_per_thread);
  cudaMemcpyAsync(TNF[cont], TNF_d[SUBP_IND],
                  global_contigs_target * n_TNF * sizeof(double),
                  cudaMemcpyDeviceToHost, _s[SUBP_IND][0]);
  cudaFreeAsync(seqs_d, _s[SUBP_IND][1]);

  for (int i = 0; i < 2; i++) cudaStreamSynchronize(_s[SUBP_IND][i]);
  // más eficiente que asignación
  seqs_kernel[SUBP_IND].clear();
}

void reader(int fpint, int id, size_t chunk, size_t _size, char *_mem) {
  size_t readSz = pread(fpint, _mem, _size, id * chunk);
  if (readSz < _size) {
    cout << " error en lectura readSz " << readSz << " chunk " << _size << endl;
  }
}

int main(int argc, char const *argv[]) {
  std::string inFile = "test.gz";
  if (argc > 2) {
    n_BLOCKS = atoi(argv[1]);
    n_THREADS = atoi(argv[2]);
    if (argc > 3) {
      inFile = argv[3];
    }
  }
  // std::cout << "n°bloques: "<< n_BLOCKS <<", n°threads:"<< n_THREADS <<
  // std::endl;

  /*
  // se inicializan los mapas
  for (int i = 0; i < 256; i++) {
    TNmap[i] = n_TNF;
    TNPmap[i] = 0;
  }
  for (int i = 0; i < n_TNF; ++i) {
    unsigned char key = get_tn(TN[i].c_str(), 0);
    TNmap[key] = i;
  }

  for (size_t i = 0; i < n_TNFP; ++i) {
    unsigned char key = get_tn(TNP[i].c_str(), 0);
    TNPmap[key] = 1;
  }
  */

  auto start_global = std::chrono::system_clock::now();
  auto start = std::chrono::system_clock::now();

  std::thread SUBPS[2];
  dim3 blkDim(n_THREADS, 1, 1);
  dim3 grdDim(n_BLOCKS, 1, 1);

  global_contigs_target = n_BLOCKS * n_THREADS * contig_per_thread;

  int SUBP_IND = 0;
  nobs_cont = 0;
  kernel_cont = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) cudaStreamCreate(&_s[i][j]);
    cudaMallocHost((void **)&seqs_kernel_index[i],
                   global_contigs_target * sizeof(size_t));
    cudaMalloc(&TNF_d[i], global_contigs_target * n_TNF * sizeof(double));
    cudaMalloc(&seqs_d_index[i], global_contigs_target * sizeof(size_t));
  }

  size_t nobs = 0;
  int nresv = 0;

  gzFile f = gzopen(inFile.c_str(), "r");
  if (f == NULL) {
    cerr << "[Error!] can't open the sequence fasta file " << inFile << endl;
    return 1;
  } else {
    // Esto si es rapido
    auto _start = std::chrono::system_clock::now();

    int nth = 4;
    int fpint = -1;

    FILE *fp = fopen(inFile.c_str(), "r");
    fseek(fp, 0L, SEEK_END);   // seek to the EOF
    size_t fsize = ftell(fp);  // get the current position
    fclose(fp);
    size_t chunk = fsize / nth;
    char *_mem = (char *)malloc(fsize);
    std::vector<std::string> contigs;

    std::cout << "tamaño total:" << fsize << std::endl;
    std::cout << "chunk:" << chunk << std::endl;

    fpint = open(inFile.c_str(), O_RDWR | O_CREAT,
                 S_IREAD | S_IWRITE | S_IRGRP | S_IROTH);

    thread readerThreads[nth];

    for (int i = 0; i < nth; i++) {
      size_t _size;
      if (i == nth - 1)
        _size = chunk;
      else
        _size = chunk + fsize % nth;
      std::cout << "tamaño _size:" << _size << std::endl;
      readerThreads[i] = thread(reader, fpint, i, chunk, _size, _mem);
    }

    for (int i = 0; i < nth; i++) {
      readerThreads[i].join();
    }

    std::string _s = "";
    for (size_t i = 0; i < 10; i++) {
      cout << _mem[i] << "[" << int(_mem[i]) << "]" << endl;
    }

    _s.push_back(0);
    if (_s != "") contigs.push_back(_s);
    std::cout << "tam strings:" << contigs.size() << std::endl;
    std::cout << contigs.at(contigs.size() - 1) << std::endl;

    close(fpint);

    auto _end = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> _duration = _end - _start;
    std::cout << "cargar archivo descomprimido:" << _duration.count() / 1000.f
              << std::endl;

    // tambien lento en secuencial 26s
    /*
    auto _start = std::chrono::system_clock::now();

    std::ifstream file(inFile.c_str(), std::ios::);
    std::string compressed_data((std::istream_iterator<char>(file)),
                                (std::istream_iterator<char>()));
    std::cout << compressed_data.size() << std::endl;

    auto _end = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> _duration = _end - _start;
    std::cout <<"cargar archivo descomprimidO"<< _duration.count() / 1000.f <<
    std::endl;
    */

    // no funciona bien cargar comprimido y descomprimir en memoria
    /*
    ////////////////////
    auto _start = std::chrono::system_clock::now();

    std::ifstream file(inFile.c_str(), std::ios::binary);
    std::string compressed_data((std::istream_iterator<char>(file)),
                                (std::istream_iterator<char>()));
    std::cout << compressed_data.size() << std::endl;
    std::string uncompressed_data(compressed_data.size() * 5, '\n');
    size_t uncompressed_size = compressed_data.size();
    int result;
    do {
      result = uncompress((Bytef *)uncompressed_data.data(), &uncompressed_size,
                          (const Bytef *)compressed_data.data(),
                          compressed_data.size());
      if(result == Z_MEM_ERROR){
        std::cout << "aumentando buffer" << std::endl;
        uncompressed_data.resize(uncompressed_data.size() * 2);
      }
    } while (result != Z_MEM_ERROR);

    if (result == Z_OK) {
      std::cout << uncompressed_data.size() << std::endl;
      uncompressed_data.resize(uncompressed_size);
      std::cout << uncompressed_size << std::endl;
    } else {

      std::cout << "error:" << result << std::endl;
    }

    auto _end = std::chrono::system_clock::now();
    std::chrono::duration<float, std::milli> _duration = _end - _start;
    std::cout << _duration.count() / 1000.f << std::endl;
    ////////////////////
    */

    const size_t contigs_target = global_contigs_target;
    kseq_t *kseq = kseq_init(f);
    int64_t len;

    while ((len = kseq_read(kseq)) > 0) {
      std::transform(kseq->seq.s, kseq->seq.s + len, kseq->seq.s, ::toupper);
      if (kseq->name.l > 0) {
        if (len >= (int)std::min(minContigByCorr, minContigByCorrForGraph)) {
          if (len < (int)minContig) {
            if (len >= (int)minContigByCorr) {
              // smallCtgs.insert(1);
            } else {
              ++nresv;
            }
          }
          seqs_kernel[SUBP_IND].append(kseq->seq.s);
          seqs_kernel_index[SUBP_IND][nobs_cont] = seqs_kernel[SUBP_IND].size();
          nobs++;
          nobs_cont++;
        } else {
          // ignored[kseq->name.s] = seqs.size();
        }
        // contig_names.push_back(kseq->name.s);
        seqs.push_back(kseq->seq.s);

        if (nobs_cont & contigs_target) {
          TNF.push_back((double *)0);
          SUBPS[SUBP_IND] = std::thread(kernel, blkDim, grdDim, SUBP_IND,
                                        kernel_cont, nobs_cont);
          SUBP_IND = (SUBP_IND + 1) & 1;
          kernel_cont++;
          nobs_cont = 0;

          // si aún no se ha terminado la ejecición la siguiente hebra se espera
          // a ella.
          if (SUBPS[SUBP_IND].joinable()) SUBPS[SUBP_IND].join();
        }
      }
    }
    kseq_destroy(kseq);
    kseq = NULL;
    gzclose(f);
  }
  if (nobs_cont != 0) {
    TNF.push_back((double *)0);
    SUBPS[SUBP_IND] =
        std::thread(kernel, blkDim, grdDim, SUBP_IND, kernel_cont, nobs_cont);
    SUBP_IND = (SUBP_IND + 1) & 2;
    kernel_cont++;
    nobs_cont = 0;
  }
  // se esperan a las hebras restantes
  for (int i = 0; i < 2; i++) {
    if (SUBPS[i].joinable()) {
      SUBPS[i].join();
    }
  }

  auto end = std::chrono::system_clock::now();
  std::chrono::duration<float, std::milli> duration = end - start;
  // std::cout <<"leer contigs + procesamiento "<< duration.count()/1000.f << "s
  // " << std::endl;

  auto end_global = std::chrono::system_clock::now();
  duration = end_global - start_global;
  std::cout << duration.count() / 1000.f << std::endl;

  std::ofstream out("TNF.bin", ios::out | ios::binary);
  if (out) {
    for (size_t i = 0; i < TNF.size(); i++) {
      if (i < (TNF.size() - 1) || nobs % (global_contigs_target) == 0)
        out.write((char *)TNF[i],
                  global_contigs_target * n_TNF * sizeof(double));
      else
        out.write((char *)TNF[i],
                  (nobs % global_contigs_target) * n_TNF * sizeof(double));
    }
    // std::cout << "TNF guardado" << std::endl;
  } else {
    // std::cout << "Error al guardar" << std::endl;
  }
  out.close();

  for (int i = 0; i < TNF.size(); i++) cudaFreeHost(TNF[i]);
  for (int i = 0; i < 2; i++) {
    cudaFreeHost(seqs_kernel_index[i]);
    cudaFree(TNF_d[i]);
    cudaFree(seqs_d_index[i]);
    for (int j = 0; j < 2; j++) cudaStreamDestroy(_s[i][j]);
  }
  return 0;
}