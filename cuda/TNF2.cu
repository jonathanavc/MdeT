// nvcc TNF.cu -lz
// ta bien
#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include "../extra/KseqReader.h"

__device__ __constant__ int n_TNF_d = 136;

__device__ const char * get_contig_d(int contig_index, const char * seqs_d,const size_t * seqs_d_index){
    size_t contig_beg = 0;
    if(contig_index != 0){
        contig_beg = seqs_d_index[contig_index-1];
    }
    return seqs_d + contig_beg;
}

__device__ __host__ unsigned char get_tn(const char * contig, size_t index){
    unsigned char tn = 0;
    for(int i = 0; i < 4; i++){
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
        
        tn = (tn<<2) + N;
    }
    return tn;
}

__device__ unsigned char get_revComp_tn_d(const char * contig, size_t index){
    unsigned char tn = 0;
    for(int i = 3; i >= 0; i--){
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
        tn = (tn<<2) + N;
    }
    return tn;
}

__global__ void get_TNF(double * TNF_d , const char * seqs_d, const size_t * seqs_d_index , size_t nobs,
    const unsigned char * TNmap, const unsigned char * TNPmap, const unsigned char * smallCtgs, size_t contigs_per_thread){
    size_t thead_id = threadIdx.x + blockIdx.x * blockDim.x;

    // inicializar valores de vector en 0
    for(size_t i = 0; i < contigs_per_thread; i++){ 
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        for(int j = 0; j < n_TNF_d; j++){
            TNF_d[contig_index * n_TNF_d + j] = 0;
        }
    }

    //__syncthreads(); 

    for(size_t i = 0; i < contigs_per_thread; i++){
        size_t contig_index = (thead_id * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        if(smallCtgs[contig_index] == 0){
            const char * contig = get_contig_d(contig_index, seqs_d, seqs_d_index);
            size_t contig_size = seqs_d_index[contig_index];
            if(contig_index != 0){
                contig_size -= seqs_d_index[contig_index - 1];
            }
            for (size_t j = 0; j < contig_size - 3; ++j) {
                unsigned char tn = get_tn(contig, j);
                //SI tn NO SE ENCUENTRA EN TNmap el complemento del palindromo sí estará
                if(TNmap[tn] != n_TNF_d){
                    ++TNF_d[contig_index * n_TNF_d + TNmap[tn]];
                }
                
                tn = get_revComp_tn_d(contig, j);

                //SALTA EL PALINDROMO PARA NO INSERTARLO NUEVAMENTE
                if (TNPmap[tn] == 0) {
                    if(TNmap[tn] != n_TNF_d){
                        ++TNF_d[contig_index * n_TNF_d + TNmap[tn]];
                    }
                }
            }
            double rsum = 0;
            for(size_t c = 0; c < n_TNF_d; ++c) {
                rsum += TNF_d[contig_index * n_TNF_d + c] * TNF_d[contig_index * n_TNF_d + c];
            }
            rsum = sqrt(rsum);
            for(size_t c = 0; c < n_TNF_d; ++c) {
                TNF_d[contig_index * n_TNF_d + c] /= rsum; //OK
            }
        }
    }
}

static const std::string TN[] = { "GGTA", "AGCC", "AAAA", "ACAT", "AGTC", "ACGA", "CATA", "CGAA", "AAGT", "CAAA",
        "CCAG", "GGAC", "ATTA", "GATC", "CCTC", "CTAA", "ACTA", "AGGC", "GCAA", "CCGC", "CGCC", "AAAC", "ACTC", "ATCC",
		"GACC", "GAGA", "ATAG", "ATCA", "CAGA", "AGTA", "ATGA", "AAAT", "TTAA", "TATA", "AGTG", "AGCT", "CCAC", "GGCC",
		"ACCC", "GGGA", "GCGC", "ATAC", "CTGA", "TAGA", "ATAT", "GTCA", "CTCC", "ACAA", "ACCT", "TAAA", "AACG", "CGAG",
		"AGGG", "ATCG", "ACGC", "TCAA", "CTAC", "CTCA", "GACA", "GGAA", "CTTC", "GCCC", "CTGC", "TGCA", "GGCA", "CACG",
		"GAGC", "AACT", "CATG", "AATT", "ACAG", "AGAT", "ATAA", "CATC", "GCCA", "TCGA", "CACA", "CAAC", "AAGG", "AGCA",
		"ATGG", "ATTC", "GTGA", "ACCG", "GATA", "GCTA", "CGTC", "CCCG", "AAGC", "CGTA", "GTAC", "AGGA", "AATG", "CACC",
		"CAGC", "CGGC", "ACAC", "CCGG", "CCGA", "CCCC", "TGAA", "AACA", "AGAG", "CCCA", "CGGA", "TACA", "ACCA", "ACGT",
		"GAAC", "GTAA", "ATGC", "GTTA", "TCCA", "CAGG", "ACTG", "AAAG", "AAGA", "CAAG", "GCGA", "AACC", "ACGG", "CCAA",
		"CTTA", "AGAC", "AGCG", "GAAA", "AATC", "ATTG", "GCAC", "CCTA", "CGAC", "CTAG", "AGAA", "CGCA", "CGCG", "AATA" };


static const std::string TNP[] = { "ACGT", "AGCT", "TCGA", "TGCA", "CATG", "CTAG", "GATC", "GTAC", "ATAT", "TATA","CGCG",
        "GCGC", "AATT", "TTAA", "CCGG", "GGCC" };

int n_THREADS = 32;
int n_BLOCKS = 128;

std::vector<std::string> seqs;
std::vector<size_t> gCtgIdx;
std::vector<unsigned char> smallCtgs;

const int n_TNF = 136;
const int n_TNFP = 16;

unsigned char TNmap[256];
unsigned char TNPmap[256];

static size_t minContig = 2500; //minimum contig size for binning
static size_t minContigByCorr = 1000; //minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000; //for graph generation purpose

double * TNF_d;
static char * seqs_d;
static size_t * seqs_d_index;
static unsigned char * TNmap_d;
static unsigned char * TNPmap_d;
static unsigned char * smallCtgs_d;
static size_t * gCtgIdx_d;

size_t nobs_cont;
size_t kernel_cont;
std::string seqs_kernel;
std::vector<double*> TNF;
size_t * gCtgIdx_kernel;
size_t * seqs_kernel_index;
unsigned char * smallCtgs_kernel;

void kernel(){
    //std::cout << "kernel: " << kernel_cont<< std::endl;
    dim3 blkDim (n_THREADS, 1, 1);
    dim3 grdDim (n_BLOCKS, 1, 1);

    cudaMalloc(&seqs_d, seqs_kernel.size());
    cudaMemcpy(seqs_d, seqs_kernel.data(), seqs_kernel.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(seqs_d_index, seqs_kernel_index, n_BLOCKS * n_THREADS  * sizeof(size_t), cudaMemcpyHostToDevice);            // seqs_index
    cudaMemcpy(smallCtgs_d, smallCtgs_kernel, n_BLOCKS * n_THREADS, cudaMemcpyHostToDevice);

    get_TNF<<<grdDim, blkDim>>>(TNF_d, seqs_d, seqs_d_index, nobs_cont, TNmap_d, TNPmap_d, smallCtgs_d, 1);
    seqs_kernel = "";
    kernel_cont++;
    nobs_cont = 0;
}

void save_tnf(){
    //std::cout << "save kernel: " << kernel_cont<< std::endl;
    cudaDeviceSynchronize();
    cudaFreeAsync(seqs_d);
    TNF.emplace_back((double *) malloc(n_BLOCKS * n_THREADS * n_TNF * sizeof(double)));
    cudaMemcpy(TNF[TNF.size() - 1], TNF_d, n_BLOCKS * n_THREADS * n_TNF * sizeof(double), cudaMemcpyDeviceToHost);
}

int main(int argc, char const *argv[]){
    if(argc > 2){
        n_BLOCKS = atoi(argv[1]);
        n_THREADS = atoi(argv[2]);
    }
    std::cout << "n°bloques: "<< n_BLOCKS <<", n°threads:"<< n_THREADS << std::endl;

    // se inicializan los mapas
    for(int i = 0; i < 256; i++){
        TNmap[i] = n_TNF;
        TNPmap[i] = 0;
    }
    for(int i = 0; i < n_TNF; ++i) {
        unsigned char key = get_tn(TN[i].c_str(), 0);
        TNmap[key] = i;
	}
    
	for(size_t i = 0; i < n_TNFP; ++i) {
		unsigned char key = get_tn(TNP[i].c_str(), 0);
        TNPmap[key] = 1;
	}

    auto start = std::chrono::system_clock::now();

    //TNmap
    cudaMalloc(&TNmap_d, 256);
    cudaMemcpy(TNmap_d, TNmap, 256, cudaMemcpyHostToDevice);

    //TNPmap
    cudaMalloc(&TNPmap_d, 256);
    cudaMemcpy(TNPmap_d, TNPmap, 256, cudaMemcpyHostToDevice);   

	size_t nobs = 0;

	int nresv = 0;
    std::string inFile = "test.gz";

    nobs_cont = 0;
    kernel_cont = 0;
    seqs_kernel_index = (size_t *) malloc(n_THREADS * n_BLOCKS * sizeof(size_t));
    smallCtgs_kernel = (unsigned char *) malloc(n_THREADS * n_BLOCKS);
   
    cudaMalloc(&TNF_d, n_BLOCKS * n_THREADS * n_TNF * sizeof(double));
    cudaMalloc(&seqs_d_index, n_BLOCKS * n_THREADS * sizeof(size_t));
    cudaMalloc(&gCtgIdx_d, n_BLOCKS * n_THREADS * sizeof(size_t));
    cudaMalloc(&smallCtgs_d, n_BLOCKS * n_THREADS);

	gzFile f = gzopen(inFile.c_str(), "r");
	if (f == NULL) {
		cerr << "[Error!] can't open the sequence fasta file " << inFile << endl;
		return 1;
	} else {
		kseq_t * kseq = kseq_init(f);
		int64_t len;
		while ((len = kseq_read(kseq)) > 0) {
			std::transform(kseq->seq.s, kseq->seq.s + len, kseq->seq.s, ::toupper);
			if (kseq->name.l > 0) {
				if(len >= (int) std::min(minContigByCorr, minContigByCorrForGraph)) {
					if(len < (int) minContig) {
						if(len >= (int) minContigByCorr){
                            smallCtgs.emplace_back(1);
                            smallCtgs_kernel[nobs_cont] = 1;
                        }
						else{
                            smallCtgs.emplace_back(0);
                            smallCtgs_kernel[nobs_cont] = 0;
							++nresv;
                        }
					}
                    else{
                        smallCtgs.emplace_back(0);
                        smallCtgs_kernel[nobs_cont] = 0;
                    }
					gCtgIdx.emplace_back(seqs.size());
                    nobs++;

                    seqs_kernel += kseq->seq.s;
                    seqs_kernel_index[nobs_cont] = seqs_kernel.size();
                    nobs_cont++;
				} else{
                    //ignored[kseq->name.s] = seqs.size();
                }	
				//contig_names.push_back(kseq->name.s);
				seqs.push_back(kseq->seq.s);

                if(nobs_cont == n_BLOCKS * n_THREADS){
                    if(kernel_cont != 0 ){
                        save_tnf();
                    }
                    kernel();
                }
			}
		}
		kseq_destroy(kseq);
		kseq = NULL;
		gzclose(f);
	}
    if(kernel_cont != 0 && nobs_cont != 0){
        save_tnf();
    }
    cudaDeviceSynchronize();
    if(nobs_cont != 0){
        kernel();
        save_tnf();
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float,std::milli> duration = end - start;
    std::cout <<"leer contigs + procesamiento "<< duration.count()/1000.f << "s " << std::endl;

    std::ofstream out("TNF.bin", ios::out | ios::binary);

	if (out) {
        for(size_t i = 0; i < TNF.size(); i++){
            if(i < (TNF.size() - 1) || nobs % (n_BLOCKS * n_THREADS)  == 0)
                out.write((char *) TNF[i], n_BLOCKS * n_THREADS * n_TNF * sizeof(double));
            else
                out.write((char *) TNF[i], (nobs % (n_BLOCKS * n_THREADS)) * n_TNF * sizeof(double));
        }
        std::cout << "TNF guardado" << std::endl;
	}
    else{
        std::cout << "Error al guardar" << std::endl;
    }
    out.close();

    for(int i = 0; i < TNF.size(); i++)
        free(TNF[i]);

    return 0;
}
