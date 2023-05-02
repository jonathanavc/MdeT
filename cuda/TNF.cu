#include <algorithm>
#include <iostream>
#include <vector>
#include "extra/KseqReader.h"

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
    for(int i = 3; i >= 0; i--){
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
            return 0;
        tn += tn<<2 + N;
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
            return 0;
        tn+= tn<<2 + N;
    }
    return tn;
}

__global__ void get_TNF(double * TNF_d , const char * seqs_d, const size_t * seqs_d_index , size_t nobs,
    const unsigned char * TNmap, const unsigned char * TNPmap, const unsigned char * smallCtgs,
    const size_t * gCtgIdx_d, size_t contigs_per_thread){
    // inicializar valores de vector en 0
    for(size_t i = 0; i < contigs_per_thread; i++){ 
        size_t contig_index = (blockIdx.x * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        for(int j = 0; j < n_TNF_d; j++){
            TNF_d[contig_index * n_TNF_d + j] = 0;
        }
    }

    //__syncthreads(); 

    for(size_t i = 0; i < contigs_per_thread; i++){
        size_t contig_index = (blockIdx.x * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        if(smallCtgs[contig_index] == 0){
            const char * contig = get_contig_d(gCtgIdx_d[contig_index], seqs_d, seqs_d_index);
            size_t contig_size = seqs_d_index[gCtgIdx_d[contig_index]];
            if(gCtgIdx_d[contig_index] != 0){
                contig_size -= seqs_d_index[gCtgIdx_d[contig_index] - 1];
            }
            for (size_t j = 0; j < contig_size - 3; ++j) {
                unsigned char tn = get_tn(contig, j);
                //SI tn NO SE ENCUENTRA EN TNmap el complemento del palindromo sí estará
                if(TNmap[tn] != n_TNF_d){
                    TNF_d[contig_index * n_TNF_d + TNmap[tn]]++;
                }
                
                tn = get_revComp_tn_d(contig, j);

                //SALTA EL PALINDROMO PARA NO INSERTARLO NUEVAMENTE
                if (TNPmap[tn] == 0) {
                    if(TNmap[tn] != n_TNF_d){
                        TNF_d[contig_index * n_TNF_d + TNmap[tn]]++;
                    }
                }
            }

            double rsum = 0;
            for(size_t c = 0; c < n_TNF_d; ++c) {
                rsum += TNF_d[contig_index * n_TNF_d + c] * TNF_d[contig_index * n_TNF_d + c];
            }
            rsum = sqrt(rsum);
            for(size_t c = 0; c < n_TNF_d; ++c) {
                //TNF_d[contig_index * n_TNF_d + c] /= rsum; //OK
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

static int n_THREADS = 100;
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

int main(int argc, char const *argv[]){
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
    for(int i = 0; i < 256; i++){
        std::cout << (int)TNmap[i] << " ";
    }
    std::cout << std::endl;

    for(int i = 0; i < 256; i++){
        std::cout << (int)TNPmap[i] << " ";
    }
    std::cout << std::endl;
    
	size_t nobs = 0;
	int nresv = 0;
    std::string inFile = "test.gz";
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
						if(len >= (int) minContigByCorr)
                            // cambio para facilitar la transferencia de smallCtgs
                            smallCtgs.emplace_back(1);
							//smallCtgs.insert(nobs);
						else
                            // cambio para facilitar la transferencia de smallCtgs
                            smallCtgs.emplace_back(0);
							++nresv;
					}
                    else
                        smallCtgs.emplace_back(0);
					//lCtgIdx[kseq->name.s] = nobs;
                    /////// cambio para facilitar la transferencia de gCtgIdx
					gCtgIdx.emplace_back(seqs.size());
                    nobs++;
                    ///////
                
				} else{
                    //ignored[kseq->name.s] = seqs.size();
                }	
				//contig_names.push_back(kseq->name.s);
				seqs.push_back(kseq->seq.s);
			}
		}
		kseq_destroy(kseq);
		kseq = NULL;
		gzclose(f);
	}

    std::cout << "nobs: " << nobs << ", small: " << smallCtgs.size() << ", gctg:" << gCtgIdx.size() << std::endl;

    std::string seqs_h;
    std::vector<size_t> seqs_h_index;
    for(std::string const& contig : seqs) {
        seqs_h += contig;
        seqs_h_index.emplace_back(seqs_h.size());
    }

    int err = cudaMalloc(&TNF_d,(nobs * n_TNF * sizeof(double)));                                                      // memoria para almacenar TNF

    err += cudaMalloc(&TNmap_d, 256);
    err += cudaMemcpy(TNmap_d, TNmap, 256, cudaMemcpyHostToDevice);                                                   // TNmap

    err += cudaMalloc(&TNPmap_d, 256);
    err += cudaMemcpy(TNPmap_d, TNPmap, 256, cudaMemcpyHostToDevice);                                                  // TNPmap 

    err += cudaMalloc(&seqs_d, seqs_h.size());
    err += cudaMemcpy(seqs_d, seqs_h.data(), seqs_h.size(), cudaMemcpyHostToDevice);

    err += cudaMalloc(&seqs_d_index, seqs_h_index.size() * sizeof(size_t));
    err += cudaMemcpy(seqs_d_index, seqs_h_index.data(), seqs_h_index.size() * sizeof(size_t), cudaMemcpyHostToDevice);// seqs_index

    err += cudaMalloc(&gCtgIdx_d, nobs * sizeof(size_t));
    err += cudaMemcpy(gCtgIdx_d, gCtgIdx.data(), nobs * sizeof(size_t), cudaMemcpyHostToDevice);                       // gCtgIdx

    err += cudaMalloc(&smallCtgs_d, nobs);
    err += cudaMemcpy(smallCtgs_d, smallCtgs.data(), nobs, cudaMemcpyHostToDevice);                                    // seqs

    std::cout << "hola" + err << std::endl;  

    size_t contigs_per_thread = 1 + ((nobs - 1) / n_THREADS);
    dim3 blkDim (n_THREADS, 1, 1);
    dim3 grdDim (1, 1, 1);

    get_TNF<<<blkDim, grdDim>>>(TNF_d, seqs_d, seqs_d_index, nobs, TNmap_d, TNPmap_d, smallCtgs_d, gCtgIdx_d, contigs_per_thread);

    cudaDeviceSynchronize();

    double * TNF = (double *)malloc(nobs * n_TNF * sizeof(double));

    cudaMemcpy(TNF, TNF_d, nobs * n_TNF * sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    for(int i = 0; i < nobs; i++){
        for(int j = 0; j < n_TNF; j++){
            std::cout << TNF[i * n_TNF + j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "chao" + err << std::endl;  

    return 0;
}
