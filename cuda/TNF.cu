#include <iostream>
#include <stdgpu/unordered_map.cuh>     // stdgpu::unordered_map
#include <stdgpu/unordered_set.cuh>     // stdgpu::unordered_set

__device__ int n_TNF = 136;

__global__ static const std::string TN[] = { "GGTA", "AGCC", "AAAA", "ACAT", "AGTC", "ACGA", "CATA", "CGAA", "AAGT", "CAAA",
        "CCAG", "GGAC", "ATTA", "GATC", "CCTC", "CTAA", "ACTA", "AGGC", "GCAA", "CCGC", "CGCC", "AAAC", "ACTC", "ATCC",
		"GACC", "GAGA", "ATAG", "ATCA", "CAGA", "AGTA", "ATGA", "AAAT", "TTAA", "TATA", "AGTG", "AGCT", "CCAC", "GGCC",
		"ACCC", "GGGA", "GCGC", "ATAC", "CTGA", "TAGA", "ATAT", "GTCA", "CTCC", "ACAA", "ACCT", "TAAA", "AACG", "CGAG",
		"AGGG", "ATCG", "ACGC", "TCAA", "CTAC", "CTCA", "GACA", "GGAA", "CTTC", "GCCC", "CTGC", "TGCA", "GGCA", "CACG",
		"GAGC", "AACT", "CATG", "AATT", "ACAG", "AGAT", "ATAA", "CATC", "GCCA", "TCGA", "CACA", "CAAC", "AAGG", "AGCA",
		"ATGG", "ATTC", "GTGA", "ACCG", "GATA", "GCTA", "CGTC", "CCCG", "AAGC", "CGTA", "GTAC", "AGGA", "AATG", "CACC",
		"CAGC", "CGGC", "ACAC", "CCGG", "CCGA", "CCCC", "TGAA", "AACA", "AGAG", "CCCA", "CGGA", "TACA", "ACCA", "ACGT",
		"GAAC", "GTAA", "ATGC", "GTTA", "TCCA", "CAGG", "ACTG", "AAAG", "AAGA", "CAAG", "GCGA", "AACC", "ACGG", "CCAA",
		"CTTA", "AGAC", "AGCG", "GAAA", "AATC", "ATTG", "GCAC", "CCTA", "CGAC", "CTAG", "AGAA", "CGCA", "CGCG", "AATA" };

__global__ static const std::string TNP[] = { "ACGT", "AGCT", "TCGA", "TGCA", "CATG", "CTAG", "GATC", "GTAC", "ATAT", "TATA","CGCG",
        "GCGC", "AATT", "TTAA", "CCGG", "GGCC" };

//__device__ static stdgpu::unordered_map<int, int> TNmap;
//__device__ static stdgpu::unordered_map<size_t, size_t> gCtgIdx;
//__device__ static stdgpu::unordered_set<int> TNPmap;
//__device__ static stdgpu::unordered_set<size_t> smallCtgs


__device__ char * get_contig_d(int contig_index,char * seqs_d, int * seqs_d_index){
    size_t contig_beg = 0;
    size_t contig_end;
    if(contig_index != 0){
        contig_beg = seqs_d_index[contig_index-1];
    }
    contig_end = seqs_d_index[contig_index];
    char contig = malloc();
    for(int i = contig_beg; i < contig_end ;i++){
        contig[i - contig_beg] = seqs_d[i];
    }
}

__device__ int get_tn_d(char * contig, int index){
    int tn = 0;
    for(int i = 0; i<4; i++){
        tn+= tn<<8 + contig[index + i];
    }
    return tn;
}

__device__ int get_revComp_tn_d(int contig_index,char * seqs_d, int * seqs_d_index){
    int tn = 0;
    for(int i = 3; i >= 0; i--){
        char N = contig[index + i];
        if (N == 'A')
			N = 'T';
		else if (N == 'T')
			N = 'A';
		else if (N == 'C')
			N = 'G';
		else if (s[i] == 'G')
		    N = 'C';
        else
            return 0;
        tn+= tn<<8 + N;
    }
    return tn;
}

__global__ void TNF(double * TNF_d , char * seqs_d, size_t * seqs_d_index , size_t nobs,
    const stdgpu::unordered_map<int, int> TNmap,
    const stdgpu::unordered_set<int> TNPmap,
    const stdgpu::unordered_set<size_t> smallCtgs,
    const stdgpu::unordered_map<size_t, size_t> gCtgIdx,
    /*
    size_t * smallCtgs_KEY, size_t smallCtgs_size, 
    size_t * gCtgIdx_KEY, size_t * gCtgIdx_VALUE, size_t gCtgIdx_size,
    */
    size_t contigs_per_thread){

    // crear mapas y sets
    /*
    if(blockIdx.x == 0){
        for(size_t = 0; i < smallCtgs_size){
            smallCtgs
        }
        for(size_t i = 0; i < n_TNF; ++i) {
            TNmap[TN[i][0]<<24 + TN[i][1]<<16 + TN[i][2]<<8 + TN[i][3]] = i;
        }
        for(size_t i = 0; i < 16; ++i) {
            TNPmap[TNP[i][0]<<24 + TNP[i][1]<<16 + TNP[i][2]<<8 + TNP[i][3]] = i;
        }

    }
    __syncthreads();
    */

    // inicializar valores de vector en 0
    for(int i = 0, i < contigs_per_thread){ 
        int contig_index = (blockIdx.x * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        for(int j = 0; j < n_TNF; j++){
            TNF_d[contig_index * n_TNF + j] = 0;
        }
    }

    __syncthreads(); 

    for(size_t i = 0, i < contigs_per_thread){
        size_t contig_index = (blockIdx.x * contigs_per_thread) + i;
        if(contig_index >= nobs) break;
        if(smallCtgs.find(contig_index) == smallCtgs.end()){
            char * contig = get_contig_d(gCtgIdx[contig_index], seqs_d, seqs_d_index);
            int contig_size = seqs_d_index[gCtgIdx[contig_index]];
            if(gCtgIdx[contig_index] != 0){
                contig_size -= seqs_d_index[gCtgIdx[contig_index] - 1];
            }
            for (size_t i = 0; i < contig_size; ++i) {
                int tn = get_contig_d(contig, i);
                stdgpu::unordered_map<int, int>::iterator it = TNmap.find(tn);
                //SI tn NO SE ENCUENTRA EN TNmap el complemento del palindromo sí estará
                if(it != TNmap.end()){
                    ++TNF_d[contig_index * n_TNF + it->second];   
                }
                
                tn = get_revComp_tn_d(contig, i);

                //SALTA EL PALINDROMO PARA NO INSERTARLO NUEVAMENTE
                if (TNPmap.find(tn) == TNPmap.end()) {
                    it = TNmap.find(tn);
                    if(it != TNmap.end()){
                        ++TNF_d(contig_index * n_TNF + it->second);
                    }
                }
            }

            double rsum = 0;
            for(size_t c = 0; c < n_TNF; ++c) {
                rsum += TNF[contig_index + c] * TNF[contig_index + c];
            }
            rsum = SQRT(rsum);
            for(size_t c = 0; c < n_TNF; ++c) {
                TNF[contig_index + c] /= rsum;
            }

        }
    }
}

static int N_THREADS = 100;
static std::unordered_map<std::string, int> TNmap;
static std::unordered_set<std::string> TNPmap;

int main(int argc, char const *argv[]){
    for(size_t i = 0; i < n_TNF; ++i) {
		TNmap[TN[i][0]<<24 + TN[i][1]<<16 + TN[i][2]<<8 + TN[i][3]] = i;
	}
	for(size_t i = 0; i < 16; ++i) {
		TNPmap[TNP[i][0]<<24 + TNP[i][1]<<16 + TNP[i][2]<<8 + TNP[i][3]] = i;
	}

    std::string seqs_h;
    std::vector<size_t> seqs_h_index;
    for(std::string const& contig : seqs) {
        seqs_h += contig;
        seqs_h_index.emplace_back(seqs_h.size());
    }

    err = cudaMemcpy(seqs_d, combined.data(), combined.size(), cudaMemcpyHostToDevice);
    err = cudaMemcpy(seqs_d_index, indexes.data(), indexes.size() * sizeof(size_t), cudaMemcpyHostToDevice);

    size_t contigs_per_thread = 1 + ((nobs - 1) / N_THREADS);

    TNF<<<1,N_THREADS>>>(TNF_d, seqs_d, seqs_d_index, nobs, TNmap, TNPmap, smallCtgs, gCtgIdx, contigs_per_thread);

    cudaDeviceSynchronize():

    return 0;
}
