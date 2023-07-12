// g++ TNF.cpp -lz -fopenmp
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../extra/KseqReader.h"

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

int n_THREADS = 32;

std::vector<std::string> seqs;
static std::unordered_map<size_t, size_t> gCtgIdx;
static std::unordered_map<std::string, size_t> lCtgIdx;
static std::unordered_set<size_t> smallCtgs;

const int n_TNF = 136;
const int n_TNFP = 16;

static size_t minContig = 2500;               // minimum contig size for binning
static size_t minContigByCorr = 1000;         // minimum contig size for recruiting (by abundance correlation)
static size_t minContigByCorrForGraph = 1000; // for graph generation purpose

std::unordered_map<std::string, int> TNmap;
std::unordered_set<std::string> TNPmap;

static bool revComp(char *s, int size)
{
    for (int i = 0; i < size; ++i)
    {
        if (s[i] == 'A')
            s[i] = 'T';
        else if (s[i] == 'T')
            s[i] = 'A';
        else if (s[i] == 'C')
            s[i] = 'G';
        else if (s[i] == 'G')
            s[i] = 'C';
        else
            return false;
    }
    return true;
}

int main(int argc, char const *argv[])
{
    std::string inFile = "test.gz";
    if (argc > 1)
    {
        n_THREADS = atoi(argv[1]);
        if (argc > 2)
        {
            inFile = argv[2];
        }
    }

    if (n_THREADS == 0)
        n_THREADS = omp_get_max_threads();
    else
        n_THREADS = std::min(n_THREADS, (int)omp_get_max_threads());
    omp_set_num_threads(n_THREADS);

    // std::cout << "n°threads:"<< n_THREADS << std::endl;

    for (size_t i = 0; i < n_TNF; ++i)
    {
        TNmap[TN[i]] = i;
    }

    for (size_t i = 0; i < n_TNFP; ++i)
    {
        TNPmap.insert(TNP[i]);
    }

    size_t nobs = 0;
    int nresv = 0;

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
                            smallCtgs.insert(nobs);
                        else
                            ++nresv;
                    }
                    //std::cout << kseq->name.s << std::endl;
                    // lCtgIdx[kseq->name.s] = nobs;
                    /////// cambio para facilitar la transferencia de gCtgIdx
                    gCtgIdx[nobs++] = seqs.size(); // global index
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
    std::cout <<"leer contigs "<< duration.count()/1000.f << "s " << std::endl;

    start = std::chrono::system_clock::now();

    double *TNF = (double *)malloc(nobs * n_TNF * sizeof(double));

    for (size_t i = 0; i < nobs * n_TNF; i++)
    {
        TNF[i] = 0;
    }

#pragma omp parallel for schedule(dynamic)
    for (size_t r = 0; r < nobs; ++r)
    {
        // omite el contig si pertenece a los smallcontigs

        if (smallCtgs.find(r) == smallCtgs.end())
        { // TNF is meaningless for small contigs
            // obtiene la secuencia del contig
            //-------------- necesito guardar seqs en la memoria de GPU
            std::string &s = seqs[gCtgIdx[r]];

            // crea la variable tn para almacenar las 4 bases
            char tn[5] = {'\0'};

            for (size_t i = 0; i < s.length() - 3; ++i)
            {
                // copia desde s a tn 4 bases partiendo desde i
                s.copy(tn, 4, i);

                // busca la secuencia encontrada en TNmap (para obtener el identificador de la secuencia)
                //---------------- necesito almacenat el mapa TNmap en GPU
                std::unordered_map<std::string, int>::iterator it = TNmap.find(tn);

                // si no es un palindromo aumenta el contador en la posición r, identificador de tn
                if (it != TNmap.end())
                    ++TNF[r * n_TNF + it->second];

                //********** aquí debería haber un continue si la condición de arriba no se cumple

                // reverse complement
                //  obtiene el reverso de tn
                std::reverse(tn, tn + 4);
                // modifica tn para obtener el complemento, error si el string no es correcto
                if (!revComp(tn, 4))
                {
                    // cout << "Unknown nucleotide letter found: " << s.substr(i, 4) << " in the row " << r + 1 << endl;
                    continue;
                }

                // si no es un palindromo aumenta el contador en la posición r, identificador de tn(complemento)
                if (TNPmap.find(tn) == TNPmap.end())
                { // if it is palindromic, then skip
                    it = TNmap.find(tn);
                    if (it != TNmap.end()) //********************** consulta innecesaria
                        ++TNF[r * n_TNF + it->second];
                }
            }

            double rsum = 0;
            for (size_t c = 0; c < n_TNF; ++c)
            {
                rsum += TNF[r * n_TNF + c] * TNF[r * n_TNF + c];
            }
            rsum = sqrt(rsum);
            for (size_t c = 0; c < n_TNF; ++c)
            {
                TNF[r * n_TNF + c] /= rsum;
            }
        }
    }

    end = std::chrono::system_clock::now();
    duration = end - start;
    std::cout <<"Crear matriz TNF "<< duration.count()/1000.f << "s " << std::endl;

    /*
    for (int i = 0; i < nobs * n_TNF; i++){
        std::cout << TNF[i] <<" "<< std::endl;
    }
    */
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
