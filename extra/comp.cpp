#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

int main(int argc, char const *argv[]){
    if(argc < 3){
        cout<< "modo de uso ~"<< argv[0]<<" file1 file2" << endl;
    }

    vector<double> TNF1;
    vector<double> TNF2;
    double value;

    ifstream file1(argv[1], ios::binary);
    ifstream file2(argv[2], ios::binary);

    while (file1.read((char*)&value, sizeof(double))){
        TNF1.push_back(value);
    }

    
    while (file2.read((char*)&value, sizeof(double))){
        TNF1.push_back(value);
    }
    file1.close();
    file2.close();

    if(TNF1.size()!= TNF2.size()){
        cout << "ERROR TAMAÑO" << endl;
        return 1;
    }
    for (size_t i = 0; i < TNF1.size(); i++){
        if(TNF1[i] != TNF2[i]) cout << TNF1[i] - TNF2[i] << endl;
    }
    
    return 0;
}