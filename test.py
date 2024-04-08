import sys
import subprocess
import json
import re
import math
import random
from datetime import datetime
import hashlib
import os

num_ex = 10

archivo = ""
archivo_abd = ""
if(len(sys.argv) < 2):
    print("No file")
    exit()
else:
    archivo = sys.argv[1]
    print("File: " + archivo)

if(len(sys.argv) < 3):
    print("No file abd")

else:
    archivo_abd = sys.argv[2]
    print("File abd: " + archivo_abd)

seeds_list = []
sim_list = []

tiempos = {
    'File': archivo,
    'abd': archivo_abd,
    "MetabatCuda2": {},
    "Metabat2": {}
}

os.system("rm -rf " + "outMetabat2")
os.system("rm -rf " + "outMetabat2Cuda")

def avg(arr):
    return sum(arr)/len(arr)

def std(arr):
    avg = sum(arr)/len(arr)
    return math.sqrt(sum([(x-avg)**2 for x in arr])/len(arr))
    #return  sum([(x-avg)**2 for x in arr])/len(arr)

def check_md5(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        file1_hash = hashlib.md5(f1.read()).hexdigest()
        file2_hash = hashlib.md5(f2.read()).hexdigest()
    return file1_hash == file2_hash

def check_all_files_in_directory(directory_path1, directory_path2):
    files = os.listdir(directory_path1)
    diff_files = len(os.listdir(directory_path1)) - len(os.listdir(directory_path2))
    if len(os.listdir(directory_path1)) > len(os.listdir(directory_path2)):
        files = os.listdir(directory_path2)
    verified_files_count = 0
    for i in range(0, len(files)):
        file1 = os.path.join(directory_path1, files[i])
        file2 = os.path.join(directory_path2, files[i])
        if check_md5(file1, file2):
            verified_files_count += 1
    os.system("rm -rf " + directory_path1)
    os.system("rm -rf " + directory_path2)
    if len(files) != 0:
        return (verified_files_count / (len(files) + diff_files)) * 100
    else:
        return 100

tiempos["Metabat2"] = {
        'READ': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'ABD': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'TNF': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'preGraph': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'Graph': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'binning':{
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'Total': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'Nbins': []
}
    

tiempos["MetabatCuda2"] = {
        'READ': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'ABD': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'TNF': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'preGraph': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'Graph': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'binning':{
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'Total': {
            'avg': 0,
            'std': 0,
            'ex': []
        },
        'Nbins': []
}
        
print("METABAT CUDA 2")
for i in range(0, num_ex):
    seed = random.randint(0, 1000000000)
    seeds_list.append(seed) 

    if(archivo_abd != ""):
        p = subprocess.Popen(['./metabatcuda2','-i' + archivo,'-a',archivo_abd, '-o'+'outMetabat2Cuda/out','--ct', '32', '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else:
        p = subprocess.Popen(['./metabatcuda2','-i' + archivo, '-o'+'outMetabat2Cuda/out','--ct', '32', '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    if(err):
        print(err)
    valores = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.\d*|\d+\.?\d*)(?:[eE][-+]?\d+)?", out)
    tiempos["MetabatCuda2"]['READ']['ex'].append(float(valores[0]))
    tiempos["MetabatCuda2"]['ABD']['ex'].append(float(valores[1]))
    tiempos["MetabatCuda2"]['TNF']['ex'].append(float(valores[2]))
    tiempos["MetabatCuda2"]['preGraph']['ex'].append(float(valores[3]))
    tiempos["MetabatCuda2"]['Graph']['ex'].append(float(valores[4]))
    tiempos["MetabatCuda2"]['Total']['ex'].append(float(valores[6]))
    tiempos["MetabatCuda2"]['binning']['ex'].append(float(valores[6]) - float(valores[4]) - float(valores[3]) - float(valores[2]) - float(valores[1]) - float(valores[0]))
    tiempos["MetabatCuda2"]['Nbins'].append(float(valores[5]))

    if(archivo_abd != ""):
        p = subprocess.Popen(['./metabat2','-i' + archivo,'-a',archivo_abd, '-o'+'outMetabat2/out', '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else:
        p = subprocess.Popen(['./metabat2','-i' + archivo, '-o'+'outMetabat2/out', '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    if(err):
        print(err)
    #valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
    valores = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.\d*|\d+\.?\d*)(?:[eE][-+]?\d+)?", out)
    
    tiempos["Metabat2"]['READ']['ex'].append(float(valores[0]))
    tiempos["Metabat2"]['ABD']['ex'].append(float(valores[1]))
    tiempos["Metabat2"]['TNF']['ex'].append(float(valores[2]))
    tiempos["Metabat2"]['preGraph']['ex'].append(float(valores[3]))
    tiempos["Metabat2"]['Graph']['ex'].append(float(valores[4]))
    tiempos["Metabat2"]['Total']['ex'].append(float(valores[6]))
    tiempos["Metabat2"]['binning']['ex'].append(float(valores[6]) - float(valores[4]) - float(valores[3]) - float(valores[2]) - float(valores[1]) - float(valores[0]))
    tiempos["Metabat2"]['Nbins'].append(float(valores[5]))

    sim_list.append(check_all_files_in_directory("outMetabat2", "outMetabat2Cuda"))
    print("[{:.1f}%] Test".format(((i + 1) / num_ex) * 100), end='\r')

tiempos["MetabatCuda2"]['READ']['avg'] = avg(tiempos["MetabatCuda2"]['READ']['ex'])
tiempos["MetabatCuda2"]['READ']['std'] = std(tiempos["MetabatCuda2"]['READ']['ex'])
tiempos["MetabatCuda2"]['ABD']['avg'] = avg(tiempos["MetabatCuda2"]['ABD']['ex'])
tiempos["MetabatCuda2"]['ABD']['std'] = std(tiempos["MetabatCuda2"]['ABD']['ex'])
tiempos["MetabatCuda2"]['TNF']['avg'] = avg(tiempos["MetabatCuda2"]['TNF']['ex'])
tiempos["MetabatCuda2"]['TNF']['std'] = std(tiempos["MetabatCuda2"]['TNF']['ex'])
tiempos["MetabatCuda2"]['preGraph']['avg'] = avg(tiempos["MetabatCuda2"]['preGraph']['ex'])
tiempos["MetabatCuda2"]['preGraph']['std'] = std(tiempos["MetabatCuda2"]['preGraph']['ex'])
tiempos["MetabatCuda2"]['Graph']['avg'] = avg(tiempos["MetabatCuda2"]['Graph']['ex'])
tiempos["MetabatCuda2"]['Graph']['std'] = std(tiempos["MetabatCuda2"]['Graph']['ex'])
tiempos["MetabatCuda2"]['Total']['avg'] = avg(tiempos["MetabatCuda2"]['Total']['ex'])
tiempos["MetabatCuda2"]['Total']['std'] = std(tiempos["MetabatCuda2"]['Total']['ex'])
tiempos["MetabatCuda2"]['binning']['avg'] = avg(tiempos["MetabatCuda2"]['binning']['ex'])
tiempos["MetabatCuda2"]['binning']['std'] = std(tiempos["MetabatCuda2"]['binning']['ex'])


tiempos["Metabat2"]['READ']['avg'] = avg(tiempos["Metabat2"]['READ']['ex'])
tiempos["Metabat2"]['READ']['std'] = std(tiempos["Metabat2"]['READ']['ex'])
tiempos["Metabat2"]['ABD']['avg'] = avg(tiempos["Metabat2"]['ABD']['ex'])
tiempos["Metabat2"]['ABD']['std'] = std(tiempos["Metabat2"]['ABD']['ex'])
tiempos["Metabat2"]['TNF']['avg'] = avg(tiempos["Metabat2"]['TNF']['ex'])
tiempos["Metabat2"]['TNF']['std'] = std(tiempos["Metabat2"]['TNF']['ex'])
tiempos["Metabat2"]['preGraph']['avg'] = avg(tiempos["Metabat2"]['preGraph']['ex'])
tiempos["Metabat2"]['preGraph']['std'] = std(tiempos["Metabat2"]['preGraph']['ex'])
tiempos["Metabat2"]['Graph']['avg'] = avg(tiempos["Metabat2"]['Graph']['ex'])
tiempos["Metabat2"]['Graph']['std'] = std(tiempos["Metabat2"]['Graph']['ex'])
tiempos["Metabat2"]['Total']['avg'] = avg(tiempos["Metabat2"]['Total']['ex'])
tiempos["Metabat2"]['Total']['std'] = std(tiempos["Metabat2"]['Total']['ex'])
tiempos["Metabat2"]['binning']['avg'] = avg(tiempos["Metabat2"]['binning']['ex'])
tiempos["Metabat2"]['binning']['std'] = std(tiempos["Metabat2"]['binning']['ex'])

tiempos["tabla"] = {
    "Lectura": {
        "MetabatCuda2": {
            "avg":tiempos["MetabatCuda2"]['READ']['avg'],
            "std": tiempos["MetabatCuda2"]['READ']['std']
        },
        "Metabat2": {
            "avg":tiempos["Metabat2"]['READ']['avg'],
            "std": tiempos["Metabat2"]['READ']['std']
        }
    },
    "ABD": {
        "MetabatCuda2": {
            "avg":tiempos["MetabatCuda2"]['ABD']['avg'],
            "std": tiempos["MetabatCuda2"]['ABD']['std']
        },
        "Metabat2": {
            "avg":tiempos["Metabat2"]['ABD']['avg'],
            "std": tiempos["Metabat2"]['ABD']['std']
        }
    },
    "TNF": {
        "MetabatCuda2": {
            "avg":tiempos["MetabatCuda2"]['TNF']['avg'],
            "std": tiempos["MetabatCuda2"]['TNF']['std']
        },
        "Metabat2": {
            "avg": tiempos["Metabat2"]['TNF']['avg'],
            "std": tiempos["Metabat2"]['TNF']['std']
        }
    },
    "PreGraph": {
        "MetabatCuda2": {
            "avg": tiempos["MetabatCuda2"]['preGraph']['avg'],
            "std": tiempos["MetabatCuda2"]['preGraph']['std']
        },
        "Metabat2": {
            "avg": tiempos["Metabat2"]['preGraph']['avg'],
            "std": tiempos["Metabat2"]['preGraph']['std']
        }
    },
    "Graph": {
        "MetabatCuda2": {
            "avg": tiempos["MetabatCuda2"]['Graph']['avg'],
            "std": tiempos["MetabatCuda2"]['Graph']['std'],
        
        },
        "Metabat2": {
            "avg": tiempos["Metabat2"]['Graph']['avg'],
            "std": tiempos["Metabat2"]['Graph']['std'],
        }
    },
    "Binning": {
        "MetabatCuda2":{
            "avg": tiempos["MetabatCuda2"]['binning']['avg'],
            "std": tiempos["MetabatCuda2"]['binning']['std'],
        },
        "Metabat2":{
            "avg": tiempos["Metabat2"]['binning']['avg'],
            "std": tiempos["Metabat2"]['binning']['std'],
        }
    },
    "Total": {
        "MetabatCuda2": {
            "avg": tiempos["MetabatCuda2"]['Total']['avg'],
            "std": tiempos["MetabatCuda2"]['Total']['std'],
        
        },
        "Metabat2": {
            "avg": tiempos["Metabat2"]['Total']['avg'],
            "std": tiempos["Metabat2"]['Total']['std'],
        }
    },
    "Nbins": {
        "Seeds": 
            seeds_list,
        "Similarity":
            sim_list,
        "MetabatCuda2":
            tiempos["MetabatCuda2"]['Nbins'],
        "Metabat2":
            tiempos["Metabat2"]['Nbins']
        
    }
}


latex_bins = '\'hline Semilla & Metabat 2 &  Metabat 2 CUDA \ \'hline '
for i in range(0, num_ex):
    latex_bins += str(seeds_list[i]) + " & " + str(int(tiempos["Metabat2"]['Nbins'][i])) + " & " + str(int(tiempos["MetabatCuda2"]['Nbins'][i])) + "\ "
latex_bins += " \'hline"
latex_bins = latex_bins.__str__()

latex_m2 = "{ "
for key in tiempos["tabla"].keys():
    if key == "Nbins":
        continue
    latex_m2 += "(" + key + "," + str(tiempos["tabla"][key]['Metabat2']['avg']) + ") +- (0," + str(tiempos["tabla"][key]['Metabat2']['std']) + ") "
latex_m2 += " };"

latex_mc2 = " { "
for key in tiempos["tabla"].keys():
    if key == "Nbins":
        continue
    latex_mc2 += "(" + key + "," + str(tiempos["tabla"][key]['MetabatCuda2']['avg']) + ") +- (0," + str(tiempos["tabla"][key]['MetabatCuda2']['std']) + ") "
latex_mc2 += " };"

tiempos["latex"] = {
    "Metabat2" : latex_m2,
    "MetabatCuda2" : latex_mc2,
    "Bins": latex_bins,
    "TimeReduction": ((tiempos["Metabat2"]['Total']['avg']-tiempos["MetabatCuda2"]['Total']['avg']) / tiempos["Metabat2"]['Total']['avg']) * 100,
    "Aceleration": tiempos["Metabat2"]['Total']['avg'] / tiempos["MetabatCuda2"]['Total']['avg']
}

print("Total CUDA: " + str(tiempos["MetabatCuda2"]['Total']['avg']))
print("Total OMP: " + str(tiempos["Metabat2"]['Total']['avg']))
print("Time Reduction: " + str(tiempos["latex"]["TimeReduction"]))
print("Aceleration: " + str(tiempos["latex"]["Aceleration"]))

#GUARDAR
_json = json.dumps(tiempos)

with open("test/" + archivo.split("/")[-1] +  datetime.now().strftime("_test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)