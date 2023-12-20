import sys
import subprocess
import json
import re
import math
import random
from datetime import datetime

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

tiempos = {
    'File': archivo,
    'abd': archivo_abd,
    "MetabatCuda2": {},
    "Metabat2": {}
}

def avg(arr):
    return sum(arr)/len(arr)

def std(arr):
    avg = sum(arr)/len(arr)
    return math.sqrt(sum([(x-avg)**2 for x in arr])/len(arr))
    #return  sum([(x-avg)**2 for x in arr])/len(arr)

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
        p = subprocess.Popen(['./metabatcuda2','-i' + archivo,'-a',archivo_abd, '-o'+'outMetabat2/out','--ct', '32', '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    else:
        p = subprocess.Popen(['./metabatcuda2','-i' + archivo, '-o'+'outMetabat2Cuda/out','--ct', '32', '--seed', str(seed)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    if(err):
        print(err)
    valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
    tiempos["MetabatCuda2"]['READ']['ex'].append(float(valores[0]))
    tiempos["MetabatCuda2"]['ABD']['ex'].append(float(valores[1]))
    tiempos["MetabatCuda2"]['TNF']['ex'].append(float(valores[2]))
    tiempos["MetabatCuda2"]['preGraph']['ex'].append(float(valores[3]))
    tiempos["MetabatCuda2"]['Graph']['ex'].append(float(valores[4]))
    tiempos["MetabatCuda2"]['Total']['ex'].append(float(valores[6]))
    tiempos["MetabatCuda2"]['binning']['ex'].append(float(valores[6]) - float(valores[4]) - float(valores[3]) - float(valores[2]) - float(valores[1]) - float(valores[0]))
    tiempos["MetabatCuda2"]['Nbins'].append(float(valores[5]))

    p = subprocess.Popen(['./metabat2','-i' + archivo,'-a',archivo_abd, '-o'+'out/out'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    if(err):
        print(err)
    valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
    tiempos["Metabat2"]['READ']['ex'].append(float(valores[0]))
    tiempos["Metabat2"]['ABD']['ex'].append(float(valores[1]))
    tiempos["Metabat2"]['TNF']['ex'].append(float(valores[2]))
    tiempos["Metabat2"]['preGraph']['ex'].append(float(valores[3]))
    tiempos["Metabat2"]['Graph']['ex'].append(float(valores[4]))
    tiempos["Metabat2"]['Total']['ex'].append(float(valores[6]))
    tiempos["Metabat2"]['binning']['ex'].append(float(valores[6]) - float(valores[4]) - float(valores[3]) - float(valores[2]) - float(valores[1]) - float(valores[0]))
    tiempos["Metabat2"]['Nbins'].append(float(valores[5]))
    
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
    "READ": {
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
    "preGraph": {
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
    "binning": {
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
        "MetabatCuda2":
            tiempos["MetabatCuda2"]['Nbins'],
        "Metabat2":
            tiempos["Metabat2"]['Nbins']
        
    }
}

print("Total CUDA: " + str(tiempos["MetabatCuda2"]['Total']['avg']))

print("Total OMP: " + str(tiempos["Metabat2"]['Total']['avg']))

print(tiempos)

#GUARDAR
_json = json.dumps(tiempos)

with open("test/" + archivo.split("/")[-1] +  datetime.now().strftime("_test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)