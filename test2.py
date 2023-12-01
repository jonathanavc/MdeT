import sys
import subprocess
import json
import re
import math
from datetime import datetime

print("File: " + sys.argv[1])
archivo = sys.argv[1]
num_ex = 10

tiempos = {
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
        'File': archivo,
        'READ': {
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
        }
}
    

tiempos["MetabatCuda2"] = {
        'File': archivo,
        'READ': {
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
        }
}
        
print("METABAT CUDA 2")
for i in range(0, num_ex):
    p = subprocess.Popen(['./metabatcuda2','-i' + archivo, '-o'+'out/out','--ct', '32'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    if(err):
        print(err)
    valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
    tiempos["MetabatCuda2"]['READ']['ex'].append(float(valores[0]))
    tiempos["MetabatCuda2"]['TNF']['ex'].append(float(valores[1]))
    tiempos["MetabatCuda2"]['preGraph']['ex'].append(float(valores[2]))
    tiempos["MetabatCuda2"]['Graph']['ex'].append(float(valores[3]))
    tiempos["MetabatCuda2"]['Total']['ex'].append(float(valores[4]))
    tiempos["MetabatCuda2"]['binning']['ex'].append(float(valores[4]) - float(valores[3]) - float(valores[2]) - float(valores[1]) - float(valores[0]))

    p = subprocess.Popen(['./metabat2','-i' + archivo, '-o'+'out/out'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    if(err):
        print(err)
    valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
    tiempos["Metabat2"]['READ']['ex'].append(float(valores[0]))
    tiempos["Metabat2"]['TNF']['ex'].append(float(valores[1]))
    tiempos["Metabat2"]['preGraph']['ex'].append(float(valores[2]))
    tiempos["Metabat2"]['Graph']['ex'].append(float(valores[3]))
    tiempos["Metabat2"]['Total']['ex'].append(float(valores[4]))
    tiempos["Metabat2"]['binning']['ex'].append(float(valores[4]) - float(valores[3]) - float(valores[2]) - float(valores[1]) - float(valores[0]))
    
    print("[{:.1f}%] Test".format(((i + 1) / num_ex) * 100), end='\r')

tiempos["MetabatCuda2"]['READ']['avg'] = avg(tiempos["MetabatCuda2"]['READ']['ex'])
tiempos["MetabatCuda2"]['READ']['std'] = std(tiempos["MetabatCuda2"]['READ']['ex'])
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
        "MetabatCuda2": tiempos["MetabatCuda2"]['READ']['avg'],
        "Metabat2": tiempos["Metabat2"]['READ']['avg']
    },
    "TNF": {
        "MetabatCuda2": tiempos["MetabatCuda2"]['TNF']['avg'],
        "Metabat2": tiempos["Metabat2"]['TNF']['avg']
    },
    "preGraph": {
        "MetabatCuda2": tiempos["MetabatCuda2"]['preGraph']['avg'],
        "Metabat2": tiempos["Metabat2"]['preGraph']['avg']
    },
    "Graph": {
        "MetabatCuda2": tiempos["MetabatCuda2"]['Graph']['avg'],
        "Metabat2": tiempos["Metabat2"]['Graph']['avg']
    },
    "binning": {
        "MetabatCuda2": tiempos["MetabatCuda2"]['binning']['avg'],
        "Metabat2": tiempos["Metabat2"]['binning']['avg']
    },
    "Total": {
        "MetabatCuda2": tiempos["MetabatCuda2"]['Total']['avg'],
        "Metabat2": tiempos["Metabat2"]['Total']['avg']
    }   
}

print("Total CUDA: " + str(tiempos["MetabatCuda2"]['Total']['avg']))

print("Total OMP: " + str(tiempos["Metabat2"]['Total']['avg']))

#GUARDAR
_json = json.dumps(tiempos)

with open(archivo.split("/")[-1] +  datetime.now().strftime("_test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)