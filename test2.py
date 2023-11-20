import sys
import subprocess
import json
import re
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
    return sum([(x-avg)**2 for x in arr])/len(arr)


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
        'Total': {
            'avg': 0,
            'std': 0,
            'ex': []
        }
}
        
print("METABAT CUDA 2")
for i in range(0, num_ex):
    p = subprocess.Popen(['./metabatcuda2','-i' + archivo, '-o'+'out/out','--ct' + '32'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    out, err = p.communicate()
    print(err)
    valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
    tiempos["MetabatCuda2"]['READ']['ex'].append(float(valores[0]))
    tiempos["MetabatCuda2"]['TNF']['ex'].append(float(valores[1]))
    tiempos["MetabatCuda2"]['preGraph']['ex'].append(float(valores[2]))
    tiempos["MetabatCuda2"]['Graph']['ex'].append(float(valores[3]))
    tiempos["MetabatCuda2"]['Total']['ex'].append(float(valores[4]))
    print("[{:.1f}%] MetabatCuda2").format(((i + 1) / num_ex) * 100)

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

#GUARDAR
_json = json.dumps(tiempos)
with open(datetime.now().strftime("Test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)