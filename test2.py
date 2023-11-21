import sys
import subprocess
import json
import re
from datetime import datetime

print("File: " + sys.argv[1])
archivo = sys.argv[1]
num_ex = 10

tiempos = {
    "Metabat2": {}
}

def avg(arr):
    return sum(arr)/len(arr)

def std(arr):
    avg = sum(arr)/len(arr)
    return sum([(x-avg)**2 for x in arr])/len(arr)


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
        'Total': {
            'avg': 0,
            'std': 0,
            'ex': []
        }
}
        
print("METABAT 2")
for i in range(0, num_ex):
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
    print("[{:.1f}%] Metabat2".format(((i + 1) / num_ex) * 100), end='\r')

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

print("Total: " + str(tiempos["Metabat2"]['Total']['avg']))

#GUARDAR
_json = json.dumps(tiempos)
with open(datetime.now().strftime("Test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)