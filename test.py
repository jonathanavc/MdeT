import sys
import subprocess
import json
import re
from datetime import datetime

print("File: " + sys.argv[1])
archivo = sys.argv[1]
num_ex = 10
threads = [1,2,4,9,18,36,72]
cuda_streams = [1,2,4,6,8,10,12]
cuda_threads = [16,32,64,128]
cuda_bloqs = [32,64,128,256,512,1024,2048]

tiempos = {}

def avg(arr):
    return sum(arr)/len(arr)

def std(arr):
    avg = sum(arr)/len(arr)
    return sum([(x-avg)**2 for x in arr])/len(arr)

print("Metabat1")
for thread in threads:
    tiempos[thread] = {
        'read': [],
        'tnf': [],
        'prob': [],
        'binning': []
    }
    for i in range(0, num_ex):
        #print("[T:"+str(thread)+']'+"metabat1 "+ str((i/num_ex) * 100) + "%", end='\r')
        p = subprocess.Popen(['./metabat1','-i' + archivo, '-o'+'out/out', '-t' + str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, err = p.communicate()
        valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
        #print(type(valores[0]))
        #print(float(valores[0]))
        tiempos[thread]['read'].append(float(valores[0]))
        tiempos[thread]['tnf'].append(float(valores[1]))
        tiempos[thread]['prob'].append(float(valores[2]))
        tiempos[thread]['binning'].append(float(valores[4]))
        #tiempos[thread] += [valores[0], valores[1], valores[3], valores[2]]
        print("[T:"+str(thread)+']'+"metabat1 "+ str(((i + 1)/num_ex) * 100) + "%" + " read: " + str(avg(tiempos[thread]['read'])) + "±" + str(std(tiempos[thread]['read'])) + " tnf: " + str(avg(tiempos[thread]['tnf']))+ "±" + str(std(tiempos[thread]['tnf'])) + " prob: " + str(avg(tiempos[thread]['prob'])) + "±" + str(std(tiempos[thread]['prob'])) + " binning: " + str(avg(tiempos[thread]['binning']  + '±' + str(std(tiempos[thread]['binning'])))), end='\r')
    
    tiempos[thread]['avg'] = {
        "read": avg(tiempos[thread]['read']),
        "tnf": avg(tiempos[thread]['tnf']),
        "prob": avg(tiempos[thread]['prob']),
        "binning": avg(tiempos[thread]['binning'])
    }
    tiempos[thread]['std'] = {
        "read": std(tiempos[thread]['read']),
        "tnf": std(tiempos[thread]['tnf']),
        "prob": std(tiempos[thread]['prob']),
        "binning": std(tiempos[thread]['binning'])
    }
    print("\n")

'''
for bloq in cuda_bloqs:
    for thread in cuda_threads:
        for stream in cuda_streams:
            for i in range(0, num_ex):
                print("[T:"+str(thread)+"/B:"+str(bloq)+"/S:"+str(stream)+"]"+"Cuda "+ str((i/num_ex) * 100) + "%", end='\r')
                p = subprocess.Popen(['./metabatcuda',"-cs "+ str(stream), "-cb " + str(bloq),"-ct" + str(thread), "-i "+ archivo, "-o out/out"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                out, err = p.communicate()
                valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
                print(valores)
                #tiempos['cuda']['n_bloqs'][str(bloq)]['n_threads'][str(thread)] += valores
'''

#GUARDAR
_json = json.dumps(tiempos)
with open(datetime.now().strftime("Test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)