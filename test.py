import subprocess
import json
import re
from datetime import datetime

num_ex = 16
threads = [1,2,4,6,8,10,12]
cuda_threads = [16,32,64,128]
cuda_bloqs = [32,64,128,256,512,1024,2048]



tiempos = {}

"""
tiempos['omp'] = {
    'n_threads':{
    }
}
for thread in threads:
    tiempos['omp']['n_threads'][str(thread)] = []
    for i in range(1, num_ex):
        print("["+str(thread)+"]"+"OMP "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
        p = subprocess.Popen(['./omp_ex', str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, err = p.communicate()
        tiempos['omp']['n_threads'][str(thread)].append(re.findall(r"[-+]?(?:\d*\.*\d+)", out))

tiempos['cuda'] = {
    'n_bloq':{
    }
}
for bloq in cuda_bloqs:
    tiempos['cuda']['n_bloq'][str(bloq)] = {
            'n_threads':{
        }
    }
    for thread in cuda_threads:
        tiempos['cuda']['n_bloq'][str(bloq)]['n_thread'][str(thread)] = []
        for i in range(1, num_ex):
            print("["+str(thread)+"/"+str(bloq)+"]"+"Cuda "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
            p = subprocess.Popen(['./cuda_ex', str(bloq), str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            out, err = p.communicate()
            tiempos['cuda']['n_bloq'][str(bloq)]['n_thread'][str(thread)].append(out)
"""

tiempos['cuda2'] = {
    'n_bloq':{
    }
}
for bloq in cuda_bloqs:
    tiempos['cuda2']['n_bloq'][str(bloq)] = {
            'n_threads':{
        }
    }
    for thread in cuda_threads: 
        tiempos['cuda2']['n_bloq'][str(bloq)]['n_thread'][str(thread)] = []
        for i in range(1, num_ex):
            print("["+str(thread)+"/"+str(bloq)+"]"+"Cuda2 "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
            p = subprocess.Popen(['./cuda2_ex', str(bloq), str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            out, err = p.communicate()
            tiempos['cuda2']['n_bloq'][str(bloq)]['n_thread'][str(thread)].append(out)

_json = json.dumps(tiempos)

with open(datetime.now().strftime("Test_%d/.%m.%Y_H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)