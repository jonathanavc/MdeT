import sys
import subprocess
import json
import re
from datetime import datetime

print(sys.argv[1])
archivo = sys.argv[1]
num_ex = 10
threads = [1,2,4,6,8,10,12]
cuda_threads = [16,32,64,128]
cuda_bloqs = [32,64,128,256,512,1024,2048]

tiempos = {}

#CUDA V3
tiempos['cuda'] = {
    'n_bloqs':{
    }
}
for bloq in cuda_bloqs:
    tiempos['cuda']['n_bloqs'][str(bloq)] = {
            'n_threads':{
        }
    }
    for thread in cuda_threads: 
        tiempos['cuda']['n_bloqs'][str(bloq)]['n_threads'][str(thread)] = []
        for i in range(0, num_ex):
            print("[T:"+str(thread)+"/B:"+str(bloq)+"]"+"Cuda "+ str((i/num_ex) * 100) + "%", end='\r')
            p = subprocess.Popen(['./metabatcuda', str(bloq), str(thread), archivo], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            out, err = p.communicate()
            valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
            tiempos['cuda']['n_bloqs'][str(bloq)]['n_threads'][str(thread)] += valores

#GUARDAR
_json = json.dumps(tiempos)
with open(datetime.now().strftime("Test_%d.%m.%Y_%H.%M.%S")+".json", "w") as outfile:
    outfile.write(_json)