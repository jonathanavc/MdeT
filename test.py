import subprocess
import json

num_ex = 20
theads = [1,2,4,6,8,10,10]
cuda_threads = [16,32,64,128]
cuda_bloqs = [32,64,128,256,512,1024,2048]

tiempos = {}

for thead in theads:
    for i in range(1, num_ex):
        print("OMP "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
        p = subprocess.call(['time', './omp_ex', str(thead)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out = p.communicate()
        tiempos["omp"][thead][i] = out
        print(tiempos)

for bloq in cuda_bloqs:
    for thead in cuda_threads:
        for i in range(1, num_ex):
            print("["+str(thead)+"/"+str(bloq)+"]"+"Cuda "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
            p = subprocess.call(['time','./cuda_ex', str(bloq), str(theads)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out = p.communicate()
            tiempos["cuda"][bloq][thead][i] = out
            
for bloq in cuda_bloqs:
    for thead in cuda_threads: 
        for i in range(1, num_ex):
            print("["+str(thead)+"/"+str(bloq)+"]"+"Cuda2 "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
            p = subprocess.call(['time','./cuda2_ex', str(bloq), str(theads)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out = p.communicate()
            tiempos["cuda2"][bloq][thead][i] = out

_json = json.dumps(tiempos)

with open("sample.json", "w") as outfile:
    outfile.write(_json)