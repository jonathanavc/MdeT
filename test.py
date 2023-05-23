import subprocess
import json

num_ex = 20
threads = [1,2,4,6,8,10,10]
cuda_threads = [16,32,64,128]
cuda_bloqs = [32,64,128,256,512,1024,2048]



tiempos = {}

tiempos['omp'] = {
    'n_threads':{
    }
}
for thread in threads:
    tiempos['omp']['n_threads'][str(thread)] = []
    for i in range(1, num_ex):
        print("OMP "+ str(((i-1)/num_ex) * 100) + "%", end='\r')
        p = subprocess.Popen(['time', './omp_ex', str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        err, out = p.communicate()
        tiempos['omp']['n_threads'][str(thread)].append(out)
        print(tiempos)

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
            p = subprocess.Popen(['time','./cuda_ex', str(bloq), str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            err, out = p.communicate()
            tiempos['cuda']['n_bloq'][str(bloq)]['n_thread'][str(thread)].append(out)

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
            p = subprocess.Popen(['time','./cuda2_ex', str(bloq), str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            err, out = p.communicate()
            tiempos['cuda2']['n_bloq'][str(bloq)]['n_thread'][str(thread)].append(out)

_json = json.dumps(tiempos)

with open("sample.json", "w") as outfile:
    outfile.write(_json)