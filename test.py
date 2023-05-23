import subprocess
import json

num_ex = 20
theads = [1,2,4,6,8,10,10]
cuda_threads = [16,32,64,128]
cuda_bloqs = [32,64,128,256,512,1024,2048]

tiempos = {}

print("Compilando archivos...",end='\r')
try:
    subprocess.call('sh compilat.sh', stderr=subprocess.DEVNULL,stdout=subprocess.DEVNULL)
except OSError:
    print("Error al compilar, verifique que gcc, cuda(12+) y zlibdev esté instalado", end='\r')
    exit()

for thead in theads:
    for i in range(1, num_ex):
        print("OMP "+ (i/num_ex) * 100 + "%", end='\r')
        p = subprocess.call(['time', './omp_ex', thead], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out = p.communicate()
        tiempos["omp"][thead][i] = out

for bloq in cuda_bloqs:
    for thead in cuda_threads:
        for i in range(1, num_ex):
            print("["+thead+"/"+bloq+"]"+"Cuda "+ (i/num_ex) * 100 + "%", end='\r')
            p = subprocess.call(['time','./cuda_ex', bloq, theads], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out = p.communicate()
            tiempos["cuda"][bloq][thead][i] = out

for bloq in cuda_bloqs:
    for thead in cuda_threads: 
        for i in range(1, num_ex):
            print("["+thead+"/"+bloq+"]"+"Cuda "+ (i/num_ex) * 100 + "%", end='\r')
            p = subprocess.call(['time','./cuda2_ex', bloq, theads], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out = p.communicate()
            tiempos["cuda2"][bloq][thead][i] = out

_json = json.dumps(tiempos)

with open("sample.json", "w") as outfile:
    outfile.write(_json)