import sys
import subprocess
import json
import re
from datetime import datetime

print("File: " + sys.argv[1])
archivo = sys.argv[1]
num_ex = 10
threads = [1,2,4,9,18,36,72]
cuda_streams = [1,4,8,12]
cuda_threads = [16,32,64]
cuda_bloqs = [128,256,512,1024,2048]

tiempos = {}

def avg(arr):
    return sum(arr)/len(arr)

def std(arr):
    avg = sum(arr)/len(arr)
    return sum([(x-avg)**2 for x in arr])/len(arr)

'''
print("Metabat1")
for thread in threads:
    tiempos[thread] = {
        'read': [],
        'tnf': [],
        'prob': [],
        'binning': []
    }
    for i in range(0, num_ex):
        p = subprocess.Popen(['./metabat1','-i' + archivo, '-o'+'out/out', '-t' + str(thread)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
        out, err = p.communicate()
        valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
        tiempos[thread]['read'].append(float(valores[0]))
        tiempos[thread]['tnf'].append(float(valores[1]))
        tiempos[thread]['prob'].append(float(valores[2]))
        tiempos[thread]['binning'].append(float(valores[4]))
        print("[T:{}] Metabat1 {:.0f}% Read: {:.4f}±{:.4f} Tnf: {:.4f}±{:.4f} Prob: {:.4f}±{:.4f} Binning: {:.4f}±{:.4f}".format(
            thread,
            ((i + 1) / num_ex) * 100,
            avg(tiempos[thread]['read']),
            std(tiempos[thread]['read']),
            avg(tiempos[thread]['tnf']),
            std(tiempos[thread]['tnf']),
            avg(tiempos[thread]['prob']),
            std(tiempos[thread]['prob']),
            avg(tiempos[thread]['binning']),
            std(tiempos[thread]['binning'])
        ), end='\r')
    
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
'''

print("Metabat CUDA")
for stream in cuda_streams:
    tiempos[stream] = {}
    tiempos[stream]["read"] = ""
    tiempos[stream]["tnf"] = ""
    tiempos[stream]["prob"] = ""
    tiempos[stream]["binning"] = ""
    for cthread in cuda_threads:
        tiempos[stream]["read"] += str(cthread) + " "
        tiempos[stream]["tnf"] += str(cthread) + " "
        tiempos[stream]["prob"] += str(cthread) + " "
        tiempos[stream]["binning"] += str(cthread) + " "
        tiempos[stream][cthread] = {}
        for cbloq in cuda_bloqs:
            tiempos[stream]["read"] += str(cbloq) + " "
            tiempos[stream]["tnf"] += str(cbloq) + " "
            tiempos[stream]["prob"] += str(cbloq) + " "
            tiempos[stream]["binning"] += str(cbloq) + " "
            tiempos[stream][cthread][cbloq] = {
                'read': [],
                'tnf': [],
                'prob': [],
                'binning': []
            }
            for i in range(0, num_ex):
                p = subprocess.Popen(['./metabatcuda','-i' + archivo, '-o'+'out/out', '--cs' + str(stream),'--ct' + str(cthread), '--cb'+ str(cbloq)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                out, err = p.communicate()
                valores = re.findall(r"[-+]?(?:\d*\.*\d+)", out)
                tiempos[stream][cthread][cbloq]['read'].append(float(valores[0]))
                tiempos[stream][cthread][cbloq]['tnf'].append(float(valores[1]))
                tiempos[stream][cthread][cbloq]['prob'].append(float(valores[2]))
                tiempos[stream][cthread][cbloq]['binning'].append(float(valores[4]))
                print("[cs:{};ct:{};cb:{}] MetabatCuda {:.0f}% Read: {:.4f}±{:.4f} Tnf: {:.4f}±{:.4f} Prob: {:.4f}±{:.4f} Binning: {:.4f}±{:.4f}".format(
                    stream,cthread,cbloq,
                    ((i + 1) / num_ex) * 100,
                    avg(tiempos[stream][cthread][cbloq]['read']),
                    std(tiempos[stream][cthread][cbloq]['read']),
                    avg(tiempos[stream][cthread][cbloq]['tnf']),
                    std(tiempos[stream][cthread][cbloq]['tnf']),
                    avg(tiempos[stream][cthread][cbloq]['prob']),
                    std(tiempos[stream][cthread][cbloq]['prob']),
                    avg(tiempos[stream][cthread][cbloq]['binning']),
                    std(tiempos[stream][cthread][cbloq]['binning'])
                ), end='\r')
            
            tiempos[stream][cthread][cbloq]['avg'] = {
                "read": avg(tiempos[stream][cthread][cbloq]['read']),
                "tnf": avg(tiempos[stream][cthread][cbloq]['tnf']),
                "prob": avg(tiempos[stream][cthread][cbloq]['prob']),
                "binning": avg(tiempos[stream][cthread][cbloq]['binning'])
            }
            tiempos[stream][cthread][cbloq]['std'] = {
                "read": std(tiempos[stream][cthread][cbloq]['read']),
                "tnf": std(tiempos[stream][cthread][cbloq]['tnf']),
                "prob": std(tiempos[stream][cthread][cbloq]['prob']),
                "binning": std(tiempos[stream][cthread][cbloq]['binning'])
            }
            tiempos[stream]["read"] += str(tiempos[stream][cthread][cbloq]['avg']['read']) + "\n"
            tiempos[stream]["tnf"] += str(tiempos[stream][cthread][cbloq]['avg']['tnf']) + "\n"
            tiempos[stream]["prob"] += str(tiempos[stream][cthread][cbloq]['avg']['prob']) + "\n"
            tiempos[stream]["binning"] += str(tiempos[stream][cthread][cbloq]['avg']['binning']) + "\n"
   



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