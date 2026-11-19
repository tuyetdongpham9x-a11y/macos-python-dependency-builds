"""Small public tensor/NumPy/operator ABI probe; no weights or application code."""
import argparse,json,os,platform,sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--architecture',choices=['x86_64','arm64'],required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 import numpy as np
 import torch,torchvision
 if platform.system()!='Darwin' or platform.machine()!=args.architecture:raise RuntimeError('Native Mac architecture mismatch')
 if sys.version_info[:3]!=(3,11,18):raise RuntimeError('Unexpected Python version')
 if not str(torch.__version__).startswith('2.13.0') or not str(torchvision.__version__).startswith('0.28.0'):raise RuntimeError('Unexpected library version')
 torch.set_num_threads(2)
 capability=torch.backends.cpu.get_cpu_capability()
 if args.architecture=='x86_64':assert os.environ.get('ATEN_CPU_CAPABILITY')=='default' and capability in ('NO AVX','DEFAULT')
 a=np.arange(16,dtype=np.float32);tensor=torch.from_numpy(a.copy());assert tensor.device.type=='cpu';assert np.array_equal(tensor.numpy(),a)
 freq=torch.fft.rfft(tensor);restored=torch.fft.irfft(freq,n=tensor.numel());assert torch.allclose(restored,tensor,atol=1e-5,rtol=1e-5)
 matrix=torch.tensor([[3.,1.],[1.,2.]],device='cpu');u,s,v=torch.linalg.svd(matrix);assert torch.allclose(u@torch.diag(s)@v,matrix,atol=1e-5,rtol=1e-5)
 boxes=torch.tensor([[0.,0.,1.,1.],[2.,2.,3.,3.]],device='cpu');indices=torchvision.ops.nms(boxes,torch.tensor([.8,.7]),.5);assert indices.tolist()==[0,1]
 assert torch.version.cuda is None
 if args.architecture=='x86_64':assert not torch.backends.mps.is_built()
 report={'schema':1,'passed':True,'architecture':platform.machine(),'python':platform.python_version(),'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'numpy':np.__version__,'cpuCapabilityRequested':os.environ.get('ATEN_CPU_CAPABILITY'),'cpuCapabilityReported':capability,'genericFallbackProbe':True,'cpuTensorNumpyBridge':True,'cpuFFT':True,'cpuLinearAlgebra':True,'nativeVisionNMS':True,'gpuOperationsPerformed':False,'mpsBuildIncluded':torch.backends.mps.is_built(),'checkpointsDownloaded':False,'applicationCompatibilityProven':False}
 args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
