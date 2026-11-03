import json, subprocess, sys
from pathlib import Path
import numpy as np
root=Path('/home/john/Strata');build=root/'build-tensor-parralize-baseline'
sys.path.insert(0,str(root/'third_party/llama.cpp/gguf-py'))
import gguf
rng=np.random.default_rng(20261104)
count=32768
raw=np.empty((count,34),dtype=np.uint8)
scales=rng.integers(1,0x6000,size=count,dtype=np.uint16)
scales|=rng.integers(0,2,size=count,dtype=np.uint16)<<15
raw[:,:2]=scales.astype('<u2').view(np.uint8).reshape(-1,2)
raw[:,2:]=rng.integers(0,256,size=(count,32),dtype=np.uint8)
raw[0]=0
raw[1:4,:2]=np.array([0x0001,0x8001,0x0400],dtype='<u2').view(np.uint8).reshape(-1,2)
raw[4:6,:2]=np.array([0x3000,0xb000],dtype='<u2').view(np.uint8).reshape(-1,2)
raw[4:6,2:]=np.tile(np.array([127,-127],dtype=np.int8).view(np.uint8),16)
source=build/'mtp-head-q8-samples.bin';source.write_bytes(raw.tobytes())
expected=gguf.quants.quantize(gguf.quants.dequantize(raw,gguf.GGMLQuantizationType.Q8_0),gguf.GGMLQuantizationType.Q4_0)
report=[]
for name in ['mtp-head-host-test','mtp-head-host-sanitized']:
    dest=build/(name+'.bin')
    subprocess.run([str(build/name),str(source),str(dest)],check=True)
    got=np.frombuffer(dest.read_bytes(),dtype=np.uint8).reshape(count,18)
    ref=expected.reshape(count,18)
    if not np.array_equal(got,ref):
        where=np.argwhere(got!=ref)
        raise AssertionError((name,where[:8].tolist(),len(where)))
    report.append({'binary':name,'reference_blocks':count,'all_bytes_equal':True})
out={'reference':'pinned gguf-py Q8_0 dequantize then Q4_0 quantize','checks':report,
     'includes':'random signed FP16 scales, zeros, subnormals, opposite signed maxima, extreme int8, row order/repetition and invalid metadata'}
(build/'mtp-head-quant-checks.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
