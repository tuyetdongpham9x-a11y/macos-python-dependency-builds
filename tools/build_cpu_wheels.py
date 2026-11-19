"""Generic pinned public CPU wheel experiment. Never imports application sources.

Only standard native public-repository runners are allowed. No source patches,
model/checkpoint downloads, private repositories, credentials or publishing.
A timeout stops the direct subprocess parent only; runner teardown is required
for descendants. No group signals or cleanup retries are used.
"""
import argparse,hashlib,json,os,platform,re,shutil,stat,struct,subprocess,tarfile,time,urllib.parse,urllib.request,zipfile
from pathlib import Path,PurePosixPath

LOCK_SHA256="6588893c7e60b484a3e8a9b0f63a095af23ed18349ae7b20e1b742e6071121b7"
DEADLINE=None
PHASES=[]
MAX_SECONDS=170*60
MAX_WHEEL_EXPANDED=2*1024*1024*1024

class BuildStopped(RuntimeError):pass

def sha(path):
 digest=hashlib.sha256()
 with path.open('rb')as stream:
  for data in iter(lambda:stream.read(1024*1024),b''):digest.update(data)
 return digest.hexdigest()

def save(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')

def run(command,log,env,seconds=3600,cwd=None):
 remaining=DEADLINE-time.monotonic()
 if remaining<=0:raise BuildStopped('170-minute driver deadline reached')
 row={'phase':log.stem,'command':list(map(str,command)),'timeoutSeconds':min(seconds,remaining),'passed':False};PHASES.append(row)
 log.parent.mkdir(parents=True,exist_ok=True);started=time.monotonic();print('PHASE '+log.stem,flush=True)
 try:
  with log.open('w',encoding='utf-8')as output:subprocess.run(row['command'],cwd=cwd,env=env,stdout=output,stderr=subprocess.STDOUT,timeout=min(seconds,remaining),check=True)
  row['passed']=True
 except subprocess.TimeoutExpired as error:row['timedOut']=True;raise BuildStopped('Timeout; direct parent stopped, descendants require runner teardown')from error
 except OSError as error:row['processError']=type(error).__name__;raise BuildStopped('Process control failure; no retry: '+str(error))from error
 except subprocess.CalledProcessError as error:row['exitCode']=error.returncode;raise RuntimeError('Command failed: '+log.stem)from error
 finally:row['elapsedSeconds']=round(time.monotonic()-started,3)
 return log.read_text(encoding='utf-8',errors='replace')

def public_url(text):
 value=urllib.parse.urlsplit(text)
 if value.scheme!='https' or value.hostname not in ('github.com','files.pythonhosted.org')or value.username or value.password or value.query or value.fragment:raise ValueError('Only named public HTTPS sources are allowed')
 return text

def download(entry,folder):
 name=entry['filename']
 if Path(name).name!=name or '\\'in name:raise ValueError('Unsafe download name')
 target=folder/name;limit=entry.get('bytes',entry.get('maxBytes',256*1024*1024))
 if target.exists():
  if sha(target)!=entry['sha256']:raise ValueError('Existing artifact hash differs')
  return target
 partial=folder/(name+'.partial');total=0
 request=urllib.request.Request(public_url(entry['url']),headers={'User-Agent':'pinned-public-cpu-wheel-build'})
 with urllib.request.urlopen(request,timeout=60)as source,partial.open('xb')as output:
  while True:
   if time.monotonic()>DEADLINE:raise BuildStopped('Deadline during public download')
   data=source.read(1024*1024)
   if not data:break
   total+=len(data)
   if total>limit:raise ValueError('Public download exceeds bound')
   output.write(data)
 if 'bytes'in entry and total!=entry['bytes']:raise ValueError('Public download size differs')
 if sha(partial)!=entry['sha256']:raise ValueError('Public download SHA256 differs')
 partial.rename(target);return target

def native_environment(work,python_root):
 # Do not forward repository tokens, runtime tokens, package index credentials,
 # or unrelated environment into third-party build/probe subprocesses.
 allow=('PATH','HOME','TMPDIR','LANG','LC_ALL','LC_CTYPE','SDKROOT','DEVELOPER_DIR')
 env={key:os.environ[key]for key in allow if key in os.environ}
 env.update(PATH=str(python_root/'bin')+os.pathsep+env.get('PATH',''),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_CACHE_DIR='1',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='false',XDG_CACHE_HOME=str(work/'cache'),TORCH_HOME=str(work/'cache/torch'),DYLD_FALLBACK_LIBRARY_PATH=str(python_root/'lib')+':/usr/lib')
 return env

def cpu_build_environment(env,version,architecture):
 # PyTorch intentionally builds a universal host protoc before its thin target.
 # Global Intel -march/-mtune flags would also reach that ARM64 bootstrap.
 result=dict(env,BUILD_VERSION=version,PYTORCH_BUILD_VERSION=version,PYTORCH_BUILD_NUMBER='1',USE_CUDA='0',USE_MPS='0',USE_NATIVE_ARCH='0',USE_OPENMP='0',USE_MKL='0',USE_DISTRIBUTED='0',USE_GLOO='0',USE_TENSORPIPE='0',USE_KINETO='0',BLAS='vecLib',BUILD_TEST='0',MAX_JOBS='2',CMAKE_BUILD_PARALLEL_LEVEL='2',MACOSX_DEPLOYMENT_TARGET='15.0',CMAKE_OSX_ARCHITECTURES=architecture,ARCHFLAGS='-arch '+architecture,TORCHVISION_USE_PNG='0',TORCHVISION_USE_JPEG='0',TORCHVISION_USE_FFMPEG='0',TORCHVISION_USE_VIDEO_CODEC='0')
 result.pop('CFLAGS',None);result.pop('CXXFLAGS',None)
 return result

def cache_values(path):
 if path.is_symlink() or not path.is_file() or path.stat().st_size>1024*1024:raise ValueError('Missing or unbounded main CMake cache')
 values={}
 for line in path.read_text(encoding='utf-8',errors='strict').splitlines():
  if not line or line.startswith(('#','//'))or'='not in line or':'not in line.split('=',1)[0]:continue
  key,value=line.split('=',1);values[key.split(':',1)[0]]=value
 return values

def verify_cpu_cache(source,architecture):
 path=source/'build/CMakeCache.txt';values=cache_values(path)
 required=('USE_CUDA','USE_MPS','USE_OPENMP','USE_NATIVE_ARCH','USE_MKL','USE_DISTRIBUTED','BUILD_TEST')
 optional=('USE_GLOO','USE_TENSORPIPE','USE_KINETO')
 false_values={'0','OFF','FALSE','NO','N','IGNORE','NOTFOUND',''}
 for key in required:
  if key not in values or values[key].upper()not in false_values:raise ValueError('Actual main cache must disable '+key)
 for key in optional:
  if key in values and values[key].upper()not in false_values:raise ValueError('Actual main cache unexpectedly enables '+key)
 if values.get('CMAKE_OSX_ARCHITECTURES')!=architecture or values.get('CMAKE_OSX_DEPLOYMENT_TARGET')!='15.0' or values.get('BLAS')!='vecLib':raise ValueError('Actual main architecture/deployment/BLAS cache differs')
 for key in ('CMAKE_C_FLAGS','CMAKE_CXX_FLAGS'):
  if key not in values or re.search(r'(?:^|\s)-(?:march|mtune|mcpu)(?:=|\s)',values[key]):raise ValueError('Global processor-specific CMake flags are not allowed')
 keys=(*required,*optional,'CMAKE_OSX_ARCHITECTURES','CMAKE_OSX_DEPLOYMENT_TARGET','BLAS','CMAKE_C_FLAGS','CMAKE_CXX_FLAGS')
 return {'passed':True,'source':'actual main build/CMakeCache.txt','cacheSHA256':sha(path),'values':{key:values.get(key)for key in keys},'scope':'Effective main configuration only; not a successful build or kernel test by itself'}

def collect_cmake_diagnostics(source,output):
 # Only generated CMake diagnostics under two known public build roots. Never
 # collect source trees, an environment dump, home, downloaded archives or keys.
 output.mkdir(parents=True,exist_ok=True)
 files=[];errors=[];total=visited=0;stop=min(DEADLINE,time.monotonic()+3);content_limit=6*1024*1024-65536
 names={'CMakeCache.txt','CMakeConfigureLog.yaml','CMakeError.log','CMakeOutput.log'}
 for area in (source/'build',source/'build_host_protoc'):
  if not area.is_dir()or area.is_symlink():continue
  for current,dirs,leaves in os.walk(area,followlinks=False):
   visited+=1;directory=Path(current)
   if time.monotonic()>=stop or visited>5000 or len(files)>=12 or total>=content_limit:break
   dirs[:]=[name for name in dirs if not(directory/name).is_symlink()and len((directory/name).relative_to(area).parts)<=5]
   for name in sorted(set(leaves)&names):
    path=directory/name
    if path.is_symlink()or not stat.S_ISREG(path.stat().st_mode):continue
    path.resolve(strict=True).relative_to(source.resolve(strict=True))
    if time.monotonic()>=stop or len(files)>=12:break
    limit=min(1024*1024,content_limit-total)
    if limit<=0:break
    size=path.stat().st_size
    with path.open('rb')as stream:
     if size>limit:
      first=limit//2;data=stream.read(first);stream.seek(max(first,size-(limit-first)));data+=stream.read(limit-first)
     else:data=stream.read(limit)
    target=output/(str(len(files)).zfill(2)+'-'+name+'.txt');target.write_bytes(data);total+=len(data)
    files.append({'source':path.relative_to(source).as_posix(),'file':target.name,'bytes':len(data),'sha256':sha(target),'truncated':size>len(data)})
  if time.monotonic()>=stop or visited>5000 or len(files)>=12 or total>=content_limit:break
 result={'files':files,'bytes':total,'fileLimit':12,'aggregateLimitBytes':6*1024*1024,'metadataReserveBytes':65536,'perFileLimitBytes':1024*1024,'directoryVisitLimit':5000,'timeLimitSeconds':3,'errors':errors}
 data=(json.dumps(result,indent=2)+'\n').encode()
 if len(data)>65536:raise ValueError('CMake diagnostic metadata bound exceeded')
 (output/'index.json').write_bytes(data);return result

def git_source(pin,target,env,out):
 # Exact commits pin the recursive gitlinks; never use --remote or floating tags.
 public_url(pin['repository']);target.mkdir()
 prefix=['git','-C',target,'-c','credential.helper=','-c','http.extraheader=','-c','protocol.file.allow=never','-c','protocol.ext.allow=never']
 run(prefix+['init'],out/'git-init.log',env,60)
 run(prefix+['remote','add','origin',pin['repository']],out/'git-remote.log',env,30)
 run(prefix+['fetch','--depth','1','origin',pin['commit']],out/'git-fetch.log',env,900)
 run(prefix+['-c','core.autocrlf=false','checkout','--detach','FETCH_HEAD'],out/'git-checkout.log',env,180)
 actual=run(prefix+['rev-parse','HEAD'],out/'git-head.log',env,30).strip()
 if actual!=pin['commit']:raise ValueError('Public source commit differs')
 if pin['pinnedRecursiveGitlinks']:
  run(prefix+['submodule','update','--init','--recursive','--jobs','2'],out/'submodules.log',env,1800)
  status=run(prefix+['submodule','status','--recursive'],out/'submodule-status.log',env,60)
  if any(line and not line.startswith(' ')for line in status.splitlines()):raise ValueError('Recursive public gitlink differs or is uninitialized')
 else:status=''
 dirty=run(prefix+['status','--porcelain','--untracked-files=no'],out/'git-status.log',env,30)
 if dirty.strip():raise ValueError('Public source is dirty before build')
 return {'repository':pin['repository'],'commit':actual,'recursiveGitlinksVerified':pin['pinnedRecursiveGitlinks'],'submoduleStatusSHA256':hashlib.sha256(status.encode()).hexdigest()}

def macho_arches(data):
 if len(data)<8:return []
 magic=struct.unpack('<I',data[:4])[0]
 if magic==0xfeedfacf:
  cpu=struct.unpack('<I',data[4:8])[0];return [{0x01000007:'x86_64',0x0100000c:'arm64'}.get(cpu,'unknown')]
 big=struct.unpack('>I',data[:4])[0]
 if big in (0xcafebabe,0xcafebabf):
  count=struct.unpack('>I',data[4:8])[0];width=32 if big==0xcafebabf else 20
  if count<1 or count>16 or len(data)<8+count*width:raise ValueError('Invalid universal Mach-O header')
  return [{0x01000007:'x86_64',0x0100000c:'arm64'}.get(struct.unpack('>I',data[8+i*width:12+i*width])[0],'unknown')for i in range(count)]
 return []

def inspect_wheel(path,architecture,notices):
 records=[];expanded=0;licenses=0;machos=0;checkpoints=0
 with zipfile.ZipFile(path)as wheel:
  names=set()
  for entry in wheel.infolist():
   name=PurePosixPath(entry.filename)
   if not entry.filename or str(name)=='.'or entry.filename in names or name.is_absolute()or'..'in name.parts or any(':'in part for part in name.parts)or'\\'in entry.filename or entry.flag_bits&1:raise ValueError('Unsafe/encrypted/duplicate wheel member')
   names.add(entry.filename)
   mode=entry.external_attr>>16
   if stat.S_ISLNK(mode):raise ValueError('Wheel symlink rejected')
   expanded+=entry.file_size
   if expanded>MAX_WHEEL_EXPANDED:raise ValueError('Expanded wheel bound exceeded')
   if entry.is_dir():continue
   if name.suffix.lower()in {'.pt','.pth','.ckpt','.onnx','.safetensors'}:checkpoints+=1
   with wheel.open(entry)as stream:prefix=stream.read(1024)
   arches=macho_arches(prefix)
   if arches:
    if architecture not in arches:raise ValueError('Wheel Mach-O architecture differs')
    if architecture=='x86_64'and arches!=['x86_64']:raise ValueError('Source-built Intel wheel is not thin x86_64')
    machos+=1;records.append({'path':entry.filename,'architectures':arches})
   leaf=name.name.upper()
   if '.dist-info/'in entry.filename and any(leaf.startswith(n)for n in ('LICENSE','COPYING','NOTICE')):
    data=wheel.read(entry)
    if len(data)>2*1024*1024:raise ValueError('Unbounded upstream license notice')
    target=notices/path.name/Path(*name.parts);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);licenses+=1
  if wheel.testzip()is not None:raise ValueError('Wheel CRC failed')
 if checkpoints:raise ValueError('Unexpected model/checkpoint file in library wheel')
 if not machos or not licenses:raise ValueError('Native extension or upstream license notices missing')
 return {'filename':path.name,'bytes':path.stat().st_size,'sha256':sha(path),'expandedBytes':expanded,'nativeMachOMembers':records,'licenseNoticeFiles':licenses,'checkpointFiles':checkpoints,'CRCVerified':True}

def main():
 global DEADLINE
 argparse.ArgumentParser().parse_args();started=time.monotonic();DEADLINE=started+MAX_SECONDS
 if os.environ.get('PUBLIC_REPOSITORY')!='true':raise SystemExit('Refuse private-repository or unconfirmed runner usage')
 arch=os.environ['EXPECTED_ARCH']
 if platform.system()!='Darwin'or arch not in ('x86_64','arm64')or platform.machine()!=arch:raise SystemExit('Matching native standard Mac runner required')
 temp=Path(os.environ['RUNNER_TEMP']).resolve(strict=True);work=temp/'public-cpu-work';out=temp/'public-cpu-evidence';share=temp/'public-cpu-wheels'
 for folder in (work,out,share):
  if folder.is_symlink():raise SystemExit('Unexpected task-directory symlink')
  folder.mkdir(exist_ok=True);folder.resolve().relative_to(temp)
 report={'schema':1,'passed':False,'architecture':arch,'publicSourcesOnly':True,'sourceBuild':arch=='x86_64','driverLimitSeconds':MAX_SECONDS,'phases':PHASES,'wheels':[],'applicationCompatibilityProven':False,'descendantCleanupGuaranteed':False,'checkpointsOrPrivateCodeIncluded':False}
 try:
  root=Path(__file__).resolve().parents[1];lock_file=root/'public-lock.json'
  if sha(lock_file)!=LOCK_SHA256:raise ValueError('Reviewed public lock changed')
  lock=json.loads(lock_file.read_text());selected=lock['architectures'][arch]
  free=shutil.disk_usage(work).free;report['initialFreeDiskBytes']=free
  if free<8*1024**3:raise RuntimeError('Less than8GiB free; source experiment not attempted')
  downloads=work/'downloads';downloads.mkdir();archive=download(selected['python'],downloads)
  python_area=work/'python';python_area.mkdir()
  if not hasattr(tarfile,'data_filter'):raise RuntimeError('Safe tar data filter unavailable')
  with tarfile.open(archive)as package:package.extractall(python_area,filter='data')
  children=list(python_area.iterdir())
  if len(children)!=1 or not children[0].is_dir():raise ValueError('Unexpected public Python archive layout')
  python_root=children[0];python=python_root/'bin/python3.11';env=native_environment(work,python_root)
  native=json.loads(run([python,'-I','-c','import json,platform;print(json.dumps({"python":platform.python_version(),"machine":platform.machine()}))'],out/'python.log',env,60))
  if native!={'python':'3.11.18','machine':arch}:raise ValueError('Public Python bootstrap version/architecture differs')
  installed=[]
  for entry in selected['bootstrapWheels']+lock['tools']:
   artifact=download(entry,downloads);installed.append(artifact)
  run([python,'-I','-m','pip','install','--no-index','--no-deps','--no-cache-dir',*installed],out/'bootstrap-install.log',env,300)
  toolchain={}
  for name,command in [('clang',['xcrun','clang','--version']),('sdk',['xcrun','--sdk','macosx','--show-sdk-version']),('cmake',[python_root/'bin/cmake','--version']),('ninja',[python_root/'bin/ninja','--version']),('pip',[python,'-I','-m','pip','--version'])]:toolchain[name]=run(command,out/(name+'-version.log'),env,60).strip()
  report['toolchain']=toolchain
  if 'cmake version 4.1.2'not in toolchain['cmake']or not toolchain['ninja'].startswith('1.13.0'):raise ValueError('Pinned build tools differ')
  wheels=[]
  if arch=='x86_64':
   for name in ('torch','torchvision'):
    pin=lock['sources'][name];source=work/(name+'-source');dest=out/name;dest.mkdir();record=git_source(pin,source,env,dest);report.setdefault('sources',{})[name]=record
    build_env=cpu_build_environment(env,pin['version'],arch)
    wheel_dir=work/(name+'-built');wheel_dir.mkdir()
    try:
     run([python,'-I','-m','pip','wheel','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',wheel_dir,source],dest/'build.log',build_env,150*60)
    except Exception:
     try:record['cmakeDiagnostics']=collect_cmake_diagnostics(source,dest/'cmake-diagnostics')
     except Exception as diagnostic_error:record['cmakeDiagnosticError']=str(diagnostic_error)
     if name=='torch':
      try:record['effectiveCPUConfiguration']=verify_cpu_cache(source,arch)
      except Exception as configuration_error:record['effectiveCPUConfiguration']={'passed':False,'error':str(configuration_error)}
     raise
    record['cmakeDiagnostics']=collect_cmake_diagnostics(source,dest/'cmake-diagnostics')
    if name=='torch':record['effectiveCPUConfiguration']=verify_cpu_cache(source,arch)
    built=list(wheel_dir.glob('*.whl'))
    if len(built)!=1:raise ValueError('Expected exactly one public source-built wheel')
    wheels.append(built[0]);run([python,'-I','-m','pip','install','--no-index','--no-deps',built[0]],dest/'install.log',env,180)
  else:
   for entry in selected['publicLibraryWheels']:
    artifact=download(entry,downloads);wheels.append(artifact);run([python,'-I','-m','pip','install','--no-index','--no-deps',artifact],out/(entry['name']+'-install.log'),env,180)
  for wheel in wheels:
   row=inspect_wheel(wheel,arch,out/'upstream-notices');report['wheels'].append(row);shutil.copyfile(wheel,share/wheel.name)
  (out/'SHA256SUMS.txt').write_text(''.join(row['sha256']+'  '+row['filename']+'\n'for row in report['wheels']),encoding='utf-8')
  run([python,'-I','-m','pip','check'],out/'pip-check.log',env,60)
  probe_env=dict(env,ATEN_CPU_CAPABILITY='default')
  run([python,'-I',root/'tools/probe_cpu.py','--architecture',arch,'--output',out/'cpu-probe.json'],out/'cpu-probe.log',probe_env,180)
  probe=json.loads((out/'cpu-probe.json').read_text())
  if probe.get('passed')is not True or probe.get('checkpointsDownloaded')is not False or probe.get('gpuOperationsPerformed')is not False:raise ValueError('Generic CPU ABI probe incomplete')
  report.update(passed=True,CPUABIProbePassed=True,finalFreeDiskBytes=shutil.disk_usage(work).free)
 except Exception as error:report['error']=str(error)
 finally:
  report['elapsedSeconds']=round(time.monotonic()-started,3);save(out/'build-report.json',report)
  if os.environ.get('GITHUB_STEP_SUMMARY'):
   with open(os.environ['GITHUB_STEP_SUMMARY'],'a')as stream:stream.write('* Public CPU dependency experiment: '+('passed'if report['passed']else'failed/incomplete')+'; no application compatibility or model inference claim.\n')
 return 0 if report['passed']else 1

if __name__=='__main__':raise SystemExit(main())
