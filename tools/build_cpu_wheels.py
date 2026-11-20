"""Generic pinned public CPU wheel experiment. Never imports application sources.

Only standard native public-repository runners are allowed. No source patches,
model/checkpoint downloads, private repositories or publishing. A CI read token
is used only by the isolated same-repository cache-admission mode; third-party
build and probe subprocesses never receive it.
A timeout stops the direct subprocess parent only; runner teardown is required
for descendants. No group signals or cleanup retries are used.
"""
import argparse,hashlib,json,os,platform,re,shutil,stat,struct,subprocess,tarfile,time,sys,urllib.parse,urllib.request,zipfile
from pathlib import Path,PurePosixPath

LOCK_SHA256="fe5b8aebd65943bbcca3b80cce97a7ab72218a493882efc2f7d68b69c977d932"
DEADLINE=None
PHASES=[]
MAX_SECONDS=280*60
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
 if remaining<=0:raise BuildStopped('280-minute driver deadline reached')
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
 allow=('PATH','HOME','TMPDIR','LANG','LC_ALL','LC_CTYPE','SDKROOT','DEVELOPER_DIR',
        'CCACHE_DIR','CCACHE_CONFIGPATH','CCACHE_TEMPDIR','CMAKE_C_COMPILER_LAUNCHER',
        'CMAKE_CXX_COMPILER_LAUNCHER','CMAKE_OBJC_COMPILER_LAUNCHER','CMAKE_OBJCXX_COMPILER_LAUNCHER')
 env={key:os.environ[key]for key in allow if key in os.environ}
 env.update(PATH=str(python_root/'bin')+os.pathsep+env.get('PATH',''),PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PIP_CONFIG_FILE=os.devnull,PIP_NO_INDEX='1',PIP_DISABLE_PIP_VERSION_CHECK='1',PIP_NO_CACHE_DIR='1',GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_TERMINAL_PROMPT='0',GIT_SSH_COMMAND='false',XDG_CACHE_HOME=str(work/'cache'),TORCH_HOME=str(work/'cache/torch'),DYLD_FALLBACK_LIBRARY_PATH=str(python_root/'lib')+':/usr/lib')
 # Pinned Torch USE_CCACHE auto-find must resolve this verified tool, not a system wrapper.
 if os.environ.get('CCACHE_EXE'):env['PATH']=str(Path(os.environ['CCACHE_EXE']).parent)+os.pathsep+env['PATH']
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
 launcher_keys=tuple('CMAKE_'+language+'_COMPILER_LAUNCHER'for language in ('C','CXX','OBJC','OBJCXX'))
 if os.environ.get('CCACHE_EXE'):
  for language,key in zip(('C','CXX','OBJC','OBJCXX'),launcher_keys):
   enabled=language in ('C','CXX')or bool(values.get('CMAKE_'+language+'_COMPILER'))or key in values
   if enabled and values.get(key)!=os.environ['CCACHE_EXE']:raise ValueError('Actual main CMake cache does not use the pinned compiler launcher: '+key)
 keys=(*required,*optional,'CMAKE_OSX_ARCHITECTURES','CMAKE_OSX_DEPLOYMENT_TARGET','BLAS','CMAKE_C_FLAGS','CMAKE_CXX_FLAGS')+launcher_keys
 return {'passed':True,'source':'actual main build/CMakeCache.txt','cacheSHA256':sha(path),'values':{key:values.get(key)for key in keys},'scope':'Effective main configuration only; not a successful build or kernel test by itself'}

def collect_cmake_diagnostics(source,output):
 # Only generated CMake diagnostics under two known public build roots. Never
 # collect source trees, an environment dump, home, downloaded archives or keys.
 output.mkdir(parents=True,exist_ok=True)
 files=[];errors=[];total=visited=0;stop=time.monotonic()+3;content_limit=6*1024*1024-65536
 names={'CMakeCache.txt','CMakeConfigureLog.yaml','CMakeError.log','CMakeOutput.log','.ninja_log'}
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
    ninja_tail={}
    with path.open('rb')as stream:
     if name=='.ninja_log':
      offset=max(0,size-limit);stream.seek(offset);data=stream.read(limit)
      if offset:data=data.partition(b'\n')[2]
      ninja_tail={'tailOnly':True,'observedSourceBytes':size,'completeLinesInBoundedTail':data.count(b'\n'),'totalLineCountKnown':False}
     elif size>limit:
      first=limit//2;data=stream.read(first);stream.seek(max(first,size-(limit-first)));data+=stream.read(limit-first)
     else:data=stream.read(limit)
    target=output/(str(len(files)).zfill(2)+'-'+name+'.txt');target.write_bytes(data);total+=len(data)
    files.append({'source':path.relative_to(source).as_posix(),'file':target.name,'bytes':len(data),'sha256':sha(target),'truncated':size>len(data),**ninja_tail})
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

# Generic build-tool cache only; no application inputs or credentials are cached.
COMPILER_CACHE_LIMIT=2*1024*1024*1024
COMPILER_CACHE_MAX_FILES=30000
COMPILER_CACHE_MANIFEST='cache-snapshot.json'
COMPILER_CACHE_POLICY='public-cpu-ccache-v1'
COMPILER_CACHE_FREE_BYTES=10_000_000_000
COMPILER_CACHE_MARGIN=128*1024*1024

def compiler_cache_digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()

def compiler_cache_task_path(name, leaf):
    temp = Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
    path = Path(os.environ[name])
    if path.absolute() != temp / leaf or path.is_symlink():
        raise ValueError('Unexpected dedicated path: ' + name)
    path.resolve().relative_to(temp)
    return path

def compiler_cache_paths():
    return (compiler_cache_task_path('CCACHE_DIR', 'public-cpu-ccache-live'),
            compiler_cache_task_path('CCACHE_SNAPSHOT_DIR', 'public-cpu-ccache-snapshot'),
            compiler_cache_task_path('CCACHE_CONSUMED_DIR', 'public-cpu-ccache-restored-consumed'))

def compiler_cache_cache_name(name):
    """Pinned ccache4.14.3: 40hex key distributed over2..4 hex directories."""
    parts = PurePosixPath(name).parts
    return (3 <= len(parts) <= 5 and all(re.fullmatch('[0-9a-f]', x) for x in parts[:-1])
            and re.fullmatch('[0-9a-f]{40}', ''.join(parts)) is not None)

def compiler_cache_selected_files(root, deadline):
    count = visited = 0
    for current, directories, files in os.walk(root, followlinks=False):
        visited += len(directories) + len(files) + 1
        if visited > 100000 or time.monotonic() >= deadline:
            raise TimeoutError("Compiler-cache scan bound reached")
        current = Path(current)
        for name in list(directories):
            path = current / name
            if path.is_symlink():
                raise ValueError('Cache directory symlink')
            # Never enter lock/tmp/config/profile directories. No raw object sidecars:
            # file_clone/hard_link=false makes result entries self-contained.
            if not re.fullmatch('[0-9a-f]', name) or len(path.relative_to(root).parts) > 4:
                directories.remove(name)
        for name in files:
            path = current / name
            if path.is_symlink():
                raise ValueError('Cache file symlink')
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                continue  # Concurrent ccache cleanup may remove an enumerated file.
            if not stat.S_ISREG(mode):
                raise ValueError('Special compiler-cache entry')
            relative = path.relative_to(root).as_posix()
            if compiler_cache_cache_name(relative):
                count += 1
                if count > COMPILER_CACHE_MAX_FILES:
                    raise ValueError('Compiler-cache file count bound exceeded')
                yield path, relative

def compiler_cache_signature(value):
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_mode, value.st_nlink)

def compiler_cache_stable_copy(source, destination, allowance, deadline, expected=None):
    """Copy a regular open inode once; discard a changed/replaced input."""
    if time.monotonic() >= deadline:
        raise TimeoutError('Cache snapshot/restore time bound reached')
    before = source.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or source.is_symlink():
        raise ValueError('Cache source is not a regular file')
    if before.st_size > allowance:
        raise ValueError('Compiler cache exceeds2GiB bound')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_BINARY', 0)
    descriptor = os.open(source, flags)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    value = hashlib.sha256()
    try:
        with os.fdopen(descriptor, 'rb') as original, destination.open('xb') as copied:
            if compiler_cache_signature(os.fstat(original.fileno())) != compiler_cache_signature(before):
                raise ValueError('Cache inode changed before opening')
            while True:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Cache snapshot/restore time bound reached')
                chunk = original.read(min(1024 * 1024, allowance - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > allowance:
                    raise ValueError('Compiler cache exceeds2GiB bound')
                value.update(chunk)
                copied.write(chunk)
            after = os.fstat(original.fileno())
        stable = (compiler_cache_signature(before) == compiler_cache_signature(after) == compiler_cache_signature(source.lstat())
                  and total == before.st_size)
        if not stable:
            destination.unlink()
            return None
        record = {'bytes': total, 'sha256': value.hexdigest()}
        if expected is not None and record != {'bytes': expected['bytes'], 'sha256': expected['sha256']}:
            raise ValueError('Restored cache content differs from closed manifest')
        os.chmod(destination, 0o600)
        return record
    except BaseException:
        if destination.exists():
            destination.unlink()
        raise

def compiler_cache_snapshot(live, target, key, limit=COMPILER_CACHE_LIMIT, seconds=180):
    if target.exists() or target.is_symlink() or live.is_symlink():
        raise ValueError('Snapshot destination already exists or cache is a symlink')
    deadline = time.monotonic() + seconds
    target.mkdir()
    records = []
    skipped = []
    total = 0
    for path, relative in compiler_cache_selected_files(live, deadline):
        try:
            row = compiler_cache_stable_copy(path, target / relative, limit - total, deadline)
        except FileNotFoundError:
            row = None  # ccache atomic rename/cleanup raced; no partial object saved.
        if row is None:
            skipped.append(relative)
            continue
        total += row['bytes']
        records.append({'path': relative, **row})
    document = {'schema': 1, 'policy': COMPILER_CACHE_POLICY, 'compilerKey': key,
                'closedSnapshot': True, 'cacheOnly': True, 'fileBytes': total,
                'files': records, 'skippedUnstableFiles': skipped}
    encoded = (json.dumps(document, indent=2) + '\n').encode()
    if total + len(encoded) > limit:
        raise ValueError('Cache entries plus manifest exceed2GiB')
    (target / COMPILER_CACHE_MANIFEST).write_bytes(encoded)
    compiler_cache_validate(target, key, limit, deadline)
    return {'saveReady': bool(records), 'files': len(records), 'bytesIncludingManifest': total + len(encoded),
            'skippedUnstableFiles': skipped, 'manifestSHA256': hashlib.sha256(encoded).hexdigest()}

def compiler_cache_validate(root, key, limit=COMPILER_CACHE_LIMIT, deadline=None):
    deadline = time.monotonic() + 180 if deadline is None else deadline
    if root.is_symlink():
        raise ValueError('Snapshot root symlink')
    metadata = root / COMPILER_CACHE_MANIFEST
    if metadata.is_symlink() or not metadata.is_file() or metadata.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Missing/unbounded cache manifest')
    document = json.loads(metadata.read_text())
    if (document.get('schema'), document.get('policy'), document.get('compilerKey'), document.get('closedSnapshot')) != (1, COMPILER_CACHE_POLICY, key, True):
        raise ValueError('Compiler cache policy/key differs')
    records = document['files']
    expected = {}
    total = metadata.stat().st_size
    for row in records:
        if not compiler_cache_cache_name(row['path']) or row['path'] in expected or not isinstance(row['bytes'], int) or row['bytes'] < 0:
            raise ValueError('Unsafe/duplicate cache manifest entry')
        expected[row['path']] = row
        total += row['bytes']
    if len(expected) > COMPILER_CACHE_MAX_FILES or total > limit:
        raise ValueError('Restored compiler cache exceeds bound')
    seen = set()
    visited = 0
    for path in root.rglob('*'):
        visited += 1
        if visited > 100000 or time.monotonic() >= deadline:
            raise TimeoutError('Closed compiler-cache verification bound reached')
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Snapshot symlink/special entry')
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == COMPILER_CACHE_MANIFEST:
            continue
        if relative not in expected:
            raise ValueError('Unexpected cache snapshot file')
        row = expected[relative]
        if path.stat().st_size != row['bytes'] or compiler_cache_digest(path) != row['sha256']:
            raise ValueError('Cache compiler_cache_snapshot file differs')
        seen.add(relative)
    if seen != set(expected) or document.get('fileBytes') != total - metadata.stat().st_size:
        raise ValueError('Cache compiler_cache_snapshot is incomplete')
    return document

def compiler_cache_restore(live, source, consumed, key):
    if live.exists() or live.is_symlink() or consumed.exists() or consumed.is_symlink():
        raise ValueError('Fresh dedicated live/consumed paths required')
    live.mkdir()
    if not source.exists():
        return {'cacheRestored': False, 'files': 0}
    document = compiler_cache_validate(source, key)
    deadline = time.monotonic() + 180
    for row in document['files']:
        copied = compiler_cache_stable_copy(source / row['path'], live / row['path'], COMPILER_CACHE_LIMIT, deadline, expected=row)
        if copied is None:
            raise ValueError('Closed restored cache changed while copying')
    # Preserve the received snapshot without deleting anything, then free this exact
    # cache-action path for a new closed snapshot on both build success and failure.
    source.rename(consumed)
    return {'cacheRestored': True, 'files': len(document['files'])}

def compiler_cache_public_intel():
 if os.environ.get('PUBLIC_REPOSITORY')!='true' or os.environ.get('EXPECTED_ARCH')!='x86_64' or platform.system()!='Darwin' or platform.machine()!='x86_64':
  raise ValueError('Only a confirmed standard public native Intel task may use this compiler cache')


def compiler_cache_prepare():
 global DEADLINE
 compiler_cache_public_intel();DEADLINE=time.monotonic()+180
 root=Path(__file__).resolve().parents[1];lock_path=root/'public-lock.json'
 if sha(lock_path)!=LOCK_SHA256:raise ValueError('Reviewed public lock changed')
 lock=json.loads(lock_path.read_text());pin=lock['compilerCache']
 if (pin['name'],pin['version'],pin['sha256'])!=('ccache','4.14.3','764dd2a8100ff83e6bbd7b9a1e2016cc401be34bee8a70ea99698186622c34d5'):
  raise ValueError('Pinned public compiler-cache tool differs')
 temp=Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
 tools=temp/'public-cpu-cache-tools';tools.mkdir(exist_ok=False)
 downloads=tools/'downloads';downloads.mkdir();archive=download(pin,downloads)
 executable=tools/'ccache'
 with tarfile.open(archive)as package:
  expected='ccache-4.14.3-darwin/ccache';member=package.getmember(expected)
  if not member.isfile()or member.size<1 or member.size>32*1024*1024:raise ValueError('Unexpected public ccache archive member')
  with package.extractfile(member)as source,executable.open('xb')as output:
   copied=0
   while True:
    data=source.read(1024*1024)
    if not data:break
    copied+=len(data)
    if copied>member.size or time.monotonic()>=DEADLINE:raise ValueError('Public compiler tool extraction bound exceeded')
    output.write(data)
  if copied!=member.size:raise ValueError('Public compiler tool is truncated')
 executable.chmod(0o700)
 # No CI tokens/runtime credentials are forwarded to public build tools.
 safe_env={key:os.environ[key]for key in ('PATH','HOME','TMPDIR','LANG','LC_ALL','SDKROOT','DEVELOPER_DIR')if key in os.environ}
 def output(args):return subprocess.check_output(list(map(str,args)),env=safe_env,text=True,timeout=20).strip()
 compiler=Path(output(['xcrun','--find','clang'])).resolve(strict=True)
 values={'policy':COMPILER_CACHE_POLICY,'compilerSHA256':sha(compiler),'compilerVersion':output([compiler,'--version']),
         'sdk':output(['xcrun','--sdk','macosx','--show-sdk-version']),'xcode':output(['xcodebuild','-version']),
         'ccacheVersion':output([executable,'--version']).splitlines()[0],'publicLockSHA256':LOCK_SHA256,'architecture':'x86_64'}
 if values['ccacheVersion']!='ccache version 4.14.3':raise ValueError('Actual compiler-cache tool version differs')
 key=hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest()
 live=temp/'public-cpu-ccache-live';snapshot=temp/'public-cpu-ccache-snapshot';consumed=temp/'public-cpu-ccache-restored-consumed'
 config=temp/'public-cpu-ccache.conf';temporary=temp/'public-cpu-ccache-tmp'
 for path in (live,snapshot,consumed,config,temporary):
  if path.exists()or path.is_symlink():raise ValueError('Fresh compiler-cache task paths required')
 temporary.mkdir()
 config.write_text('cache_dir = '+str(live)+'\ntemporary_dir = '+str(temporary)+
                   '\nmax_size = 1800Mi\nmax_files = 25000\ncompression = true\ndirect_mode = false\ncompiler_check = content\n'
                   'hash_dir = true\nfile_clone = false\nhard_link = false\nnamespace = public-pinned-cpu\n')
 env={'CCACHE_DIR':str(live),'CCACHE_SNAPSHOT_DIR':str(snapshot),'CCACHE_CONSUMED_DIR':str(consumed),
      'CCACHE_CONFIGPATH':str(config),'CCACHE_TEMPDIR':str(temporary),'CCACHE_EXE':str(executable),'CCACHE_COMPILER_KEY':key}
 for language in ('C','CXX','OBJC','OBJCXX'):env['CMAKE_'+language+'_COMPILER_LAUNCHER']=str(executable)
 with open(os.environ['GITHUB_ENV'],'a')as stream:
  for name,value in env.items():stream.write(name+'='+value+'\n')
 with open(os.environ['GITHUB_OUTPUT'],'a')as stream:stream.write('compiler-key='+key+'\n')
 return {'compilerKey':key,'inputs':values,'maxSnapshotBytes':COMPILER_CACHE_LIMIT,'liveCleanupTargetMiB':1800,'toolSHA256':sha(executable)}


def compiler_cache_admission(usage,repository):
 if not isinstance(usage,dict)or usage.get('full_name')!=repository:raise ValueError('Cache-usage repository differs')
 used=usage.get('active_caches_size_in_bytes');count=usage.get('active_caches_count')
 if type(used)is not int or type(count)is not int or used<0 or count<0:raise ValueError('Actual repository cache usage is missing/invalid')
 return {'admitted':used+COMPILER_CACHE_LIMIT+COMPILER_CACHE_MARGIN<=COMPILER_CACHE_FREE_BYTES,
         'repository':repository,'activeCacheBytes':used,'activeCacheCount':count,'newSnapshotReserveBytes':COMPILER_CACHE_LIMIT,
         'archiveAndConcurrencyMarginBytes':COMPILER_CACHE_MARGIN,'conservativeFreeCeilingBytes':COMPILER_CACHE_FREE_BYTES,
         'projectedReservedBytes':used+COMPILER_CACHE_LIMIT+COMPILER_CACHE_MARGIN,'cacheLimitChanged':False,'paidUseAuthorized':False}


class CompilerCacheNoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,request,fp,code,msg,headers,newurl):
  raise ValueError('Cache-usage redirects are refused; token is never forwarded')


def compiler_cache_admit():
 # This is the only mode that reads the ephemeral same-repository CI read token.
 # It never runs a subprocess or sends the token to public source/build services.
 result={'admitted':False,'actualUsageRead':False,'cacheLimitChanged':False,'paidUseAuthorized':False}
 try:
  compiler_cache_public_intel()
  repository=os.environ['GITHUB_REPOSITORY']
  if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository):raise ValueError('Invalid same-repository identity')
  event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
  info=event.get('repository',{})
  if info.get('full_name')!=repository or info.get('visibility')!='public' or info.get('private')is not False:
   raise ValueError('Confirmed same public repository is required')
  token=os.environ.get('GH_CACHE_READ_TOKEN')
  if not token:raise ValueError('Same-repository CI read token unavailable')
  _,snapshot,_=compiler_cache_paths();compiler_cache_validate(snapshot,os.environ['CCACHE_COMPILER_KEY'])
  url='https://api.github.com/repos/'+repository+'/actions/cache/usage'
  request=urllib.request.Request(url,headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json',
                                            'X-GitHub-Api-Version':'2026-03-10','User-Agent':'public-compiler-cache-admission'})
  opener=urllib.request.build_opener(CompilerCacheNoRedirect())
  with opener.open(request,timeout=20)as response:
   if response.status!=200 or response.geturl()!=url:raise ValueError('Unexpected cache-usage response')
   data=response.read(65537)
   if len(data)>65536:raise ValueError('Unbounded cache-usage response')
  result=compiler_cache_admission(json.loads(data),repository);result['actualUsageRead']=True
 except Exception as error:
  # Deliberately record only the type: never request/header/token/error-body data.
  result['failureType']=type(error).__name__
 finally:
  with open(os.environ['GITHUB_OUTPUT'],'a')as stream:stream.write('save-admitted='+str(result['admitted']).lower()+'\n')
 return result


def compiler_cache_smoke():
 # Keep every inherited CI token and unreviewed CCACHE_* override out of tools.
 compiler_cache_public_intel()
 temp=Path(os.environ['RUNNER_TEMP']).resolve(strict=True)
 executable=Path(os.environ['CCACHE_EXE'])
 if executable.is_symlink()or executable.absolute()!=temp/'public-cpu-cache-tools/ccache':raise ValueError('Unexpected compiler-cache executable')
 compiler_cache_task_path('CCACHE_CONFIGPATH','public-cpu-ccache.conf')
 work=temp/'public-cpu-cache-smoke'
 if work.exists()or work.is_symlink():raise ValueError('Fresh compiler-cache smoke directory required')
 work.mkdir()
 env=native_environment(work,temp/'public-cpu-work/python/python')
 defaults={}
 for key in('remote_storage','sloppiness'):
  value=subprocess.check_output([str(executable),'--get-config',key],env=env,stderr=subprocess.PIPE,text=True,timeout=20)
  if value.strip():raise ValueError('Effective compiler-cache '+key+' must be empty')
  defaults[key]=True
 compiler=Path(subprocess.check_output(['xcrun','--find','clang++'],env=env,stderr=subprocess.PIPE,text=True,timeout=20).strip()).resolve(strict=True)
 source=work/'compiler-cache-probe.cpp';target=work/'compiler-cache-probe.o'
 source.write_text('int compiler_cache_probe(){return 42;}\n',encoding='utf-8')
 result=subprocess.run([str(executable),str(compiler),'-std=c++17','-c',str(source),'-o',str(target)],env=env,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60,check=False)
 output=result.stdout[:65536]
 evidence=temp/'public-cpu-evidence';evidence.mkdir(exist_ok=True)
 (evidence/'compiler-cache-smoke-compile.log').write_bytes(output)
 if result.returncode:raise ValueError('Compiler-cache C++ object preflight failed; see bounded compile log')
 if target.is_symlink()or not target.is_file()or target.stat().st_size<1:raise ValueError('Compiler-cache preflight did not produce an object')
 with target.open('rb')as stream:architectures=macho_arches(stream.read(1024))
 if architectures!=['x86_64']:raise ValueError('Compiler-cache preflight object architecture differs')
 return {'effectiveRemoteStorageEmpty':defaults['remote_storage'],'effectiveSloppinessEmpty':defaults['sloppiness'],
         'filteredEnvironmentUsed':True,'tinyCXXObjectCompiled':True,'objectBytes':target.stat().st_size,
         'objectSHA256':sha(target),'objectArchitectures':architectures,'compileOutputTruncated':len(result.stdout)>65536,
         'applicationCompatibilityProven':False}


def compiler_cache_cli(mode):
 if mode=='prepare':result=compiler_cache_prepare()
 elif mode=='admit':result=compiler_cache_admit()
 elif mode=='smoke':result=compiler_cache_smoke()
 else:
  compiler_cache_public_intel();live,target,consumed=compiler_cache_paths();key=os.environ['CCACHE_COMPILER_KEY']
  if mode=='restore':result=compiler_cache_restore(live,target,consumed,key)
  else:
   result=compiler_cache_snapshot(live,target,key)
   with open(os.environ['GITHUB_OUTPUT'],'a')as stream:stream.write('save-ready='+str(result['saveReady']).lower()+'\n')
 print(json.dumps(result,indent=2));return 0

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
     run([python,'-I','-m','pip','wheel','--no-index','--no-deps','--no-build-isolation','--no-cache-dir','--wheel-dir',wheel_dir,source],dest/'build.log',build_env,(270 if name=='torch' else 30)*60)
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

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='--compiler-cache':
  parser=argparse.ArgumentParser();parser.add_argument('--compiler-cache',choices=('prepare','restore','snapshot','smoke','admit'),required=True)
  raise SystemExit(compiler_cache_cli(parser.parse_args().compiler_cache))
 raise SystemExit(main())
