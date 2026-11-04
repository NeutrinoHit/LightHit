"""Prepared runtime tables: file check and compact transport (``lighthit-tables``).

A runtime bundle (``bundle.json`` with the SHA256 of every file) can be
distributed in three forms, all read by the same loaders once installed:

* full: the bundle itself;
* exact (``pack``): primary arrays only, the derived arrays are rebuilt at
  installation by LightHit's own builders; event answers change only by
  floating-point round-off;
* reduced (``pack --tolerance 1e-7 --drop-rte-order1``): primary arrays also
  in reduced form, with a recorded maximum row error.

``verify`` checks the files of a runtime against ``bundle.json``; after
``unpack`` it checks the exact files and lists the rebuilt and reduced ones.

``pack`` keeps the primary arrays and drops every array that LightHit
rebuilds from them with its own builders: the Q1 time integrals, the coarse
Q1 time atlas, the spectral Q1 bands, the order >=2 spline pack and its band
fold. This step is exact up to the platform's floating-point libraries.

With ``--tolerance`` the large primary arrays are also stored in reduced
form. Smooth CDF tables (the geometry hit branches and the Q1 reference) are
POD factors along their time axis; the remaining CDFs and the complex RTE
coefficients are row-scaled single precision. Each row keeps its float64
scale and, for CDFs, its exact first and last values; rows that were
monotone or nonnegative are projected back onto that set, which cannot
increase the maximum error. The muon atlases pin each row file by SHA256
and are always kept exact. The recorded error is the maximum over rows of
the absolute error divided by the row's largest magnitude.

``--drop-rte-order1`` omits the order-1 moments of the RTE tables. The
glass route takes Q1 from the first-entry tables and only reads the RTE
metadata at event time; the order >=2 moments are kept to rebuild the
spline pack. The unpacked tables then hold NaN in place of order 1, so any
other use fails visibly instead of silently.

``--deflate`` stores the remaining .npy arrays and all factors byte-shuffled
and deflated with the standard library; decoding is bit-exact.

``unpack`` writes a complete runtime directory; loaders and event code are
unchanged. It refuses to finish if an exactly transported file differs from
its recorded SHA256. ``compare`` reports array differences between two
runtimes. Pack and unpack record time, peak memory and file SHA256 against
the source.
"""
import argparse,hashlib,io,json,os,resource,shutil,sys,time,zipfile,zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

SCHEMA='lighthit/compact-bundle/1'
FIRST='tables/first-entry/'
DERIVED={
    'q1-integral':[FIRST+'q1-reference/'+b+'_integral.npy' for b in ('hit','miss')],
    'q1-adaptive':[FIRST+'q1-adaptive/'+b+'_'+n+'.npy' for b in ('hit','miss')
                   for n in ('cdf','integral','charge','time_error','quad_coarse','quad_fine')],
    'q1-bands':[FIRST+'q1-bands/'+b+'_'+n+'.npy' for b in ('hit','miss')
                for n in ('cdf','charge','error','quad_coarse','quad_fine')],
    'ge2':['tables/ge2-pack/array-000.npy','tables/ge2-pack/array-001.npy','tables/ge2-band-fold/folded.npy']}
ORDER=('q1-integral','q1-adaptive','q1-bands','ge2')
INPUTS={'q1-integral':FIRST+'q1-reference/','q1-adaptive':FIRST+'q1-reference/',
        'q1-bands':FIRST+'geometry/','ge2':'tables/rte-cache/'}
# (group, glob, npz members or None, row axis, codec, kind); kind 'cdf' keeps endpoints.
GROUPS=[
    ('geometry-hit',FIRST+'geometry/hit_cdf.npy',None,2,'pod','cdf'),
    ('geometry-primary-hit',FIRST+'geometry-primary/hit_cdf.npy',None,2,'pod','cdf'),
    ('q1-reference-hit',FIRST+'q1-reference/hit_cdf.npy',None,2,'pod','cdf'),
    ('geometry-miss',FIRST+'geometry/miss_cdf.npy',None,2,'scaled32','cdf'),
    ('geometry-primary-miss',FIRST+'geometry-primary/miss_cdf.npy',None,2,'scaled32','cdf'),
    ('q1-reference-miss',FIRST+'q1-reference/miss_cdf.npy',None,2,'pod','cdf'),
    # The muon atlases are absent on purpose: their manifests pin every row
    # file by SHA256 and the loaders verify it, so they travel exactly.
    ('rte-cache','tables/rte-cache/directional-*.npz',('coefficients',),0,'scaled32','field')]
SIDE=('scale','first','last','flags')


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1<<22),b''):h.update(block)
    return h.hexdigest()


def peak_bytes():
    scale=1 if sys.platform=='darwin' else 1024
    own=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*scale
    children=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss*scale
    return dict(self=int(own),largest_child=int(children))


# ---------------------------------------------------------------- shuffled deflate store

MAGIC=b'LHSHZ1\n'
CHUNK=1<<26


def shz_write(path,array,prefix=b'',level=6):
    """Byte-shuffled deflate of one C-contiguous array; ``prefix`` keeps an .npy header."""
    a=np.asarray(array);flat=a.reshape(-1);item=a.dtype.itemsize;step=max(1,CHUNK//item);sizes=[]
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(path)+'.tmp')
    with open(tmp,'wb') as f:
        f.write(MAGIC)
        for b in range(0,len(flat),step):
            part=np.ascontiguousarray(flat[b:b+step]).view(np.uint8).reshape(-1,item).T.tobytes()
            blob=zlib.compress(part,level);f.write(blob);sizes.append(len(blob))
        meta=json.dumps(dict(dtype=a.dtype.str,shape=list(a.shape),step=step,sizes=sizes,prefix=prefix.hex())).encode()
        f.write(meta);f.write(len(meta).to_bytes(8,'little'))
    tmp.replace(path)


def shz_meta(path):
    with open(path,'rb') as f:
        f.seek(-8,2);n=int.from_bytes(f.read(8),'little');f.seek(-8-n,2);return json.loads(f.read(n))


def shz_read(path,out=None):
    m=shz_meta(path);dtype=np.dtype(m['dtype']);item=dtype.itemsize
    a=np.empty(m['shape'],dtype) if out is None else out;flat=a.reshape(-1)
    with open(path,'rb') as f:
        if f.read(len(MAGIC))!=MAGIC:raise ValueError('not a shuffled array file: '+str(path))
        for i,size in enumerate(m['sizes']):
            raw=np.frombuffer(zlib.decompress(f.read(size)),np.uint8).reshape(item,-1)
            b=i*m['step'];flat[b:b+raw.shape[1]]=np.ascontiguousarray(raw.T).view(dtype).reshape(-1)
    return a


def npy_to_shz(source,target):
    with open(source,'rb') as f:
        version=np.lib.format.read_magic(f)
        (np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0)(f)
        offset=f.tell();f.seek(0);prefix=f.read(offset)
    shz_write(target,np.load(source,mmap_mode='r'),prefix=prefix);return str(target)


def shz_to_npy(source,target):
    m=shz_meta(source);prefix=bytes.fromhex(m['prefix']);dtype=np.dtype(m['dtype'])
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True);tmp=Path(str(target)+'.tmp')
    with open(tmp,'wb') as f:f.write(prefix);f.truncate(len(prefix)+int(np.prod(m['shape']))*dtype.itemsize)
    out=np.memmap(tmp,dtype=dtype,mode='r+',offset=len(prefix),shape=tuple(m['shape']))
    shz_read(source,out);out.flush();del out;tmp.replace(target);return str(target)


def save_factors(directory,payload,deflate):
    directory.mkdir(parents=True)
    for key,value in payload.items():
        name=key.replace('/','-')
        if deflate:shz_write(directory/(name+'.shz'),np.ascontiguousarray(value))
        else:np.save(directory/(name+'.npy'),value,allow_pickle=False)


def load_factors(directory):
    out={}
    for p in directory.iterdir():
        key=p.stem.replace('-','/',1)
        out[key]=shz_read(p) if p.suffix=='.shz' else np.load(p,allow_pickle=False)
    return out


def _parallel(function,pairs,jobs):
    if not pairs:return
    with ProcessPoolExecutor(max_workers=max(1,min(jobs,len(pairs)))) as pool:
        for result in pool.map(function,*zip(*pairs)):pass


# ---------------------------------------------------------------- row codecs

def _rows(a,axis):
    return np.moveaxis(np.asarray(a),axis,-1).reshape(-1,a.shape[axis])


def _unrows(rows,shape,axis):
    moved=list(shape);n=moved.pop(axis);moved.append(n)
    return np.ascontiguousarray(np.moveaxis(rows.reshape(moved),-1,axis))


def _side(rows,kind):
    scale=np.abs(rows).max(axis=1);scale=np.where(scale>0,scale,1.)
    if kind!='cdf':return dict(scale=scale)
    flags=(np.all(np.diff(rows,axis=1)>=0,axis=1).astype(np.uint8)|(2*np.all(rows>=0,axis=1)).astype(np.uint8))
    return dict(scale=scale,first=rows[:,0].copy(),last=rows[:,-1].copy(),flags=flags)


def _finish(normal,side):
    """Scaled rows with exact CDF endpoints and monotone/nonnegative projection."""
    rows=normal*side['scale'][:,None]
    if 'flags' not in side:return rows
    first,last,flags=side['first'],side['last'],side['flags']
    rows[:,0]=first;rows[:,-1]=last
    mono=(flags&1).astype(bool)
    if mono.any():
        m=np.maximum.accumulate(rows[mono],axis=1)
        rows[mono]=np.minimum(np.maximum(m,first[mono,None]),last[mono,None])
    nonneg=(flags&2).astype(bool)
    if nonneg.any():rows[nonneg]=np.maximum(rows[nonneg],0.)
    return rows


def _decode(entry,basis):
    coef=entry['coef']
    normal=coef.astype(np.complex128 if coef.dtype.kind=='c' else float)
    if basis is not None:normal=normal@basis.T
    return _finish(normal,{k:entry[k] for k in SIDE if k in entry})


def _slice(side,sl):return {k:v[sl] for k,v in side.items()}


def _single(rows):return np.complex64 if np.iscomplexobj(rows) else np.float32


def _error(parts,basis,k,dtype):
    worst=0.;v=None if basis is None else basis[:,:k]
    for side,rows in parts:
        dt=dtype or _single(rows)
        for b in range(0,len(rows),1<<15):
            sl=slice(b,b+(1<<15));s=_slice(side,sl);normal=rows[sl]/s['scale'][:,None]
            coef=(normal if v is None else normal@v).astype(dt)
            got=_decode(dict(s,coef=coef),v)
            worst=max(worst,float(np.max(np.abs(got-rows[sl])/s['scale'][:,None])))
    return worst


def encode(arrays,axis,codec,kind,tolerance):
    """Factors of a group of arrays sharing the row axis; returns metadata and per-array parts."""
    parts=[]
    for a in arrays:
        rows=_rows(a,axis);parts.append((_side(rows,kind),rows))
    single=_single(parts[0][1])
    if codec=='pod':
        from numpy.linalg import qr,svd
        rs=[]
        for side,rows in parts:
            for b in range(0,len(rows),1<<16):
                chunk=rows[b:b+(1<<16)]/side['scale'][b:b+(1<<16),None]
                rs.append(qr(chunk,mode='r') if len(chunk)>chunk.shape[1] else chunk)
        stacked=np.vstack(rs);r=qr(stacked,mode='r') if len(stacked)>stacked.shape[1] else stacked
        s,basis=svd(r,full_matrices=False)[1:];basis=np.ascontiguousarray(basis.T)
        n=basis.shape[0];chosen=None
        for dtype in (single,np.float64):
            if _error(parts,basis,n,dtype)>tolerance:continue
            lo,hi=1,n
            while lo<hi:
                mid=(lo+hi)//2
                if _error(parts,basis,mid,dtype)<=tolerance:hi=mid
                else:lo=mid+1
            chosen=(lo,dtype);break
        if chosen is None:raise ValueError('tolerance is below the representable row error')
        k,dtype=chosen;basis=np.ascontiguousarray(basis[:,:k])
        if k*np.dtype(dtype).itemsize>=n*np.dtype(single).itemsize:basis=None
        else:meta=dict(codec='pod',rank=int(k),dtype=np.dtype(dtype).name,singular_values=s[:min(len(s),128)].tolist())
    else:basis=None
    if basis is None:dtype=None;meta=dict(codec='scaled32',rank=None,dtype='single')
    error=_error(parts,basis,None if basis is None else basis.shape[1],dtype)
    out=[]
    for side,rows in parts:
        normal=rows/side['scale'][:,None]
        out.append(dict(side,coef=(normal if basis is None else normal@basis).astype(dtype or _single(rows))))
    return dict(meta,max_row_error=error),basis,out


# ---------------------------------------------------------------- pack

def _npy_shape(path):
    with open(path,'rb') as f:
        version=np.lib.format.read_magic(f)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        return reader(f)[0]


def _member(path,name,drop_order1):
    with np.load(path,allow_pickle=False) as d:a=d[name]
    if drop_order1 and 'rte-cache' in str(path):a=a[...,1]
    return a


def pack(args):
    began=time.perf_counter();source=args.bundle.resolve();out=args.output
    if out.exists():raise FileExistsError('fresh output directory required')
    record=json.loads((source/'bundle.json').read_text())
    if not record.get('complete'):raise ValueError('incomplete source bundle')
    files=record['files'];derived={p:name for name,paths in DERIVED.items() for p in paths}
    missing=[p for p in derived if p not in files]
    if missing:raise ValueError('source lacks derived products: '+', '.join(missing))
    if args.drop_rte_order1 and args.tolerance is None:raise ValueError('--drop-rte-order1 needs the reduced form')
    coded={}
    if args.tolerance is not None:
        for spec in GROUPS:
            if args.groups and spec[0] not in args.groups:continue
            for path in sorted(source.glob(spec[1])):coded[str(path.relative_to(source))]=spec
    (out/'files').mkdir(parents=True);(out/'factors').mkdir()
    entries=[];groups={};shrink=[]
    every=sorted(str(p.relative_to(source)) for p in source.rglob('*') if p.is_file())
    for rel in every:
        path=source/rel;info=dict(path=rel,size=path.stat().st_size,sha256=files.get(rel,{}).get('sha256') or digest(path))
        if rel in derived:entries.append(dict(info,kind='derived',recipe=derived[rel]));continue
        if rel in coded:
            group,_,members,axis,codec,kind=coded[rel]
            if members is None:
                shape=list(_npy_shape(path))
                entries.append(dict(info,kind='coded',group=group,shape=shape,axis=axis%len(shape)))
                groups.setdefault(group,[]).append((rel,None))
            else:
                with zipfile.ZipFile(path) as z:order=[(i.filename,i.compress_type) for i in z.infolist()]
                target=out/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True)
                with zipfile.ZipFile(path) as z,zipfile.ZipFile(target,'w') as w:
                    for name,method in order:
                        if name[:-4] not in members:w.writestr(z.getinfo(name),z.read(name),compress_type=method)
                shapes={}
                with np.load(path,allow_pickle=False) as d:
                    for m in members:shapes[m]=list(d[m].shape)
                for m in members:groups.setdefault(group,[]).append((rel,m))
                entries.append(dict(info,kind='npz',members=[[n,k] for n,k in order],coded=list(members),
                    shapes=shapes,group=group,axis=axis,order1='nan' if args.drop_rte_order1 and group=='rte-cache' else None))
            continue
        target=out/'files'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if args.deflate and rel.endswith('.npy') and info['size']>(1<<20):
            shrink.append((path,Path(str(target)+'.shz')));entries.append(dict(info,kind='file',stored='shz'));continue
        shutil.copy2(path,target)
        entries.append(dict(info,kind='file'))
    t=time.perf_counter();_parallel(npy_to_shz,shrink,args.jobs);deflate_seconds=time.perf_counter()-t
    reports={}
    for spec in GROUPS:
        group,_,_,axis,codec,kind=spec
        if group not in groups:continue
        t=time.perf_counter();items=groups[group]
        arrays=[np.load(source/rel,allow_pickle=False) if m is None else _member(source/rel,m,args.drop_rte_order1)
                for rel,m in items]
        meta,basis,parts=encode(arrays,axis%arrays[0].ndim,codec,kind,args.tolerance)
        payload={} if basis is None else {'basis':basis}
        for i,part in enumerate(parts):
            for key,value in part.items():payload[f'{i}/{key}']=value
        save_factors(out/'factors'/group,payload,args.deflate)
        reports[group]=dict(meta,items=[[r,m] for r,m in items],row_length=int(arrays[0].shape[axis]),
            source_bytes=int(sum(a.nbytes for a in arrays)),stored_bytes=sum(p.stat().st_size for p in (out/'factors'/group).iterdir()),
            seconds=time.perf_counter()-t)
        print(json.dumps(dict(group=group,codec=meta['codec'],rank=meta['rank'],dtype=meta['dtype'],
            max_row_error=meta['max_row_error'],ratio=reports[group]['stored_bytes']/reports[group]['source_bytes'],
            seconds=reports[group]['seconds'])),flush=True)
        del arrays,parts
    stored=sum((out/'files'/(e['path']+('.shz' if e.get('stored')=='shz' else ''))).stat().st_size
               for e in entries if e['kind'] in ('file','npz'))
    stored+=sum(p.stat().st_size for p in (out/'factors').rglob('*') if p.is_file())
    manifest=dict(schema=SCHEMA,form='exact' if args.tolerance is None else 'reduced',
        source_bundle_sha256=digest(source/'bundle.json'),tolerance=args.tolerance,
        drop_rte_order1=bool(args.drop_rte_order1),entries=entries,groups=reports,
        source_bytes=int(sum(e['size'] for e in entries)),transport_bytes=int(stored),
        deflate=bool(args.deflate),deflate_seconds=deflate_seconds,
        pack_seconds=time.perf_counter()-began,pack_peak=peak_bytes())
    manifest['transport_files']=seal(out)
    (out/'compact.json').write_text(json.dumps(manifest,indent=1)+'\n')
    print(json.dumps(dict(source_bytes=manifest['source_bytes'],transport_bytes=stored,
        ratio=stored/manifest['source_bytes'],seconds=manifest['pack_seconds'],peak=manifest['pack_peak'])),flush=True)
    return 0


def seal(out):
    """SHA256 of every stored transport file, checked before unpack."""
    return {str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*'))
            if p.is_file() and p.name!='compact.json'}


def check_transport(source,manifest):
    stored=manifest.get('transport_files')
    if stored is None:raise ValueError('transport without file checksums; pack it again')
    bad=[rel for rel,sha in stored.items() if not (source/rel).is_file() or digest(source/rel)!=sha]
    if bad:raise ValueError('transport files missing or changed: '+', '.join(bad[:10]))


# ---------------------------------------------------------------- derived recipes

def _save(path,array):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp.npy')
    np.save(tmp,np.ascontiguousarray(array),allow_pickle=False);tmp.replace(path)


def rebuild_q1_integral(root,jobs):
    directory=root/FIRST/'q1-reference';m=json.loads((directory/'manifest.json').read_text())
    for branch in ('hit','miss'):
        cdf=np.load(directory/(branch+'_cdf.npy'),mmap_mode='r')
        target=directory/(branch+'_integral.npy');tmp=target.with_suffix('.tmp.npy')
        out=np.lib.format.open_memmap(tmp,mode='w+',dtype=cdf.dtype,shape=cdf.shape)
        # Same expression as FirstEntryAtlas.build; elementwise along radius.
        u_grid=np.linspace(0.,m['time_u_max'],cdf.shape[2]);du=np.diff(u_grid)
        first=np.diff(np.sinh(u_grid));second=du*np.sinh(u_grid[1:])-np.diff(np.cosh(u_grid))
        for i in range(0,len(cdf),8):
            c=np.asarray(cdf[i:i+8]);integral=np.zeros_like(c)
            slope=np.diff(c,axis=2)/du[None,None,:,None,None]
            increments=c[:,:,:-1]*first[None,None,:,None,None]+slope*second[None,None,:,None,None]
            integral[:,:,1:]=np.cumsum(increments,axis=2);out[i:i+8]=integral
        out.flush();del out;tmp.replace(target)


def rebuild_q1_adaptive(root,jobs):
    from lighthit.receiver_first_atlas import FirstEntryAtlas
    from lighthit.receiver_first_adaptive import AdaptiveFirstEntry
    directory=root/FIRST;fine=FirstEntryAtlas.load(directory/'q1-reference')
    stride=json.loads((directory/'q1-adaptive/manifest.json').read_text())['time_stride']
    atlas=AdaptiveFirstEntry.of(fine,stride=stride);target=directory/'q1-adaptive'
    for branch in ('hit','miss'):
        for name in ('cdf','integral','charge'):_save(target/f'{branch}_{name}.npy',getattr(atlas.coarse,f'{branch}_{name}'))
        _save(target/f'{branch}_time_error.npy',getattr(atlas,branch+'_error'))
        for mode in ('coarse','fine'):_save(target/f'{branch}_quad_{mode}.npy',getattr(atlas,f'{branch}_quad_{mode}'))


def rebuild_q1_bands(root,jobs):
    from lighthit.receiver_first_spectral import SpectralFirstEntry,SpectralFirstEntryConfig
    from lighthit.receiver_geometry import ReceiverGeometry
    from lighthit.receiver_first_atlas import _optics
    directory=root/FIRST;m=json.loads((directory/'q1-bands/manifest.json').read_text())
    work=root/'.unpack/q1-bands';shutil.rmtree(work,ignore_errors=True)
    atlas=SpectralFirstEntry.build_parallel(geometry_directory=directory/'geometry',radius_m=m['radius_m'],
        tables=_optics(m['tables']),output=work,jobs=jobs,config=SpectralFirstEntryConfig(**m['config']),
        geometry_indices=np.load(directory/'q1-bands/geometry_indices.npy'),keep_blocks=False)
    if m.get('quadratic_time'):atlas.prepare_quadratic(ReceiverGeometry.load(directory/'geometry'),work)
    for name in DERIVED['q1-bands']:(work/Path(name).name).replace(root/name)
    shutil.rmtree(work)


def rebuild_ge2(root,jobs):
    from lighthit import load_model
    from lighthit.transport import KernelConfig
    from lighthit.receiver_data import decode_settings
    from lighthit.band_fold import BandFold,fold_bundle,save_folded
    from lighthit.shower.api import FREQUENCIES_PER_NS,READOUT_PER_NS
    tables=root/'tables';manifest=json.loads((tables/'shower-data.json').read_text())
    settings=decode_settings(manifest['kernel_settings'],tables)
    config=KernelConfig(omega_per_ns=FREQUENCIES_PER_NS,readout_omega_per_ns=READOUT_PER_NS,
        cache_directory=tables/manifest['rte_cache'],cache_policy='require',**settings)
    kernel=load_model(root/'model/bgvd-model.npz').kernel(config)
    store=kernel.ge2_kernel_store();indices=np.arange(len(kernel.config.omega_per_ns))
    bundle=store.bundle(kernel,indices)
    stationary=store.bundle(kernel,indices[:1])
    stationary['coefficients']=np.ascontiguousarray(stationary['coefficients'])
    work=root/'.unpack/ge2';shutil.rmtree(work,ignore_errors=True);store.save(work/'pack')
    shipped=json.loads((tables/manifest['ge2_pack']/'pack.json').read_text())
    written=json.loads((work/'pack/pack.json').read_text())
    for item in shipped['bundles']:
        # Wavelength names are optical fingerprints; last-bit platform differences
        # may rename one, and the loader adopts such aliases. Match by basis.
        match=[w for w in written['bundles'] if w['key'][1:]==item['key'][1:] and len(w['key'][0])==len(item['key'][0])]
        if len(match)!=1:raise ValueError('rebuilt order >=2 pack does not match the shipped pack')
        renamed=[[a,b] for a,b in zip(item['key'][0],match[0]['key'][0]) if a!=b]
        if renamed:print(json.dumps(dict(ge2_fingerprint_aliases=renamed)),flush=True)
        (work/'pack'/match[0]['file']).replace(tables/manifest['ge2_pack']/item['file'])
    meta=json.loads((tables/manifest['ge2_band_fold']/'folded.json').read_text())
    fold=BandFold.of(kernel,rank=meta['band_fold']['rank'],bands=tuple(meta['band_fold']['bands']))
    folded=fold_bundle(bundle,fold,subdivide=meta['subdivide'])
    save_folded(folded,work/'fold');(work/'fold/folded.npy').replace(tables/manifest['ge2_band_fold']/'folded.npy')
    shutil.rmtree(work)


RECIPES=dict(zip(ORDER,(rebuild_q1_integral,rebuild_q1_adaptive,rebuild_q1_bands,rebuild_ge2)))


# ---------------------------------------------------------------- unpack

def _write_npz(target,packed,entry,arrays):
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix('.tmp.npz')
    with zipfile.ZipFile(packed) as z,zipfile.ZipFile(tmp,'w') as w:
        for name,method in entry['members']:
            if name[:-4] in arrays:
                buffer=io.BytesIO();np.save(buffer,arrays[name[:-4]],allow_pickle=False)
                info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=method
                w.writestr(info,buffer.getvalue())
            else:w.writestr(z.getinfo(name),z.read(name),compress_type=method)
    tmp.replace(target)


def unpack(args):
    began=time.perf_counter();source=args.input.resolve();root=args.output
    m=json.loads((source/'compact.json').read_text())
    if m['schema']!=SCHEMA:raise ValueError('unknown compact bundle schema')
    if root.exists():raise FileExistsError('fresh output directory required')
    timings={};t=time.perf_counter();check_transport(source,m);timings['transport_sha256']=time.perf_counter()-t
    root.mkdir(parents=True)
    t=time.perf_counter()
    expand=[(source/'files'/(e['path']+'.shz'),root/e['path']) for e in m['entries'] if e.get('stored')=='shz']
    _parallel(shz_to_npy,expand,args.jobs)
    for e in m['entries']:
        if e['kind']!='file' or e.get('stored')=='shz':continue
        target=root/e['path'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/'files'/e['path'],target)
    timings['copy']=time.perf_counter()-t;t=time.perf_counter()
    index={(rel,member):(group,i) for group,info in m['groups'].items() for i,(rel,member) in enumerate(info['items'])}
    cache={}
    def decoded(rel,member,shape,axis):
        group,i=index[(rel,member)]
        if group not in cache:cache.clear();cache[group]=load_factors(source/'factors'/group)
        g=cache[group];basis=g.get('basis')
        entry={k:g[f'{i}/{k}'] for k in SIDE+('coef',) if f'{i}/{k}' in g}
        return _unrows(_decode(entry,basis),shape,axis)
    nan_order1=[]
    for e in m['entries']:
        if e['kind']=='coded':_save(root/e['path'],decoded(e['path'],None,e['shape'],e['axis']))
        elif e['kind']=='npz':
            arrays={}
            for name in e['coded']:
                shape=e['shapes'][name]
                if e.get('order1'):
                    full=np.zeros(shape,np.complex128);full[...,1]=decoded(e['path'],name,shape[:-1],e['axis']%(len(shape)-1))
                    arrays[name]=full;nan_order1.append(e)
                else:arrays[name]=decoded(e['path'],name,shape,e['axis']%len(shape))
            _write_npz(root/e['path'],source/'files'/e['path'],e,arrays)
    cache.clear();timings['decode']=time.perf_counter()-t
    for name in ORDER:
        if not any(e.get('recipe')==name for e in m['entries']):continue
        t=time.perf_counter();RECIPES[name](root,args.jobs);timings[name]=time.perf_counter()-t
        print(json.dumps(dict(recipe=name,seconds=timings[name],peak=peak_bytes())),flush=True)
    t=time.perf_counter()
    for e in nan_order1:
        # Order 1 is absent from the transport; mark it after the >=2 pack is built.
        with np.load(root/e['path'],allow_pickle=False) as d:arrays={k:d[k] for k in e['coded']}
        for name in arrays:arrays[name][...,0]=np.nan
        _write_npz(root/e['path'],root/e['path'],e,arrays)
    timings['order1']=time.perf_counter()-t
    shutil.rmtree(root/'.unpack',ignore_errors=True)
    t=time.perf_counter();check=[]
    for e in m['entries']:
        check.append(dict(path=e['path'],kind=e['kind'],identical=digest(root/e['path'])==e['sha256']))
    timings['sha256']=time.perf_counter()-t
    corrupt=[c['path'] for c in check if c['kind']=='file' and not c['identical']]
    report=dict(schema=SCHEMA,form=m.get('form','reduced' if m['tolerance'] is not None else 'exact'),
        source_bundle_sha256=m['source_bundle_sha256'],tolerance=m['tolerance'],
        drop_rte_order1=m['drop_rte_order1'],seconds=time.perf_counter()-began,
        timings=timings,peak=peak_bytes(),jobs=args.jobs,files=check,
        identical={k:[sum(c['identical'] for c in check if c['kind']==k),sum(c['kind']==k for c in check)]
                   for k in ('file','coded','npz','derived')},
        differing=[c['path'] for c in check if not c['identical']],corrupt=corrupt,
        runtime_bytes=int(sum((root/e['path']).stat().st_size for e in m['entries'])))
    (root/'unpack-report.json').write_text(json.dumps(report,indent=1)+'\n')
    print(json.dumps({k:report[k] for k in ('form','seconds','timings','peak','identical','runtime_bytes')}),flush=True)
    if corrupt:raise ValueError('exactly transported files differ from their SHA256: '+', '.join(corrupt))
    return 0


# ---------------------------------------------------------------- verify

def verify(args):
    """SHA256 of every runtime file against bundle.json (and unpack-report.json)."""
    root=args.bundle.resolve();record=json.loads((root/'bundle.json').read_text())
    if not record.get('complete'):raise ValueError('bundle.json does not describe a complete bundle')
    files=record['files'];report_path=root/'unpack-report.json'
    expected={}
    if report_path.exists():
        unpacked=json.loads(report_path.read_text())
        if unpacked.get('source_bundle_sha256')!=digest(root/'bundle.json'):
            raise ValueError('unpack-report.json belongs to another bundle')
        expected={c['path']:c['kind'] for c in unpacked['files'] if c['kind']!='file'}
    missing=[];changed=[];rebuilt=[]
    for rel,info in sorted(files.items()):
        path=root/rel
        if not path.is_file():missing.append(rel);continue
        same=path.stat().st_size==info['size'] and digest(path)==info['sha256']
        if same:continue
        (rebuilt if rel in expected else changed).append(rel)
    form=('full' if not report_path.exists() else unpacked.get('form'))
    result=dict(bundle=str(root),form=form,files=len(files),missing=missing,changed=changed,
                rebuilt_or_reduced=rebuilt,passed=not missing and not changed)
    print(json.dumps(result,indent=1))
    return 0 if result['passed'] else 2


# ---------------------------------------------------------------- compare

def compare(args):
    """Largest absolute and relative differences of every table array."""
    a=args.reference.resolve();b=args.candidate.resolve();rows=[]
    def arrays(path):
        if path.suffix=='.npy':return {'':np.load(path,mmap_mode='r')}
        d=np.load(path,allow_pickle=False);return {k:d[k] for k in d.files}
    for p in sorted(a.rglob('*')):
        rel=p.relative_to(a)
        if not p.is_file() or p.suffix not in ('.npy','.npz') or rel.parts[0]!='tables':continue
        x=arrays(p);y=arrays(b/rel)
        for k in x:
            u=np.asarray(x[k]);v=np.asarray(y[k])
            if u.dtype.kind not in 'fc' or u.ndim==0:
                rows.append(dict(path=str(rel),member=k,equal=bool(np.array_equal(u,v))));continue
            d=0.;scale=0.;nan=0
            for i in range(0,max(len(u),1),16):
                du=np.abs(u[i:i+16]-v[i:i+16]);nan+=int(np.isnan(du).sum())
                d=max(d,float(np.nanmax(du,initial=0.)));scale=max(scale,float(np.max(np.abs(u[i:i+16]),initial=0.)))
            rows.append(dict(path=str(rel),member=k,max_abs=d,max_abs_over_max=d/scale if scale else d,nan=nan))
    args.report.write_text(json.dumps(rows,indent=1)+'\n')
    for r in rows:
        if r.get('max_abs',0)>0 or r.get('equal') is False or r.get('nan'):print(json.dumps(r))
    return 0


def main(argv=None):
    p=argparse.ArgumentParser(prog='lighthit-tables',description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    v=sub.add_parser('verify',help='check runtime files against bundle.json');v.add_argument('--bundle',type=Path,required=True)
    a=sub.add_parser('pack',help='write the exact or reduced transport of a complete runtime')
    a.add_argument('--bundle',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--tolerance',type=float,help='maximum relative row error of the reduced primaries; omit for the exact form')
    a.add_argument('--drop-rte-order1',action='store_true',help='omit the order-1 RTE moments (unused by the glass route)')
    a.add_argument('--groups',nargs='+',help='reduce only these groups (default: all)')
    a.add_argument('--deflate',action='store_true',help='byte-shuffled deflate of stored arrays (standard library only)')
    a.add_argument('--jobs',type=int,default=os.cpu_count() or 1)
    u=sub.add_parser('unpack',help='rebuild a complete runtime from a transport')
    u.add_argument('--input',type=Path,required=True);u.add_argument('--output',type=Path,required=True)
    u.add_argument('--jobs',type=int,default=os.cpu_count() or 1)
    c=sub.add_parser('compare',help='largest array differences between two runtimes')
    c.add_argument('--reference',type=Path,required=True);c.add_argument('--candidate',type=Path,required=True)
    c.add_argument('--report',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='pack' and args.tolerance is not None and not 0<args.tolerance<1:p.error('tolerance in (0,1) required')
    return dict(verify=verify,pack=pack,unpack=unpack,compare=compare)[args.command](args)


if __name__=='__main__':raise SystemExit(main())
