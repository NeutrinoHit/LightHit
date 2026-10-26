#!/usr/bin/env python3
"""Build a portable first-entry shower/muon cache from one serialized model.

Run once before event calculations. The default recipe retains the production
32-band source and three-band fold, physical beta-time anchors, and the
separately checked positive wavelength rule. No charged event enters building.
Use --plan to inspect the expensive preparation; repeat the same command to
resume. Existing centre-reference bundles are never changed.
"""
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
RECIPE = 'lighthit/first-entry-bundle/1'
PRIMARY_RADII = [3,5,8,12,15,18,21.5,25,30,35,42.5,50,60,70,85,100,150,220,300]
TIME_RADII = [3,4,5,5.75,6.5,8,9,10,11,12,15,18,21.5,25,30,35,42.5,50,60,70,85,100,150,220,300]
TIME_CHECK_RADII = [3.5,6.,10.,13.5,27.5,55.,80.,180.]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def inventory(root,paths):
    """Hash named products; per-row construction checkpoints are not runtime data."""
    root=Path(root).resolve();result={};seen={}
    for product in paths:
        product=Path(product)
        if not product.exists():raise FileNotFoundError(f'missing build product: {product}')
        files=sorted(product.rglob('*')) if product.is_dir() else [product]
        for path in files:
            parts=path.relative_to(root).parts
            temporary=(path.name.endswith(('.tmp','.tmp.npz'))
                or any(p.startswith(('table-','preview-','fluence-')) for p in parts[:-1]))
            if (not path.is_file() or temporary
                    or any(p in ('blocks','__pycache__') for p in parts)):continue
            resolved=path.resolve()
            if not resolved.is_relative_to(root):raise ValueError(f'product escapes bundle: {path}')
            st=path.stat();key=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
            if key not in seen:seen[key]=digest(path)
            result[str(path.relative_to(root))]=dict(size=st.st_size,sha256=seen[key])
    if not result:raise ValueError('build stage produced no files')
    return result


def verify_inventory(root,record,*,quick=False):
    root=Path(root).resolve()
    if not record:raise ValueError('empty product inventory')
    for name,info in record.items():
        path=(root/name).resolve()
        if not path.is_relative_to(root):raise ValueError('inventory path escapes bundle')
        if not path.is_file() or path.stat().st_size!=info['size']:
            raise ValueError(f'missing or truncated build product: {name}')
        if not quick and digest(path)!=info['sha256']:
            raise ValueError(f'changed build product: {name}; use a new output directory')


@contextmanager
def build_lock(directory):
    """A process lock is released automatically after interruption or failure."""
    import fcntl
    path=Path(directory)/'builder.lock';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another builder owns this output') from None
        try:yield
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


@dataclass
class Stage:
    name: str
    products: tuple
    commands: list = field(default_factory=list)
    action: object = None
    validate: object = None
    recover: object = None


class Builder:
    def __init__(self,root,identity,*,runner=None):
        self.root=Path(root).resolve();self.work=self.root/'.prepare'
        self.identity=identity;self.runner=runner or self.execute

    def execute(self,command,log):
        env=dict(os.environ,NUMBA_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
                 PYTHONPATH=str(REPOSITORY/'src')+os.pathsep+os.environ.get('PYTHONPATH',''))
        # Large temporary Bessel blocks belong on the chosen output volume.
        env['TMPDIR']=str(self.work/'tmp');Path(env['TMPDIR']).mkdir(exist_ok=True)
        env['NUMBA_CACHE_DIR']=str(self.work/'numba')
        with Path(log).open('a') as stream:
            subprocess.run(command,cwd=REPOSITORY,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

    def run(self,stages,*,finalize=None):
        self.work.mkdir(parents=True,exist_ok=True)
        with build_lock(self.work):
            state_path=self.work/'build.json'
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state['identity']!=self.identity:
                    raise ValueError('model, recipe, code or numerical environment changed; choose a new output directory')
            else:
                if any(p.name!='.prepare' for p in self.root.iterdir()):
                    raise FileExistsError('output is not empty and has no matching build record')
                state=dict(identity=self.identity,completed={});atomic_json(state_path,state)
            logs=self.work/'logs';logs.mkdir(exist_ok=True)
            for stage in stages:
                normalized=[]
                for command in stage.commands:
                    command=list(command)
                    for i,value in enumerate(command[:-1]):
                        if value in ('--workers','--jobs'):command[i+1]='{workers}'
                    normalized.append(command)
                signature=hashlib.sha256(json.dumps(normalized,sort_keys=True).replace(str(self.root),'{bundle}').encode()).hexdigest()
                if stage.name in state['completed']:
                    old=state['completed'][stage.name]
                    if old['signature']!=signature:raise ValueError('completed stage command changed: '+stage.name)
                    verify_inventory(self.root,old['files'])
                    if stage.validate:stage.validate()
                    print('REUSED',stage.name,flush=True);continue
                print('BUILD',stage.name,'log:',logs/(stage.name+'.log'),flush=True)
                started=time.perf_counter()
                recovered=stage.recover is not None and stage.recover()
                if not recovered:
                    for command in stage.commands:self.runner(command,logs/(stage.name+'.log'))
                    if stage.action:stage.action()
                if stage.validate:stage.validate()
                files=inventory(self.root,stage.products)
                state['completed'][stage.name]=dict(signature=signature,files=files,seconds=time.perf_counter()-started)
                atomic_json(state_path,state)
                print('DONE',stage.name,flush=True)
            if finalize:finalize()


def _grids(radius,maximum):
    step=.05
    near=radius+np.geomspace(radius*.001,3-radius,int(np.ceil(np.log((3-radius)/(radius*.001))/step))+1)
    far=np.geomspace(3.,maximum,int(np.ceil(np.log(maximum/3)/step))+1)
    return np.unique(np.r_[near,far])


def _check_beta(path):
    with np.load(path,allow_pickle=False) as data:m=json.loads(str(data['metadata']))
    if not m.get('synthetic_validation_passed'):raise ValueError('beta correction failed synthetic checks: '+str(path))


def _recover_bank(directory,degree):
    path=Path(directory)/'manifest.json'
    if not path.exists():return False
    try:record=json.loads(path.read_text())
    except json.JSONDecodeError:
        path.replace(path.with_suffix('.interrupted.json'));return False
    if (record.get('schema')!='lighthit/first-entry-receiver-bank/1'
            or record['settings']['spatial_degree']!=degree or len(record['bands'])!=32):
        raise ValueError('completed directional bank has another recipe')
    for band in record['bands']:
        if digest(Path(directory)/band['file'])!=band['sha256']:raise ValueError('directional bank checksum differs')
    # The owning build record has already matched the model, code and recipe.
    return True


def _time_manifest(root,base,anchor,report):
    validation=json.loads(Path(report).read_text())
    if not validation['passed']:raise ValueError('physical beta-time interpolation failed validation')
    nodes=[]
    for beta,path in [(.999,anchor),(1.,base)]:
        manifest=Path(path)/'manifest.json';m=json.loads(manifest.read_text())
        if m['status']!='complete' or m['physics']['beta']!=beta:raise ValueError('incomplete physical beta-time anchor')
        nodes.append(dict(beta=beta,directory=os.path.relpath(path,root),manifest_sha256=digest(manifest)))
    atomic_json(root/'first-time-beta.json',dict(schema='lighthit/first-time-beta/1',validation_passed=True,
        nodes=nodes,validation=validation))


def preparation(model,root,jobs):
    """One fixed numerical recipe; changing jobs does not change the physics."""
    from lighthit import load_model
    from lighthit.receiver import SphereSurface
    m=load_model(model);surface=m.detector.receiver_surface
    if not isinstance(surface,SphereSurface):raise ValueError('model must supply spherical glass')
    radii=surface.radii(len(m.detector));radius=float(radii[0])
    if not np.allclose(radii,radius,rtol=1e-12,atol=0) or not 0<radius<3:
        raise ValueError('one common positive glass radius below 3 m required')
    if m.kernel().acceptance()['degree']!=3:raise ValueError('this fast recipe requires cubic OM acceptance')
    root=Path(root).resolve();work=root/'.prepare';tables=root/'tables';first=tables/'first-entry'
    model_out=root/'model/bgvd-model.npz';muon=tables/'muon'
    scripts=REPOSITORY/'tools/tables';experiments=REPOSITORY/'experiments'
    def command(script,*args):return [sys.executable,str(script),*map(str,args)]
    def receiver(kind,output,*args):
        return command(experiments/'receiver_surface.py',kind,'--model',model_out,'--output',output,
                       '--jobs',jobs,'--path-nodes',256,'--radial-nodes',400,*args,'--force')
    def track(output,beta,axis,*args):
        return command(scripts/'build_track_first_atlas.py','--model',model_out,'--first-entry-tables',first,
            '--output',output,'--workers',jobs,'--beta',beta,'--radii',*axis,'--check-rows',*args)
    def bootstrap():
        model_out.parent.mkdir(parents=True,exist_ok=True);work.mkdir(exist_ok=True)
        if Path(model).resolve()!=model_out:shutil.copyfile(model,model_out)
        np.save(work/'radii.npy',_grids(radius,1000.),allow_pickle=False)
        maximum=float(np.hypot(300.,300.*8.5))
        # Retain every ordinary geometry node; twenty additional intervals
        # cover the primary atlas's existing |u/rho| <= 8.5 support.
        primary=np.r_[_grids(radius,1000.),np.geomspace(1000.,maximum,21)[1:]]
        np.save(work/'primary-radii.npy',primary,allow_pickle=False)
        from lighthit.band_fold import BandFold
        fold=BandFold.of(m.kernel(),bands=(10,16,21))
        atomic_json(work/'fold-template.json',dict(band_fold=fold.metadata()))
    stages=[Stage('inputs',(model_out,work/'radii.npy',work/'primary-radii.npy',work/'fold-template.json'),action=bootstrap)]
    stages += [Stage('q1-reference',(first/'q1-reference',),[receiver('q1',first/'q1-reference','--radial-grid',work/'radii.npy')]),
        Stage('q1-adaptive',(first/'q1-adaptive',),[receiver('q1-time',first/'q1-adaptive','--fine-atlas',first/'q1-reference')]),
        Stage('geometry',(first/'geometry',),[receiver('geometry',first/'geometry','--radial-grid',work/'radii.npy')]),
        Stage('geometry-primary',(first/'geometry-primary',),[receiver('geometry',first/'geometry-primary','--radial-grid',work/'primary-radii.npy')]),
        Stage('q1-bands',(first/'q1-bands',),[
            receiver('q1-bands',first/'q1-bands','--geometry',first/'geometry','--time-nodes',65),
            receiver('q1-band-time',first/'q1-bands','--geometry',first/'geometry')])]
    stages += [Stage('rte-bank',(work/'bank',),[receiver('bank',work/'bank')],
        recover=lambda:_recover_bank(work/'bank',32)),
        Stage('fast-packs',(tables/'rte-cache',tables/'ge2-pack',tables/'ge2-band-fold',tables/'shower-data.pending.json'),[
            command(experiments/'prepare_receiver_route.py','--model',model_out,'--bank',work/'bank',
                '--first-entry-tables',first,'--output',tables,'--fold-template',work/'fold-template.json',
                '--manifest-name','shower-data.pending.json')]),
        Stage('point-reference-bank',(work/'point-bank',),[
            receiver('bank',work/'point-bank','--source-degree',128,'--frequencies',0)],
            recover=lambda:_recover_bank(work/'point-bank',128)),
        Stage('point-reference-pack',(work/'point-route/kernel-data.json',work/'point-route/ge2-pack',work/'point-route/rte-cache'),[
            command(experiments/'prepare_receiver_route.py','--model',model_out,'--bank',work/'point-bank',
                '--first-entry-tables',first,'--output',work/'point-route','--fold-template',work/'fold-template.json','--stationary-only')])]
    stages += [Stage('preview-proposal',(muon/'preview-proposal.npz',),[
        command(scripts/'build_positive_fluence.py','--model',model_out,'--output',muon/'preview-proposal.npz',
                '--count',6000000,'--seed',81,'--rho-bins',96,'--z-bins',384,'--extent-m',500,'--resume')]),
        Stage('surface-preview',(muon/'surface-preview',),[
            command(scripts/'build_surface_preview.py','--model',model_out,'--pack',work/'point-route/ge2-pack',
                '--cache',work/'point-route/rte-cache','--first-entry-tables',first,'--template',muon/'preview-proposal.npz',
                '--output',muon/'surface-preview','--workers',jobs,'--phi-panel',.5,'--spatial-order',4)])]
    stages += [Stage('primary-q1',(muon/'q1-atlas',),[track(muon/'q1-atlas',1.,PRIMARY_RADII)]),
        Stage('primary-ge2',(muon/'ge2-atlas',),[
            command(scripts/'build_track_ge2_atlas.py','--model',model_out,'--pack',tables/'ge2-pack',
                '--cache',tables/'rte-cache','--output',muon/'ge2-atlas','--workers',jobs)]),
        Stage('charge-cubature',(work/'cubature',),[
            command(scripts/'build_response_cubature.py','--model',model_out,'--output',work/'cubature')]),
        Stage('glass-cubature',(muon/'surface-rule.npz',muon/'surface-rule.json'),[
            command(experiments/'validate_receiver_cubature.py','--model',model_out,'--first-entry-tables',first,
                '--rule',work/'cubature/basis-rank24.npz','--output',muon/'surface-rule.npz','--waveform',
                '--radii',10,80,'--betas',.95,.995,'--cuts','root','edge','--resume')]),
        Stage('first-beta',(muon/'first-beta',),[
            command(scripts/'build_first_beta_charge.py','--model',model_out,'--first-entry-tables',first,
                '--base',muon/'q1-atlas','--cubature',muon/'surface-rule.npz','--output',muon/'first-beta','--workers',jobs)],
            validate=lambda:_check_beta(muon/'first-beta/first-beta-charge.npz')),
        Stage('ge2-beta',(muon/'ge2-beta',),[
            command(scripts/'build_ge2_beta_charge.py','--model',model_out,'--base',muon/'ge2-atlas',
                '--pack',tables/'ge2-pack','--cache',tables/'rte-cache','--output',muon/'ge2-beta')],
            validate=lambda:_check_beta(muon/'ge2-beta/beta-charge.npz')),
        Stage('time-beta1',(muon/'time-beta1',),[track(muon/'time-beta1',1.,TIME_RADII,'--reuse-rows-from',muon/'q1-atlas')]),
        Stage('time-beta0999',(muon/'time-beta0999',),[track(muon/'time-beta0999',.999,TIME_RADII)]),
        Stage('time-validation',(muon/'time-validation.json',),[
            command(experiments/'validate_receiver_time_beta.py','--model',model_out,'--first-entry-tables',first,
                '--base',muon/'time-beta1','--anchor',muon/'time-beta0999','--output',muon/'time-validation.json',
                '--align-glass-front','--radii',*TIME_CHECK_RADII)]),
        Stage('time-manifest',(muon/'first-time-beta.json',),action=lambda:_time_manifest(muon,muon/'time-beta1',muon/'time-beta0999',muon/'time-validation.json'))]
    return stages


def publish(root):
    """Load the real runtime interfaces before publishing either final manifest."""
    from lighthit import load_model
    from lighthit.shower import ShowerData,ShowerTransport
    from lighthit.muon import MuonData,MuonTransport,write_manifest
    from lighthit.receiver_prompt import FirstEntryPrompt
    root=Path(root);tables=root/'tables';muon=tables/'muon'
    shower=ShowerData.load(tables/'shower-data.pending.json')
    settings=shower.kernel_settings
    if not Path(settings['first_entry_tables']).resolve().is_relative_to(root.resolve()):
        raise ValueError('first-entry tables escape bundle')
    entries=dict(rte_cache=shower.rte_cache,ge2_pack=shower.ge2_pack,ge2_band_fold=shower.ge2_band_fold,
        kernel_settings=settings,primary_first_atlas=muon/'q1-atlas',primary_ge2_atlas=muon/'ge2-atlas',
        primary_first_beta=muon/'first-beta/first-beta-charge.npz',primary_ge2_beta=muon/'ge2-beta/beta-charge.npz',
        preview_atlas=muon/'surface-preview/surface-preview.npz',preview_proposal_atlas=muon/'preview-proposal.npz',
        primary_first_time_beta=muon/'first-time-beta.json',primary_first_cubature=muon/'surface-rule.npz')
    for key,path in entries.items():
        if key!='kernel_settings' and not Path(path).resolve().is_relative_to(root.resolve()):raise ValueError('runtime input escapes bundle')
    write_manifest(tables/'muon-data.pending.json',**entries)
    model=load_model(root/'model/bgvd-model.npz')
    engine=ShowerTransport(model.kernel(),shower);engine.tables();FirstEntryPrompt(engine.kernel)
    transport=MuonTransport(model.kernel(),MuonData.load(tables/'muon-data.pending.json'));transport.tables()
    for name in ('shower','muon'):
        atomic_json(tables/(name+'-data.json'),json.loads((tables/(name+'-data.pending.json')).read_text()))


def identity(model):
    paths=sorted((REPOSITORY/'src/lighthit').rglob('*.py'))+sorted((REPOSITORY/'tools/tables').glob('*.py'))
    paths+=sorted((REPOSITORY/'experiments').glob('*receiver*.py'))
    code={str(p.relative_to(REPOSITORY)):digest(p) for p in paths}
    import scipy,numba
    return dict(schema=RECIPE,model_sha256=digest(model),code=code,
                python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,numba=numba.__version__)


def check(root,*,quick=False):
    root=Path(root).resolve();path=root/'bundle.json'
    if not path.is_file():raise ValueError('bundle is incomplete: bundle.json has not been published')
    record=json.loads(path.read_text())
    if record.get('schema')!=RECIPE or not record.get('complete'):raise ValueError('incomplete or unsupported bundle')
    verify_inventory(root,record['files'],quick=quick)
    from lighthit.shower import ShowerData
    from lighthit.muon import MuonData
    ShowerData.load(root/'tables/shower-data.json');MuonData.load(root/'tables/muon-data.json')
    print('READY',root,'(sizes and manifests)' if quick else '(SHA256 verified)',flush=True)
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model',type=Path,help='serialized water, OM response and glass geometry')
    parser.add_argument('--output',type=Path,required=True,help='new standalone bundle directory')
    parser.add_argument('--jobs',type=int,default=1,help='independent preparation workers; not event threads')
    parser.add_argument('--plan',action='store_true',help='print stages without writing files or building tables')
    parser.add_argument('--check',action='store_true',help='verify a completed, possibly relocated bundle')
    parser.add_argument('--quick',action='store_true',help='with --check, verify existence and sizes instead of reading every array')
    args=parser.parse_args(argv)
    if args.jobs<1:parser.error('--jobs must be positive')
    if args.quick and not args.check:parser.error('--quick requires --check')
    if args.plan and args.check:parser.error('--plan and --check are mutually exclusive')
    try:
        if args.check:check(args.output,quick=args.quick);return 0
        if args.model is None:parser.error('--model is required for preparation')
        root=args.output.resolve();stages=preparation(args.model,root,args.jobs)
        if args.plan:
            print(json.dumps(dict(schema=RECIPE,stages=[dict(name=s.name,commands=s.commands,products=list(map(str,s.products))) for s in stages]),indent=2));return 0
        signature=identity(args.model)
        if (root/'bundle.json').exists():
            old=check(root)
            if old['identity']!=signature:raise ValueError('completed bundle has another model, recipe or environment')
            return 0
        def finalize():
            publish(root)
            atomic_json(root/'bundle.json',dict(schema=RECIPE,complete=True,identity=signature,
                validation_scope='component and synthetic interpolation checks; full-event benchmark is separate',
                files=inventory(root,[root/'model',root/'tables'])))
        Builder(root,signature).run(stages,finalize=finalize)
        check(root,quick=True)
        print('Set LIGHTHIT_DATA='+str(root),flush=True)
        print('Set LIGHTHIT_SHOWER_DATA='+str(root/'tables/shower-data.json'),flush=True)
        print('Set LIGHTHIT_MUON_DATA='+str(root/'tables/muon-data.json'),flush=True)
        return 0
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        print('Build stopped:',error,file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
