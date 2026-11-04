#!/usr/bin/env python3
"""Export the runtime closure of checked first-entry tables without rebuilding.

All array bytes and primary-atlas manifests are preserved. Only directory
references in route/pack/beta-time manifests are relocated. --plan measures
the actual runtime closure; construction blocks and held-out build jobs are
not part of that closure.
"""
from pathlib import Path
import argparse,copy,json,os,shutil,sys
from tempfile import TemporaryDirectory

from build_receiver_bundle import (RECIPE,atomic_json,build_lock,digest,inventory,
                                   publish,verify_inventory,check)


class Export:
    def __init__(self,source_root,output):
        self.source_root=Path(source_root).resolve();self.output=Path(output).resolve();self.files={};self.documents={}

    def source(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.source_root):raise ValueError('source dependency escapes --source-root: '+str(path))
        if not path.is_file():raise FileNotFoundError(str(path))
        return path

    def add(self,source,name):
        source=self.source(source);name=Path(name)
        if name.is_absolute() or '..' in name.parts:raise ValueError('invalid export destination')
        key=str(name)
        if key in self.files and self.files[key]!=source:raise ValueError('conflicting export destination')
        self.files[key]=source

    def document(self,name,value):self.documents[str(name)]=value

    def atlas(self,source,destination):
        source=Path(source);m=json.loads(self.source(source/'manifest.json').read_text())
        if m.get('status')!='complete':raise ValueError('incomplete source track atlas')
        self.add(source/'manifest.json',Path(destination)/'manifest.json')
        for row in m['rows']:
            for key in ('file','charge_file'):
                if row.get(key):self.add(source/row[key],Path(destination)/row[key])
        if m.get('charge_file'):
            name=m['charge_file']['file'];self.add(source/name,Path(destination)/name)

    def plan(self,model,shower_manifest,muon_manifest):
        from lighthit.shower import ShowerData
        from lighthit.muon import MuonData
        from lighthit.receiver_data import encode_settings
        shower=ShowerData.load(self.source(shower_manifest));muon=MuonData.load(self.source(muon_manifest))
        if shower.kernel_settings.get('scattered_receiver')!='first-entry' or muon.kernel_settings.get('scattered_receiver')!='first-entry':
            raise ValueError('this exporter requires explicitly selected first-entry routes')
        self.add(model,'model/bgvd-model.npz')
        header=json.loads(self.source(shower.ge2_pack/'pack.json').read_text())
        other=json.loads(self.source(muon.ge2_pack/'pack.json').read_text())
        if header!=other:raise ValueError('shower and muon must use the same prepared pack')
        packed=copy.deepcopy(header);keys=set()
        for i,item in enumerate(packed['bundles']):
            old=header['bundles'][i]['file'];new=f'array-{i:03d}.npy';self.add(shower.ge2_pack/old,'tables/ge2-pack/'+new);item['file']=new
            keys.update(item['key'][0])
        self.document('tables/ge2-pack/pack.json',packed)
        for key in sorted(keys):self.add(shower.rte_cache/f'directional-{key}.npz',f'tables/rte-cache/directional-{key}.npz')
        for name in ('folded.json','folded.npy'):self.add(shower.ge2_band_fold/name,'tables/ge2-band-fold/'+name)
        first=Path(shower.kernel_settings['first_entry_tables'])
        if first!=Path(muon.kernel_settings['first_entry_tables']):raise ValueError('Q1 roots differ between routes')
        for name in ('q1-reference','q1-adaptive','geometry','geometry-primary','q1-bands'):
            directory=first/name
            if not (directory/'manifest.json').exists():raise FileNotFoundError(str(directory/'manifest.json'))
            for path in sorted(directory.iterdir()):
                if path.name=='manifest.json' or path.suffix=='.npy':self.add(path,'tables/first-entry/'+name+'/'+path.name)
        self.atlas(muon.primary_first_atlas,'tables/muon/q1-atlas');self.atlas(muon.primary_ge2_atlas,'tables/muon/ge2-atlas')
        for source,dest in [(muon.primary_first_beta,'first-beta/first-beta-charge.npz'),(muon.primary_ge2_beta,'ge2-beta/beta-charge.npz'),
                            (muon.preview_atlas,'surface-preview/surface-preview.npz'),(muon.preview_proposal_atlas,'preview-proposal.npz'),
                            (muon.primary_first_cubature,'surface-rule.npz')]:
            if source is None:raise ValueError('source route is missing a required fast input: '+dest)
            self.add(source,'tables/muon/'+dest)
        original=json.loads(self.source(muon.primary_first_time_beta).read_text());time_manifest=copy.deepcopy(original)
        for node in time_manifest['nodes']:
            old=Path(muon.primary_first_time_beta).parent/node['directory']
            if digest(self.source(old/'manifest.json'))!=node['manifest_sha256']:raise ValueError('source beta-time anchor hash differs')
            name='time-beta'+str(node['beta']).replace('.','')
            self.atlas(old,'tables/muon/'+name);node['directory']=name
        time_manifest['source_manifest_sha256']=digest(muon.primary_first_time_beta)
        self.document('tables/muon/first-time-beta.json',time_manifest)
        settings=dict(shower.kernel_settings,first_entry_tables=self.output/'tables/first-entry')
        self.document('tables/shower-data.pending.json',dict(rte_cache='rte-cache',ge2_pack='ge2-pack',ge2_band_fold='ge2-band-fold',
            kernel_settings=encode_settings(settings,self.output/'tables')))
        return dict(model_sha256=digest(model),source_shower_manifest_sha256=digest(shower_manifest),
                    source_muon_manifest_sha256=digest(muon_manifest),model_code='first-entry runtime export; no array fitting')

    def run(self,provenance,*,hardlink=False):
        root=self.output;work=root/'.export';work.mkdir(parents=True,exist_ok=True)
        with build_lock(work):
            state_path=work/'progress.json'
            mapping={name:str(path.relative_to(self.source_root)) for name,path in self.files.items()}
            key=dict(provenance=provenance,mapping=mapping,documents=self.documents)
            if state_path.exists():
                state=json.loads(state_path.read_text())
                if state['identity']!=key:raise ValueError('export inputs changed; choose a new directory')
            else:
                if any(p.name!='.export' for p in root.iterdir()):raise FileExistsError('export output must be new or have a matching progress record')
                state=dict(identity=key,copied={});atomic_json(state_path,state)
            for name,source in self.files.items():
                dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
                if name in state['copied']:
                    verify_inventory(root,{name:state['copied'][name]});continue
                with TemporaryDirectory(prefix='export-',dir=dest.parent) as directory:
                    temporary=Path(directory)/'data'
                    if hardlink:
                        try:os.link(source,temporary)
                        except OSError:shutil.copyfile(source,temporary)
                    else:shutil.copyfile(source,temporary)
                    info=dict(size=source.stat().st_size,sha256=digest(source))
                    if temporary.stat().st_size!=info['size'] or digest(temporary)!=info['sha256']:raise ValueError('export copy checksum differs')
                    temporary.replace(dest)
                state['copied'][name]=info;atomic_json(state_path,state)
                print('COPIED',name,info['size'],flush=True)
            for name,value in self.documents.items():atomic_json(root/name,value)
            publish(root)
            atomic_json(root/'bundle.json',dict(schema=RECIPE,complete=True,identity=provenance,
                validation_scope='byte-preserving export and runtime component loading; event check recorded separately',
                files=inventory(root,[root/'model',root/'tables'])))
            check(root,quick=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-root','model','shower-data','muon-data','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan',action='store_true');p.add_argument('--hardlink',action='store_true',help='use same-volume links where possible; copied arrays are otherwise verified')
    a=p.parse_args(argv)
    try:
        export=Export(a.source_root,a.output);provenance=export.plan(a.model,a.shower_data,a.muon_data)
        sizes=[v.stat().st_size for v in export.files.values()]
        print(json.dumps(dict(files=len(sizes),runtime_bytes=sum(sizes),largest_file_bytes=max(sizes),source=provenance)),flush=True)
        if not a.plan:export.run(provenance,hardlink=a.hardlink)
        return 0
    except (OSError,ValueError,RuntimeError) as error:print('Export stopped:',error,file=sys.stderr);return 1


if __name__=='__main__':raise SystemExit(main())
