"""Restartable training; test split is NEVER scored in this module."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import gc
import random
import time
import numpy as np
import torch
from common import ROOT, RUNS, DATA, read, write, sha, now
from data import Store, batches, fixed_noise
from models import build, EMA, sample
from objectives import deterministic_loss, generative_loss
from metrics import stats, concatenate, selection_values, select, rows
from normalization import get as get_normalization

def convergence_decision(history,minimum=48,patience=16,delta=.001):
    best=-float('inf');last_improvement=0
    for r in history:
        if 'selection_CSI' in r and r['selection_CSI']>best+delta:
            best=r['selection_CSI'];last_improvement=r['epoch']
    epoch=history[-1]['epoch'] if history else 0
    return {'stop':epoch>=minimum and epoch-last_improvement>=patience,
            'best_screen_CSI_at_minimum_delta':best,'last_improvement_epoch':last_improvement,
            'patience_used_epochs':epoch-last_improvement,'minimum_epochs':minimum,
            'patience_epochs':patience,'minimum_CSI_improvement':delta}

def configure():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)

def load_model(run_dir, device='cuda'):
    record = read(Path(run_dir)/'summary.json')
    path = Path(run_dir)/'selected.pt'
    assert sha(path) == record['checkpoint_sha256']
    ck = torch.load(path, map_location='cpu', weights_only=False)
    m = build(record['method']).to(device)
    m.load_state_dict(ck['model'])
    return m.eval().requires_grad_(False), record

@torch.no_grad()
def evaluate(model, store, starts, method, base_model=None, batch_size=128, steps=16, member=0):
    model.eval()
    parts=[]
    for starts_batch in batches(starts, batch_size):
        x,y = store.batch(starts_batch)
        n = fixed_noise(starts_batch, member) if method != 'deterministic' else None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            base = base_model(x).float() if base_model is not None else None
            p = sample(model, x, n, method, base, steps)
        # Scores correspond to the FP16 dBZ products that are exported in the final stage.
        p = (p*50).to(torch.float16).float()
        parts.append(stats(p, y*50))
    return concatenate(parts)

def train_run(store, round_id, method, recipe, seed, phase, epochs, base_run=None, batch_size=128):
    configure()
    rd = RUNS/round_id/phase/method/recipe/str(seed)
    rd.mkdir(parents=True, exist_ok=True)
    source_names = ['common.py','data.py','models.py','objectives.py','metrics.py','normalization.py','reuse.py','train.py']
    normalization=get_normalization(round_id)
    formal_policy=read(ROOT/'protocol.json')['formal']
    source_hashes={n:sha(Path(__file__).parent/n) for n in source_names}
    c={'round':round_id,'method':method,'recipe':recipe,'seed':seed,'phase':phase,'epochs':epochs,
       'batch_size':batch_size,'dataset_id':store.manifest['dataset_id'],'source_hashes':source_hashes,
       'protocol_sha256':sha(ROOT/'protocol.json'),'base_run':str(base_run) if base_run else None,
       'learning_rate':.001 if method=='deterministic' else .0003, 'weight_decay':.0001,
       'precision':'BF16 forward, FP32 losses and EMA; exported/scored prediction FP16 dBZ',
       'deterministic_algorithms':True, 'validation_steps':16, 'selection_member':0,
       'target_normalization':normalization if method!='deterministic' else 'physical range only, unchanged control',
       'target_normalization_sha256':sha(ROOT/'inputs'/f'{round_id}_target_normalization.json'),
       'minimum_epochs':epochs if phase=='screen' else formal_policy['min_epochs'],
       'patience_epochs':None if phase=='screen' else formal_policy['patience_epochs'],
       'min_CSI_improvement':None if phase=='screen' else formal_policy['min_CSI_improvement']}
    if (rd/'config.json').exists():
        assert read(rd/'config.json') == c, 'Config/source mismatch: '+str(rd)
    else:
        write(rd/'config.json', c)
    if (rd/'summary.json').exists():
        s=read(rd/'summary.json')
        assert sha(rd/'selected.pt')==s['checkpoint_sha256']
        return rd, s
    from reuse import try_reuse
    reused=try_reuse(store,rd,c)
    if reused is not None:return reused
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=build(method).cuda()
    if method!='deterministic':model.set_normalization(normalization['mean_normalized'],normalization['std_normalized'])
    ema=EMA(model)
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    base_model=None
    if method=='resdiff':
        assert base_run
        base_model,_=load_model(base_run)
    training_starts=store.starts(round_id,'train')
    validation_screen=store.starts(round_id,'screen')
    history=[];first=1;steps_done=0;elapsed_before=0.
    last=rd/'last.pt'
    if last.exists():
        ck=torch.load(last,map_location='cpu',weights_only=False)
        assert ck['config']==c
        model.load_state_dict(ck['model']);ema.model.load_state_dict(ck['ema']);ema.updates=ck['ema_updates']
        opt.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
        history=ck['history'];first=ck['epoch']+1;steps_done=ck['steps_done'];elapsed_before=ck['seconds']
        torch.set_rng_state(ck['torch_rng']);torch.cuda.set_rng_state_all(ck['cuda_rng'])
        random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
    run_begin=time.perf_counter()
    if phase=='formal' and history and convergence_decision(history,c['minimum_epochs'],c['patience_epochs'],c['min_CSI_improvement'])['stop']:
        first=epochs+1
    for epoch in range(first,epochs+1):
        begin=time.perf_counter();model.train();total=0.;n_seen=0
        # This same issue order is used for every method/loss within a seed and epoch.
        for batch_number, starts_batch in enumerate(batches(training_starts,batch_size,True,seed+epoch*100003)):
            x,y=store.batch(starts_batch)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                with torch.no_grad():
                    bp=base_model(x).float() if base_model is not None else None
                loss=deterministic_loss(model(x),y,recipe) if method=='deterministic' else generative_loss(model,x,y,method,recipe,bp)
            assert torch.isfinite(loss), (method,recipe,epoch,batch_number)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert torch.isfinite(norm)
            opt.step();ema.update(model)
            steps_done+=1;n_seen+=len(x);total+=float(loss.detach())*len(x)
            if batch_number%50==0:
                write(rd/'status.json',{'status':'training','pid':os.getpid(),'updated_at':now(),'epoch':epoch,'epochs':epochs,
                       'batch':batch_number+1,'samples_seen':n_seen,'train_samples':len(training_starts),
                       'train_loss_so_far':total/n_seen,'steps_done':steps_done})
        scheduler.step()
        record={'epoch':epoch,'train_loss':total/n_seen,'samples':n_seen,'steps_done':steps_done,
                'learning_rate_next':scheduler.get_last_lr()[0],'train_epoch_seconds':time.perf_counter()-begin}
        assert n_seen==len(training_starts)
        if epoch%4==0 or epoch==epochs:
            vs=evaluate(ema.model,store,validation_screen,method,base_model,batch_size)
            record.update(selection_values(vs))
            save(rd/'candidates'/f'epoch_{epoch:03}.pt',{'model':ema.model.state_dict(),'epoch':epoch,'config':c})
            np.savez_compressed(rd/'candidates'/f'epoch_{epoch:03}_screen.npz',**vs)
        history.append(record)
        save(last,{'config':c,'model':model.state_dict(),'ema':ema.model.state_dict(),'ema_updates':ema.updates,
                   'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'epoch':epoch,'history':history,
                   'steps_done':steps_done,'seconds':elapsed_before+time.perf_counter()-run_begin,
                   'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                   'numpy_rng':np.random.get_state(),'python_rng':random.getstate()})
        write(rd/'history.json',history)
        print('EPOCH',round_id,phase,method,recipe,seed,record,flush=True)
        if phase=='formal' and epoch%4==0:
            audit=convergence_decision(history,c['minimum_epochs'],c['patience_epochs'],c['min_CSI_improvement'])
            write(rd/'convergence.json',audit)
            if audit['stop']:
                print('VALIDATION_EARLY_STOP',round_id,method,recipe,seed,audit,flush=True)
                break
    trained_epochs=history[-1]['epoch']
    # Full validation re-ranks the three best screen checkpoints plus the final epoch.
    remaining=[h for h in history if 'selection_CSI' in h];shortlist=[]
    while remaining and len(shortlist)<3:
        chosen=select(remaining);shortlist.append(chosen);remaining=[h for h in remaining if h['epoch']!=chosen['epoch']]
    if all(h['epoch']!=trained_epochs for h in shortlist):
        shortlist.append(history[-1])
    full=[]
    for checkpoint in shortlist:
        epoch=checkpoint['epoch']; path=rd/'candidates'/f'epoch_{epoch:03}.pt'
        ck=torch.load(path,map_location='cpu',weights_only=False);ema.model.load_state_dict(ck['model'])
        sp=rd/'candidates'/f'epoch_{epoch:03}_full_validation.npz';receipt=sp.with_suffix('.json')
        if receipt.exists():
            rec=read(receipt);assert rec['checkpoint_sha256']==sha(path) and rec['stats_sha256']==sha(sp)
            v={k:a for k,a in np.load(sp).items()}
        else:
            write(rd/'status.json',{'status':'full_validation','pid':os.getpid(),'updated_at':now(),'epoch':epoch,'epochs':epochs})
            v=evaluate(ema.model,store,store.starts(round_id,'valid'),method,base_model,batch_size)
            np.savez_compressed(sp,**v)
            write(receipt,{'checkpoint_sha256':sha(path),'stats_sha256':sha(sp),'created_at':now()})
        full.append({'epoch':epoch,**selection_values(v)})
        print('FULL_VALIDATION',round_id,phase,method,recipe,seed,full[-1],flush=True)
    decision=select(full);epoch=decision['epoch']
    ck=torch.load(rd/'candidates'/f'epoch_{epoch:03}.pt',map_location='cpu',weights_only=False)
    save(rd/'selected.pt',ck)
    stat=np.load(rd/'candidates'/f'epoch_{epoch:03}_full_validation.npz')
    np.savez_compressed(rd/'validation_stats.npz',**dict(stat.items()))
    from common import csvwrite
    csvwrite(rd/'validation_curves.csv',rows(dict(stat.items()),round=round_id,phase=phase,method=method,recipe=recipe,seed=seed))
    summary={'status':'complete','created_at':now(),**c,'parameters':sum(p.numel() for p in model.parameters()),
             'trained_epochs':trained_epochs,'selected_epoch':epoch,'checkpoint_sha256':sha(rd/'selected.pt'),
             'selection':decision,'full_validation_candidates':full,
             'optimization_seconds':elapsed_before+time.perf_counter()-run_begin,
             'checkpoint_selected_on_full_validation_only':True,'test_seen':False,
             'base_checkpoint_sha256':sha(Path(base_run)/'selected.pt') if base_run else None}
    # A late selected epoch is recorded for the later convergence audit, not silently called convergence.
    summary['late_selected_checkpoint']=epoch>=trained_epochs-4
    summary['stopped_early_on_validation']=trained_epochs<epochs
    summary['convergence']=convergence_decision(history,c['minimum_epochs'],c['patience_epochs'],c['min_CSI_improvement']) if phase=='formal' else None
    write(rd/'summary.json',summary)
    write(rd/'status.json',{'status':'complete','pid':os.getpid(),'updated_at':now(),'selected_epoch':epoch})
    del model,ema,opt,scheduler,base_model
    gc.collect();torch.cuda.empty_cache()
    return rd,summary
