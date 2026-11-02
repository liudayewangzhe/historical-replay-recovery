"""P0 covariance-rule comparison, with optional isolated population probes."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from run_recovery import (Engine, load_data, make_schedule, parser as base_parser,
                          partition, sync, write_csv)
from run_paired import CLIENT_OFFSET, observed_history


DEFAULT_METHODS = ([f'{p}{s}' for p in ['P','V','S'] for s in [1,2,4]]+
                   [f'{p}{s}' for p in ['Lag','Switch','Fit'] for s in [2,4]]+
                   ['Moment1','Moment2','Moment4','filter','exact'])


def observed_history_cpu(engine,start,clients,attack,delay,args):
    state=start.clone();history=[]
    for j in range(delay):
        t=args.onset+j
        updates=torch.empty((len(clients),state.numel()),dtype=state.dtype,device='cpu')
        honest_sum=torch.zeros_like(state)
        for m in clients:
            delta=engine.delta(state,t,m,poison=attack=='backdoor' and m<args.malicious)
            updates[m].copy_(delta.detach(),non_blocking=False)
            if m>=args.malicious: honest_sum.add_(delta)
        h=honest_sum/(len(clients)-args.malicious)
        if attack=='signflip':
            updates[:args.malicious].mul_(-args.sign_scale)
        elif attack=='ipm':
            updates[:args.malicious].copy_((-args.ipm_epsilon*h).cpu().expand(args.malicious,-1))
        elif attack=='backdoor':
            updates[:args.malicious].mul_(args.backdoor_scale)
        w=state.detach().cpu().clone()
        aggregate=torch.zeros_like(state)
        for m in clients: aggregate.add_(updates[m].to(state.device))
        state-=aggregate/len(clients)
        h_cpu=h.detach().cpu().clone()
        variance=sum((updates[m]-h_cpu).square().sum().item() for m in clients if m>=args.malicious)/(len(clients)-args.malicious)
        history.append({'w':w,'h':h_cpu,'updates':updates,'variance':variance,'after':state.detach().cpu().clone()})
    return history


def aggregate_stream(engine,state,t,clients):
    """Sequential sum avoids holding a full client's parameter vectors on GPU."""
    total=torch.zeros_like(state)
    for m in clients: total.add_(engine.delta(state,t,m))
    return total/len(clients)


def history_variance_cpu(record,honest):
    if 'variance' not in record:
        h=record['h']
        record['variance']=sum((record['updates'][m]-h).square().sum().item() for m in honest)/len(honest)
    return record['variance']


def kind_size(method):
    if 'Init1_' in method:
        kind,size=method.split('Init1_',1)
        return kind,int(size)
    for prefix in ['Switch','Moment','Lag','Fit','P','V','S']:
        if method.startswith(prefix):
            return prefix,int(method[len(prefix):])
    return method,0


def moments(x,y,denominator):
    xc,yc=x-x.mean(0),y-y.mean(0)
    return {'vx':(xc.square().sum()/denominator).item(),
            'vy':(yc.square().sum()/denominator).item(),
            'cov':((xc*yc).sum()/denominator).item()}


def fitted_beta(previous):
    if previous is None or previous['vy']==0:
        return 0.0
    return float(np.clip(previous['cov']/previous['vy'],0,1))


def schedule_for_run(parts,args,seed,device):
    if args.local_epochs==0:
        return make_schedule(parts,args,seed,device)
    rng=np.random.default_rng(seed+7201)
    batches_flat=[]
    lengths=[]
    layout=[]
    for _ in range(args.rounds):
        round_layout=[]
        for part in parts:
            client_layout=[]
            for _ in range(args.local_epochs):
                order=rng.permutation(part)
                for i in range(0,len(order),args.batch):
                    batch=order[i:i+args.batch]
                    batches_flat.append(batch)
                    client_layout.append(len(lengths))
                    lengths.append(len(batch))
            round_layout.append(client_layout)
        layout.append(round_layout)
    if not batches_flat:
        return []
    flat=torch.from_numpy(np.concatenate(batches_flat)).to(device=device,dtype=torch.long)
    batches=list(flat.split(lengths))
    return [[[batches[i] for i in client] for client in round_layout] for round_layout in layout]


def replay(engine,start,history,honest,method,permutations,args,common):
    kind,size=kind_size(method)
    if kind=='exact': size=len(honest)
    initial_beta=1.0 if 'Init1_' in method else args.initial_beta
    state=start.clone()
    previous=None
    previous_moment=None
    v0=(history_variance_cpu(history[0],honest) if args.history_cpu else
        (history[0]['updates'][honest]-history[0]['h']).square().sum(1).mean().item())
    algorithm_calls=diagnostic_calls=0
    algorithm_seconds=client_seconds=diagnostic_seconds=0.0
    traces,diagnostics=[],[]
    for j,stored in enumerate(history):
        record=({**stored,'h':stored['h'].to(start.device),'w':stored['w'].to(start.device)}
                if args.history_cpu else stored)
        sync(start.device);tic=time.perf_counter()
        old_all=record['updates'][honest] if not args.history_cpu else None
        vy_history=(history_variance_cpu(stored,honest) if args.history_cpu else
                    (old_all-record['h']).square().sum(1).mean().item())
        selected=[];new_sample=None;sample_stats=None;crossmoment=None
        if j==0:
            beta=1.0;estimate=record['h']
        else:
            if kind=='P': beta=1.0
            elif kind=='V': beta=min(1.0,(v0/vy_history)**.5) if vy_history else 1.0
            elif kind=='Lag': beta=initial_beta if j==1 else fitted_beta(previous)
            elif kind=='Moment': beta=initial_beta if j==1 else fitted_beta(previous_moment)
            elif kind=='Switch': beta=initial_beta if j==1 else float(previous is not None and 2*previous['cov']>previous['vy'])
            else: beta=0.0
            # All methods share this pre-generated subset. Full queries use the reference order.
            selected=honest if size==len(honest) else permutations[j-1][:size].tolist()
            before=engine.calls
            sync(start.device);query_tic=time.perf_counter()
            if args.history_cpu and size==len(honest):
                # Welford's update records the same centered sample moments
                # without retaining all full-model vectors on the accelerator.
                mx=torch.zeros_like(state);my=torch.zeros_like(state)
                m2x=m2y=cross=0.0;cross_sum=0.0
                sample_cpu=[] if args.diagnostics else None
                for count,m in enumerate(selected,1):
                    x=engine.delta(state,args.onset+j,m)
                    y=record['updates'][m].to(start.device)
                    dx=x-mx;dy=y-my
                    mx.add_(dx/count);my.add_(dy/count)
                    m2x+=torch.dot(dx,x-mx).item();m2y+=torch.dot(dy,y-my).item()
                    cross+=torch.dot(dx,y-my).item()
                    cross_sum+=torch.dot(x-y,y-record['h']).item()
                    if sample_cpu is not None: sample_cpu.append(x.detach().cpu())
                new_sample=None
                sample_stats=({'vx':m2x/(size-1),'vy':m2y/(size-1),'cov':cross/(size-1)}
                              if size>=2 else None)
                crossmoment=vy_history+cross_sum/size
                estimate=mx
                if kind=='Fit': beta=fitted_beta(sample_stats)
            else:
                new_sample=torch.stack([engine.delta(state,args.onset+j,m) for m in selected])
            sync(start.device);client_seconds+=time.perf_counter()-query_tic
            algorithm_calls+=engine.calls-before
            if new_sample is not None:
                old_sample=record['updates'][selected].to(start.device)
                crossmoment=vy_history+((new_sample-old_sample)*(old_sample-record['h'])).sum(1).mean().item()
                if size>=2:
                    sample_stats=moments(new_sample,old_sample,size-1)
                if kind=='Fit': beta=fitted_beta(sample_stats)
                if size==len(honest): estimate=new_sample.mean(0)
                elif kind=='P': estimate=record['h']+(new_sample-old_sample).mean(0)
                elif kind=='S': estimate=new_sample.mean(0)
                else: estimate=new_sample.mean(0)+beta*(record['h']-old_sample.mean(0))
        next_state=state-estimate
        sync(start.device);algorithm_seconds+=time.perf_counter()-tic
        traces.append(dict(common,method=method,step=j,round=args.onset+j,
            selected_clients=';'.join(map(str,selected)),query_count=len(selected),beta=beta,
            beta_rule=kind,beta_chosen_before_current_sample=kind!='Fit',
            sample_covariance=sample_stats['cov'] if sample_stats else float('nan'),
            sample_old_variance=sample_stats['vy'] if sample_stats else float('nan'),
            previous_sample_covariance=previous['cov'] if previous else float('nan'),
            previous_sample_old_variance=previous['vy'] if previous else float('nan'),
            sample_crossmoment_covariance=crossmoment if crossmoment is not None else float('nan'),
            previous_crossmoment_covariance=previous_moment['cov'] if previous_moment else float('nan'),
            previous_full_history_variance=previous_moment['vy'] if previous_moment else float('nan'),
            beta_source=('previous_crossmoment_and_full_old_variance' if kind=='Moment' else
                         'previous_centered_sample' if kind in ['Lag','Switch'] else
                         'current_centered_sample_biased' if kind=='Fit' else 'fixed_or_saved_history'),
            history_variance=vy_history,initial_history_variance=v0,
            replay_anchor_distance=torch.linalg.vector_norm(state-record['w']).item()))
        if args.diagnostics:
            before=engine.calls
            sync(start.device);probe_tic=time.perf_counter()
            if args.history_cpu:
                old_all=stored['updates'][honest]
            if j==0:
                new_all=old_all
            elif args.history_cpu and size==len(honest):
                new_all=torch.stack(sample_cpu)
            elif args.history_cpu:
                known={m:new_sample[i].detach().cpu() for i,m in enumerate(selected)}
                new_all=torch.stack([known[m] if m in known else engine.delta(state,args.onset+j,m).detach().cpu()
                                     for m in honest])
            else:
                known={m:new_sample[i] for i,m in enumerate(selected)}
                new_all=torch.stack([known[m] if m in known else engine.delta(state,args.onset+j,m)
                                     for m in honest])
            population=moments(new_all,old_all,len(honest))
            gain=beta*beta*population['vy']-2*beta*population['cov']
            factor=(len(honest)-size)/(size*(len(honest)-1)) if j else 0.0
            centered_residual=(new_all-new_all.mean(0))-beta*(old_all-old_all.mean(0))
            residual_population_variance=centered_residual.square().sum(1).mean().item()
            diagnostics.append(dict(common,method=method,step=j,round=args.onset+j,beta=beta,
                VX=population['vx'],VY=population['vy'],C=population['cov'],
                fixed_beta_variance_gain=gain,fixed_beta_gain_negative=gain<0,
                finite_population_factor=factor,
                fixed_beta_estimator_mse=factor*residual_population_variance,
                fixed_beta_variance_formula_valid=kind!='Fit',
                realized_squared_estimation_error=(estimate.to(new_all.device)-new_all.mean(0)).square().sum().item(),
                full_population_beta_star=float(np.clip(population['cov']/population['vy'],0,1)) if population['vy'] else 0.0,
                diagnostic_extra_client_updates=engine.calls-before))
            sync(start.device);diagnostic_seconds+=time.perf_counter()-probe_tic
            diagnostic_calls+=engine.calls-before
        # Only the queried sample, never the population probe, enters the next rule.
        previous=sample_stats
        previous_moment={'cov':crossmoment,'vy':vy_history} if crossmoment is not None else None
        state=next_state
    costs=dict(recovery_seconds=algorithm_seconds,client_compute_seconds=client_seconds,
               server_estimation_seconds=algorithm_seconds-client_seconds,
               extra_client_updates=algorithm_calls,diagnostic_extra_client_updates=diagnostic_calls,
               diagnostic_seconds=diagnostic_seconds,planned_client_updates=size*(len(history)-1))
    return state,costs,traces,diagnostics


def zero_cost():
    return dict(recovery_seconds=0.0,client_compute_seconds=0.0,server_estimation_seconds=0.0,
                extra_client_updates=0,diagnostic_extra_client_updates=0,diagnostic_seconds=0.0,
                planned_client_updates=0)


def run(args):
    device=torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    honest=list(range(args.malicious,args.clients))
    for method in args.methods:
        kind,size=kind_size(method)
        if kind in ['Lag','Switch','Fit'] and size<2:
            raise ValueError('Covariance rules require at least two sampled clients.')
        if size>len(honest): raise ValueError('Sample size exceeds retained client count.')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    x,y,tx,ty=load_data(args,device)
    checkpoint=(torch.load(args.warmup_checkpoint,map_location=device,weights_only=True)
                if args.warmup_checkpoint else None)
    cifar_resnet_models={'ResNet18-GroupNorm-CIFAR','ResNet18-GroupNorm-CIFAR-ImageNet1K-init'}
    if checkpoint and checkpoint.get('model') in cifar_resnet_models:
        mean=torch.tensor([.4914,.4822,.4465],device=device).view(1,3,1,1)
        std=torch.tensor([.247,.243,.261],device=device).view(1,3,1,1)
        x.sub_(mean).div_(std);tx.sub_(mean).div_(std)
    metadata=dict(vars(args),torch=torch.__version__,python=platform.python_version(),
        checkpoint_model=checkpoint.get('model') if checkpoint else None,
        checkpoint_initialization=checkpoint.get('initialization') if checkpoint else None,
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        train_samples=len(y),test_samples=len(ty),retained_clients=len(honest),
        coefficient_rules={'P':'1','V':'min(1,sqrt(V_initial/V_j)); V_j=0 gives 1','S':'0',
            'Lag':'clip(previous queried sample covariance / previous queried sample old variance,0,1)',
            'Moment':'clip(previous crossmoment covariance / previous full historical variance,0,1); Chat=VY_full+mean_Q dot(X-Y,Y-h)',
            'Switch':'1 iff 2*previous queried sample covariance > previous queried sample old variance',
            'Fit':'clip(current queried sample covariance / current queried sample old variance,0,1); sample-dependent biased control'},
        lag_initialization='beta=initial_beta at first queried step; Init1_ aliases override it to 1; no previous sample is fabricated',
        covariance_conventions='centered queried sample uses s-1; full-population diagnostic uses K; crossmoment uses full historical VY plus an s-sample mean, valid for s=1',
        diagnostics_policy='reuse algorithm queries and query remaining clients only for probes; never update beta from probes; exclude probe time/calls from method cost',
        fixed_beta_variance='(K-s)/(s*(K-1)) * (VX + beta^2 VY - 2 beta C); not a conditional variance identity for Fit',
        local_training='complete shuffled local epochs' if args.local_epochs else 'fixed number of sampled local minibatches')
    (out/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    rows,trajectories,traces,diagnostics,warmups=[],[],[],[],[]
    for seed in args.seeds:
        torch.manual_seed(seed);np.random.seed(seed)
        parts=partition(y.cpu().numpy(),args.clients,args.alpha,seed)
        schedule=schedule_for_run(parts,args,seed,device)
        engine=Engine(args,x,y,tx,ty,schedule)
        if checkpoint and checkpoint.get('model') in cifar_resnet_models:
            from torch import nn
            from torchvision.models import resnet18
            model=resnet18(num_classes=10,norm_layer=lambda c:nn.GroupNorm(8,c))
            model.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False)
            model.maxpool=nn.Identity()
            engine.model=model.to(device)
        clients=list(range(args.clients))
        rng=np.random.default_rng(seed+CLIENT_OFFSET+args.replay_seed_offset)
        permutations=[rng.permutation(honest) for _ in range(max(args.delays)-1)]
        start=engine.vector()
        if args.warmup_checkpoint:
            if (checkpoint['seed'],checkpoint['round'],checkpoint['local_epochs'],checkpoint['lr'],checkpoint['clients'],checkpoint['alpha'],checkpoint['batch']) != (seed,args.onset,args.local_epochs,args.lr,args.clients,args.alpha,args.batch):
                raise ValueError('Warmup checkpoint does not match the experiment protocol')
            start=checkpoint['state'].to(device)
        else:
            for t in range(args.onset): start-=engine.aggregate(start,t,clients)
        warmups.append(dict(seed=seed,round=args.onset,accuracy=(engine.predictions(start)==ty).float().mean().item(),
                            client_update_calls=(args.onset*args.clients if args.warmup_checkpoint else engine.calls),
                            warmup_source=('checkpoint' if args.warmup_checkpoint else 'fresh'),training_samples=len(y)))
        write_csv(out/'warmup.csv',warmups)
        clean_states={}
        if 'clean' in args.methods:
            clean=start.clone()
            horizon=args.onset+max(args.delays) if args.recovery_only else args.rounds
            for t in range(args.onset,horizon):
                clean-=(aggregate_stream(engine,clean,t,clients) if args.history_cpu else engine.aggregate(clean,t,clients))
                if t+1-args.onset in args.delays or t+1==args.rounds: clean_states[t+1]=clean.clone()
        for attack in args.attacks:
            history_all=(observed_history_cpu if args.history_cpu else observed_history)(engine,start,clients,attack,max(args.delays),args)
            for delay in args.delays:
                end=args.onset+delay;history=history_all[:delay];corrupt=history[-1]['after'].to(device).clone()
                common=dict(dataset=args.dataset if not args.smoke else 'synthetic_smoke',seed=seed,
                    replay_seed_offset=args.replay_seed_offset,attack=attack,delay=delay)
                exact,exact_cost,exact_trace,exact_diag=replay(engine,start,history,honest,'exact',permutations,args,common)
                states,costs={},{}
                for method in args.methods:
                    if method=='exact':
                        states[method],costs[method]=exact.clone(),exact_cost
                        traces.extend(exact_trace);diagnostics.extend(exact_diag)
                    elif method in ['filter','rollback','clean']:
                        source={'filter':corrupt,'rollback':start,'clean':clean_states.get(end)}[method]
                        states[method],costs[method]=source.clone(),zero_cost()
                    else:
                        state,cost,trace,diagnostic=replay(engine,start,history,honest,method,permutations,args,common)
                        states[method],costs[method]=state,cost
                        traces.extend(trace);diagnostics.extend(diagnostic)
                denominator=torch.linalg.vector_norm(corrupt-exact).item()
                base=dict(common,onset=args.onset,detected_round=end,parameters=len(start),
                          buffer_bytes=delay*(args.clients+1)*len(start)*start.element_size())
                summary={}
                for method,state in states.items():
                    metrics=engine.metrics(state,exact,attack)
                    rows.append(dict(base,method=method,stage='recovery',round=end,
                        recovery_fraction=(1-metrics['parameter_residual']/denominator) if denominator else float('nan'),
                        bitwise_equal_exact=bool(torch.equal(state,exact)),**costs[method],**metrics))
                    trajectories.append(dict(base,method=method,round=end,**metrics))
                    summary[method]={'residual':metrics['parameter_residual'],'calls':costs[method]['extra_client_updates'],
                                     'diagnostic_calls':costs[method]['diagnostic_extra_client_updates']}
                write_csv(out/'results.csv',rows);write_csv(out/'trajectories.csv',trajectories)
                write_csv(out/'coefficient_trace.csv',traces)
                if args.diagnostics: write_csv(out/'covariance_diagnostics.csv',diagnostics)
                print(json.dumps(dict(seed=seed,attack=attack,delay=delay,stage='recovery',rows_saved=len(rows))),flush=True)
                if not args.recovery_only:
                    states.pop('clean',None)
                    if 'exact' not in states: states['exact']=exact.clone()
                    for t in range(end,args.rounds):
                        for method in states:
                            states[method]-=(aggregate_stream(engine,states[method],t,honest)
                                             if args.history_cpu else engine.aggregate(states[method],t,honest))
                        if (t+1-end)%args.eval_every==0 or t+1==args.rounds:
                            for method in args.methods:
                                if method!='clean': trajectories.append(dict(base,method=method,round=t+1,
                                    **engine.metrics(states[method],states['exact'],attack)))
                    if 'clean' in args.methods: states['clean']=clean_states[args.rounds]
                    for method in args.methods:
                        metrics=engine.metrics(states[method],states['exact'],attack)
                        rows.append(dict(base,method=method,stage='final',round=args.rounds,
                            recovery_fraction=float('nan'),bitwise_equal_exact=bool(torch.equal(states[method],states['exact'])),
                            **costs[method],**metrics))
                write_csv(out/'results.csv',rows);write_csv(out/'trajectories.csv',trajectories)
                write_csv(out/'coefficient_trace.csv',traces)
                if args.diagnostics: write_csv(out/'covariance_diagnostics.csv',diagnostics)
                print(json.dumps(dict(seed=seed,attack=attack,delay=delay,recovery=summary,rows_saved=len(rows))),flush=True)
    return rows


def parser():
    p=base_parser()
    p.description=__doc__
    p.set_defaults(fraction=1.0,clients=50,malicious=10,delays=[8,16,32],onset=100,rounds=132,
                   ipm_epsilon=.5,backdoor_scale=1.0,seeds=[101,202,303,404,505],attacks=['ipm'])
    p.add_argument('--methods',nargs='+',default=DEFAULT_METHODS,
        choices=['filter','rollback','clean','exact']+[f'{k}{s}' for k in ['P','V','S'] for s in [1,2,3,4,8,16,40]]+
                [f'{k}{s}' for k in ['Lag','Switch','Fit'] for s in [2,3,4,8,16,40]]+
                [f'Moment{s}' for s in [1,2,3,4,8,16,40]]+
                ['LagInit1_2','MomentInit1_1','MomentInit1_2','SwitchInit1_2'])
    p.add_argument('--diagnostics',action='store_true')
    p.add_argument('--recovery-only',action='store_true')
    p.add_argument('--replay-seed-offset',type=int,default=0)
    p.add_argument('--initial-beta',type=float,default=0.0)
    p.add_argument('--local-epochs',type=int,default=0,help='0 uses --local-steps; positive values train complete shuffled local epochs')
    p.add_argument('--warmup-checkpoint',default=None,help='Resume from a saved matching federated warmup checkpoint')
    p.add_argument('--history-cpu',action='store_true',help='Keep historical client vectors in host memory; stream per-client updates')
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.smoke:
        args.seeds,args.attacks,args.delays=[7],['ipm'],[1,2,4]
        args.rounds,args.onset,args.clients,args.malicious=7,1,4,1
        args.local_steps,args.batch,args.eval_every=1,16,1
        args.methods=['P1','V1','S1','Moment1','P2','V2','S2','Lag2','Moment2','Switch2','Fit2',
                      'P3','V3','S3','Lag3','Moment3','Switch3','Fit3','filter','exact']
    run(args)
