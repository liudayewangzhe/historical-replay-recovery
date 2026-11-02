"""Paired client correction and controls matched by exact client-call budget."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from run_recovery import (Engine, load_data, make_schedule, parser as base_parser,
                          partition, secant_transport, sync, write_csv)


CLIENT_OFFSET = 854003
ROUND_OFFSET = 935021
DEFAULT_METHODS = [f'{prefix}{size}' for prefix in ['P','S','BUniform','BRand']
                   for size in [2,4,8]]+['exact']


def method_kind_size(method):
    for prefix in ['BUniform','BRand','P','S','V']:
        if method.startswith(prefix):
            return prefix,int(method[len(prefix):])
    return method,None


def observed_history(engine,start,clients,attack,delay,args):
    state,history = start.clone(),[]
    for j in range(delay):
        t = args.onset+j
        updates = [engine.delta(state,t,m,poison=attack=='backdoor' and m<args.malicious)
                   for m in clients]
        h = torch.stack(updates[args.malicious:]).mean(0)
        if attack=='signflip':
            for m in range(args.malicious): updates[m] = -args.sign_scale*updates[m]
        elif attack=='ipm':
            for m in range(args.malicious): updates[m] = -args.ipm_epsilon*h
        elif attack=='backdoor':
            for m in range(args.malicious): updates[m] *= args.backdoor_scale
        history.append({'w':state.clone(),'h':h.clone(),'updates':torch.stack(updates)})
        state -= torch.stack(updates).mean(0)
        history[-1]['after'] = state.clone()
    return history


def query_plan(method, delay, retained, priorities):
    """Replay indices are 0,...,delay-1; step zero is already exact."""
    n = delay-1
    if method == 'exact':
        return {j:retained for j in range(1,delay)}
    if method in ['E1','E2']:
        return {}
    kind,size = method_kind_size(method)
    if kind in ['P','S','V']:
        return {j:size for j in range(1,delay)}
    budget = size*n
    full, remainder = divmod(budget,retained)
    if kind == 'BUniform':
        # floor(i*n/full), i=1,...,full; includes the last replay step.
        chosen = [(i*n)//full for i in range(1,full+1)]
    else:
        chosen = (np.argsort(priorities[:n])[:full]+1).tolist()
    plan = {j:retained for j in chosen}
    if remainder:
        partial_step = max(set(range(1,delay))-set(chosen))
        plan[partial_step] = remainder
    return dict(sorted(plan.items()))


def empty_cost():
    return dict(recovery_seconds=0.0, client_compute_seconds=0.0,
                server_estimation_seconds=0.0, extra_client_updates=0,
                planned_client_updates=0, full_query_steps=0, partial_query_steps=0,
                query_plan='[]', max_secant_operator_norm=0.0)


def replay(engine, start, history, honest, method, permutations, priorities, args):
    plan = query_plan(method,len(history),len(honest),priorities)
    state, trace, norms = start.clone(), [], []
    calls0, client_seconds = engine.calls, 0.0
    sync(start.device)
    begin = time.perf_counter()
    baseline_variance = ((history[0]['updates'][honest]-history[0]['h']).square().sum(1).mean().item()
                         if method.startswith('V') else None)
    for j, record in enumerate(history):
        size = plan.get(j,0)
        if size:
            selected = permutations[j-1][:size].tolist()
            beta = 0.0 if method.startswith('S') else 1.0
            historical_variance = None
            if method.startswith('V'):
                historical_variance = (record['updates'][honest]-record['h']).square().sum(1).mean().item()
                beta = min(1.0,(baseline_variance/historical_variance)**0.5) if historical_variance else 1.0
            sync(start.device)
            query_begin = time.perf_counter()
            if size == len(honest):
                # Algebraically identical PCC; retain the reference summation order.
                update = engine.aggregate(state,args.onset+j,honest)
            elif method.startswith('S'):
                update = torch.stack([engine.delta(state,args.onset+j,m) for m in selected]).mean(0)
            elif method.startswith('V'):
                current_mean = torch.stack([engine.delta(state,args.onset+j,m) for m in selected]).mean(0)
                stored_mean = record['updates'][selected].mean(0)
                update = current_mean+beta*(record['h']-stored_mean)
            else:
                differences = [engine.delta(state,args.onset+j,m)-record['updates'][m]
                               for m in selected]
                update = record['h']+torch.stack(differences).mean(0)
            sync(start.device)
            client_seconds += time.perf_counter()-query_begin
            trace.append({'step':j,'count':size,'clients':honest if size==len(honest) else selected,
                          'beta':beta,'historical_variance':historical_variance,
                          'baseline_historical_variance':baseline_variance})
        elif method == 'E2' and j:
            correction, op_norm = secant_transport(history,j,state-record['w'],args)
            update = record['h']+correction
            norms.append(op_norm)
        else:
            update = record['h']
        state -= update
    sync(start.device)
    seconds = time.perf_counter()-begin
    cost = dict(recovery_seconds=seconds,client_compute_seconds=client_seconds,
                server_estimation_seconds=seconds-client_seconds,
                extra_client_updates=engine.calls-calls0,
                planned_client_updates=sum(plan.values()),
                full_query_steps=sum(v==len(honest) for v in plan.values()),
                partial_query_steps=sum(v<len(honest) for v in plan.values()),
                query_plan=json.dumps(trace,separators=(',',':')),
                max_secant_operator_norm=max(norms,default=0.0))
    return state,cost


def run(args):
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    honest = list(range(args.malicious,args.clients))
    sample_sizes = [method_kind_size(m)[1] for m in args.methods if method_kind_size(m)[1] is not None]
    if sample_sizes and (min(sample_sizes)<1 or max(sample_sizes)>len(honest)):
        raise ValueError('sample sizes must be between 1 and the retained-client count')
    out = Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    x,y,tx,ty = load_data(args,device)
    methods = args.methods
    metadata = dict(vars(args),methods=methods,torch=torch.__version__,python=platform.python_version(),
                    gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                    train_samples=len(y),test_samples=len(ty),
                    protocol='controlled_oracle_detection_full_participation',
                    pcc='h_j + mean_{m in Q_j}(U_m,j(r_j)-stored_U_m,j(w_j))',
                    sampled_mean_control='S_s directly averages U_m,j(r_j) over the same Q_j; same call count',
                    shrinkage='V_s: current_sample_mean + beta*(h_j-stored_sample_mean); beta=min(1,sqrt(V_0/V_j)); V_j=mean_m norm(stored_u_mj-h_j)^2; beta=1 if V_j=0',
                    client_sampling=f'independent numpy RNG seed+{CLIENT_OFFSET}+replay_seed_offset, one permutation per replay step; prefixes nested by size',
                    round_sampling=f'independent priorities from numpy RNG seed+{ROUND_OFFSET}+replay_seed_offset; choose lowest priorities in available steps',
                    uniform_rounds='floor(i*(delay-1)/full_rounds), i=1,...,full_rounds; remainder uses last unselected step',
                    timing='synchronized replay incl paired delta arithmetic; excludes retained aggregate formation and test evaluation')
    (out/'config.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    rows,trajectories = [],[]
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        parts = partition(y.cpu().numpy(),args.clients,args.alpha,seed)
        schedule = make_schedule(parts,args,seed,device)
        engine = Engine(args,x,y,tx,ty,schedule)
        clients = list(range(args.clients))
        client_rng = np.random.default_rng(seed+CLIENT_OFFSET+args.replay_seed_offset)
        permutations = [client_rng.permutation(honest) for _ in range(max(args.delays)-1)]
        priorities = np.random.default_rng(seed+ROUND_OFFSET+args.replay_seed_offset).random(max(args.delays)-1)
        start = engine.vector()
        for t in range(args.onset):
            start -= engine.aggregate(start,t,clients)
        clean_states = {}
        if 'clean' in methods:
            clean = start.clone()
            horizon = args.onset+max(args.delays) if args.recovery_only else args.rounds
            for t in range(args.onset,horizon):
                clean -= engine.aggregate(clean,t,clients)
                if t+1-args.onset in args.delays or t+1==args.rounds:
                    clean_states[t+1] = clean.clone()
        for attack in args.attacks:
            history_all = observed_history(engine,start,clients,attack,max(args.delays),args)
            for delay in args.delays:
                end = args.onset+delay
                history = history_all[:delay]
                corrupt = history[-1]['after'].clone()
                exact,exact_cost = replay(engine,start,history,honest,'exact',permutations,priorities,args)
                states,costs = {},{}
                for method in methods:
                    if method=='exact':
                        states[method],costs[method] = exact.clone(),exact_cost
                    elif method in ['filter','rollback','clean']:
                        source = {'filter':corrupt,'rollback':start,'clean':clean_states.get(end)}[method]
                        states[method],costs[method] = source.clone(),empty_cost()
                    else:
                        states[method],costs[method] = replay(
                            engine,start,history,honest,method,permutations,priorities,args)
                denominator = torch.linalg.vector_norm(corrupt-exact).item()
                base = dict(dataset=args.dataset if not args.smoke else 'synthetic_smoke',
                            seed=seed,replay_seed_offset=args.replay_seed_offset,attack=attack,delay=delay,onset=args.onset,detected_round=end,
                            parameters=len(start),buffer_bytes=delay*(args.clients+1)*len(start)*start.element_size())
                summary = {}
                for method,state in states.items():
                    metrics = engine.metrics(state,exact,attack)
                    bitwise = bool(torch.equal(state,exact))
                    rows.append(dict(base,method=method,stage='recovery',round=end,
                        recovery_fraction=(1-metrics['parameter_residual']/denominator) if denominator else float('nan'),
                        bitwise_equal_exact=bitwise,**costs[method],**metrics))
                    trajectories.append(dict(base,method=method,round=end,**metrics))
                    summary[method] = {'residual':metrics['parameter_residual'],
                                       'calls':costs[method]['extra_client_updates']}
                if not args.recovery_only:
                    states.pop('clean',None)
                    if 'exact' not in states:
                        states['exact'] = exact.clone()
                    for t in range(end,args.rounds):
                        for method in states:
                            states[method] -= engine.aggregate(states[method],t,honest)
                        if (t+1-end)%args.eval_every==0 or t+1==args.rounds:
                            for method in methods:
                                if method!='clean':
                                    trajectories.append(dict(base,method=method,round=t+1,
                                        **engine.metrics(states[method],states['exact'],attack)))
                    if 'clean' in methods:
                        states['clean'] = clean_states[args.rounds]
                    for method in methods:
                        metrics = engine.metrics(states[method],states['exact'],attack)
                        rows.append(dict(base,method=method,stage='final',round=args.rounds,
                            recovery_fraction=float('nan'),bitwise_equal_exact=bool(torch.equal(states[method],states['exact'])),
                            **costs[method],**metrics))
                write_csv(out/'results.csv',rows)
                write_csv(out/'trajectories.csv',trajectories)
                print(json.dumps({'seed':seed,'attack':attack,'delay':delay,
                                  'recovery':summary,'rows_saved':len(rows)}),flush=True)
    return rows


if __name__=='__main__':
    parser = base_parser()
    parser.description = __doc__
    parser.add_argument('--methods',nargs='+',default=DEFAULT_METHODS,
        choices=['filter','rollback','E1','E2','exact','clean']+
                [f'{prefix}{size}' for prefix in ['P','S','V','BUniform','BRand'] for size in [1,2,3,4,8,16]])
    parser.add_argument('--recovery-only',action='store_true')
    parser.add_argument('--replay-seed-offset',type=int,default=0)
    args = parser.parse_args()
    if args.smoke:
        args.seeds,args.attacks,args.delays=[7],['signflip'],[1,2,4]
        args.rounds,args.onset,args.clients,args.malicious=7,1,4,1
        args.local_steps,args.batch,args.eval_every=1,16,1
        args.methods=['filter','E1','E2','exact']+[f'{prefix}{size}'
                     for prefix in ['P','S','V','BUniform','BRand'] for size in [1,2,3]]
    run(args)
