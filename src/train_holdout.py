"""Federated training with train-only stratified validation model selection."""
import csv,json,time
from pathlib import Path
import numpy as np
import torch
from run_recovery import Engine,partition,parser as base_parser
from run_covariance import schedule_for_run,aggregate_stream
from holdout_data import datasets,stratified_indices,tensors,normalize_cifar,normalize_for


def main():
    p=base_parser()
    p.set_defaults(fraction=1.)
    p.add_argument('--local-epochs',type=int,default=1)
    p.add_argument('--eval-interval',type=int,default=5)
    p.add_argument('--target-accuracy',type=float,default=.85)
    p.add_argument('--split-seed',type=int,default=20261030)
    p.add_argument('--validation-fraction',type=float,default=.1)
    p.add_argument('--examples-per-class',type=int,default=0)
    p.add_argument('--initialization',choices=['random','imagenet1k_gn'],default='random')
    args=p.parse_args()
    device=torch.device(args.device);torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train_data,test_data=datasets(args.dataset,args.data)
    indices,val_indices=stratified_indices(train_data.targets,args.split_seed,args.validation_fraction,args.examples_per_class)
    x,y=tensors(train_data,indices,device);vx,vy=tensors(train_data,val_indices,device)
    if args.dataset in ('cifar10','cifar100'): normalize_for(args.dataset,x);normalize_for(args.dataset,vx)
    metadata=dict(dataset=args.dataset,split_seed=args.split_seed,validation_fraction=args.validation_fraction,
                  examples_per_class=args.examples_per_class,train_size=len(indices),validation_size=len(val_indices),
                  official_test_used_for_selection=False,full_participation=True,
                  random_schedule='seed+7201; same client split and batches for all replay methods',
                  args=vars(args))
    (out/'protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    for seed in args.seeds:
        torch.manual_seed(seed);np.random.seed(seed)
        parts=partition(y.cpu().numpy(),args.clients,args.alpha,seed)
        schedule=schedule_for_run(parts,args,seed,device)
        engine=Engine(args,x,y,vx,vy,schedule)
        model_name='SmallCNN-FashionMNIST';init={'source':'random'}
        if args.dataset in ('cifar10','cifar100'):
            n_classes=100 if args.dataset=='cifar100' else 10
            if args.initialization=='imagenet1k_gn':
                from imagenet_gn_init import build_imagenet_gn
                net,init=build_imagenet_gn(num_classes=n_classes);model_name='ResNet18-GroupNorm-CIFAR-ImageNet1K-init'
            else:
                from torchvision.models import resnet18
                from torch import nn
                net=resnet18(num_classes=n_classes,norm_layer=lambda c:nn.GroupNorm(8,c))
                net.conv1=nn.Conv2d(3,64,3,stride=1,padding=1,bias=False);net.maxpool=nn.Identity()
                model_name='ResNet18-GroupNorm-CIFAR'
            engine.model=net.to(device)
        state=engine.vector();best=-1.;best_round=0;start=time.monotonic()
        checkpoint=out/f'checkpoint_seed{seed}.pt'
        path=out/'validation_curve.csv'
        for t in range(args.rounds):
            state-=aggregate_stream(engine,state,t,list(range(args.clients)))
            if (t+1)%args.eval_interval==0 or t+1==args.rounds:
                accuracy=(engine.predictions(state)==vy).float().mean().item()
                row=dict(seed=seed,round=t+1,validation_accuracy=accuracy,client_updates=engine.calls,
                         elapsed=time.monotonic()-start,local_epochs=args.local_epochs,lr=args.lr)
                with path.open('a',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row))
                    if f.tell()==0:writer.writeheader()
                    writer.writerow(row)
                if accuracy>best:
                    best=accuracy;best_round=t+1
                    torch.save(dict(state=state.cpu(),seed=seed,round=best_round,validation_accuracy=best,
                        local_epochs=args.local_epochs,local_steps=args.local_steps,lr=args.lr,clients=args.clients,
                        alpha=args.alpha,batch=args.batch,model=model_name,initialization=init,dataset=args.dataset,
                        split_seed=args.split_seed,validation_fraction=args.validation_fraction,
                        examples_per_class=args.examples_per_class,
                        train_indices=torch.from_numpy(indices.copy()),validation_indices=torch.from_numpy(val_indices.copy()),
                        full_participation=True),checkpoint)
                print(json.dumps(row),flush=True)
                if accuracy>=args.target_accuracy:break
        chosen=torch.load(checkpoint,map_location=device,weights_only=True)
        tx,ty=tensors(test_data,None,device)
        if args.test_samples:tx,ty=tx[:args.test_samples],ty[:args.test_samples]
        if args.dataset in ('cifar10','cifar100'):normalize_for(args.dataset,tx)
        engine.tx,engine.ty=tx,ty
        test_accuracy=(engine.predictions(chosen['state'])==ty).float().mean().item()
        chosen['official_test_accuracy']=test_accuracy
        chosen['state']=chosen['state'].cpu()
        torch.save(chosen,checkpoint)
        report=dict(seed=seed,selected_round=best_round,validation_accuracy=best,
                    official_test_accuracy=test_accuracy,selection='first validation target, else best validation',
                    official_test_used_for_selection=False)
        (out/f'selected_seed{seed}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
