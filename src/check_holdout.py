"""Verify saved split, per-class counts and replay schedule prefix."""
import argparse
import numpy as np
import torch
from holdout_data import datasets,stratified_indices
from run_recovery import partition
from run_covariance import schedule_for_run


def main():
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('--data',default='data')
    a=p.parse_args();c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    dataset,_=datasets(c['dataset'],a.data);labels=np.asarray(dataset.targets)
    train=c['train_indices'].numpy();val=c['validation_indices'].numpy()
    expected_train,expected_val=stratified_indices(labels,c['split_seed'],c['validation_fraction'],c['examples_per_class'])
    assert np.array_equal(train,expected_train) and np.array_equal(val,expected_val)
    assert len(np.intersect1d(train,val))==0
    print('train class counts',np.bincount(labels[train],minlength=10).tolist())
    print('validation class counts',np.bincount(labels[val],minlength=10).tolist())
    parts=partition(labels[train],c['clients'],c['alpha'],c['seed'])
    assert np.array_equal(np.sort(np.concatenate(parts)),np.arange(len(train)))
    class Args:pass
    args=Args();args.local_epochs=c['local_epochs'];args.local_steps=c['local_steps']
    args.batch=c['batch'];args.rounds=c['round']+2
    full=schedule_for_run(parts,args,c['seed'],torch.device('cpu'))
    args.rounds=c['round'];prefix=schedule_for_run(parts,args,c['seed'],torch.device('cpu'))
    for t in range(args.rounds):
        for m in range(c['clients']):
            assert all(torch.equal(a,b) for a,b in zip(full[t][m],prefix[t][m]))
    print('checkpoint split, disjoint indices, partition and schedule-prefix checks passed')


if __name__=='__main__':main()
