import numpy as np
import bayes_opt
import draccus
import yaml
from copy import copy, deepcopy

import peft
import train
import plot

    
def maximize_hyper(cfg):
    def pmap(alpha, gamma, peak_learning_rate):
        return (float(np.log(alpha)), float(np.log(gamma)+0.5*np.log(peak_learning_rate)), float(np.log(peak_learning_rate)))
    def imap(logalpha,loggammalr,loglr):
        return (float(np.exp(logalpha)), float(np.exp(loggammalr-0.5*loglr)), float(np.exp(loglr)))
    def lossmap(loss):
        return -min(loss,2.)+1
    
    def f(logalpha,loggammalr,loglr):
        c = copy(cfg)
        c.peft_config.alpha, c.peft_config.gamma, c.peak_learning_rate = imap(logalpha,loggammalr,loglr)
        
        cur = yaml.dump(draccus.encode(c), default_flow_style=False, sort_keys=False)
        outf = 'configs/'+hex(abs(hash(cur))).lstrip('0x')+'.yaml'
        print(f'writing tested config to {outf}')
        open(outf,'w').write(cur)

        _, train_loss, test_loss = train.train_peft(c)
        return lossmap(train_loss)

    theta0 = (cfg.peft_config.alpha, cfg.peft_config.gamma, cfg.peak_learning_rate)
    ptheta0 = pmap(*theta0)
    ptheta0 = (ptheta0[0], 0., ptheta0[2])
    
    bo = bayes_opt.BayesianOptimization(
            f=f,
            # acquisition_function=bayes_opt.acquisition.ProbabilityOfImprovement(xi=1e-4), # prefer exploitation
            # acquisition_function=bayes_opt.acquisition.ExpectedImprovement(xi=0.0), # prefer exploitation
            acquisition_function=bayes_opt.acquisition.UpperConfidenceBound(kappa=0.1), # prefer exploitation
            pbounds = {"logalpha": (2, 7),
                       "loggammalr": (-1, 1.5),
                       "loglr": (-12.7, -8.1)},
            allow_duplicate_points = True,
            # bounds_transformer = bayes_opt.SequentialDomainReductionTransformer(minimum_window=0.5)
            )
    
    def normalize_cfg(cfg_dict):
        cfg_dict = deepcopy(cfg_dict)
        del cfg_dict['peft_config']['alpha']
        del cfg_dict['peft_config']['gamma']
        del cfg_dict['peak_learning_rate']
        return cfg_dict
    
    cfg_dict = draccus.encode(cfg)
    data, _  = plot.load_data()
    cfg_norm = normalize_cfg(cfg_dict)

    probe_theta0_needed = True
    for ri,r in data.iterrows():
        cur_cfg_dict = eval(r['cfg'])
        if normalize_cfg(cur_cfg_dict) == cfg_norm:
            theta = (cur_cfg_dict['peft_config']['alpha'],
                      cur_cfg_dict['peft_config']['gamma'],
                      cur_cfg_dict['peak_learning_rate'])
            ptheta = pmap(*theta)
            if all(abs(e-f) < 1e-5 for e,f in zip(ptheta, ptheta0)):
                probe_theta0_needed = False
            bo.register(ptheta, lossmap(r['train_loss']))

    if probe_theta0_needed:
        bo.probe(ptheta0)

    bo.maximize(init_points=0,n_iter=4)
    print(bo.max)
    
if __name__ == '__main__':
    cfg = draccus.parse(config_class=train.PeftTrainConfig)
    trainer = maximize_hyper(cfg)
