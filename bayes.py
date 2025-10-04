import numpy as np
import bayes_opt
import draccus
import yaml
from copy import copy, deepcopy

import peft
import train
import plot

    
def maximize_hyper(cfg):
    def normalize_cfg(cfg_dict):
        cfg_dict = deepcopy(cfg_dict)
        del cfg_dict['peft_config']['alpha']
        del cfg_dict['peft_config']['gamma']
        del cfg_dict['peak_learning_rate']
        return cfg_dict
    
    cfg_dict = draccus.encode(cfg)
    data, _  = plot.load_data()
    cfg_norm = normalize_cfg(cfg_dict)

    sols = {}
    for ri,r in data.iterrows():
        cur_cfg_dict = eval(r['cfg'])
        if normalize_cfg(cur_cfg_dict) == cfg_norm:
            sols[(np.log(cur_cfg_dict['peft_config']['alpha']),
                  np.log(cur_cfg_dict['peft_config']['gamma']),
                  np.log(cur_cfg_dict['peak_learning_rate']))] = r['train_loss']
    print('previous solutions', sols)
    
    def f(logalpha, loggamma, loglr):
        if (sol := sols.get((logalpha, loggamma, loglr),None)) != None:
            return -sol
        c = copy(cfg)
        c.peft_config.alpha = float(np.exp(logalpha))
        c.peft_config.gamma = float(np.exp(loggamma))
        c.peak_learning_rate = float(np.exp(loglr))
        
        cur = yaml.dump(draccus.encode(c), default_flow_style=False, sort_keys=False)
        open('configs/'+hex(abs(hash(cur))).lstrip('0x')+'.yaml','w').write(cur)

        _, train_loss, test_loss = train.train_peft(c)
        return -train_loss

    logalpha0 = np.log(cfg.peft_config.alpha)
    loggamma0 = np.log(cfg.peft_config.gamma)
    loglr0 = np.log(cfg.peak_learning_rate)
    
    bo = bayes_opt.BayesianOptimization(
            f=f,
            pbounds = {"logalpha": (-1, 7),
                       "loggamma": (0, 7.5),
                       "loglr": (loglr0 - 2.3, loglr0 + 2.3)},
            )

    toprobe = set([(logalpha0, loggamma0, loglr0)]).union(sols)
    for alpha, gamma, lr in toprobe:
        bo.probe({"logalpha": alpha,
                  "loggamma": gamma,
                  "loglr": lr})
    bo.maximize(10)
    print(bo.max)
    
if __name__ == '__main__':
    cfg = draccus.parse(config_class=train.PeftTrainConfig)
    trainer = maximize_hyper(cfg)
