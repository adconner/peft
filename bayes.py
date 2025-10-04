import numpy as np
import bayes_opt
import draccus
import yaml
from copy import copy

import peft
import train
import plot

    
def maximize_hyper(cfg):
    data, _  = plot.load_data()
    def f(logalpha, loggamma, loglr):
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
    bo.probe({"logalpha": logalpha0,
              "loggamma": loggamma0,
              "loglr": loglr0})
    bo.maximize(10)
    print(bo.max)
    
if __name__ == '__main__':
    cfg = draccus.parse(config_class=train.PeftTrainConfig)
    trainer = maximize_hyper(cfg)
