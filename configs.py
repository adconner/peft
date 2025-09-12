import train
import peft
import yaml
import draccus

def write_configs():
    def write1(cfg):
        cur = yaml.dump(draccus.encode(cfg), default_flow_style=False, sort_keys=False)
        open('configs/'+hex(abs(hash(cur))).lstrip('0x')+'.yaml','w').write(cur)
        
    c = train.PeftTrainConfig()
    lr = c.peak_learning_rate
    for l in [lr, lr/10, lr*10]:
        c.peak_learning_rate = l
        
        for r in [4,8,16]:
            for l in [2,4,8,16]:
                for postmult in [False, True]:
                    c.peft_config = peft.TensorEmbeddingConfig(a=r, b=r, l=l, postmult=postmult)
                    write1(c)
                        
        for r in [4,8,16]:
            for postmult in [False, True]:
                c.peft_config = peft.TiedLoraExtraConfig(a=r, b=r, postmult=postmult)
                write1(c)
                    
        for r in [4,8,16]:
            for postmult in [False, True]:
                c.peft_config = peft.TiedLoraConfig(r=r, postmult=postmult)
                write1(c)
                    
        for r in [4,8,16]:
            for (la,lb) in [(2,2), (4,4), (8,8), (8,1), (1,8)]:
                c.peft_config = peft.PartiallyTiedLoraConfig(r=r,la=la,lb=lb)
                write1(c)
                            
        for r in [4,8,16]:
            c.peft_config = peft.LoraConfig(r=r)
            write1(c)
                        
        for r in [4,8,16]:
            for transpose in [False,True]:
                c.peft_config = peft.DoraConfig(r=r,transpose=transpose)
                write1(c)
                c.peft_config = peft.SimpleDoraConfig(r=r,transpose=transpose)
                write1(c)
                
        for r in [2,4,8,16]:
            c.peft_config = peft.SvdoraConfig(rU=r,rV=r)
            write1(c)


if __name__=='__main__':
    write_configs()
