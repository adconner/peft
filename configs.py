import train
import peft
import yaml
import draccus

def write_configs():
    def write1(cfg):
        cur = yaml.dump(draccus.encode(cfg), default_flow_style=False, sort_keys=False)
        open('configs/'+hex(abs(hash(cur))).lstrip('0x')+'.yaml','w').write(cur)
        
    c = train.PeftTrainConfig()
    
    for r in [8,32,128]:
        for l in [4,8,16]:
            for postmult in [False, True]:
                c.peft_config = peft.TensorEmbeddingConfig(a=r, b=r, l=l, postmult=postmult)
                write1(c)
                    
    for r in [8,32,128]:
        for postmult in [False, True]:
            c.peft_config = peft.TiedLoraExtraConfig(a=r, b=r, postmult=postmult)
            write1(c)
                
    for r in [8,32,128]:
        c.peft_config = peft.TiedLoraConfig(r=r, postmult=True)
        write1(c)
                
    for r in [8,32,128]:
        for (la,lb) in [(2,2), (4,4), (8,8)]:
            c.peft_config = peft.PartiallyTiedLoraConfig(r=r,la=la,lb=lb)
            write1(c)
                        
    for r in [4,8,16]:
        c.peft_config = peft.LoraConfig(r=r,gamma=1.)
        write1(c)
        c.peft_config = peft.LoraConfig(r=r,gamma=750.)
        write1(c)
        c.peft_config = peft.NormedLoraConfig(r=r,gamma=1.)
        write1(c)
        c.peft_config = peft.NormedLoraConfig(r=r,gamma=750.)
        write1(c)
        c.peft_config = peft.StrongGammaLoraConfig(r=r,gamma=750.)
        write1(c)
                    
    for r in [4,8,16]:
        c.peft_config = peft.DoraConfig(r=r,transpose=False)
        write1(c)
        c.peft_config = peft.SimpleDoraConfig(r=r,transpose=False)
        write1(c)
            
    # for r in [2,4,8]:
    #     c.peft_config = peft.SvdoraConfig(rU=r,rV=r)
    #     write1(c)


if __name__=='__main__':
    write_configs()
