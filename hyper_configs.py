import train
import peft
import yaml
import draccus

def write_configs():
    def write1(cfg,pre=''):
        cur = yaml.dump(draccus.encode(cfg), default_flow_style=False, sort_keys=False)
        open('hyper_configs/'+pre+hex(abs(hash(cur))).lstrip('0x')+'.yaml','w').write(cur)
        
    c = train.PeftTrainConfig()
    
    # for r in [8,32,128]:
    #     for l in [4,8,16]:
    #         for postmult in [False, True]:
    #             c.peft_config = peft.TensorEmbeddingConfig(a=r, b=r, l=l, postmult=postmult)
    #             write1(c)
                    
    # for r in [8,32,128]:
    #     for postmult in [False, True]:
    #         c.peft_config = peft.TiedLoraExtraConfig(a=r, b=r, postmult=postmult)
    #         write1(c)
                
    # for r in [8,32,128]:
    #     c.peft_config = peft.TiedLoraConfig(r=r, postmult=True)
    #     write1(c)
                
    # for r in [8,32,128]:
    #     for (la,lb) in [(2,2), (4,4), (8,8)]:
    #         c.peft_config = peft.PartiallyTiedLoraConfig(r=r,la=la,lb=lb)
    #         write1(c)
                        
    c.peft_config = peft.LoraConfig(r=8,gamma=1.)
    write1(c,'0')
    c.peft_config = peft.NormedLoraConfig(r=8)
    write1(c,'1')
    c.peft_config = peft.StrongGammaLoraConfig(r=8)
    write1(c,'2')
    c.peft_config = peft.TiedLoraConfig(r=128)
    write1(c,'3')
    c.peft_config = peft.SimpleDoraConfig(r=8)
    write1(c,'4')
    c.peft_config = peft.DoraConfig(r=8)
    write1(c,'5')
    c.peft_config = peft.PartiallyTiedLoraConfig(r=32,la=4,lb=4)
    write1(c,'6')
    c.peft_config = peft.TiedLoraExtraConfig(a=128,b=128)
    write1(c,'7')
    c.peft_config = peft.NormedLoraConfig(r=16)
    write1(c,'8')
    c.peft_config = peft.TensorEmbeddingConfig(a=128,b=128,l=8)
    write1(c,'9')
                    
    # for r in [4,8,16]:
    #     c.peft_config = peft.DoraConfig(r=r,transpose=False)
    #     write1(c)
    #     c.peft_config = peft.SimpleDoraConfig(r=r,transpose=False)
    #     write1(c)
            
    # for r in [2,4,8]:
    #     c.peft_config = peft.SvdoraConfig(rU=r,rV=r)
    #     write1(c)


if __name__=='__main__':
    write_configs()
