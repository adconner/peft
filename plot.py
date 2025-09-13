import os
from glob import glob
import yaml
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap, factor_mark, linear_cmap, log_cmap
from bokeh.models import HoverTool, OpenURL, TapTool
from bokeh.embed import file_html
import math

def load_data(smoothing_half_life=250):
    lam = math.pow(0.5,1/smoothing_half_life)
    data = {
        'hash': [],
        'peft_type' : [],
        'train_loss' : [],
        'test_loss' : [],
        'model_params' : [],
        'peft_params' : [],
        'peft_cfg' : [],
        'peak_learning_rate' : [],
    }

    # hash keyed, values are dicts with keys 'iteration', 'loss', 'type'
    iterations_by_hash = {}

    for out in glob('data/*.out'):
        print('reading',out)
        base = out[5:-4]
        cfg_file = f'configs/{base}.yaml'
        cfg = yaml.safe_load(open(cfg_file))
        peft_cfg = cfg['peft_config']

        iterations = { 'iteration' : [], 'loss' : [], 'type' : [] } 
        test_loss = 100.0
        loss = 0.0
        with open(out) as outf:
            l1 = outf.readline().strip()
            if l1 == '':
                continue
            info = eval(l1)
            iteration = 1
            for l in outf:
                l = eval(l)
                iterations['iteration'].append(iteration)
                if 'eval_loss' in l:
                    test_loss = min(test_loss,l['eval_loss'])
                    iterations['loss'].append(l['eval_loss'])
                    iterations['type'].append('test')
                else:
                    curloss = l['loss'] / l['tokens']
                    loss = curloss/iteration + (iteration-1)*loss / iteration if \
                            iteration < smoothing_half_life else (1-lam) * curloss  + lam * loss
                    iterations['loss'].append(loss)
                    iterations['type'].append('train')
                    iteration += 1
        if len(iterations['loss']) < 2:
            continue
        if loss > 2.0 or test_loss > 2.0:
            continue
        iterations_by_hash[base] = iterations
                    
        data['hash'].append(base)
        data['peft_type'].append(peft_cfg['type'])
        data['train_loss'].append(loss)
        data['test_loss'].append(test_loss)
        data['model_params'].append(info['model_params'])
        data['peft_params'].append(info['peft_params'])
        data['peft_cfg'].append(yaml.dump(peft_cfg, default_flow_style=False, sort_keys=False))
        data['peak_learning_rate'].append(cfg['peak_learning_rate'])

    return data, iterations_by_hash
        
def plot_run(iterations):
    p = figure(title = None, background_fill_color="#fafafa", tooltips="#@iteration: @loss")
              # tools="hover", tooltips="@iteration: @loss")
    p.xaxis.axis_label = 'iteration'
    p.yaxis.axis_label = 'cross entropy per token (bits)'
    p.scatter('iteration', 'loss', source=iterations,
              color = factor_cmap('type', 'Category10_3', ['train', 'test']))
    return p

def plot_all_runs(data):
    tools='hover,tap,box_select,box_zoom,wheel_zoom,reset'
    p1 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params @peft_cfg lr: @peak_learning_rate")
    p1.xaxis.axis_label = 'train loss'
    p1.yaxis.axis_label = 'test loss'
    p1.scatter('train_loss', 'test_loss', source=data, size=5,
             color = log_cmap('peft_params', 'Turbo256', min(data['peft_params']), max(data['peft_params'])))
              # color = factor_cmap('peft_type', 'Category10_6', sorted(set(data['peft_type']))))
    p1.select(type=TapTool).callback = OpenURL(url="runs/@hash.html")
              
    p2 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params @peft_cfg lr: @peak_learning_rate")
    p2.xaxis.axis_label = 'peft params'
    p2.yaxis.axis_label = 'test loss'
    p2.scatter('peft_params', 'test_loss', source=data, size=5,
             color = log_cmap('peft_params', 'Turbo256', min(data['peft_params']), max(data['peft_params'])))
              # color = factor_cmap('peft_type', 'Category10_6', sorted(set(data['peft_type']))))
    p2.select(type=TapTool).callback = OpenURL(url="runs/@hash.html")
    
    p3 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params @peft_cfg lr: @peak_learning_rate")
    p3.xaxis.axis_label = 'peft params'
    p3.yaxis.axis_label = 'train loss'
    p3.scatter('peft_params', 'train_loss', source=data, size=5,
             color = log_cmap('peft_params', 'Turbo256', min(data['peft_params']), max(data['peft_params'])))
              # color = factor_cmap('peft_type', 'Category10_6', sorted(set(data['peft_type']))))
    p3.select(type=TapTool).callback = OpenURL(url="runs/@hash.html")
    
    return row(p1, p2, p3)

def main(prefix='.'):
    os.makedirs(f'{prefix}/runs',exist_ok=True)
    data, iterations_by_hash = load_data()
    model = plot_all_runs(data)
    outf = f'{prefix}/index.html'
    print(f'writing {outf}')
    with open(outf,'w') as f:
        f.write(file_html(model,title="Peft runs"))
    for hash, peft_type, train_loss, test_loss, model_params, peft_params, peft_cfg, peak_learning_rate in zip(data['hash'], data['peft_type'], data['train_loss'], data['test_loss'], data['model_params'], data['peft_params'], data['peft_cfg'], data['peak_learning_rate']):
        iterations = iterations_by_hash[hash]
        model = plot_run(iterations)
        outf = f'{prefix}/runs/{hash}.html'
        print(f'writing {outf}')
        with open(outf,'w') as f:
            f.write(file_html(model, title=peft_cfg,
                              template_variables={'inner_body' : 
                              f'<code>{open('configs/'+hash+'.yaml').read()}</code>'}))

if __name__ == '__main__':
    main()
