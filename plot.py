import os
from glob import glob
import yaml
import bokeh
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap, factor_mark, linear_cmap, log_cmap
from bokeh.models import HoverTool, OpenURL, TapTool, CDSView, ColumnDataSource, GroupFilter, IntersectionFilter, UnionFilter, AllIndices, Select, MultiChoice, CustomJS
from bokeh.embed import file_html, components
from bokeh.resources import CDN, INLINE
# from bokeh.palettes import Spectral6
from jinja2 import Template
import pandas as pd
import numpy as np
import math
import subprocess
from copy import deepcopy

def load_data(smoothing_half_life=250):
    lam = math.pow(0.5,1/smoothing_half_life)
    data = {}
    iterations = {}

    # hash keyed, values are dicts with keys 'iteration', 'loss', 'type'

    best_train_loss = {}
    best_test_loss = {}
    def normalize_cfg(cfg_dict,previous):
        cfg_dict = deepcopy(cfg_dict)
        try:
            del cfg_dict['peft_config']['alpha']
            del cfg_dict['peft_config']['gamma']
        except KeyError:
            pass
        del cfg_dict['peak_learning_rate']
        del cfg_dict['weight_decay']
        del cfg_dict['out_dir']
        del cfg_dict['batchsize']
        del cfg_dict['logging_steps']
        del cfg_dict['seed']
        del cfg_dict['dataset_randomize']
        del cfg_dict['seq_len']
        cfg_dict['peft_config'] = tuple(cfg_dict['peft_config'].items())
        return (tuple(cfg_dict.items()),previous)

    for out in glob('data/*.out'):
        print('reading',out)
        hash = out[5:-4]
        cfg_file = f'configs/{hash}.yaml'
        cfg_yaml = open(cfg_file).read()
        cfg = yaml.safe_load(cfg_yaml)
        peft_cfg = cfg['peft_config']
        if peft_cfg['type'] == 'normed_lora':
            peft_cfg['type'] = 'lora'

        test_loss = 100.0
        loss = 0.0
        diverged = False
        with open(out) as outf:
            l1 = outf.readline().strip()
            if l1 == '':
                continue
            info = eval(l1)
            iteration = 1
            for l in outf:
                if 'nan' in l:
                    diverged = True
                    break
                l = eval(l)
                if 'eval_loss' in l:
                    test_loss = min(test_loss,l['eval_loss'])
                    iterations.setdefault('hash',[]).append(hash)
                    iterations.setdefault('iteration',[]).append(iteration)
                    iterations.setdefault('loss',[]).append(l['eval_loss'])
                    iterations.setdefault('type',[]).append('test')
                    if l['eval_loss'] > 2.0:
                        diverged = True
                        break
                else:
                    curloss = l['loss'] / l['tokens']
                    loss = curloss/iteration + (iteration-1)*loss / iteration if \
                            iteration < smoothing_half_life else (1-lam) * curloss  + lam * loss
                    iterations.setdefault('hash',[]).append(hash)
                    iterations.setdefault('iteration',[]).append(iteration)
                    iterations.setdefault('loss',[]).append(loss)
                    iterations.setdefault('type',[]).append('train')
                    iteration += 1
                    if loss > 2.0:
                        diverged = True
                        break
        if iteration == 1:
            print(cfg_yaml, 'no iterations (errored?), skipping')
                    
        data.setdefault('hash',[]).append(hash)
        data.setdefault('cfg',[]).append(str(cfg))
        data.setdefault('cfg_yaml',[]).append(cfg_yaml)
        data.setdefault('peft_cfg_yaml',[]).append(yaml.dump(peft_cfg, default_flow_style=False, sort_keys=False))
        data.setdefault('peft_type',[]).append(peft_cfg['type'])
        data.setdefault('train_loss',[]).append(loss)
        data.setdefault('test_loss',[]).append(test_loss)
        data.setdefault('model_name',[]).append(cfg['model_name'])
        data.setdefault('model_params',[]).append(info['model_params'])
        data.setdefault('peft_params',[]).append(info['peft_params'])
        data.setdefault('peak_learning_rate',[]).append(cfg['peak_learning_rate'])
        data.setdefault('alpha',[]).append(peft_cfg.get('alpha',None))
        data.setdefault('gamma',[]).append(peft_cfg.get('gamma',None))
        previous = peft_cfg.get('gamma',1.) == 1. and (peft_cfg['type'] in ['lora','full','tied-lora'] or 
                            (peft_cfg['type'] == 'dora' and not peft_cfg['transpose']))
        data.setdefault('work',[]).append('previous' if previous else 'current')
        data.setdefault('diverged',[]).append(diverged)
        
        cfg_norm = normalize_cfg(cfg,previous)
        if loss < best_train_loss.get(cfg_norm,(None, 100.))[1]:
            best_train_loss[cfg_norm] = (hash, loss)
        if test_loss < best_test_loss.get(cfg_norm,(None, 100.))[1]:
            best_test_loss[cfg_norm] = (hash, test_loss)

    besthashes = set(hash for hash, _ in best_train_loss.values())
    # besthashes = set(hash for hash, _ in best_train_loss.values()).union(set(hash for hash, _ in best_test_loss.values()))
    data['best'] = ['True' if hash in besthashes else 'False' for hash in data['hash']]

    data, iterations = pd.DataFrame(data), pd.DataFrame(iterations)
    data['percent'] = 100. * data['peft_params'] / data['model_params']
    data['diverged_str'] = np.where(data['diverged'], 'True', 'False')
    data['True'] = 'True'
    return data, iterations
        
def plot_run(iterations, row):
    iterations = ColumnDataSource(iterations[iterations['hash'] == row['hash']])
    p = figure(title = None, background_fill_color="#fafafa", tooltips="#@iteration: @loss")
              # tools="hover", tooltips="@iteration: @loss")
    p.xaxis.axis_label = 'iteration'
    p.yaxis.axis_label = 'cross entropy per token (bits)'
    p.scatter('iteration', 'loss', source=iterations,
              color = factor_cmap('type', 'Category10_3', ['train', 'test']),
               # size = pd.Series(2, index=iterations['type'].index).where(iterations['type'] == 'test',1)
              )
    
    template = '''    
{% from macros import embed %}
<!DOCTYPE html>
<html lang="en">
  {% block head %}
  <head>
  {% block inner_head %}
    <meta charset="utf-8">
    <title>{% block title %}{{ title | e if title else "Bokeh Plot" }}{% endblock %}</title>
  {%  block preamble -%}{%- endblock %}
  {%  block resources %}
    <style>
      html, body {
        box-sizing: border-box;
        display: flow-root;
        height: 100%;
        margin: 0;
        padding: 0;
      }
    </style>
  {%   block css_resources -%}
    {{- bokeh_css if bokeh_css }}
  {%-  endblock css_resources %}
  {%   block js_resources -%}
    {{  bokeh_js if bokeh_js }}
  {%-  endblock js_resources %}
  {%  endblock resources %}
  {%  block postamble %}{% endblock %}
  {% endblock inner_head %}
  </head>
  {% endblock head%}
  {% block body %}
  <body>
  {{ pre_content }}
  {%  block inner_body %}
  {%    block contents %}
  {%      for doc in docs %}
  {{        embed(doc) if doc.elementid }}
  {%-       for root in doc.roots %}
  {%          block root scoped %}
  {{            embed(root) }}
  {%          endblock %}
  {%        endfor %}
  {%      endfor %}
  {%    endblock contents %}
  {{ plot_script | indent(4) }}
  {%  endblock inner_body %}
  {{ post_content }}
  </body>
  {% endblock body%}
</html>
'''
    return file_html(p, title=row['peft_cfg_yaml'], template=template,
                              template_variables={'pre_content' : 
                              f'<pre><code>{row['cfg_yaml']}</code></pre>',
                                                  'post_content' : ''})

def plot_all_runs(data, source=None, filt=None):
    if source is None:
        source = ColumnDataSource(data)
    if filt is None:
        filt = AllIndices()
        
    filt = IntersectionFilter(operands=[filt,GroupFilter(column_name='diverged_str', group='False')])
    # data['new'] = np.where(data['gamma'] == 960., 'yes', 'no')
    tools='hover,tap,pan,box_select,wheel_zoom,reset'
    p1 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params (@percent %) @peft_cfg_yaml lr: @peak_learning_rate",
                x_axis_type='log')
    p1.xaxis.axis_label = 'peft params'
    p1.yaxis.axis_label = 'train loss (bits)'
              
    p2 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params (@percent %) @peft_cfg_yaml lr: @peak_learning_rate",
                x_axis_type='log')
    p2.xaxis.axis_label = 'peft params'
    p2.yaxis.axis_label = 'test loss (bits)'
    
    p3 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params (@percent %) @peft_cfg_yaml lr: @peak_learning_rate")
    p3.xaxis.axis_label = 'train loss (bits)'
    p3.yaxis.axis_label = 'test loss (bits)'
    
    for work in ['current', 'previous']:
        curfilt = GroupFilter(column_name='work', group=work)
        view = CDSView(filter=IntersectionFilter(operands=[filt,curfilt])) 
        s1 = p1.scatter('peft_params', 'train_loss', source=source, view=view, size=8, legend_label=work, line_color='black',
                 color = log_cmap('peft_params', 'Turbo256', data['peft_params'].min(), data['peft_params'].max()),
                        # color=factor_cmap('new', palette='Spectral3', factors=['yes','no']),
                   marker = factor_mark('work',['circle', 'triangle'],['current', 'previous']))
        s2 = p2.scatter('peft_params', 'test_loss', source=source, view=view, size=8, line_color='black',
                 color = log_cmap('peft_params', 'Turbo256', data['peft_params'].min(), data['peft_params'].max()),
                        # color=factor_cmap('new', palette='Spectral3', factors=['yes','no']),
                   marker = factor_mark('work',['circle', 'triangle'],['current', 'previous']))
        s3 = p3.scatter('train_loss', 'test_loss', source=source, view=view, size=8, line_color='black',
                 color = log_cmap('peft_params', 'Turbo256', data['peft_params'].min(), data['peft_params'].max()),
                        # color=factor_cmap('new', palette='Spectral3', factors=['yes','no']),
                   marker = factor_mark('work',['circle', 'triangle'],['current', 'previous']))
        # s1.js_link('hidden', s2, 'hidden')
        # s1.js_link('hidden', s3, 'hidden')
        s1.js_link('visible', s2, 'visible')
        s1.js_link('visible', s3, 'visible')
        
    # p1.legend.location="top_left"
    p1.legend.click_policy='hide'
    
    tt = p1.select(type=TapTool)
    # tt.behavior = 'inspect'
    tt.callback = OpenURL(url="runs/@hash.html")
        
    tt = p2.select(type=TapTool)
    # tt.behavior = 'inspect'
    tt.callback = OpenURL(url="runs/@hash.html")
    
    tt = p3.select(type=TapTool)
    # tt.behavior = 'inspect'
    tt.callback = OpenURL(url="runs/@hash.html")
    
    # return p1,p2,p3
    return row(p1, p2, p3)

def make_main(data):
    body = subprocess.check_output('pandoc --katex -f markdown-smart -t html doc/main.md', shell=True).decode()
    template = '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Parameter Efficient Fine Tuning</title>
<script defer="" src="https://cdn.jsdelivr.net/npm/katex@latest/dist/katex.min.js"></script>
<script>document.addEventListener("DOMContentLoaded", function () {
var mathElements = document.getElementsByClassName("math");
var macros = [];
for (var i = 0; i < mathElements.length; i++) {
var texText = mathElements[i].firstChild;
if (mathElements[i].tagName == "SPAN") {
katex.render(texText.data, mathElements[i], {
displayMode: mathElements[i].classList.contains('display'),
throwOnError: false,
macros: macros,
fleqn: false
});
}}});
</script>
<link rel="stylesheet"
href="https://cdn.jsdelivr.net/npm/katex@latest/dist/katex.min.css" />
{{ resources }}
{{ script }}
<style>
.embed-wrapper {
    display: flex;
    justify-content: space-evenly;
}
</style>
</head>
<body>
    ''' + body + '''
</body>
</html>
    '''
    resources = CDN.render()
    scripts = []

    model_options=[('google/gemma-2b', 'google/gemma-2b'), ('meta-llama/Llama-3.1-8B', 'meta-llama/Llama-3.1-8B')]
    model_select = Select(value='google/gemma-2b', options=model_options)
    model_filter = GroupFilter(column_name='model_name', group=model_select.value)
    model_select.js_link('value',model_filter,'group')
    
    best_options=[('best', 'Best runs only'), ('True', 'All runs')]
    best_select = Select(value='best', options=best_options)
    best_filter = GroupFilter(column_name=best_select.value, group='True')
    best_select.js_link('value',best_filter,'column_name')
    
    peft_types = ['full', 'lora', 'dora', 'tied_lora', 'strong_gamma_lora', 'partially_tied_lora', 'simple_dora', 'tensor_embedding', 'tied_lora_extra']
    type_boxes = {}
    peft_filts = [GroupFilter(column_name='peft_type', group=ptype) for ptype in peft_types]
    peft_filt = UnionFilter(operands=peft_filts)
    peft_select = MultiChoice(value=peft_types, options=peft_types)
    peft_select.js_on_change('value', CustomJS(args=dict(peft_filts=dict(zip(peft_types, peft_filts)), peft_filt=peft_filt), 
                                               code='peft_filt.operands = this.value.map(ptype => peft_filts[ptype])'))
    
    
    # work_options=[('previous', 'Existing approaches only'), ('current', 'New approaches')]
    # work_select = Select(value='current', options=work_options)
    # work_filter = GroupFilter(column_name='work', group=work_select.value)
    # work_select.js_link('value',work_filter,'group')
    
    filt = IntersectionFilter(operands=[model_filter, best_filter, peft_filt])
    
    source = ColumnDataSource(data)
    main_plots = plot_all_runs(data, source, filt)
    
            # value=['best_train_loss'],options=['best_train_loss','best_test_loss','all'])
    # mc.js_link
    
    script, plots = components({'main_plots' : main_plots,
                                'model_select': model_select, 
                                'best_select': best_select, 
                                'peft_select' : peft_select })
    template = Template(template)
    html = template.render(resources=resources, script=script, **plots)
    return html

def main(prefix='.'):
    os.makedirs(f'{prefix}/runs',exist_ok=True)
    data, iterations = load_data()
    html = make_main(data)
    outf = f'{prefix}/index.html'
    print(f'writing {outf}')
    with open(outf,'w') as f:
        f.write(html)
    for ix,row in data.iterrows():
        if row['diverged']:
            continue
        html = plot_run(iterations, row)
        outf = f'{prefix}/runs/{row["hash"]}.html'
        print(f'writing {outf}')
        with open(outf,'w') as f:
            f.write(html)

if __name__ == '__main__':
    main()
