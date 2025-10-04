import os
from glob import glob
import yaml
from bokeh.plotting import figure, show
from bokeh.layouts import column, row
from bokeh.transform import factor_cmap, factor_mark, linear_cmap, log_cmap
from bokeh.models import HoverTool, OpenURL, TapTool, CDSView, ColumnDataSource, GroupFilter
from bokeh.embed import file_html
import pandas as pd
import math

def load_data(smoothing_half_life=250):
    lam = math.pow(0.5,1/smoothing_half_life)
    data = {}
    iterations = {}

    # hash keyed, values are dicts with keys 'iteration', 'loss', 'type'

    for out in glob('data/*.out'):
        print('reading',out)
        hash = out[5:-4]
        cfg_file = f'configs/{hash}.yaml'
        cfg_yaml = open(cfg_file).read()
        cfg = yaml.safe_load(cfg_yaml)
        peft_cfg = cfg['peft_config']

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
        data.setdefault('model_params',[]).append(info['model_params'])
        data.setdefault('peft_params',[]).append(info['peft_params'])
        data.setdefault('peak_learning_rate',[]).append(cfg['peak_learning_rate'])
        data.setdefault('alpha',[]).append(peft_cfg['alpha'])
        data.setdefault('gamma',[]).append(peft_cfg['gamma'])
        data.setdefault('work',[]).append('previous' if peft_cfg['type'] == 'lora' or 
                            (peft_cfg['type'] == 'dora' and not peft_cfg['transpose']) else 'current')
        data.setdefault('diverged',[]).append(diverged)

    return pd.DataFrame(data), pd.DataFrame(iterations)
        
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

def plot_all_runs(data):
    peft_types = sorted(data['peft_type'].unique())
    source = ColumnDataSource(data)
    tools='hover,tap,box_select,box_zoom,wheel_zoom,reset'
    p1 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params @peft_cfg lr: @peak_learning_rate")
    p1.xaxis.axis_label = 'train loss'
    p1.yaxis.axis_label = 'test loss'
              
    p2 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params @peft_cfg lr: @peak_learning_rate",
                x_axis_type='log')
    p2.xaxis.axis_label = 'peft params'
    p2.yaxis.axis_label = 'test loss'
    
    p3 = figure(title = None, background_fill_color="#fafafa", tools=tools, 
                tooltips="params: @peft_params @peft_cfg lr: @peak_learning_rate",
                x_axis_type='log')
    p3.xaxis.axis_label = 'peft params'
    p3.yaxis.axis_label = 'train loss'
    
    for peft_type in peft_types:
        view = CDSView(filter=GroupFilter(column_name='peft_type', group=peft_type)) 
        s1 = p1.scatter('train_loss', 'test_loss', source=source, view=view, size=8, legend_label=peft_type, line_color='black',
                 color = log_cmap('peft_params', 'Turbo256', data['peft_params'].min(), data['peft_params'].max()),
                   marker = factor_mark('work',['circle', 'triangle'],['current', 'previous']))
        s2 = p2.scatter('peft_params', 'test_loss', source=source, view=view, size=8, line_color='black',
                 color = log_cmap('peft_params', 'Turbo256', data['peft_params'].min(), data['peft_params'].max()),
                   marker = factor_mark('work',['circle', 'triangle'],['current', 'previous']))
        s3 = p3.scatter('peft_params', 'train_loss', source=source, view=view, size=8, line_color='black',
                 color = log_cmap('peft_params', 'Turbo256', data['peft_params'].min(), data['peft_params'].max()),
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

def main(prefix='.'):
    os.makedirs(f'{prefix}/runs',exist_ok=True)
    data, iterations = load_data()
    model = plot_all_runs(data)
    outf = f'{prefix}/index.html'
    print(f'writing {outf}')
    with open(outf,'w') as f:
        f.write(file_html(model,title="Peft runs"))
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
