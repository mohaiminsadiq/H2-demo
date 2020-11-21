from pathlib import Path
import os
from PIL import Image
import numpy as np
from bokeh.plotting import Figure, show
from bokeh.models import ColumnDataSource
from bokeh.models import CustomJS, Slider
from bokeh.layouts import column, row
from bokeh.models.glyphs import Quadratic
from bokeh.models import Div
import pandas as pd
from python_util.surface3d import Surface3d


def generate_genetic_plot():
    my_path = os.path.abspath(os.path.dirname(__file__))
    im_dir = Path(my_path, "..", "static", 'images', 'genetic', '2d_data.csv')
    data = pd.read_csv(im_dir)
    dict_data = {}
    avg_fit_dict = {}
    max_fit_dict = {}
    for i in range(100):
        temp = data[(data['generation'] == i+1) & 
                    (data['type'] == 'cfacts')].sort_values('fitness', axis=0, ascending=False)
        dict_data[str(i)+"x_best"] = [temp['X2'].iloc[0]] * 100
        dict_data[str(i)+"y_best"] = [temp['X1'].iloc[0]] * 100
        dict_data[str(i)+"x_good"] = temp['X2'].iloc[1:int(len(temp)/2)]
        dict_data[str(i)+"y_good"] = temp['X1'].iloc[1:int(len(temp)/2)]
        dict_data[str(i)+"x_okay"] = temp['X2'].iloc[int(len(temp)/2):]
        dict_data[str(i)+"y_okay"] = temp['X1'].iloc[int(len(temp)/2):]        
        dict_data[str(i)+"x_bad"] = data[(data['generation'] == i+1) & 
                                (data['type'] == 'bads')]['X2']
        dict_data[str(i)+"y_bad"] = data[(data['generation'] == i+1) & 
                                (data['type'] == 'bads')]['X1']
                                
        avg_fit_dict[str(i)] = [data[(data['generation'] == i+1) & 
                                    (data['type'] == 'cfacts')].iloc[:int(len(temp)/2)].fitness.mean()]
                                    
        max_fit_dict[str(i)] = [data[(data['generation'] == i+1) & 
                                    (data['type'] == 'cfacts')].fitness.max()]
    
    dict_data['dispx_best'] = dict_data['0x_best']
    dict_data['dispy_best'] = dict_data['0y_best']
    dict_data['dispx_good'] = dict_data['0x_good']
    dict_data['dispy_good'] = dict_data['0y_good']
    dict_data['dispx_okay'] = dict_data['0x_okay']
    dict_data['dispy_okay'] = dict_data['0y_okay']
    dict_data['dispx_bad'] = dict_data['0x_bad']
    dict_data['dispy_bad'] = dict_data['0y_bad']
    
    src=ColumnDataSource(dict_data)
    avg_fit_src=ColumnDataSource(avg_fit_dict)
    max_fit_src=ColumnDataSource(max_fit_dict)
    div_avg_text = Div(text="Avergage Fitness: ")
    div_avg = Div(text=("%3.2f" % avg_fit_dict['0'][0]))
    div_max_text = Div(text="Maximum Fitness: ")
    div_max = Div(text=("%3.2f" % max_fit_dict['0'][0]))
    
    slider = Slider(start=0, end=99, value=0, step=1, title="Generation")
    callback = CustomJS(args=dict(source=src, offset=slider, avg_fits=avg_fit_src, 
                                  max_fits=max_fit_src, avg_div=div_avg, max_div=div_max),
                code="""
                        const data = source.data;
                        const B = offset.value;
                        const avg_data = avg_fits.data[String(B)][0]
                        const max_data = max_fits.data[String(B)][0]
                        max_div.text = String(max_data.toFixed(2))
                        avg_div.text = String(avg_data.toFixed(2))
                        console.log(avg_data[String(B)])
                        console.log(max_data[String(B)])
                        data['dispx_best'] = data[String(B)+"x_best"];
                        data['dispy_best'] = data[String(B)+"y_best"];
                        data['dispx_good'] = data[String(B)+"x_good"];
                        data['dispy_good'] = data[String(B)+"y_good"];
                        data['dispx_okay'] = data[String(B)+"x_okay"];
                        data['dispy_okay'] = data[String(B)+"y_okay"];
                        data['dispx_bad'] = data[String(B)+"x_bad"];
                        data['dispy_bad'] = data[String(B)+"y_bad"];
                        source.change.emit();
                    """)
    slider.js_on_change('value', callback)
    
    x = np.arange(0, .866, .001)[::2]
    y = x**2 + .25

    p = Figure(x_range=(0,1),y_range=(0,1),width=500, height=500)
    
    p.line(x=x, y=y)
    p.line(x=[0]*2, y=[.25,1])
    p.line(x=[0,.866]*2, y=[1,1])
    
    p.scatter(x='dispx_good', y='dispy_good',source=src, color='blue'  , size=5, legend_label="Top 50% C-Fact")
    p.scatter(x='dispx_okay', y='dispy_okay',source=src, color='green' , size=5, legend_label="Bottom 50% C-Fact")
    p.scatter(x='dispx_bad',  y='dispy_bad', source=src, color='red'   , size=5, legend_label="Same Class")
    p.scatter(x='dispx_best', y='dispy_best',source=src, color='purple', size=5, legend_label="Best C-Fact")
    p.square (x=.25,          y=.25,                     color='black' , size=5, legend_label="Individual")
    
    p.legend.location = "top_right"
    p.legend.click_policy="hide"
    
    return column(row(column(slider, width=100), p), row(column(width=100), div_avg_text, div_avg, div_max_text, div_max))

def generate_genetic_plot_3D():
    my_path = os.path.abspath(os.path.dirname(__file__))
    im_dir = Path(my_path, "..", "static", 'images', 'genetic', '3d_data.csv')
    data = pd.read_csv(im_dir)
    dict_data = {}
    avg_fit_dict = {}
    max_fit_dict = {}
    for i in range(25):
        temp = data[(data['generation'] == i+1) & 
                    (data['type'] == 'cfacts')].sort_values('fitness', axis=0, ascending=False)
        dict_data[str(i)+"x_best"] = [temp['juv_fel_count'].iloc[0]] * 10
        dict_data[str(i)+"y_best"] = [temp['priors_count'].iloc[0]] * 10
        dict_data[str(i)+"z_best"] = [temp['race_African_American'].iloc[0]] * 10
        
        dict_data[str(i)+"x_good"] = temp['juv_fel_count'].iloc[1:11]
        dict_data[str(i)+"y_good"] = temp['priors_count'].iloc[1:11]
        dict_data[str(i)+"z_good"] = temp['race_African_American'].iloc[1:11]
        
        dict_data[str(i)+"x_okay"] = temp['juv_fel_count'].iloc[11:21]
        dict_data[str(i)+"y_okay"] = temp['priors_count'].iloc[11:21]     
        dict_data[str(i)+"z_okay"] = temp['race_African_American'].iloc[11:21]
        
        dict_data[str(i)+"x_bad"] = data[(data['generation'] == i+1) & 
                                (data['type'] == 'bads')]['juv_fel_count'].iloc[:10]
        dict_data[str(i)+"y_bad"] = data[(data['generation'] == i+1) & 
                                (data['type'] == 'bads')]['priors_count'].iloc[:10]
        dict_data[str(i)+"z_bad"] = data[(data['generation'] == i+1) & 
                                (data['type'] == 'bads')]['race_African_American'].iloc[:10]
                                
        avg_fit_dict[str(i)] = [data[(data['generation'] == i+1) & 
                                    (data['type'] == 'cfacts')].iloc[:int(len(temp)/2)].fitness.mean()]
                                    
        max_fit_dict[str(i)] = [data[(data['generation'] == i+1) & 
                                    (data['type'] == 'cfacts')].fitness.max()]
    
    dict_data['dispx_best'] = dict_data['0x_best']
    dict_data['dispy_best'] = dict_data['0y_best']
    dict_data['dispz_best'] = dict_data['0y_best']
    
    dict_data['dispx_good'] = dict_data['0x_good']
    dict_data['dispy_good'] = dict_data['0y_good']
    dict_data['dispz_good'] = dict_data['0z_good']
    
    dict_data['dispx_okay'] = dict_data['0x_okay']
    dict_data['dispy_okay'] = dict_data['0y_okay']
    dict_data['dispz_okay'] = dict_data['0z_okay']
    
    dict_data['dispx_bad'] = dict_data['0x_bad']
    dict_data['dispy_bad'] = dict_data['0y_bad']
    dict_data['dispz_bad'] = dict_data['0z_bad']
    
    src=ColumnDataSource(dict_data)
    avg_fit_src=ColumnDataSource(avg_fit_dict)
    max_fit_src=ColumnDataSource(max_fit_dict)
    div_avg_text = Div(text="Avergage Fitness: ")
    div_avg = Div(text=("%3.2f" % avg_fit_dict['0'][0]))
    div_max_text = Div(text="Maximum Fitness: ")
    div_max = Div(text=("%3.2f" % max_fit_dict['0'][0]))
    
    slider = Slider(start=0, end=24, value=0, step=1, title="Generation")
    callback = CustomJS(args=dict(source=src, offset=slider, avg_fits=avg_fit_src, 
                                  max_fits=max_fit_src, avg_div=div_avg, max_div=div_max),
                code="""
                        const data = source.data;
                        const B = offset.value;
                        const avg_data = avg_fits.data[String(B)][0]
                        const max_data = max_fits.data[String(B)][0]
                        max_div.text = String(max_data.toFixed(2))
                        avg_div.text = String(avg_data.toFixed(2))
                        console.log(avg_data[String(B)])
                        console.log(max_data[String(B)])
                        data['dispx_best'] = data[String(B)+"x_best"];
                        data['dispy_best'] = data[String(B)+"y_best"];
                        data['dispz_best'] = data[String(B)+"z_best"];
                        
                        data['dispx_good'] = data[String(B)+"x_good"];
                        data['dispy_good'] = data[String(B)+"y_good"];
                        data['dispz_good'] = data[String(B)+"z_good"];
                        
                        data['dispx_okay'] = data[String(B)+"x_okay"];
                        data['dispy_okay'] = data[String(B)+"y_okay"];
                        data['dispz_okay'] = data[String(B)+"z_okay"];
                        
                        data['dispx_bad'] = data[String(B)+"x_bad"];
                        data['dispy_bad'] = data[String(B)+"y_bad"];
                        data['dispz_bad'] = data[String(B)+"z_bad"];
                        
                        source.change.emit();
                    """)
    slider.js_on_change('value', callback)
    
    p = Surface3d(x_good="dispx_good", y_good="dispy_good", z_good="dispz_good", 
                  x_best="dispx_best", y_best="dispy_best", z_best="dispz_best",
                  x_okay="dispx_okay", y_okay="dispy_okay", z_okay="dispz_okay",
                  x_bad="dispx_bad",   y_bad="dispy_bad",   z_bad="dispz_bad",
                  x_indiv=0, y_indiv=0, z_indiv=0,
                  data_source=src, width=600, height=600)
    
    return column(row(column(slider, width=100), p), row(column(width=100), div_avg_text, div_avg, div_max_text, div_max))
  
if __name__ == "__main__":
    #show(generate_genetic_plot())
    show(generate_genetic_plot_3D())