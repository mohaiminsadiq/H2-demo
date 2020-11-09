from python_util.API import API
import cvxpy as cvx
import numpy as np
import pandas as pd
from collections import namedtuple, Counter, OrderedDict
from bokeh.plotting import figure, output_file, show

if __name__ == "__main__":
    api_s = API("shap")
    api_b = API("burden")
    api_l = API("loco")
    
    for x in api_b.get_model_results():
        print (x)
    for x in api_s.get_model_results():
        print (x)
    for x in api_l.get_model_results():
        print (x)

        
    show(api_b.get_repr_graph())
    input()
    show(api_b.get_demoParity_graph())
    input()
    show(api_l.get_repr_graph())
    input()
    show(api_l.get_demoParity_graph())
    input()
    show(api_l.get_scatter_plot())