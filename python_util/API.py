import cvxpy as cvx
import numpy as np
import pandas as pd
from collections import namedtuple, Counter, OrderedDict
from python_util.modules import ShapModel, BurdenModel, LocoModel
from bokeh.plotting import figure, output_file, show


class API:
    def __init__(self, debias_method="shap", calibrated=False, dataset="compas"):
        self.method = debias_method.lower()
        self.calib = calibrated
        self.dataset = "compas"
        if self.method.lower() == "shap":
            self.model = ShapModel.ShapModel()
        elif self.method.lower() == "burden":
            self.model = BurdenModel.BurdenModel()
        elif self.method.lower() == "loco":
            self.model = LocoModel.LocoModel() 
        else:
            print(self.method.lower())
            raise ValueError()
        
    def get_model_results(self, calibrated=False):
        if self.method == "shap":
            return self.model.equalized_odds_shap(self.dataset, True)
        elif self.method == "burden":
            return self.model.equalized_odds_burden(self.dataset)
        elif self.method == "loco":
            return self.model.equalized_odds_loco(self.dataset)
        else: 
            return None
            
    def get_repr_graph(self):
        if self.method == "burden":
            return self.model.get_burden_graph(self.dataset)
        elif self.method == "loco":
            return self.model.get_loco_graph(self.dataset) 
        return None
            
    def get_demoParity_graph(self):
        if self.method == "burden":
            return self.model.get_burden_demoParity(self.dataset)
        if self.method == "loco":
            return self.model.get_loco_demoParity(self.dataset)
        return None
            
    def get_scatter_plot(self):
        if self.method == "burden":
            return self.model.get_burden_scatter(self.dataset)
        if self.method == "loco":
            return self.model.get_loco_scatter(self.dataset)
        return None
            
    def get_conv_hulls():
        pass
    
    def get_delta_demo():
        pass
        
    def get_instance_graph():
        pass
        
    def get_cfacts():
        if self.method is "shap":
            return None
            raise ValueError

        
    