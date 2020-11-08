import cvxpy as cvx
import numpy as np
import pandas as pd
from collections import namedtuple, Counter, OrderedDict
from modules import ShapModel, BurdenModel
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
        else:
            print(self.method.lower())
            raise ValueError()
        
    def get_model_results(self, calibrated=False):
        if self.method == "shap":
            return self.model.equalized_odds_shap(self.dataset, True)
        elif self.method == "burden":
            return self.model.equalized_odds_burden(self.dataset)
        else: 
            return None
            
    def get_repr_graph(self):
        if self.method == "burden":
            return self.model.get_burden_graph(self.dataset)
         
        return None
            
    def get_demoParity_graph(self):
        if self.method == "burden":
            return self.model.get_burden_demoParity(self.dataset)
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

if __name__ == "__main__":
    api_s = API("shap")
    api_b = API("burden")
    
    for x in api_b.get_model_results():
        print (x)
    for x in api_s.get_model_results():
        print (x)
        
    show(api_b.get_repr_graph())
    show(api_b.get_demoParity_graph())

        
    