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
        self.dataset = dataset
        if self.method.lower() == "shap":
            self.model = ShapModel.ShapModel()
        elif self.method.lower() == "burden":
            self.model = BurdenModel.BurdenModel()
        elif self.method.lower() == "loco":
            self.model = LocoModel.LocoModel() 
        else:
            print(self.method.lower())
            raise ValueError()
        
    def get_model_results(self, calibrated=False, shap_enabled=False):
        if self.method == "shap":
            if calibrated:
                return self.model.calibrated_equalized_odds_shap(self.dataset, 'weighted', shap_enabled)
            else:    
                return self.model.equalized_odds_shap(self.dataset, shap_enabled)
        elif self.method == "burden":
            return self.model.equalized_odds_burden(self.dataset)
        elif self.method == "loco":
            return self.model.equalized_odds_loco(self.dataset)
        else: 
            return None
            
    def get_repr_graph(self):
        if self.method == "burden":
            return self.model.get_burden_graph(self.dataset)
        elif self.method == "shap":
            return self.model.plot_shap_summaryplot(self.dataset)
        elif self.method == "loco":
            return self.model.get_loco_graph(self.dataset) 
        return None
            
    def get_demoParity_graph(self):
        if self.method == "burden":
            return self.model.get_burden_demoParity(self.dataset)
        elif self.method == "shap":
            return self.model.get_shap_demoParity(self.dataset)
        if self.method == "loco":
            return self.model.get_loco_demoParity(self.dataset)
        return None
            
    def get_eq_odds_graphs(self):
        if self.method == "burden":
            pass
        elif self.method == "shap":
            #nothing to return, will save the plots as images, html will pull from correct folder
            self.model.plot_shap_kdeplots(self.dataset)
        elif self.method == "loco":
            pass
        return None

    def get_calib_eq_odds_graph(self, t_ceo_g0, t_ceo_g1, f_ceo_g0, f_ceo_g1):
        if self.method == "burden":
            pass
        elif self.method == "shap":
            return self.model.draw_shap_calib_eq_odds_plot(self.dataset, t_ceo_g0, t_ceo_g1, f_ceo_g0, f_ceo_g1)
        elif self.method == "loco":
            pass
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

        
    