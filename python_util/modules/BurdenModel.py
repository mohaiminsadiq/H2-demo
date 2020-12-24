import cvxpy as cvx
import numpy as np
import pandas as pd
from collections import namedtuple, Counter, OrderedDict
import os
from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearInterpolator, Span, TickFormatter
from bokeh.io import output_notebook
from bokeh.embed import components
from bokeh.models import Range1d, HoverTool
from bokeh.io import show
from bokeh.models import CustomJS, Slider
from  bokeh.models.sources import ColumnDataSource
from bokeh.layouts import column, row
import tabulate

def pair_wise_avg(iter):
        ret = []
        for x, obj in enumerate(iter):
            ret.append((obj[0] + obj[1])/2)
        return ret

class BurdenModel:

    def burden_val_test_split(self, dataset):
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_burden_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_burden_postprocess.csv")
        elif dataset == 'user':
            data_filepath = os.path.join(my_path, "../data/user.csv")
            
        test_and_val_data = pd.read_csv(data_filepath)
        test_and_val_data.index = test_and_val_data.id
        np.random.seed(42)
        # Randomly split the data into two sets - one for computing the fairness constants
        order = np.random.permutation(len(test_and_val_data)) #randomizes the list of indices
        val_indices = order[0::2] #get even index elements (the elements themselves are the original indices), i.e. starting from 0 with a step of 2
        test_indices = order[1::2] #get odd numbered index elements, i.e. starting from 1 with a step of 2
        val_data = test_and_val_data.iloc[val_indices]
        test_data = test_and_val_data.iloc[test_indices]

        # Create model objects - one for each group, validation and test
        group_0_val_data = val_data[val_data['group'] == 0]
        group_1_val_data = val_data[val_data['group'] == 1]
        group_0_test_data = test_data[test_data['group'] == 0]
        group_1_test_data = test_data[test_data['group'] == 1]

        self.group_0_val_model = Model(group_0_val_data)
        self.group_1_val_model = Model(group_1_val_data)
        self.group_0_test_model = Model(group_0_test_data)
        self.group_1_test_model = Model(group_1_test_data)


    def equalized_odds_burden(self, dataset):
        self.burden_val_test_split(dataset)

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.burden_eq_odds(self.group_0_val_model, self.group_1_val_model)

        # Apply the mixing rates to the test models
        self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model = Model.burden_eq_odds(self.group_0_test_model,
                                                                                self.group_1_test_model, 
                                                                                mix_rates)

        return self.group_0_test_model, self.group_1_test_model, self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model

    def equalized_odds_part_burden(self, dataset):
        self.burden_val_test_split(dataset)

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.burden_eq_opp(self.group_0_val_model, self.group_1_val_model)

        # Apply the mixing rates to the test models
        self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model = Model.burden_eq_opp(self.group_0_test_model,
                                                                                self.group_1_test_model, 
                                                                                mix_rates)

        return self.group_0_test_model, self.group_1_test_model, self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model

    def equalized_odds_random(self, dataset):
        self.burden_val_test_split(dataset)

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.eq_odds(self.group_0_val_model, self.group_1_val_model)

        # Apply the mixing rates to the test models
        self.eq_odds_group_0_test_model_rand, self.eq_odds_group_1_test_model_rand = Model.eq_odds(self.group_0_test_model,
                                                                                self.group_1_test_model, 
                                                                                mix_rates)

        return self.group_0_test_model, self.group_1_test_model, self.eq_odds_group_0_test_model_rand, self.eq_odds_group_1_test_model_rand

    
    def get_burden_scatter(self, dataset):
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_burden_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_burden_postprocess.csv")
        elif dataset == 'user':
            data_filepath = os.path.join(my_path, "../data/user.csv")
            
        full_data = pd.read_csv(data_filepath)
        
        g0_burden = 1/full_data[full_data['group'] == 0].fitness
        g0_wiggle = 0.5*np.sin(np.arange(g0_burden.shape[0])) + .5
        
        g1_burden = 1/full_data[full_data['group'] == 1].fitness
        g1_wiggle = 0.5 * np.sin(np.arange(g1_burden.shape[0])) + .5
        g1_offset = (g1_wiggle)+2
        data = {'x':g1_burden, 'y':g1_offset}
        source = ColumnDataSource(data)
        
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="Burden between two groups")#Plotting
        p.circle(y='y',                            #categories
               x='x',
               source=source, #bar heights
               width = .9,
               fill_alpha = .5,
               fill_color = 'red',
               line_alpha = 0,
               legend_label="group 1"
          )#Signing the axis
          
        p.circle(y=g0_wiggle,                            #categories
               x=g0_burden,                      #bar heights
               width = .9,
               fill_alpha = .5,
               fill_color = 'blue',
               line_alpha = 0,
               legend_label="group 0"
          )#Signing the axis
        
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        p.yaxis.axis_label='Burdened Group'
        
        p.sizing_mode = 'scale_width'
        p.legend.click_policy = "hide"
        
        p.yaxis.axis_label_text_font_size='16pt'
        p.yaxis.major_label_text_font_size='16pt'
        p.xaxis.axis_label_text_font_size='16pt'
        
        p.yaxis.axis_label_text_font_style='bold'
        p.xaxis.axis_label_text_font_style='normal'
        
        p.xaxis.axis_label = "Distance (Lower is Better)"
        
        slider = Slider(start=0, end=2, value=2, step=.01, title="offset")
        callback = CustomJS(args=dict(source=source, offset=slider),
                    code="""
                            const data = source.data;
                            const y = data['y'];
                            const B = offset.value;
                            for (var i = 0; i < y.length; i++) {
                                y[i] = B + (0.5*Math.sin(i)+.5);
                            }
                            source.change.emit();
                        """)
        slider.js_on_change('value', callback)
        
        return row(column(slider, width=100), p)
    
    def plot_burden_kdeplots(self, dataset):
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_burden_postprocess.csv") 
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_burden_postprocess.csv")
        elif dataset == 'user':
            data_filepath = os.path.join(my_path, "../data/user.csv")

        df = pd.read_csv(data_filepath)
        
        hover = HoverTool(
            tooltips=[
                ("group", "@group"),
                ("(dist, prob)", "(@x, @y)")
            ]
        )
        
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="PMF of fitness between two groups",
                    tools=[hover])
        
        smooth_fact = 50
        
        loco_1 = 1/df[df['group']==1].fitness
        ind1, loco_1_pmf = np.histogram(loco_1, bins=int(len(loco_1)/smooth_fact), density=True)
        loco_0 = 1/df[df['group']==0].fitness
        ind0, loco_0_pmf = np.histogram(loco_0, bins=int(len(loco_0)/smooth_fact), density=True)
                        
        source1 = ColumnDataSource({'x' : loco_1_pmf[:-1], 'y' : ind1, 'group': [1]*len(ind1)})
        source0 = ColumnDataSource({'x' : loco_0_pmf[:-1], 'y' : ind0, 'group': [0]*len(ind0)})
        
        p.line(x='x', y='y', line_color='red', legend_label='group 1', source=source1)
        p.line(x='x', y='y', line_color='blue', legend_label='group 0', source=source0)
    
        p.xaxis.axis_label = "Distance (Lower is Better)"
        p.yaxis.axis_label = 'Portion of Individuals With Specific Distance'
        p.sizing_mode = 'scale_width'
        
        p.yaxis.axis_label_text_font_size='12pt'
        p.yaxis.major_label_text_font_size='16pt'
        p.xaxis.axis_label_text_font_size='16pt'
        
        p.yaxis.axis_label_text_font_style='bold'
        p.xaxis.axis_label_text_font_style='normal'
        
        p.legend.click_policy="hide"
        return p#components(p)
    
    def get_burden_graph(self, dataset): 
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_burden_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_burden_postprocess.csv")
        elif dataset == 'user':
            data_filepath = os.path.join(my_path, "../data/user.csv")
            
        full_data = pd.read_csv(data_filepath)

        years = [0]
        values = [(1/full_data[full_data['group'] == 0].fitness).mean()-
                  (1/full_data[full_data['group'] == 1].fitness).mean()]
        
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="Burden between two groups")#Plotting
        p.vbar(years,                            #categories
              top = values,                      #bar heights
               width = .9,
               fill_alpha = .5,
               fill_color = ['blue'],
               line_alpha = .5,
               line_color='green',
               line_dash='solid',              
          )#Signing the axis
        
        p.y_range = Range1d(-abs(values[0])*1.1, abs(values[0])*1.1)
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        p.renderers.extend([hline])
        p.yaxis.axis_label='Delta Burden'
        
        labels = ['Group 1', 'Group 0']
        y_tick_locs = (-abs(values[0]), abs(values[0]))
        p.yaxis.ticker = y_tick_locs
        label_dict = {}
        for x, lab in zip(y_tick_locs, labels):
            label_dict[x] = lab
        p.yaxis.major_label_overrides = label_dict
        p.sizing_mode = 'scale_width'
        
        p.yaxis.axis_label_text_font_size='16pt'
        p.yaxis.major_label_text_font_size='16pt'
        p.xaxis.axis_label_text_font_size='16pt'
        
        p.yaxis.axis_label_text_font_style='bold'
        p.xaxis.axis_label_text_font_style='normal'

        
        return p
    
    def get_burden_table(self, dataset):
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_burden_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_burden_postprocess.csv")
        elif dataset == 'user':
            data_filepath = os.path.join(my_path, "../data/user.csv")
            
        full_data = pd.read_csv(data_filepath)
        
        before0, before1, g0, g1 = self.equalized_odds_burden(dataset)
        _, _, pg0, pg1 = self.equalized_odds_part_burden(dataset)
        _, _, rand0, rand1 = self.equalized_odds_random(dataset)

        table = [["None", 0, before0.accuracy(), 
          before0.fp_cost(), 
          before0.fn_cost(), 
          before0.base_rate(), 
          before0.pred.mean()],
         ["None", 1, before1.accuracy(), 
          before1.fp_cost(), 
          before1.fn_cost(), 
          before1.base_rate(), 
          before1.pred.mean()],
         ["Random", 0, rand0.accuracy(), 
          rand0.fp_cost(), 
          rand0.fn_cost(), 
          rand0.base_rate(), 
          rand0.pred.mean()],
         ["Random", 1, rand1.accuracy(), 
          rand1.fp_cost(), 
          rand1.fn_cost(), 
          rand1.base_rate(), 
          rand1.pred.mean()],
         ["Burden", 0, g0.accuracy(), 
          g0.fp_cost(), 
          g0.fn_cost(), 
          g0.base_rate(), 
          g0.pred.mean()],
         ["Burden", 1, g1.accuracy(), 
          g1.fp_cost(), 
          g1.fn_cost(), 
          g1.base_rate(), 
          g1.pred.mean()],
         ["Partial Burden", 0, pg0.accuracy(), 
          pg0.fp_cost(), 
          pg0.fn_cost(), 
          pg0.base_rate(), 
          pg0.pred.mean()],
         ["Partial Burden", 1, pg1.accuracy(), 
          pg1.fp_cost(), 
          pg1.fn_cost(), 
          pg1.base_rate(), 
          pg1.pred.mean()]]
        head = ['Method', 'Group', 'Accuracy', 'F.P. Cost', 'F.N. cost', 'Base rate', 'Avg. score']
        return tabulate.tabulate(table, tablefmt='html', headers=head)
        
    def get_burden_demoParity(self, dataset): 
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_burden_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_burden_postprocess.csv")
        elif dataset == 'user':
            data_filepath = os.path.join(my_path, "../data/user.csv")
            
        full_data = pd.read_csv(data_filepath)
        
        _, _, g0, g1 = self.equalized_odds_burden(dataset)
        _, _, pg0, pg1 = self.equalized_odds_part_burden(dataset)
        
        width = .35
        padd = .45
        
        x_1 = [width, width*2+padd, width*3+padd*2]
        x_0 = [0, width+padd, width*2+padd*2]
        values_1 = [
                  full_data[(full_data['group'] == 1)].prediction.round().mean(),
                  g1.datadf.prediction.round().mean(),
                  pg1.datadf.prediction.round().mean()
                  ]
        values_0 = [full_data[(full_data['group'] == 0)].prediction.round().mean(),
                    g0.datadf.prediction.round().mean(),
                    pg0.datadf.prediction.round().mean()
        ]
        
        source0 = ColumnDataSource({'x' : x_0, 'y' : values_0})
        source1 = ColumnDataSource({'x' : x_1, 'y' : values_1})
        
        hover = HoverTool(
            tooltips=[
                ("Probability", "@y")
            ]
        )
        labels = ['None', 'Burden', 'Partial Burden']
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="Demographic Parity",
                    tools=[hover])#Plotting
        p.vbar(x='x',                            #categories
              top = 'y',                      #bar heights
               width = width,
               fill_alpha = .5,
               fill_color = 'blue',
               line_alpha = .5,
               line_color='green',
               line_dash='solid',   
               legend_label="group 0",
               source=source0
          )#Signing the axis
        p.vbar(x='x',                            #categories
            top = 'y',                      #bar heights
            width = width,
            fill_alpha = .5,
            fill_color = 'red',
            line_alpha = .5,
            line_color='green',
            line_dash='solid',   
            legend_label="group 1",
            source=source1
        )#Signing the axis
        
        
        p.sizing_mode = 'scale_width'
        
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        p.renderers.extend([hline])
        p.yaxis.axis_label='Probability'
        p.xaxis.axis_label='Method'
        x_tick_locs = pair_wise_avg(zip(x_0, x_1))
        p.xaxis.ticker = x_tick_locs
        
        p.yaxis.axis_label_text_font_size='16pt'
        p.yaxis.major_label_text_font_size='14pt'
        p.xaxis.axis_label_text_font_size='16pt'
        p.xaxis.major_label_text_font_size='14pt'
        
        p.yaxis.axis_label_text_font_style='bold'
        p.xaxis.axis_label_text_font_style='bold'
        
        label_dict = {}
        for x, lab in zip(x_tick_locs, labels):
            label_dict[x] = lab
        p.xaxis.major_label_overrides = label_dict
        
        return p    
    
    
        
    def calibrated_equalized_odds_shap(self, dataset, cost_constraint, shap_enabled=False):
        pass
    #Plotting the convex hulls
    def plot_convex_hulls_equalized_odds(self, calibrated=False):
        pass
    @staticmethod
    def drawBokehGraph(dataset, calib_eq_odds_group_0_test_model_shap, calib_eq_odds_group_1_test_model_shap, calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model):  
        pass
    @staticmethod
    def plot_shap_summaryplot(dataset):
        pass

class Model:
    
    def __init__(self, datadf):
        self.pred = datadf['prediction'].copy()
        self.label = datadf['label'].copy()
        self.label = datadf['label'].copy()
        self.datadf = datadf.copy()
        self.datadf['cfact_dist'] = 1/datadf['fitness']
        self.datadf = self.datadf.query("cfact_dist == cfact_dist")
                    
    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label
    
    def get_burden(self):
        return self.datadf['cfact_dist'].mean()

    def burden_eq_opp(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)
                   
        print(1-sp2p, sn2p, 1-op2p, on2p, "pburden")
        # find indices close to the border and flip them
        if self.get_burden() > othr.get_burden():
            burdened_predictor = "self"
        else:
            burdened_predictor = "other"
        
        if burdened_predictor is "other":
            priv_df = self.datadf.copy()
            disa_df = othr.datadf.copy()
            priv_n2p = sn2p[0]
            disa_n2p = on2p[0]
        else:
            priv_df = othr.datadf.copy()
            disa_df = self.datadf.copy()
            priv_n2p = on2p[0]
            disa_n2p = sn2p[0]
                
        priv_neg = priv_df[priv_df['prediction'].round() == 0]
        disa_neg = disa_df[disa_df['prediction'].round() == 0]
                
        num_priv_n2p = int(priv_n2p * priv_df.shape[0]) 
        priv_ind = np.asarray(priv_neg.sort_values('cfact_dist', ascending = True).id)[:num_priv_n2p]     
        num_disa_n2p = int(disa_n2p * disa_df.shape[0]) 
        disa_ind = np.asarray(disa_neg.sort_values('cfact_dist', ascending = True).id)[:num_disa_n2p]


        priv_df.loc[priv_ind, "prediction"] = 1 - priv_df.loc[priv_ind, "prediction"]
        disa_df.loc[disa_ind, "prediction"] = 1 - disa_df.loc[disa_ind, "prediction"]
        
        fair_self = None
        fair_othr = None
                
        if burdened_predictor is "other":
            fair_self = Model(priv_df)
            fair_othr = Model(disa_df)
        else:
            fair_self = Model(disa_df)
            fair_othr = Model(priv_df)
                            
        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr
    
    def burden_eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)
                   
        print(1-sp2p, sn2p, 1-op2p, on2p, "burden")
        # find indices close to the border and flip them
        if self.get_burden() > othr.get_burden():
            burdened_predictor = "self"
        else:
            burdened_predictor = "other"
        
        if burdened_predictor is "self":
            priv_df = self.datadf
            disa_df = othr.datadf
        else:
            priv_df = othr.datadf
            disa_df = self.datadf
        
        self_df = self.datadf.copy(deep=True)
        self_df_pos = self_df[self_df["prediction"].round() == 1]
        self_df_neg = self_df[self_df["prediction"].round() == 0]
        othr_df = othr.datadf.copy(deep=True)
        othr_df_pos = othr_df[othr_df["prediction"].round() == 1]
        othr_df_neg = othr_df[othr_df["prediction"].round() == 0]
    
            
        num_sn2p = int(self_df.shape[0] *  (sn2p[0]))
        sn2p_indices = np.asarray(self_df_neg.sort_values('cfact_dist', ascending = True).id)[:num_sn2p]
        
        num_on2p = int(othr_df.shape[0] *  (on2p[0]))
        on2p_indices = np.asarray(othr_df_neg.sort_values('cfact_dist', ascending = True).id)[:num_on2p]
        
        num_sp2n = int(self_df.shape[0] *  (1 - sp2p[0]))
        sp2n_indices = np.asarray(self_df_pos.sort_values('cfact_dist', ascending = True).id)[:num_sp2n]
        
        num_op2n = int(othr_df.shape[0] *  (1 - op2p[0]))
        op2n_indices = np.asarray(othr_df_pos.sort_values('cfact_dist', ascending = True).id)[:num_op2n]
        
        # flip those values
        self_df.loc[sn2p_indices, 'prediction'] = 1 - self_df.loc[sn2p_indices, 'prediction']      
        self_df.loc[sp2n_indices, 'prediction'] = 1 - self_df.loc[sp2n_indices, 'prediction']      

        othr_df.loc[on2p_indices, 'prediction'] = 1 - othr_df.loc[on2p_indices, 'prediction']
        othr_df.loc[op2n_indices, 'prediction'] = 1 - othr_df.loc[op2n_indices, 'prediction']      
        
        fair_self = Model(self_df)
        fair_othr = Model(othr_df)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr
        
    def eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)
        
        # select random indices to flip in our model
        self_fair_pred = self.datadf.copy()
        self_pp_indices = self_fair_pred[self_fair_pred["prediction"] >= .5]
        self_pn_indices = self_fair_pred[self_fair_pred["prediction"] < .5]
        self_pp_indices = self_pp_indices.sample(frac=(1-sp2p[0]))
        self_pn_indices = self_pn_indices.sample(frac=sn2p[0])
        
        # flip randomly the predictions in our model
        self_fair_pred.loc[self_pn_indices.id] = 1 - self_fair_pred.loc[self_pn_indices.id]
        self_fair_pred.loc[self_pp_indices.id] = 1 - self_fair_pred.loc[self_pp_indices.id]

        # select random indices to flip in the other model
        othr_fair_pred = othr.datadf.copy()
        othr_pp_indices = othr_fair_pred[othr_fair_pred["prediction"] >= .5]
        othr_pn_indices = othr_fair_pred[othr_fair_pred["prediction"] < .5]
        othr_pp_indices = othr_pp_indices.sample(frac=(1-op2p[0]))
        othr_pn_indices = othr_pn_indices.sample(frac=on2p[0])

        # dlip randomly the precitions of the other model
        othr_fair_pred.loc[othr_pn_indices.id] = 1 - othr_fair_pred.loc[othr_pn_indices.id]
        othr_fair_pred.loc[othr_pp_indices.id] = 1 - othr_fair_pred.loc[othr_pp_indices.id]

        # create new model objects with the now fair predictions
        
        fair_self = Model(self_fair_pred)
        fair_othr = Model(othr_fair_pred)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr   
        
        
    def eq_odds_optimal_mix_rates(self, othr):
        sbr = float(self.base_rate())
        obr = float(othr.base_rate())

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = self.fpr() * sp2p + self.tnr() * sn2p
        sfnr = self.fnr() * sn2n + self.tpr() * sp2n
        ofpr = othr.fpr() * op2p + othr.tnr() * on2p
        ofnr = othr.fnr() * on2n + othr.tpr() * op2n
        error = sfpr + sfnr + ofpr + ofnr

        sflip = 1 - self.pred
        sconst = self.pred
        oflip = 1 - othr.pred
        oconst = othr.pred

        sm_tn = np.logical_and(self.pred.round() == 0, self.label == 0)
        sm_fn = np.logical_and(self.pred.round() == 0, self.label == 1)
        sm_tp = np.logical_and(self.pred.round() == 1, self.label == 1)
        sm_fp = np.logical_and(self.pred.round() == 1, self.label == 0)

        om_tn = np.logical_and(othr.pred.round() == 0, othr.label == 0)
        om_fn = np.logical_and(othr.pred.round() == 0, othr.label == 1)
        om_tp = np.logical_and(othr.pred.round() == 1, othr.label == 1)
        om_fp = np.logical_and(othr.pred.round() == 1, othr.label == 0)

        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / sbr + \
                      (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / sbr

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - sbr) + \
                      (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - sbr)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / obr + \
                      (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / obr

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - obr) + \
                      (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - obr)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
        return res

    def distance(self, othr):
        x = (self.fpr() - othr.fpr()) ** 2
        y = (self.tpr() - othr.tpr()) ** 2
        return math.sqrt(x + y)
    
    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])