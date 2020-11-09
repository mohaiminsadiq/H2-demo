from collections import namedtuple
import cvxpy as cvx
import numpy as np
import pandas as pd
from collections import namedtuple, Counter, OrderedDict
import os
from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearInterpolator, Span, TickFormatter
from bokeh.io import output_notebook
from bokeh.embed import components
from bokeh.models import Range1d
from bokeh.io import show
from bokeh.models import CustomJS, Slider
from  bokeh.models.sources import ColumnDataSource
from bokeh.layouts import column, row

def pair_wise_avg(iter):
        ret = []
        for x, obj in enumerate(iter):
            ret.append((obj[0] + obj[1])/2)
        return ret

class LocoModel:

    def loco_val_test_split(self, dataset):
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_loco_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_loco_postprocess.csv")

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

        self.group_0_val_model = Model(group_0_val_data['prediction'], group_0_val_data['label'], group_0_val_data['loco'])
        self.group_1_val_model = Model(group_1_val_data['prediction'], group_1_val_data['label'], group_1_val_data['loco'])
        self.group_0_test_model = Model(group_0_test_data['prediction'], group_0_test_data['label'], group_0_test_data['loco'])
        self.group_1_test_model = Model(group_1_test_data['prediction'], group_1_test_data['label'], group_1_test_data['loco'])


    def equalized_odds_loco(self, dataset):
        self.loco_val_test_split(dataset)

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.loco_eq_odds(self.group_0_val_model, self.group_1_val_model)

        # Apply the mixing rates to the test models
        self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model = Model.loco_eq_odds(self.group_0_test_model,
                                                                                self.group_1_test_model, 
                                                                                mix_rates)

        return self.group_0_test_model, self.group_1_test_model, self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model

    def equalized_odds_part_loco(self, dataset):
        self.loco_val_test_split(dataset)

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.loco_eq_opp(self.group_0_val_model, self.group_1_val_model)

        # Apply the mixing rates to the test models
        self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model = Model.loco_eq_opp(self.group_0_test_model,
                                                                                self.group_1_test_model, 
                                                                                mix_rates)

        return self.group_0_test_model, self.group_1_test_model, self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model

    
    def get_loco_scatter(self, dataset):
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_loco_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_loco_postprocess.csv")

        full_data = pd.read_csv(data_filepath)
        
        g0_loco = full_data[full_data['group'] == 0].loco
        g0_wiggle = 0.5*np.sin(np.arange(g0_loco.shape[0])) + .5
        
        g1_loco = full_data[full_data['group'] == 1].loco
        g1_wiggle = 0.5 * np.sin(np.arange(g1_loco.shape[0])) + .5
        g1_offset = (g1_wiggle)+2
        data = {'x':g1_loco, 'y':g1_offset}
        source = ColumnDataSource(data)
        
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="loco between two groups")#Plotting
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
               x=g0_loco,                      #bar heights
               width = .9,
               fill_alpha = .5,
               fill_color = 'blue',
               line_alpha = 0,
               legend_label="group 0"
          )#Signing the axis
        
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        p.yaxis.axis_label='Delta loco'
        
        p.sizing_mode = 'scale_width'
        p.legend.click_policy = "hide"
        
        p.yaxis.axis_label_text_font_size='16pt'
        p.yaxis.major_label_text_font_size='16pt'
        p.xaxis.axis_label_text_font_size='16pt'
        
        p.yaxis.axis_label_text_font_style='bold'
        p.xaxis.axis_label_text_font_style='normal'
        
        slider = Slider(start=0, end=2, value=2, step=.01, title="offset")
        callback = CustomJS(args=dict(source=source, offset=slider),
                    code="""
                            console.log("piss")
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
    
    def get_loco_graph(self, dataset): 
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_loco_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_loco_postprocess.csv")

        full_data = pd.read_csv(data_filepath)
        
        years = [0, 1]
        values = [(full_data[full_data['group'] == 0].loco).mean(),
                  (full_data[full_data['group'] == 1].loco).mean()]
        
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="loco between two groups")#Plotting
        p.vbar(years,                            #categories
               top = values,                      #bar heights
               width = .9,
               fill_alpha = .5,
               fill_color = ['blue', 'red'],
               line_alpha = .5,
               line_color='green',
               line_dash='solid',              
          )#Signing the axis
        
        hline = Span(location=0, dimension='width', line_color='black', line_width=1)
        p.renderers.extend([hline])
        p.yaxis.axis_label='loco score'
        
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
               
    def get_loco_demoParity(self, dataset): 
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_loco_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_loco_postprocess.csv")

        full_data = pd.read_csv(data_filepath)
        
        _, _, g0, g1 = self.equalized_odds_loco(dataset)
        
        width = .35
        padd = .45
        
        x_1 = [width, width*2+padd]
        x_0 = [0, width+padd]
        values_1 = [
                  full_data[(full_data['group'] == 1) & (full_data['label'] == 1)].prediction.round().mean(),
                  g1.pred.round().mean(),
                  ]
        values_0 = [full_data[(full_data['group'] == 0) & (full_data['label'] == 1)].prediction.round().mean(),
                    g0.pred.round().mean(),
        ]
        labels = ['default', 'loco']
        p = figure( plot_height=200, 
                    plot_width=400,
                    title="Demographic Parity")#Plotting
        p.vbar(x_0,                            #categories
              top = values_0,                      #bar heights
               width = width,
               fill_alpha = .5,
               fill_color = 'blue',
               line_alpha = .5,
               line_color='green',
               line_dash='solid',   
               legend_label="group 0"
          )#Signing the axis
        p.vbar(x_1,                            #categories
            top = values_1,                      #bar heights
            width = width,
            fill_alpha = .5,
            fill_color = 'red',
            line_alpha = .5,
            line_color='green',
            line_dash='solid',   
            legend_label="group 1"
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


class Model(namedtuple('Model', 'pred label loco')):
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

    def loco_eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)
                
        self_data = pd.DataFrame({'loco': self.loco, 'pred': self.pred, 'label': self.label})
                
        self_data['outcome'] = self_data['pred'].round()
        self_original_pos = self_data[self_data['outcome'] == 1]
        self_original_neg = self_data[self_data['outcome'] == 0]
        
        other_data = pd.DataFrame({'loco': othr.loco, 'pred': othr.pred, 'label': othr.label})
                
        other_data['outcome'] = other_data['pred'].round()
        other_original_pos = other_data[other_data['outcome'] == 1]
        other_original_neg = other_data[other_data['outcome'] == 0]
        
        # find indices with high loco score in race to flip
        # people with high race loco should be moved to the negative class as its artificlially pushing them positive
        num_sp2n= int(self_original_pos.shape[0] *  (1 - sp2p))
        self_original_pos['pred_except_race_loco'] = self_original_pos['pred']-self_original_pos['loco']
        sp2n_indices = np.asarray(self_original_pos.sort_values('pred_except_race_loco', ascending = True).index[:num_sp2n])
        
        num_op2n= int(other_original_pos.shape[0] *  (1 - op2p))
        other_original_pos['pred_except_race_loco'] = other_original_pos['pred']-other_original_pos['loco']
        op2n_indices = np.asarray(other_original_pos.sort_values('pred_except_race_loco', ascending = True).index[:num_op2n])
                
        # flip those values
        self_data.loc[sp2n_indices, 'pred'] = 1 - self_data.loc[sp2n_indices, 'pred']      
        other_data.loc[op2n_indices, 'pred'] = 1 - other_data.loc[op2n_indices, 'pred']
        
        
        # find indices with highly negative loco score in race to flip
        # people with highly negative race loco should be moved to the positive class as its artificlially pushing them negative
        num_sn2p= int(self_original_pos.shape[0] *  (sn2p))
        self_original_neg['pred_except_race_loco'] = self_original_neg['pred']-self_original_neg['loco']
        sn2p_indices = np.asarray(self_original_neg.sort_values('pred_except_race_loco', ascending = False).index[:num_sn2p])
        
        num_on2p= int(other_original_pos.shape[0] *  (on2p))
        other_original_neg['pred_except_race_loco'] = other_original_neg['pred']-other_original_neg['loco']
        on2p_indices = np.asarray(other_original_neg.sort_values('pred_except_race_loco', ascending = False).index[:num_on2p])
        
        # flip those values
        self_data.loc[sn2p_indices, 'pred'] = 1 - self_data.loc[sn2p_indices, 'pred']      
        other_data.loc[on2p_indices, 'pred'] = 1 - other_data.loc[on2p_indices, 'pred']

        fair_self = Model(np.array(self_data['pred']), self.label, self.loco)
        fair_othr = Model(np.array(other_data['pred']), othr.label, othr.loco)

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
        self_fair_pred = self.pred.copy()
        self_pp_indices, = np.nonzero(self.pred.round())
        self_pn_indices, = np.nonzero(1 - self.pred.round())
        np.random.shuffle(self_pp_indices)
        np.random.shuffle(self_pn_indices)

        # flip randomly the predictions in our model
        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = 1 - self_fair_pred[n2p_indices]
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = 1 - self_fair_pred[p2n_indices]

        # select random indices to flip in the other model
        othr_fair_pred = othr.pred.copy()
        othr_pp_indices, = np.nonzero(othr.pred.round())
        othr_pn_indices, = np.nonzero(1 - othr.pred.round())
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        # dlip randomly the precitions of the other model
        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = 1 - othr_fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = 1 - othr_fair_pred[p2n_indices]

        # create new model objects with the now fair predictions
        fair_self = Model(self_fair_pred, self.label, self.loco)
        fair_othr = Model(othr_fair_pred, othr.label, othr.loco)

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