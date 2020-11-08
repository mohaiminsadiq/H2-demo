import cvxpy as cvx
import numpy as np
import pandas as pd
from collections import namedtuple, Counter, OrderedDict
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from bokeh.plotting import figure, output_file, show
from bokeh.models import LinearInterpolator, Span
from bokeh.io import output_notebook
from bokeh.embed import components

class ShapModel:

    def shap_val_test_split(self, dataset):
        # Load the validation set scores from csvs
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_shap_postprocess.csv")
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_shap_postprocess.csv")

        test_and_val_data = pd.read_csv(data_filepath)
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

        self.group_0_val_model = Model(group_0_val_data['id'].to_numpy(),  group_0_val_data['shap'].to_numpy(), 
                                    group_0_val_data['prediction'].to_numpy(), group_0_val_data['label'].to_numpy())
        self.group_1_val_model = Model(group_1_val_data['id'].to_numpy(),  group_1_val_data['shap'].to_numpy(),
                                    group_1_val_data['prediction'].to_numpy(), group_1_val_data['label'].to_numpy())
        self.group_0_test_model = Model(group_0_test_data['id'].to_numpy(), group_0_test_data['shap'].to_numpy(),
                                    group_0_test_data['prediction'].to_numpy(), group_0_test_data['label'].to_numpy())
        self.group_1_test_model = Model(group_1_test_data['id'].to_numpy(), group_1_test_data['shap'].to_numpy(),
                                    group_1_test_data['prediction'].to_numpy(), group_1_test_data['label'].to_numpy())


    def equalized_odds_shap(self, dataset, shap_enabled=False):
        self.shap_val_test_split(dataset)

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.eq_odds(self.group_0_val_model, self.group_1_val_model)

        # Apply the mixing rates to the test models
        self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model = Model.eq_odds(self.group_0_test_model,
                                                                                self.group_1_test_model, 
                                                                                shap_enabled,
                                                                                mix_rates)

        return self.group_0_test_model, self.group_1_test_model, self.eq_odds_group_0_test_model, self.eq_odds_group_1_test_model



        
    def calibrated_equalized_odds_shap(self, dataset, cost_constraint, shap_enabled=False):

        self.shap_val_test_split(dataset)
        # Cost constraint
        cost_constraint = cost_constraint#'weighted'
        if cost_constraint not in ['fnr', 'fpr', 'weighted']:
            raise RuntimeError('cost_constraint (arg #2) should be one of fnr, fpr, weighted')

        if cost_constraint == 'fnr':
            fn_rate = 1
            fp_rate = 0
        elif cost_constraint == 'fpr':
            fn_rate = 0
            fp_rate = 1
        elif cost_constraint == 'weighted':
            fn_rate = 1
            fp_rate = 1

        # Find mixing rates for equalized odds models
        _, _, mix_rates = Model.calib_eq_odds(self.group_0_val_model, self.group_1_val_model, fp_rate, fn_rate)

        # Apply the mixing rates to the test models
        self.calib_eq_odds_group_0_test_model, self.calib_eq_odds_group_1_test_model = Model.calib_eq_odds(self.group_0_test_model,
                                                                                                    self.group_1_test_model,
                                                                                                    fp_rate, fn_rate, 
                                                                                                shap_enabled,
                                                                                                    mix_rates)

        return self.calib_eq_odds_group_0_test_model, self.calib_eq_odds_group_1_test_model

    #Plotting the convex hulls
    def plot_convex_hulls_equalized_odds(self, calibrated=False):
        points = np.array([(0,0), (self.group_0_test_model.fpr(), self.group_0_test_model.tpr()), (1-self.group_0_test_model.fpr(), 1-self.group_0_test_model.tpr()) , (1,1)])
        hull = ConvexHull(points)
        mylabel = 'group 0'
        for simplex in hull.simplices:
            plt.plot(points[simplex,0], points[simplex,1], 'k-', label=mylabel)
            mylabel = "_nolegend_"

        points = np.array([(0,0), (self.group_1_test_model.fpr(), self.group_1_test_model.tpr()), (1-self.group_1_test_model.fpr(), 1-self.group_1_test_model.tpr()), (1,1)])
        hull = ConvexHull(points)
        mylabel = 'group 1'
        for simplex in hull.simplices:
            plt.plot(points[simplex,0], points[simplex,1], 'r-', label=mylabel)
            mylabel = "_nolegend_"
            

        plt.plot(self.group_0_test_model.fpr(), self.group_0_test_model.tpr(), 'k+', label="Original Group 0")
        plt.plot(self.group_1_test_model.fpr(), self.group_1_test_model.tpr(), 'r+', label="Original Group 1")
        
        if calibrated:
            plt.plot(self.calib_eq_odds_group_0_test_model.fpr(), self.calib_eq_odds_group_0_test_model.tpr(), 'k*', label="Calib Eq Odds Group 0")
            plt.plot(self.calib_eq_odds_group_1_test_model.fpr(), self.calib_eq_odds_group_1_test_model.tpr(), 'r*', label="Calib Eq Odds Group 1")
        else:
            plt.plot(self.eq_odds_group_0_test_model.fpr(), self.eq_odds_group_0_test_model.tpr(), 'k*', label="Eq Odds Group 0")
            plt.plot(self.eq_odds_group_1_test_model.fpr(), self.eq_odds_group_1_test_model.tpr(), 'r*', label="Eq Odds Group 1")

        plt.plot(0, 0, 'g+')

        plt.ylabel('Y = 1')
        plt.xlabel('Y = 0')
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.show()

        print("TPR and FPR difference before post processing {} {}".format((self.group_1_test_model.tpr() - self.group_0_test_model.tpr()), ((self.group_1_test_model.fpr() - self.group_0_test_model.fpr()))))
        if calibrated:
            print("TPR and FPR difference after post processing {} {}".format((self.calib_eq_odds_group_1_test_model.tpr() - self.calib_eq_odds_group_0_test_model.tpr()), ((self.calib_eq_odds_group_1_test_model.fpr() - self.calib_eq_odds_group_0_test_model.fpr()))))
        else:
            print("TPR and FPR difference after post processing {} {}".format((self.eq_odds_group_1_test_model.tpr() - self.eq_odds_group_0_test_model.tpr()), ((self.eq_odds_group_1_test_model.fpr() - self.eq_odds_group_0_test_model.fpr()))))
        print("FP Cost and FN Cost difference before post processing {} {}".format((self.group_1_test_model.fp_cost() - self.group_0_test_model.fp_cost()), ((self.group_1_test_model.fn_cost() - self.group_0_test_model.fn_cost()))))
        if calibrated:
            print("FP Cost and FN Cost difference after post processing {} {}".format((self.calib_eq_odds_group_1_test_model.fp_cost() - self.calib_eq_odds_group_0_test_model.fp_cost()), ((self.calib_eq_odds_group_1_test_model.fn_cost() - self.calib_eq_odds_group_0_test_model.fn_cost()))))
        else:
            print("FP Cost and FN Cost difference after post processing {} {}".format((self.eq_odds_group_1_test_model.fp_cost() - self.eq_odds_group_0_test_model.fp_cost()), ((self.eq_odds_group_1_test_model.fn_cost() - self.eq_odds_group_0_test_model.fn_cost()))))

    @staticmethod
    def drawBokehGraph(dataset, calib_eq_odds_group_0_test_model_shap, calib_eq_odds_group_1_test_model_shap, calib_eq_odds_group_0_test_model, calib_eq_odds_group_1_test_model):  
        my_path = os.path.abspath(os.path.dirname(__file__))
        if dataset == 'compas':
            data_filepath = os.path.join(my_path, "../data/compas_shap_postprocess.csv") 
        elif dataset == 'adult':
            data_filepath = os.path.join(my_path, "../data/adult_shap_postprocess.csv")

        test_and_val_data = pd.read_csv(data_filepath)
        test_and_val_data.reset_index(drop=True, inplace=True)
        shap_changed_ids = calib_eq_odds_group_1_test_model_shap.get_changed_id()
        shap_pred = test_and_val_data.loc[shap_changed_ids, 'prediction']
        shap_shap = test_and_val_data.loc[shap_changed_ids, 'shap']

        randomized_changed_ids =  calib_eq_odds_group_1_test_model.get_changed_id()
        random_pred = test_and_val_data.loc[randomized_changed_ids, 'prediction']
        random_shap = test_and_val_data.loc[randomized_changed_ids, 'shap']

        base_rate =  calib_eq_odds_group_1_test_model_shap.base_rate() 

        common_ids = set(shap_changed_ids).intersection(set(randomized_changed_ids))

        weights_shap = [200*i for i in Counter(shap_shap).values() for j in range(i)]
        weights_random = [200*i for i in Counter(random_shap).values() for j in range(i)]


        TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
        shap_df = pd.DataFrame(list(zip(shap_shap, shap_pred,weights_shap)), columns=["shap","pred","weights"])
        random_shap_df = pd.DataFrame(list(zip(random_shap, random_pred,weights_random)), columns=["shap","pred","weights"])
        shap_df.to_csv("shap_df.csv", index = False)
        random_shap_df.to_csv("random_shap_df.csv", index = False)
        size_mapper=LinearInterpolator(
            x=[shap_df.weights.min(), shap_df.weights.max()],
            y=[5,50]
        )

        random_size_mapper=LinearInterpolator(
            x=[random_shap_df.weights.min(), random_shap_df.weights.max()],
            y=[5,50]
        )
        p = figure(tools=TOOLS)

        p.scatter("shap", "pred", source=shap_df, fill_color = 'blue', fill_alpha = 0.6, size = {'field':'weights', 'transform': size_mapper}, legend_label='shap based')
        p.scatter("shap", "pred", source=random_shap_df, fill_color = 'red', fill_alpha = 0.6, size = {'field':'weights', 'transform': size_mapper}, legend_label='randomized')
        p.xaxis.axis_label = "Shap Scores"
        p.yaxis.axis_label = "Predictions"
        vline = Span(location=0, dimension='height', line_color='red', line_width=3, line_dash="dotted")
        hline = Span(location=base_rate, dimension='width', line_color='red', line_width=3,line_dash="dotted")

        p.renderers.extend([vline, hline])
        p.legend.click_policy="hide"
        #show(p)  # open a browser
        return components(p)

    @staticmethod
    def plot_shap_summaryplot(dataset):
        my_path = os.path.abspath(os.path.dirname(__file__))

        if dataset == "compas":
            shap_values = pd.read_csv(os.path.join(my_path, "../data/compas_shap_random_allfeatures.csv"))
            data_final = pd.read_csv(os.path.join(my_path, "../data/compas_processed.csv"))
            independent_columns  = ['priors_count','age_cat_25_45', 'age_cat_Greaterthan45',
            'age_cat_Lessthan25','race_random_African_American',
            'race_random_Caucasian',
            'sex_Female', 'sex_Male', 'c_charge_degree_M','c_charge_degree_F']

            X =  data_final.loc[:, independent_columns]
            sns.set()
            shap.summary_plot(shap_values.values, X)

            shap_values = pd.read_csv(os.path.join(my_path, "../data/compas_shap_normal_allfeatures.csv"))
            independent_columns  = ['priors_count','age_cat_25_45', 'age_cat_Greaterthan45',
            'age_cat_Lessthan25', 'race_African_American',
            'race_Caucasian',
            'sex_Female', 'sex_Male', 'c_charge_degree_M','c_charge_degree_F']

            X =  data_final.loc[:, independent_columns]

            sns.set()
            shap.summary_plot(shap_values.values, X)

        elif dataset == "adult":
            shap_values = pd.read_csv(os.path.join(my_path, "../data/adult_shap_random_allfeatures.csv"))
            data_final = pd.read_csv(os.path.join(my_path, "../data/adult_processed.csv"))
            independent_columns  = list(data_final)

            X =  data_final.loc[:, independent_columns]

            sns.set()
            shap.summary_plot(shap_values.values, X, max_display=shap_values.shape[1])

class Model(namedtuple('Model', 'id shap pred label')):
  
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
    
    def trivial(self):
        """
        Given a classifier, produces the trivial classifier
        (i.e. a model that just returns the base rate for every prediction)
        """
        base_rate = self.base_rate()
        pred = np.ones(len(self.pred)) * base_rate
        return Model(self.id, self.shap, pred, self.label)

    def weighted_cost(self, fp_rate, fn_rate, w_fp = 1, w_fn = 1):
        """
        Returns the weighted cost
        If fp_rate = 1 and fn_rate = 0, returns self.fp_cost
        If fp_rate = 0 and fn_rate = 1, returns self.fn_cost
        If fp_rate and fn_rate are nonzero, returns fp_rate * self.fp_cost * (1 - self.base_rate) +
            fn_rate * self.fn_cost * self.base_rate
        """
        norm_const = float(w_fp *fp_rate + w_fn *fn_rate) if (fp_rate != 0 and fn_rate != 0) else 1
        res = w_fp*fp_rate / norm_const * self.fp_cost() * (1 - self.base_rate()) + \
            w_fn*fn_rate / norm_const * self.fn_cost() * self.base_rate()
        return res
    
    def calib_eq_odds(self, other, fp_rate, fn_rate, shap_enabled =  False, mix_rates=None):
        if mix_rates is None:
          if fn_rate == 0:
              self_cost = self.fp_cost()
              other_cost = other.fp_cost()
              self_trivial_cost = self.trivial().fp_cost()
              other_trivial_cost = other.trivial().fp_cost()
          elif fp_rate == 0:
              self_cost = self.fn_cost()
              other_cost = other.fn_cost()
              self_trivial_cost = self.trivial().fn_cost()
              other_trivial_cost = other.trivial().fn_cost()
          else:
              w_fp =1
              w_fn =1
              self_cost = self.weighted_cost(fp_rate, fn_rate, w_fp, w_fn)
              other_cost = other.weighted_cost(fp_rate, fn_rate, w_fp, w_fn)
              self_trivial_cost = self.trivial().weighted_cost(fp_rate, fn_rate, w_fp, w_fn)
              other_trivial_cost = other.trivial().weighted_cost(fp_rate, fn_rate, w_fp, w_fn)
              print("Self_cost: {}, other_cost:{}, self_trivial_cost:{}, other_trivial_cost: {}".format(self_cost, other_cost, self_trivial_cost, other_trivial_cost))

          other_costs_more = other_cost > self_cost
          self_mix_rate = (other_cost - self_cost) / (self_trivial_cost - self_cost) if other_costs_more else 0
          other_mix_rate = 0 if other_costs_more else (self_cost - other_cost) / (other_trivial_cost - other_cost)
          print("The mix_rates are ", self_mix_rate, other_mix_rate)
        else:
          self_mix_rate = mix_rates[0]
          other_mix_rate = mix_rates[1]
        # New classifiers
        if shap_enabled:
          self_changed_id, self_new_pred = self.shap_fair_individuals_calibrated(self_mix_rate, advantaged  = True) #changed true to false here 
        else:
          self_changed_id, self_new_pred = self.randomized_fair_individuals_calibrated(self_mix_rate)
        
        
       
          
        calib_eq_odds_self = Model(self_changed_id, self.shap, self_new_pred, self.label)
        
        if self_mix_rate >0:
          print("Changing SELF Model--------------- for self mix rate ", self_mix_rate)
          df = self.get_pandas_df()          
          df.plot.scatter(x ='shap', y='pred')
          plt.axhline(y=self.base_rate(), color='r', linestyle='-')
          plt.title("Choosing " +str(len(self_changed_id)) + " Individuals from " + str(len(self_changed_id)) + "individuals")
          plt.show()
        else:
          print("Number of people with equal predictions in self " , np.sum(self_new_pred==self.pred))

        if shap_enabled:
          other_changed_id, other_new_pred = other.shap_fair_individuals_calibrated(other_mix_rate, advantaged = False)
        else:
          other_changed_id, other_new_pred = other.randomized_fair_individuals_calibrated(other_mix_rate)
        sns.set(rc={'figure.figsize':(5.7,5.27)})

        
          
        calib_eq_odds_other = Model(other_changed_id, other.shap, other_new_pred, other.label)
    
        if other_mix_rate >0:
          print("Changing Other Model--------------- for other mix rate ", other_mix_rate)
          df = other.get_pandas_df()
          sns.set(rc={'figure.figsize':(12.7,8.27)})
          df.plot.scatter(x ='shap', y='pred')
          plt.axhline(y=other.base_rate(), color='g', linestyle='--', linewidth = 5)
          plt.axvline(x=0, color='g', linestyle='--', linewidth = 5)
          plt.xlim(-0.1, 0.1)
          plt.title("Need to choose " +str(len(other_changed_id)) + " people from " + str(len(other.shap)) + " individuals")
          plt.show()
        else:
          print("Number of people with equal predictions in self " , np.sum(other_new_pred==other.pred))
        

        if mix_rates is None:
            return calib_eq_odds_self, calib_eq_odds_other, (self_mix_rate, other_mix_rate)
        else:
            return calib_eq_odds_self, calib_eq_odds_other
          
    def shap_fair_individuals_calibrated(self, mix_rate , advantaged):
        total = len(self.id)
        num_changed =int(mix_rate * total)
        df = self.get_pandas_df()
        print("In SHAP Method Number of people changed %d and total number %d " % (num_changed, total) )

        base_rate = self.base_rate()
        if advantaged:
          if mix_rate > 0:
            print("Redistributing Predictions for Advantaged Group")
          changed_indices = df.sort_values('shap', ascending = False).index[:num_changed]
          df.loc[changed_indices, 'pred'] = self.base_rate()

        else:
          if mix_rate > 0:
            print("Redistributing Predictions for Dis Advantaged Group")
            
          quadrant_1 = df[(df['shap'] > 0) & (df['pred'] > base_rate)].index
          quadrant_2 = df[(df['shap'] < 0) & (df['pred'] > base_rate)].index
          quadrant_3 = df[(df['shap'] <= 0) & (df['pred'] <= base_rate)].index
          quadrant_4 = df[(df['shap'] >= 0)& (df['pred'] <= base_rate)].index
          df['distance'] =  np.sqrt(np.square(df['shap']) + np.square(df['pred'] - base_rate))  
          print("Number of people in each quadrant ", len(quadrant_1),len(quadrant_2),len(quadrant_3),len(quadrant_4))
          
          required_changed_ids = num_changed
          print("Required_changed_ids/ num_changed: {}".format(required_changed_ids))
          changed_indices1 = (list(df.loc[(quadrant_1.union(quadrant_3))].sort_values('distance', ascending = False).index[:required_changed_ids]))
          print("People taken from Quadrant 1 and 3 ", len(changed_indices1))
          required_changed_ids -= len(changed_indices1)
          print("ID 's left to take  Adfter Quadrant 1 and 3 ", required_changed_ids)
          changed_indices2 = list(df.loc[quadrant_2.union(quadrant_4)].sort_values('distance', ascending = True).index[:required_changed_ids])
          required_changed_ids -= len(changed_indices2)
          print("Number of ids changed in the end ", len(changed_indices2))
          changed_indices = changed_indices1+ changed_indices2
            
          df.loc[changed_indices, 'pred'] = self.base_rate()
        return changed_indices, df.pred.copy()
    
    
    
    def randomized_fair_individuals_calibrated(self, mix_rate):
        total = len(self.id)
        changed_indices = np.random.permutation(self.id)[:int(mix_rate * total)]

        df = self.get_pandas_df()
        
        df.loc[changed_indices, 'pred'] = self.base_rate()

        return changed_indices, df.pred.copy() 

    def randomized_fair_individuals_equalized(self, p2p, n2p):
        from sklearn.utils import shuffle

        df = self.get_pandas_df()
        df['pred_outcome'] = df['pred'].round()
        original_pos =  df[df['pred_outcome'] == 1]
        # Changing positive predictions
        no_p2n= int(original_pos.shape[0] *  (1 - p2p))
        original_pos =  shuffle(original_pos)     
        p2n_indices = original_pos.head(no_p2n).index
        df.loc[p2n_indices, 'pred'] = 1 - df.loc[p2n_indices, 'pred']      
        
        #Changing negative predictions
        original_neg = df[df['pred_outcome']==0]
        no_n2p= int(original_neg.shape[0] *  (n2p))
        n2p_indices = original_neg.head(no_n2p).index
        df.loc[n2p_indices, 'pred'] = 1 - df.loc[n2p_indices, 'pred']      

        changed_ids = np.append(n2p_indices, p2n_indices)
        return changed_ids, df.pred.copy()

    def shap_fair_individuals_equalized(self, p2p, n2p):

        df = self.get_pandas_df()
        df['pred_outcome'] = df['pred'].round()
        original_pos =  df[df['pred_outcome'] == 1]
        # Changing positive predictions
        # Flipping positive predictions for people with lowest prediction without their race
        no_p2n= int(original_pos.shape[0] *  (1 - p2p))
        original_pos['pred_except_race_shap'] = original_pos['pred']-original_pos['shap']
        p2n_indices = np.asarray(original_pos.sort_values('pred_except_race_shap', ascending = True).index[:no_p2n])
        df.loc[p2n_indices, 'pred'] = 1 - df.loc[p2n_indices, 'pred']      
        
        #Changing negative predictions
        # Flipping Negative Predictions for people with highest contribution except SHAP
        original_neg = df[df['pred_outcome']==0]
        no_n2p= int(original_neg.shape[0] *  (n2p))
        original_neg['pred_except_race_shap'] = original_neg['pred']-original_neg['shap']
        n2p_indices = np.asarray(original_neg.sort_values('pred_except_race_shap', ascending = False).index[:no_n2p])
        df.loc[n2p_indices, 'pred'] = 1 - df.loc[n2p_indices, 'pred']      

        changed_ids = np.append(n2p_indices, p2n_indices)
        return changed_ids, df.pred.copy()
    
    def get_pandas_df(self):
        a = pd.DataFrame({'id':self.id, 'shap':self.shap,'label':self.label, 'pred':self.pred})
        a.set_index('id', inplace =  True)
        return a
        
    def get_changed_id(self):
      return self.id

    def eq_odds(self, other, shap_enabled =  False, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(other)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)

        df_self = self.get_pandas_df()
        if shap_enabled:
           print("using SHAP based selection of individuals for self ")
           self_changed_id, self_fair_pred = self.shap_fair_individuals_equalized(sp2p, sn2p)
      
        else:
          print("using random based selection of individuals for self ")
          self_changed_id, self_fair_pred = self.randomized_fair_individuals_equalized(sp2p, sn2p)

        df_other =  other.get_pandas_df()

        if shap_enabled:
          print("using SHAP based selection of individuals for other ")
          other_changed_id, other_fair_pred = other.shap_fair_individuals_equalized(op2p,on2p)
        else:
          print("using  random selection of individuals for other ")
          other_changed_id, other_fair_pred = other.randomized_fair_individuals_equalized(op2p,on2p)

        fair_self = Model(self_changed_id, self.shap, self_fair_pred, self.label)
        fair_other = Model(other_changed_id, other.shap, other_fair_pred, other.label)
  
        if not has_mix_rates:
            return fair_self, fair_other, mix_rates
        else:
            return fair_self, fair_other

    def eq_odds_optimal_mix_rates(self, othr):
      
        """Function to calculate mix rates for equalized odds"""
        np.random.seed(34)
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

      

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])   

    def results_dict(self):         
        return {'Accuracy': round(self.accuracy(),3),
            'F.P. cost': round(self.fp_cost(),3),
            'F.N. cost': round(self.fn_cost(),3),
            'Base rate': round(self.base_rate(),3),
            'Avg. score': round(self.pred.mean(),3)
            }  