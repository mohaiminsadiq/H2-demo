from python_util.API import API
from python_util.modules.ShapModel import ShapModel

from flask import Flask, render_template, request, redirect, url_for, make_response
from bokeh.embed import components

app = Flask(__name__)

@app.route('/bokeh_self_check', methods=['POST', 'GET'])
def bokeh_self_check():
	
	src, plt = randomBar.get_bokeh_chart()

	return render_template('bokeh_test.html',
							bokeh_plot=plt,
							bokeh_src=src)

@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Start Demo':
            return redirect(url_for('dataset'))
        else:
            pass
    #elif request.method == 'GET':
    else:
        return render_template('home.html')

@app.route('/dataset', methods=['POST', 'GET'])
def dataset():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Go back':
            return redirect(url_for('home'))
        elif request.form['submit_button'] == 'Pick dataset':
            resp = make_response(redirect(url_for('method')))
            resp.set_cookie('dataset', request.form['dataset'])
            return resp

    else:
        return render_template('dataset.html')

@app.route('/method', methods=['POST', 'GET'])
def method():
    if request.method == 'POST':
        app.logger.debug("method got a POST")
        app.logger.debug(str(request.form))
        if request.form['submit_button'] == 'Go back':
            return redirect(url_for('dataset'))
        elif request.form['submit_button'] == 'Pick method':
            resp = make_response(redirect(url_for('results')))
            resp.set_cookie('method', request.form['method'])
            return resp
    else:
        return render_template('method.html')

@app.route('/results', methods=['POST', 'GET'])
def results():
    method = request.cookies.get('method').lower()
    dataset = request.cookies.get('dataset').lower()
    if method is None or dataset is None:
        return render_template('result_error.html')
    if request.method == 'POST':
        if request.form['submit_button'] == 'Go back':
            return redirect(url_for('method'))
        elif request.form['submit_button'] == 'Pick method':
            pass
            #return redirect(url_for(''))
    else:
        graph_handle = API(debias_method=method, dataset=dataset)
        if method == 'burden':
            burden_plt, burden_src = components(graph_handle.get_repr_graph())
            demo_plt, demo_src = components(graph_handle.get_demoParity_graph())
            scatter_plt, scatter_src = components(graph_handle.get_scatter_plot())
            return render_template('results_burden.html',
                                    burden_plt=burden_plt,
                                    burden_src=burden_src,
                                    demo_plt=demo_plt,
                                    demo_src=demo_src,
                                    scat_plt=scatter_plt,
                                    scat_src=scatter_src,
                                    burden_table=graph_handle.get_model_results())
        elif method == 'shap':
            #detection
            shap_plt, shap_src = components(graph_handle.get_repr_graph())
            graph_handle.get_demoParity_graph()
            graph_handle.get_eq_odds_graphs()

            #mitigation, this needs two API objects

            #Randomized regular and calibrated equalized odds
            randomized_model = API(debias_method=method, dataset=dataset)

            randomized_model.get_model_results(calibrated=False, shap_enabled=False)
            randomized_model.get_model_results(calibrated=True, shap_enabled=False)

            rand_eo_g0, rand_eo_g1, rand_ceo_g0, rand_ceo_g1 = randomized_model.model.dicts_compiler()

            #Shap-based regular and calibrated equalized odds
            shap_model = API(debias_method=method, dataset=dataset)

            shap_model.get_model_results(calibrated=False, shap_enabled=True)
            shap_model.get_model_results(calibrated=True, shap_enabled=True)

            shap_eo_g0, shap_eo_g1, shap_ceo_g0, shap_ceo_g1 = shap_model.model.dicts_compiler()

            calib_eq_odds_plt, calib_eq_odds_src = components(graph_handle.get_calib_eq_odds_graph(shap_model.model.calib_eq_odds_group_0_test_model, shap_model.model.calib_eq_odds_group_1_test_model, 
            randomized_model.model.calib_eq_odds_group_0_test_model, randomized_model.model.calib_eq_odds_group_1_test_model))
            # demo_plt, demo_src = components(graph_handle.get_demoParity_graph())
            # scatter_plt, scatter_src = components(graph_handle.get_scatter_plot())

            return render_template('results_shap.html',
                                    shap_plt=shap_plt,
                                    shap_src=shap_src,
                                    rand_eq_odds_results_group_0 = rand_eo_g0,
                                    rand_eq_odds_results_group_1 = rand_eo_g1,
                                    rand_calib_eq_odds_results_group_0 = rand_ceo_g0, 
                                    rand_calib_eq_odds_results_group_1 = rand_ceo_g1,
                                    shap_eq_odds_results_group_0 = shap_eo_g0,
                                    shap_eq_odds_results_group_1 = shap_eo_g1,
                                    shap_calib_eq_odds_results_group_0 = shap_ceo_g0, 
                                    shap_calib_eq_odds_results_group_1 = shap_ceo_g1,
                                    calib_eq_odds_plt=calib_eq_odds_plt,
                                    calib_eq_odds_src=calib_eq_odds_src)
        elif method == 'loco':
            loco_plt, loco_src = components(graph_handle.get_repr_graph())
            demo_plt, demo_src = components(graph_handle.get_demoParity_graph())
            scatter_plt, scatter_src = components(graph_handle.get_scatter_plot())
            return render_template('results_loco.html',
                                    loco_plt=loco_plt, 
                                    loco_src=loco_src,
                                    demo_plt=demo_plt,
                                    demo_src=demo_src,
                                    scatter_plt=scatter_plt, 
                                    scatter_src=scatter_src,
                                    loco_table=graph_handle.get_model_results())

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r


if __name__=='__main__':
    app.run(debug=True)
