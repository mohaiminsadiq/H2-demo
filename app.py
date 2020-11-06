from python_util import randomBar

from flask import Flask, render_template, request, redirect, url_for, make_response
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
    method = request.cookies.get('method')
    dataset = request.cookies.get('dataset')
    if method is None or dataset is None:
        return render_template('result_error.html')
    src, div = randomBar.get_bokeh_chart(method,
                                         dataset)
    if request.method == 'POST':
        if request.form['submit_button'] == 'Go back':
            return redirect(url_for('method'))
        elif request.form['submit_button'] == 'Pick method':
            pass
            #return redirect(url_for(''))
    else:
        return render_template('bokeh_test.html',
                                bokeh_plot=div,
                                bokeh_src=src)



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
