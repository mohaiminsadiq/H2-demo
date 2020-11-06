from flask import Flask, render_template, request, redirect, url_for
app = Flask(__name__)

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
            return redirect(url_for('method'))
    else:
        return render_template('dataset.html')

@app.route('/method', methods=['POST', 'GET'])
def method():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Go back':
            return redirect(url_for('dataset'))
        elif request.form['submit_button'] == 'Pick method':
            pass
            #return redirect(url_for(''))
    else:
        return render_template('method.html')



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
