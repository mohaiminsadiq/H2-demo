import json

def test_valid_home(app,client):
	res = client.get('/')
	assert 200 <= res.status_code < 300
def test_valid_bokeh(app,client):
	res = client.get('/bokeh_self_check')
	assert 200 <= res.status_code < 300
def test_valid_data(app,client):
	res = client.get('/dataset')
	assert 200 <= res.status_code < 300
def test_valid_method(app,client):
	res = client.get('/method')
	assert 200 <= res.status_code < 300
def test_valid_result(app,client):	
	res = client.get('/results')
	assert 200 <= res.status_code < 300
def test_valid_test(app,client):
	res = client.get('/testing')
	assert 200 <= res.status_code < 300
	
	
def test_invalid(app,client):
	res = client.get('/invalid')
	assert 400 <= res.status_code < 600
def test_cookei_corrupt(app,client):
	to_set = {"1":2}
	client.set_cookie('/','results',json.dumps(to_set))
	res = client.get('/results')
	assert 400 <= res.status_code < 600
	
