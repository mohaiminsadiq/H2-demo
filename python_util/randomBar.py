import numpy as np

from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import factor_cmap

def get_bokeh_chart(y_lab="y", x_lab="x"):
	# Creating a list of categories
	years = np.arange(15)
	values = np.arange(15)
	
	p = figure( plot_height=300, 
				plot_width=400,
				title="Measles in the USA 2000-2015")#Plotting
	p.vbar(years,                            #categories
		  top = values,                      #bar heights
		   width = .9,
		   fill_alpha = .5,
		   fill_color = 'salmon',
		   line_alpha = .5,
		   line_color='green',
		   line_dash='dashed'
		  
	  )#Signing the axis
	p.xaxis.axis_label=x_lab
	p.yaxis.axis_label=y_lab
	
	return components(p)