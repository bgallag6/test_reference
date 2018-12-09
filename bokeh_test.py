# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:07:32 2018

@author: Brendan
"""


from bokeh.io import output_file, show
from bokeh.layouts import row, widgetbox
from bokeh.plotting import figure
from bokeh.models.widgets import Button, RadioButtonGroup, Select, Slider, RangeSlider
from bokeh.models import ColumnDataSource, CDSView, BooleanFilter

output_file("test10.html")

x = list(range(11))
y0 = x
y1 = [10 - i for i in x]
y2 = [abs(i - 5) for i in x]

# create a column data source for the plots to share
source = ColumnDataSource(data=dict(x=x, y0=y0, y1=y1))

# create a view of the source for one plot to use
view = CDSView(source=source, filters=[BooleanFilter([True if y < 250 else False for y in y1])])

#TOOLS = "box_zoom,box_select,lasso_select,tap,hover,reset,help"
TOOLS = "box_zoom,box_select,tap,reset,save"

# create a new plot *and add a renderer
s1 = figure(tools=TOOLS, plot_width=250, plot_height=250, title=None, toolbar_location="above")
s1.circle('x', 'y0', size=10, color="navy", hover_color="firebrick", alpha=0.5, source=source)
s1.toolbar.active_tap = None

# create another new plot, *add a renderer that uses the view of the data source
s2 = figure(tools=TOOLS, plot_width=250, plot_height=250, title=None)
s2.triangle('x', 'y1', size=10, color="firebrick", hover_color="navy", alpha=0.5, source=source, view=view)

# create and another
s3 = figure(plot_width=250, plot_height=250, title=None)
s3.square(x, y2, size=10, color="olive", alpha=0.5)

# create some widgets
slider = Slider(start=0, end=10, value=1, step=.1, title="Slider")
button_group = RadioButtonGroup(labels=["Option 1", "Option 2", "Option 3"], active=0)
select = Select(title="Option:", value="foo", options=["foo", "bar", "baz", "quux"])
button_1 = Button(label="Button 1")

range_slider = RangeSlider(start=0, end=10, value=(1,9), step=.1, title="Stuff")

widg1 = widgetbox(button_1, slider, button_group, select, range_slider, width=300)

# put the results in a row
show(row(widg1, s1, s2))

