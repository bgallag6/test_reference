# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:23:21 2018

@author: Brendan
"""

import numpy as np
from bokeh.io import show, output_file
from bokeh.plotting import figure
from bokeh import events
from bokeh.models import CustomJS, Div, Button
from bokeh.layouts import column, row

def display_event(div, attributes=[], style = 'float:left;clear:left;font_size=10pt'):
    "Build a suitable CustomJS to display the current event in the div model."
    return CustomJS(args=dict(div=div), code="""
        var attrs = %s; var args = [];
        for (var i = 0; i<attrs.length; i++) {
            args.push(attrs[i] + '=' + Number(cb_obj[attrs[i]]).toFixed(2));
        }
        var line = "<span style=%r><b>" + cb_obj.event_name + "</b>(" + args.join(", ") + ")</span>\\n";
        var text = div.text.concat(line);
        var lines = text.split("\\n")
        if (lines.length > 35)
            lines.shift();
        div.text = lines.join("\\n");
    """ % (attributes, style))

x = np.random.random(size=4000) * 100
y = np.random.random(size=4000) * 100
radii = np.random.random(size=4000) * 1.5
colors = ["#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)]

p = figure(tools="pan,wheel_zoom,zoom_in,zoom_out,reset")
p.scatter(x, y, radius=np.random.random(size=4000) * 1.5,
          fill_color=colors, fill_alpha=0.6, line_color=None)

div = Div(width=1000)
button = Button(label="Button", button_type="success")
layout = column(button, row(p, div))

point_attributes = ['x','y','sx','sy']                     # Point events

p.js_on_event(events.Tap, display_event(div, attributes=point_attributes))

output_file("js_events.html", title="JS Events Example")
show(layout)