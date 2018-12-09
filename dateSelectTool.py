# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 09:46:11 2018

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import datetime
  
def update(val):
    global text1, text2
    
    YI = sYearI.val
    mI = sMonthI.val
    DI = sDayI.val
    HI = sHourI.val
    MI = sMinuteI.val
    SI = sSecondI.val
    
    YF = sYearF.val
    mF = sMonthF.val
    DF = sDayF.val
    HF = sHourF.val
    MF = sMinuteF.val
    SF = sSecondF.val
    
    dt1 = datetime.datetime(int(YI), int(mI), int(DI), int(HI), int(MI), int(SI))
    dt2 = datetime.datetime(int(YF), int(mF), int(DF), int(HF), int(MF), int(SF))
    x = np.array([dt1,dt2], dtype=object)
    t1.set_xdata(x)
    
    text1.set_text('Start Date: %s' % dt1)
    text2.set_text('End Date: %s' % dt2)

def reset(event):
    sYearI.reset()
    sMonthI.reset()
    sDayI.reset()
    sHourI.reset()
    sMinuteI.reset()
    sSecondI.reset()
    
    sYearF.reset()
    sMonthF.reset()
    sDayF.reset()
    sHourF.reset()
    sMinuteF.reset()
    sSecondF.reset()
    
    YI = sYearI.val
    mI = sMonthI.val
    DI = sDayI.val
    HI = sHourI.val
    MI = sMinuteI.val
    SI = sSecondI.val
    
    YF = sYearF.val
    mF = sMonthF.val
    DF = sDayF.val
    HF = sHourF.val
    MF = sMinuteF.val
    SF = sSecondF.val
    
    dt1 = datetime.datetime(int(YI), int(mI), int(DI), int(HI), int(MI), int(SI))
    dt2 = datetime.datetime(int(YF), int(mF), int(DF), int(HF), int(MF), int(SF))
    x = np.array([dt1,dt2], dtype=object)
    t1.set_xdata(x)
    
    text1.set_text('Start Date: %s' % dt1)
    text2.set_text('End Date: %s' % dt2)


# create figure with heatmap and spectra side-by-side subplots
fig1 = plt.figure(figsize=(16,9))
plt.subplots_adjust(bottom=0.4)

ax1 = plt.gca()
 
ax1.set_title(r'Date Span Selector', y = 1.01, fontsize=17)

dt10 = datetime.datetime(1999, 1, 1, 1, 0, 0)
dt20 = datetime.datetime(1999, 1, 1, 1, 0, 0)
x = np.array([dt10,dt20], dtype=object)
y = [0.0, 0.0]
t1, = ax1.plot(x, y, '*')

global text1, text2
text1, = ([plt.text(datetime.datetime(1980,1,1,0,0,0), 0.02, 'Start Date: %s' % dt10, fontsize=12)])
text2, = ([plt.text(datetime.datetime(1980,1,1,0,0,0), 0.01, 'End Date:   %s' % dt20, fontsize=12)])

dateMin = datetime.datetime(1900, 1, 1, 0, 0, 0)
dateMax = datetime.datetime(2100, 12, 31, 23, 59, 59)
ax1.set_xlim(dateMin, dateMax)   

# designate axes object for sliders
axYearI = plt.axes([0.15, 0.23, 0.3, 0.02])
axMonthI = plt.axes([0.15, 0.19, 0.3, 0.02])
axDayI = plt.axes([0.15, 0.15, 0.3, 0.02])
axHourI = plt.axes([0.15, 0.11, 0.3, 0.02])
axMinuteI = plt.axes([0.15, 0.07, 0.3, 0.02])
axSecondI = plt.axes([0.15, 0.03, 0.3, 0.02])

axYearF = plt.axes([0.55, 0.23, 0.3, 0.02])
axMonthF = plt.axes([0.55, 0.19, 0.3, 0.02])
axDayF = plt.axes([0.55, 0.15, 0.3, 0.02])
axHourF = plt.axes([0.55, 0.11, 0.3, 0.02])
axMinuteF = plt.axes([0.55, 0.07, 0.3, 0.02])
axSecondF = plt.axes([0.55, 0.03, 0.3, 0.02])

axStart = plt.axes([0.23, 0.27, 0.15, 0.03])
axEnd = plt.axes([0.63, 0.27, 0.15, 0.03])

axreset = plt.axes([0.47, 0.29, 0.05, 0.05])


# make sliders and add update function 
sYearI = Slider(axYearI, 'Year', 1900, 2100, valinit=1999, valfmt='%0.0f')
sYearI.on_changed(update)
sMonthI = Slider(axMonthI, 'Month', 1, 12, valinit=1, valfmt='%0.0f')
sMonthI.on_changed(update)
sDayI = Slider(axDayI, 'Day', 1, 31, valinit=1, valfmt='%0.0f')
sDayI.on_changed(update)
sHourI = Slider(axHourI, 'Hour', 0, 23, valinit=1, valfmt='%0.0f')
sHourI.on_changed(update)
sMinuteI = Slider(axMinuteI, 'Minute', 0, 59, valinit=0, valfmt='%0.0f')
sMonthI.on_changed(update)
sSecondI = Slider(axSecondI, 'Second', 0, 59, valinit=0, valfmt='%0.0f')
sSecondI.on_changed(update)

sYearF = Slider(axYearF, 'Year', 1900, 2100, valinit=1999, valfmt='%0.0f')
sYearF.on_changed(update)
sMonthF = Slider(axMonthF, 'Month', 1, 12, valinit=1, valfmt='%0.0f')
sMonthF.on_changed(update)
sDayF = Slider(axDayF, 'Day', 1, 31, valinit=1, valfmt='%0.0f')
sDayF.on_changed(update)
sHourF = Slider(axHourF, 'Hour', 0, 23, valinit=1, valfmt='%0.0f')
sHourF.on_changed(update)
sMinuteF = Slider(axMinuteF, 'Minute', 0, 59, valinit=0, valfmt='%0.0f')
sMonthF.on_changed(update)
sSecondF = Slider(axSecondF, 'Second', 0, 59, valinit=0, valfmt='%0.0f')
sSecondF.on_changed(update)

bStart = Button(axStart, 'Start', hovercolor='lightgray')
bEnd = Button(axEnd, 'End')

breset = Button(axreset, 'Reset')
breset.on_clicked(reset)

