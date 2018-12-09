# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 06:52:01 2018

@author: Brendan
"""

import numpy as np
import matplotlib.pyplot as plt
import datetime
import jdcal

dates = ['20101028', '20110207', '20110601', '20110909', '20111030', '20120111',
         '20120319', '20120621', '20120914', '20121227', '20130301', '20130618',
         '20131015', '20140130', '20140409', '20140601', '20140724', '20141205',
         '20150218', '20150530', '20150915', '20151231', '20160319', '20160602',
         '20160917', '20170111', '20170603', '20170830', '20171104', '20171230', 
         '20180315', '20180614']

fmt = '%Y%m%d'
    
greg_date = []
jul_date = []

for q in range(len(dates)):
    greg_date_temp = '%s/%s/%s' % (dates[q][0:4], dates[q][4:6], dates[q][6:8])
    greg_date = np.append(greg_date, greg_date_temp)
    dt = datetime.datetime.strptime(dates[q][0:10],fmt)
    jul_date_temp = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day)) + dt.hour/24.
    jul_date = np.append(jul_date, jul_date_temp)
    
    
plt.figure()
plt.plot(jul_date, np.arange(len(dates)))
#plt.plot(np.arange(len(dates)), np.arange(len(dates)))
plt.xticks(jul_date, greg_date, rotation=60)