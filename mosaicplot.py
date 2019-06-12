# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 00:30:33 2019

@author: thimo
"""

# Create a mosaicplot for the titanic dataset

import pandas as pd
import numpy as np
import os
from statsmodels.graphics.mosaicplot import mosaic
os.chdir('C:/Users/thimo/Dropbox/corsi/machine_learning/real-world-machine-learning-master')
df = pd.read_csv("data/titanic.csv")

# Mosaic plot
mosaic(df, ['Sex', 'Survived'])
# boxplot
df.boxplot(column='Age', by='Survived')


