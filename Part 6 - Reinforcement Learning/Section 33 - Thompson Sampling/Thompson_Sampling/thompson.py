# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 18:22:06 2019

@author: Alpish
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')
#Implementing Thompson Sampling
import random
N=10000
d=10
ads_selected=[]
total_reward=0
numbers_of_rewards_1=[0]*d
numbers_of_rewards_0=[0]*d
for n in range(0,N):
    ad=0
    max_random=0
    for i in range(0,d):
        random_beta=random.betavariate(numbers_of_rewards_1[i]+1,numbers_of_rewards_0[i]+1)
        if random_beta>max_random:
            max_random=random_beta
            ad=i
    ads_selected.append(i)
    reward=dataset.values[n,ad]
    if reward==1:
        numbers_of_rewards_1[ad]+=1
    else:
        numbers_of_rewards_0[ad]+=1
    total_reward+=reward
    
#visualising the results
plt.hist(ads_selected)
plt.title("histogram of ads selections")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad was selected")
plt.show()
