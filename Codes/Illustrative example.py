# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:48:52 2021

@author: MSI-PC
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pulp as pl
import cplex
from math import radians, cos, sin, asin, sqrt
import os
os.chdir('C:/Users/MSI-PC/Desktop/')
import time
import warnings
warnings.filterwarnings('ignore')




''''''''''''''''''''''''''''''''
'''LP model (inversed optimization)'''
''''''''''''''''''''''''''''''''
from gurobipy import *

dataset = pd.DataFrame.from_dict(
            {"individual_id": [1,2,3,4,5,6,7,8,9,10],
             "home_x": [4,4,4,4,4,10,10,10,10,10],
             "home_y": [10,10,10,10,10,8,8,8,8,8],
             "shopping_id": [1,1,2,2,3,1,1,2,3,3],
             "dinner_id":[1,3,3,4,3,2,3,3,2,3]
            })

dataset_shopping = pd.DataFrame.from_dict(
            {"individual_id": [1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10],
             "home_x": [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
             "home_y": [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
             "shopping_id": [1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3,1,2,3],
             "dinner_id": [1,1,1,3,3,3,3,3,3,4,4,4,3,3,3,2,2,2,3,3,3,3,3,3,2,2,2,3,3,3],
             "chosen":[1,0,0,1,0,0,0,1,0,0,1,0,0,0,1,1,0,0,1,0,0,0,1,0,0,0,1,0,0,1]
            })

dataset_dinner = pd.DataFrame.from_dict(
            {"individual_id": [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,9,9,9,9,10,10,10,10],
             "home_x": [4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10],
             "home_y": [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8],
             "shopping_id": [1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,3,3,3,3,1,1,1,1,1,1,1,1,2,2,2,2,3,3,3,3,3,3,3,3],
             "dinner_id":[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4],
             "chosen": [1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,1,0]
            })

shop = pd.DataFrame.from_dict(
            {"shopping_id": [1,2,3],
             "shop_x": [8,2,12],
             "shop_y": [12,4,2],
             "n_store": [30,40,50]
            })

dinner = pd.DataFrame.from_dict(
            {"dinner_id": [1,2,3,4],
             "dinner_x": [2,14,8,6],
             "dinner_y": [14,10,6,2],
             "n_restaurant": [10,50,20,30]
            })

def calculate_data(dataset,shop,dinner):
    results = dataset.copy(deep=True)
    results = pd.merge(results,shop,on="shopping_id")
    results = pd.merge(results,dinner,on="dinner_id")
    results['dist_hs'] = ((results['home_x']-results['shop_x'])**2 + (results['home_y']-results['shop_y'])**2)**0.5
    results['dist_sd'] = ((results['dinner_x']-results['shop_x'])**2 + (results['dinner_y']-results['shop_y'])**2)**0.5
    results['dist_dh'] = ((results['dinner_x']-results['home_x'])**2 + (results['dinner_y']-results['home_y'])**2)**0.5
    return results

shopping_dataset = calculate_data(dataset_shopping,shop,dinner)
dinner_dataset = calculate_data(dataset_dinner,shop,dinner)



#Shopping_IO 
def solve_IO_shopping(aa,initialize,epsilon_c,iid,safe_boundary):
    dist_hs = float(aa[aa['chosen']==True]['dist_hs'])
    n_store = float(aa[aa['chosen']==True]['n_store'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(2,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['shopping_id'].iloc[j] != aa['shopping_id'][aa['chosen']==True].values[0]:
            m.addConstr((n_store-aa['n_store'].iloc[j])*x[0] + (dist_hs-aa['dist_hs'].iloc[j])*x[1] >= safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2#add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        variables = [0,0]
        Z = 0
    return np.array(variables),Z



#Shopping IO fixed variable
def solve_IO_shopping_fixvar(aa,initialize,shared_cof,epsilon_c,iid,safe_boundary):
    dist_hs = float(aa[aa['chosen']==True]['dist_hs'])
    n_store = float(aa[aa['chosen']==True]['n_store'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(1,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['shopping_id'].iloc[j] != aa['shopping_id'][aa['chosen']==True].values[0]:
            m.addConstr((dist_hs-aa['dist_hs'].iloc[j])*shared_cof + (n_store-aa['n_store'].iloc[j])*x[0] >=safe_boundary)
    obj = (x[0]-initialize[0])**2 #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        variables = np.append(variables,shared_cof)
        Z = m.ObjVal
    except:
        variables = [0,0]
        Z = 0
    return np.array(variables),Z


#Dinner_IO 
def solve_IO_dinner(aa,initialize,epsilon_c,iid,safe_boundary):
    dist_sd = float(aa[aa['chosen']==True]['dist_sd'])
    dist_dh= float(aa[aa['chosen']==True]['dist_dh'])
    n_restaurant = float(aa[aa['chosen']==True]['n_restaurant'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(2,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['dinner_id'].iloc[j] != aa['dinner_id'][aa['chosen']==True].values[0]:
            m.addConstr((n_restaurant-aa['n_restaurant'].iloc[j])*x[0] + (dist_sd-aa['dist_sd'].iloc[j])*x[1] + (dist_dh-aa['dist_dh'].iloc[j])*x[1]>=safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2#add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        variables = [0,0]
        Z = 0
    return np.array(variables),Z


#Dinner IO fixed variable
def solve_IO_dinner_fixvar(aa,initialize,shared_cof,epsilon_c,iid,safe_boundary):
    dist_sd = float(aa[aa['chosen']==True]['dist_sd'])
    dist_dh= float(aa[aa['chosen']==True]['dist_dh'])
    n_restaurant = float(aa[aa['chosen']==True]['n_restaurant'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(1,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['dinner_id'].iloc[j] != aa['dinner_id'][aa['chosen']==True].values[0]:
            m.addConstr((n_restaurant-aa['n_restaurant'].iloc[j])*x[0] + (dist_sd-aa['dist_sd'].iloc[j])*shared_cof + (dist_dh-aa['dist_dh'].iloc[j])*shared_cof>=safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        variables = np.append(variables,shared_cof)
        Z = m.ObjVal
    except:
        variables = [0,0]
        Z = 0
    return np.array(variables),Z



#test_IO
np.random.seed(8521)
shuffle = np.random.permutation(10)+1

alter_num_s = 3
alter_num_d = 4
epsilon_s = np.random.gumbel(0,1,10*alter_num_s).reshape(10,alter_num_s)
epsilon_d = np.random.gumbel(0,1,10*alter_num_d).reshape(10,alter_num_d)


aa = shopping_dataset[shopping_dataset['individual_id']==1]
variable,Z = solve_IO_shopping(aa,[0,0],epsilon_s,iid=1,safe_boundary=2)
print(variable)
variable2,Z2 = solve_IO_shopping_fixvar(aa,[0],variable[-1],epsilon_s,iid=1,safe_boundary=2)
print(variable2)


aa = dinner_dataset[dinner_dataset['individual_id']==1]
variable,Z = solve_IO_dinner(aa,[0,0],epsilon_d,iid=1,safe_boundary=2)
print(variable)
variable2,Z2 = solve_IO_dinner_fixvar(aa,[0],variable[-1],epsilon_d,iid=1,safe_boundary=2)
print(variable2)


#######################
#####Main Fucntion#####
#######################
def One_iteration(shopping_dataset,dinner_dataset,epsilon_s,epsilon_d, x_k,bound, boundary_max,boundary_min,step):
    ##0.get initial parameters
    x_k_s = np.append(x_k[0],x_k[2])
    x_k_d = np.append(x_k[1],x_k[2])
    
    ##1.Run three IO models
    parameter_i_lst_s = []
    parameter_i_lst_d = []
    sb_s = []
    sb_d = []
    for i in range(1,11):
        #shopping
        aa = shopping_dataset[shopping_dataset['individual_id']==i]
        safe_boundary_s = boundary_max
        parameter_i_s,Z = solve_IO_shopping(aa,x_k_s,epsilon_s,i,safe_boundary_s)
        while ((parameter_i_s.max()>bound)or(parameter_i_s.min()<-bound)or(parameter_i_s.sum()==0)) and (safe_boundary_s>boundary_min):
            safe_boundary_s -= step
            parameter_i_s,Z = solve_IO_shopping(aa,x_k_s,epsilon_s,i,safe_boundary_s)
        parameter_i_lst_s.append(parameter_i_s)
        sb_s.append(safe_boundary_s)
        #dinner
        bb = dinner_dataset[dinner_dataset['individual_id']==i]
        safe_boundary_d = boundary_max
        parameter_i_d,Z = solve_IO_dinner(bb,x_k_d,epsilon_d,i,safe_boundary_d)
        while ((parameter_i_d.max()>bound)or(parameter_i_d.min()<-bound)or(parameter_i_d.sum()==0)) and (safe_boundary_d>boundary_min):
            safe_boundary_d -= step
            parameter_i_d,Z = solve_IO_dinner(bb,x_k_d,epsilon_d,i,safe_boundary_d)
        parameter_i_lst_d.append(parameter_i_d)
        sb_d.append(safe_boundary_d)
    #shopping
    parameter_i_lst_0_s = np.array(parameter_i_lst_s)
    parameter_i_lst_s = parameter_i_lst_0_s[(parameter_i_lst_0_s.max(axis=1)<bound)&
                                      (parameter_i_lst_0_s.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_s.sum(axis=1)!=0)]
    #dinner
    parameter_i_lst_0_d = np.array(parameter_i_lst_d)
    parameter_i_lst_d = parameter_i_lst_0_d[(parameter_i_lst_0_d.max(axis=1)<bound)&
                                      (parameter_i_lst_0_d.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_d.sum(axis=1)!=0)]

    
    ##2.Uniform the parameter of ln_dwork
    travel_dist_2lst = np.array([parameter_i_lst_0_s[:,-1],parameter_i_lst_0_d[:,-1]])
    travel_dist_lst = travel_dist_2lst.mean(axis=0)

    #3.Fix ln_work and update the rest parameters
    parameter_i_lst_s = []
    parameter_i_lst_d = []
    sb_s2 = []
    sb_d2 = []
    for j,i in enumerate(range(1,11)):
        #shopping
        aa = shopping_dataset[shopping_dataset['individual_id']==i]
        safe_boundary_s = sb_s[j]
        parameter_i_s,Z = solve_IO_shopping_fixvar(aa,x_k_s[:-1],travel_dist_lst[j],epsilon_s,i,safe_boundary_s)
        while ((parameter_i_s.max()>bound)or(parameter_i_s.min()<-bound)or(parameter_i_s.sum()==0)) and (safe_boundary_s>boundary_min):
            safe_boundary_s -= step
            parameter_i_s,Z = solve_IO_shopping_fixvar(aa,x_k_s[:-1],travel_dist_lst[j],epsilon_s,i,safe_boundary_s)
        parameter_i_lst_s.append(parameter_i_s)
        sb_s2.append(safe_boundary_s)
        #lunch
        bb = dinner_dataset[dinner_dataset['individual_id']==i]
        safe_boundary_d = sb_d[j]
        parameter_i_d,Z = solve_IO_dinner_fixvar(bb,x_k_d[:-1],travel_dist_lst[j],epsilon_d,i,safe_boundary_d)
        while ((parameter_i_d.max()>bound)or(parameter_i_d.min()<-bound)or(parameter_i_d.sum()==0)) and (safe_boundary_d>boundary_min):
            safe_boundary_d -= step
            parameter_i_d,Z = solve_IO_dinner_fixvar(bb,x_k_d[:-1],travel_dist_lst[j],epsilon_d,i,safe_boundary_d)
        parameter_i_lst_d.append(parameter_i_d)
        sb_d2.append(safe_boundary_d)
    #shopping
    parameter_i_lst_0_s = np.array(parameter_i_lst_s)
    parameter_i_lst_s = parameter_i_lst_0_s[(parameter_i_lst_0_s.max(axis=1)<bound)&
                                      (parameter_i_lst_0_s.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_s.sum(axis=1)!=0)]
    #dinner
    parameter_i_lst_0_d = np.array(parameter_i_lst_d)
    parameter_i_lst_d = parameter_i_lst_0_d[(parameter_i_lst_0_d.max(axis=1)<bound)&
                                      (parameter_i_lst_0_d.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_d.sum(axis=1)!=0)]
    
    ##4.Calculate y_0
    y_k_s = parameter_i_lst_s.mean(axis=0)
    y_k_d = parameter_i_lst_d.mean(axis=0)
    y_k = np.append(y_k_s[:-1],y_k_d[:-1])
    travel_dist = (y_k_s[-1]+y_k_d[-1])/2
    y_k = np.append(y_k,travel_dist)
    
    return y_k, parameter_i_lst_0_s, parameter_i_lst_0_d,sb_s2,sb_d2


#test One_iteration
x_k = [0,0,0]
y_k, parameter_i_lst_0_s, parameter_i_lst_0_d,sb_s2,sb_d2 = One_iteration(shopping_dataset,dinner_dataset,epsilon_s,epsilon_d,x_k,bound=30,boundary_max=3,boundary_min=1,step=0.4)

check1 = pd.DataFrame(parameter_i_lst_0_s)
check2 = pd.DataFrame(parameter_i_lst_0_d)




def whole_experiment_SR(shopping_dataset,dinner_dataset,
                        x_0,tal=1.8,gama=0.3,bound=30,
                        boundary_max=3,boundary_min=-3,step=0.5,plot=True):
    ##0.Data Processing
    alter_num_s = 3
    alter_num_d = 4

    ##1.initialization
    start_time = time.time()
    params_track_x = []
    params_track_y = []
    k=1
    beta_k = 1
    #iteration 1
    epsilon_s = np.random.gumbel(0,1,10*alter_num_s).reshape(10,alter_num_s)
    epsilon_d = np.random.gumbel(0,1,10*alter_num_d).reshape(10,alter_num_d)
    x_0 = np.array(x_0)
    y_0, parameter_i_lst_0_s, parameter_i_lst_0_d,sb_s2,sb_d2 = One_iteration(shopping_dataset,dinner_dataset,epsilon_s,epsilon_d,x_0,bound,boundary_max,boundary_min,step)
    params_track_x.append(x_0) #record x
    params_track_y.append(y_0) #record y
    #iteration 2
    epsilon_s = np.random.gumbel(0,1,10*alter_num_s).reshape(10,alter_num_s)
    epsilon_d = np.random.gumbel(0,1,10*alter_num_d).reshape(10,alter_num_d)
    x_1 = y_0
    y_1, parameter_i_lst_0_s, parameter_i_lst_0_d,sb_s2,sb_d2 = One_iteration(shopping_dataset,dinner_dataset,epsilon_s,epsilon_d,x_1,bound,boundary_max,boundary_min,step)
    params_track_x.append(x_1) #record x
    params_track_y.append(y_1) #record y
    change = (x_1-x_0)/(x_0+1e-8)

    ##3.Main iteration
    while np.sum(np.abs(change))>0.003:
        #calculate x_k_next
        epsilon_s = np.random.gumbel(0,1,10*alter_num_s).reshape(10,alter_num_s)
        epsilon_d = np.random.gumbel(0,1,10*alter_num_d).reshape(10,alter_num_d)
        x_k = params_track_x[-1]
        y_k = params_track_y[-1]
        x_k_last = params_track_x[-2]
        y_k_last = params_track_y[-2]
        if np.linalg.norm(y_k-x_k)>= np.linalg.norm(y_k_last-x_k_last):
            beta_k = beta_k + tal
        else:
            beta_k = beta_k + gama 
        alpha_k = 1/beta_k
        x_k_next = x_k + alpha_k*(y_k - x_k)
        #calculate y_k_next
        y_k_next,parameter_i_lst_0_s,parameter_i_lst_0_d,sb_s2,sb_d2 = One_iteration(shopping_dataset,dinner_dataset,epsilon_s,epsilon_d,x_k_next,bound,boundary_max,boundary_min,step)
        #record x,y
        params_track_x.append(x_k_next) 
        params_track_y.append(y_k_next) 
        #update k, change
        k = k+1
        change = (x_k_next-x_k)/(x_k+1e-8)    
        
        if (k%10==0) and k>0:
            print('%i iteration finished'%k)
            print(np.sum(np.abs(change))) 
    
    ##4.Get Results
    theta_0 = pd.DataFrame(params_track_x)
    theta_ni_s = pd.DataFrame(parameter_i_lst_0_s)
    theta_ni_d = pd.DataFrame(parameter_i_lst_0_d)
    end_time = time.time()
    print('model running time: %is'%int(end_time-start_time))
    print('Meet the coveragence standard at %i-th iteration'%k)  
    
    if plot==True:
        #shopping
        fig,ax = plt.subplots()
        theta_0[0].plot(ax = ax,label='n_store')
        theta_0[2].plot(ax = ax,label='travel_dist')
        plt.legend()
        plt.title('Values of theta_0 in each iteration (shopping)')
        plt.xlabel('iteration_num')
        plt.ylabel('value')
        fig,ax = plt.subplots()
        theta_ni_s[0].hist(ax = ax,bins=10,label='n_store')
        theta_ni_s[1].hist(ax = ax,bins=10,label='travel_dist')
        plt.legend()
        plt.title('Distribution of theta_i after the final iteration (shopping)')
        plt.xlabel('value')
        plt.ylabel('frequency')
        #lunch
        fig,ax = plt.subplots()
        theta_0[1].plot(ax = ax,label='n_restaurant')
        theta_0[2].plot(ax = ax,label='travel_dist')
        plt.legend()
        plt.title('Values of theta_0 in each iteration (dinner)')
        plt.xlabel('iteration_num')
        plt.ylabel('value')
        fig,ax = plt.subplots()
        theta_ni_d[0].hist(ax = ax,bins=30,label='n_restaurant')
        theta_ni_d[1].hist(ax = ax,bins=30,label='travel_dist')
        plt.legend()
        plt.title('Distribution of theta_i after the final iteration (dinner)')
        plt.xlabel('value')
        plt.ylabel('frequency')
    
    ##5.Outputs
    theta_ni_s['individual_id'] = range(1,11)
    theta_ni_s['safe_boundary'] = sb_s2
    theta_ni_d['individual_id'] = range(1,11)
    theta_ni_d['safe_boundary'] = sb_d2
    return theta_0, theta_ni_s, theta_ni_d
        

#Run the model
# from sklearn.preprocessing import MinMaxScaler
# ms = MinMaxScaler()
# shopping_dataset.iloc[:,7:] = ms.fit_transform(shopping_dataset.iloc[:,7:].values)
# dinner_dataset.iloc[:,7:] = ms.fit_transform(dinner_dataset.iloc[:,7:].values)




x_0 = [0,0,0]
theta_0,theta_ni_s,theta_ni_d = whole_experiment_SR(shopping_dataset,dinner_dataset,x_0
                                                    ,tal=1.8,gama=0.3,bound=30,boundary_max=1,
                                                    boundary_min=-1,step=0.5,plot=True)     


theta_0.iloc[-1]
# theta_0.to_csv('toy_sample_theta_0.csv',index=False)
# theta_ni_s.to_csv('toy_sample_theta_ni_s.csv',index=False)
# theta_ni_d.to_csv('toy_sample_theta_ni_d.csv',index=False)



shopping_dataset2 = shopping_dataset[shopping_dataset['chosen']==True]
dinner_dataset2 = dinner_dataset[dinner_dataset['chosen']==True]

results = pd.merge(theta_ni_s,shopping_dataset2,on='individual_id')
results['V_shopping'] = results[0]*results['n_store']+results[1]*results['dist_hs']

results2 = pd.merge(theta_ni_d,dinner_dataset2,on='individual_id')
results2['V_dinner'] = results2[0]*results2['n_restaurant']+results2[1]*(results2['dist_sd']+results2['dist_dh'])


aa = results['V_shopping']+results2['V_dinner']
results2.mean()









