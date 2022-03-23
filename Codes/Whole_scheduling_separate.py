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
Commuting_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\2.初步模型试验\Commuting_choice_0507.csv")
Lunch_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\2.初步模型试验\Lunch_choice_0507.csv")
Afterwork_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\2.初步模型试验\Afterwork_choice_0507.csv")




#Commuting_IO 
def solve_IO_random_commuting(aa,initialize,epsilon_c,iid,safe_boundary):
    t_commute = float(aa[aa['chosen']==True]['t_commute'])
    c_commute = float(aa[aa['chosen']==True]['c_commute'])
    M_commute = float(aa[aa['chosen']==True]['M_commute2'])
    SDE_work = float(aa[aa['chosen']==True]['SDE_work'])
    SDL_work = float(aa[aa['chosen']==True]['SDL_work'])
    PL_work = float(aa[aa['chosen']==True]['PL_work'])
    ln_dwork = float(aa[aa['chosen']==True]['ln_dwork'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(7,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['alternative'].iloc[j] != str(aa['alternative'][aa['chosen']==True].values[0]):
            m.addConstr((t_commute-aa['t_commute'].iloc[j])*x[0] + (c_commute-aa['c_commute'].iloc[j])*x[1] + (M_commute-aa['M_commute2'].iloc[j])*x[2] + (SDE_work-aa['SDE_work'].iloc[j])*x[3] + (SDL_work-aa['SDL_work'].iloc[j])*x[4] + (PL_work-aa['PL_work'].iloc[j])*x[5] + (ln_dwork-aa['ln_dwork'].iloc[j])*x[6] >=epsilon_c[iid-1,j] - epsilon_c[iid-1,chosen_i] + safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2 + (x[2]-initialize[2])**2 + (x[3]-initialize[3])**2 + (x[4]-initialize[4])**2 + (x[5]-initialize[5])**2 + (x[6]-initialize[6])**2#add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        variables = [0,0,0,0,0,0,0]
        Z = 0
    return np.array(variables),Z

#Commuting_IO fixed variable
def solve_IO_random_commuting_fixvar(aa,initialize,cof_ln_dwork,epsilon_c,iid,safe_boundary):
    t_commute = float(aa[aa['chosen']==True]['t_commute'])
    c_commute = float(aa[aa['chosen']==True]['c_commute'])
    M_commute = float(aa[aa['chosen']==True]['M_commute2'])
    SDE_work = float(aa[aa['chosen']==True]['SDE_work'])
    SDL_work = float(aa[aa['chosen']==True]['SDL_work'])
    PL_work = float(aa[aa['chosen']==True]['PL_work'])
    ln_dwork = float(aa[aa['chosen']==True]['ln_dwork'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(6,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['alternative'].iloc[j] != str(aa['alternative'][aa['chosen']==True].values[0]):
            m.addConstr((t_commute-aa['t_commute'].iloc[j])*x[0] + (c_commute-aa['c_commute'].iloc[j])*x[1] + (M_commute-aa['M_commute2'].iloc[j])*x[2] + (SDE_work-aa['SDE_work'].iloc[j])*x[3] + (SDL_work-aa['SDL_work'].iloc[j])*x[4] + (PL_work-aa['PL_work'].iloc[j])*x[5] + (ln_dwork-aa['ln_dwork'].iloc[j])*cof_ln_dwork >= epsilon_c[iid-1,j] - epsilon_c[iid-1,chosen_i] + safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2 + (x[2]-initialize[2])**2 + (x[3]-initialize[3])**2 + (x[4]-initialize[4])**2 + (x[5]-initialize[5])**2 #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        variables = np.append(variables,cof_ln_dwork)
        Z = m.ObjVal
    except:
        variables = [0,0,0,0,0,0,0]
        Z = 0
    return np.array(variables),Z


#Lunch_IO
def solve_IO_random_lunch(aa,initialize,epsilon_l,iid,safe_boundary):
    SDE_lunch = float(aa[aa['chosen']==True]['SDE_lunch'])
    SDL_lunch = float(aa[aa['chosen']==True]['SDL_lunch'])
    K_lunch1 = float(aa[aa['chosen']==True]['K_lunch1'])
    K_lunch2 = float(aa[aa['chosen']==True]['K_lunch2'])
    t_worklunch = float(aa[aa['chosen']==True]['t_worklunch'])
    ln_dwork = float(aa[aa['chosen']==True]['ln_dwork'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(6,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['alternative'].iloc[j] != str(aa['alternative'][aa['chosen']==True].values[0]):
            m.addConstr((SDE_lunch-aa['SDE_lunch'].iloc[j])*x[0] + (SDL_lunch-aa['SDL_lunch'].iloc[j])*x[1] + (K_lunch1-aa['K_lunch1'].iloc[j])*x[2] + (K_lunch2-aa['K_lunch2'].iloc[j])*x[3] + (t_worklunch-aa['t_worklunch'].iloc[j])*x[4] + (ln_dwork-aa['ln_dwork'].iloc[j])*x[5] >= epsilon_l[iid-1,j] - epsilon_l[iid-1,chosen_i] + safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2 + (x[2]-initialize[2])**2 + (x[3]-initialize[3])**2 + (x[4]-initialize[4])**2 + (x[5]-initialize[5])**2 #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        variables = [0,0,0,0,0,0]
        Z = 0
    return np.array(variables),Z

#Lunch_IO fixed variable
def solve_IO_random_lunch_fixvar(aa,initialize,cof_ln_dwork,epsilon_l,iid,safe_boundary):
    SDE_lunch = float(aa[aa['chosen']==True]['SDE_lunch'])
    SDL_lunch = float(aa[aa['chosen']==True]['SDL_lunch'])
    K_lunch1 = float(aa[aa['chosen']==True]['K_lunch1'])
    K_lunch2 = float(aa[aa['chosen']==True]['K_lunch2'])
    t_worklunch = float(aa[aa['chosen']==True]['t_worklunch'])
    ln_dwork = float(aa[aa['chosen']==True]['ln_dwork'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(5,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['alternative'].iloc[j] != str(aa['alternative'][aa['chosen']==True].values[0]):
            m.addConstr((SDE_lunch-aa['SDE_lunch'].iloc[j])*x[0] + (SDL_lunch-aa['SDL_lunch'].iloc[j])*x[1] + (K_lunch1-aa['K_lunch1'].iloc[j])*x[2] + (K_lunch2-aa['K_lunch2'].iloc[j])*x[3] + (t_worklunch-aa['t_worklunch'].iloc[j])*x[4] + (ln_dwork-aa['ln_dwork'].iloc[j])*cof_ln_dwork >=epsilon_l[iid-1,j] - epsilon_l[iid-1,chosen_i] + safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2 + (x[2]-initialize[2])**2 + (x[3]-initialize[3])**2 + (x[4]-initialize[4])**2  #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        variables = np.append(variables,cof_ln_dwork)
        Z = m.ObjVal
    except:
        variables = [0,0,0,0,0,0]
        Z = 0
    return np.array(variables),Z

#Afterwork_IO
def solve_IO_random_afterwork(aa,initialize,epsilon_a,iid,safe_boundary):
    ln_dafterwork = float(aa[aa['chosen']==True]['ln_dafterwork'])
    ln_dwork_ln_afterwork = float(aa[aa['chosen']==True]['ln_dwork*ln_afterwork'])
    ln_dwork = float(aa[aa['chosen']==True]['ln_dwork'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(3,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['alternative'].iloc[j] != str(aa['alternative'][aa['chosen']==True].values[0]):
            m.addConstr((ln_dafterwork-aa['ln_dafterwork'].iloc[j])*x[0] + (ln_dwork_ln_afterwork-aa['ln_dwork*ln_afterwork'].iloc[j])*x[1] + (ln_dwork-aa['ln_dwork'].iloc[j])*x[2] >= epsilon_a[iid-1,j] - epsilon_a[iid-1,chosen_i] + safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2 + (x[2]-initialize[2])**2 #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        Z = m.ObjVal
    except:
        variables = [0,0,0]
        Z = 0
    return np.array(variables),Z


#Afterwork_IO fixed variable
def solve_IO_random_afterwork_fixvar(aa,initialize,cof_ln_dwork,epsilon_a,iid,safe_boundary):
    ln_dafterwork = float(aa[aa['chosen']==True]['ln_dafterwork'])
    ln_dwork_ln_afterwork = float(aa[aa['chosen']==True]['ln_dwork*ln_afterwork'])
    ln_dwork = float(aa[aa['chosen']==True]['ln_dwork'])
    aa = aa.reset_index()
    chosen_i = int(aa[aa['chosen']==True].index.values)
    m = Model()
    x = m.addVars(2,lb=-float('inf') ,ub=float('inf') , vtype=GRB.CONTINUOUS, name='x') #add decision variables
    for j in range(len(aa)):
        if aa['alternative'].iloc[j] != str(aa['alternative'][aa['chosen']==True].values[0]):
            m.addConstr((ln_dafterwork-aa['ln_dafterwork'].iloc[j])*x[0] + (ln_dwork_ln_afterwork-aa['ln_dwork*ln_afterwork'].iloc[j])*x[1] + (ln_dwork-aa['ln_dwork'].iloc[j])*cof_ln_dwork >= epsilon_a[iid-1,j] - epsilon_a[iid-1,chosen_i] + safe_boundary)  #add constraints
    obj = (x[0]-initialize[0])**2 + (x[1]-initialize[1])**2 #add objective function
    m.setObjective(obj,GRB.MINIMIZE)
    m.update()
    m.Params.LogToConsole = 0
    m.optimize()
    try:
        variables = np.array(m.getAttr('X', m.getVars()))
        variables = np.append(variables,cof_ln_dwork)
        Z = m.ObjVal
    except:
        variables = [0,0,0]
        Z = 0
    return np.array(variables),Z



#test_IO
np.random.seed(8521)
shuffle = np.random.permutation(26149)+1
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
Commuting_choice_ms = Commuting_choice.copy(deep=True)
Commuting_choice_ms.iloc[:,3:-1] = ms.fit_transform(Commuting_choice_ms.iloc[:,3:-1].values)
Lunch_choice_ms = Lunch_choice.copy(deep=True)
Lunch_choice_ms.iloc[:,3:-1] = ms.fit_transform(Lunch_choice_ms.iloc[:,3:-1].values)
Afterwork_choice_ms = Afterwork_choice.copy(deep=True)
Afterwork_choice_ms.iloc[:,3:-1] = ms.fit_transform(Afterwork_choice_ms.iloc[:,3:-1].values)



alter_num_c = int(Commuting_choice_ms.groupby('iid').agg({'hw_od':'count'}).mean().values)
alter_num_l = int(Lunch_choice_ms.groupby('iid').agg({'hw_od':'count'}).mean().values)
alter_num_a = int(Afterwork_choice_ms.groupby('iid').agg({'hw_od':'count'}).mean().values)
np.random.seed(8521)
epsilon_c = np.random.gumbel(0,1,26149*alter_num_c).reshape(26149,alter_num_c)
epsilon_l = np.random.gumbel(0,1,26149*alter_num_l).reshape(26149,alter_num_l)
epsilon_a = np.random.gumbel(0,1,26149*alter_num_a).reshape(26149,alter_num_a)


aa = Commuting_choice_ms[Commuting_choice_ms['iid']==100]
variable,Z = solve_IO_random_commuting(aa,[0,0,0,0,0,0,0],epsilon_c,iid=100,safe_boundary=2)
variable

variable2,Z2 = solve_IO_random_commuting_fixvar(aa,[0,0,0,0,0,0]
                                                ,variable[-1],epsilon_c,iid=100,safe_boundary=2)
variable2




#######################
#####Main Fucntion#####
#######################
def One_iteration(Commuting_choice_ms,Lunch_choice_ms,Afterwork_choice_ms,shuffle,epsilon_c,epsilon_l,epsilon_a, x_k, sample_size,bound, boundary_max,boundary_min,step):
    ##0.get initial parameters
    x_k_c = np.append(x_k[:6],x_k[-1])
    x_k_l = np.append(x_k[6:11],x_k[-1])
    x_k_a = x_k[11:]
    
    ##1.Run three IO models
    parameter_i_lst_c = []
    parameter_i_lst_l = []
    parameter_i_lst_a = []
    sb_c = []
    sb_l = []
    sb_a = []
    for i in shuffle[:sample_size]:
        #commuting
        aa = Commuting_choice_ms[Commuting_choice_ms['iid']==i]
        safe_boundary_c = boundary_max
        parameter_i_c,Z = solve_IO_random_commuting(aa,x_k_c,epsilon_c,i,safe_boundary_c)
        while ((parameter_i_c.max()>bound)or(parameter_i_c.min()<-bound)or(parameter_i_c.sum()==0)) and (safe_boundary_c>boundary_min):
            safe_boundary_c -= step
            parameter_i_c,Z = solve_IO_random_commuting(aa,x_k_c,epsilon_c,i,safe_boundary_c)
        parameter_i_lst_c.append(parameter_i_c)
        sb_c.append(safe_boundary_c)
        #lunch
        bb = Lunch_choice_ms[Lunch_choice_ms['iid']==i]
        safe_boundary_l = boundary_max
        parameter_i_l,Z = solve_IO_random_lunch(bb,x_k_l,epsilon_l,i,safe_boundary_l)
        while ((parameter_i_l.max()>bound)or(parameter_i_l.min()<-bound)or(parameter_i_l.sum()==0)) and (safe_boundary_l>boundary_min):
            safe_boundary_l -= step
            parameter_i_l,Z = solve_IO_random_lunch(bb,x_k_l,epsilon_l,i,safe_boundary_l)
        parameter_i_lst_l.append(parameter_i_l)
        sb_l.append(safe_boundary_l)
        #Afterwork
        cc = Afterwork_choice_ms[Afterwork_choice_ms['iid']==i]
        safe_boundary_a = boundary_max
        parameter_i_a,Z = solve_IO_random_afterwork(cc,x_k_a,epsilon_a,i,safe_boundary_a)
        while ((parameter_i_a.max()>bound)or(parameter_i_a.min()<-bound)or(parameter_i_a.sum()==0)) and (safe_boundary_a>boundary_min):
            safe_boundary_a -= step
            parameter_i_a,Z = solve_IO_random_afterwork(cc,x_k_a,epsilon_a,i,safe_boundary_a)
        parameter_i_lst_a.append(parameter_i_a)
        sb_a.append(safe_boundary_a)
    #commuting
    parameter_i_lst_0_c = np.array(parameter_i_lst_c)
    parameter_i_lst_c = parameter_i_lst_0_c[(parameter_i_lst_0_c.max(axis=1)<bound)&
                                      (parameter_i_lst_0_c.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_c.sum(axis=1)!=0)]
    #lunch
    parameter_i_lst_0_l = np.array(parameter_i_lst_l)
    parameter_i_lst_l = parameter_i_lst_0_l[(parameter_i_lst_0_l.max(axis=1)<bound)&
                                      (parameter_i_lst_0_l.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_l.sum(axis=1)!=0)]
    #Afterwork
    parameter_i_lst_0_a = np.array(parameter_i_lst_a) #用于算theta_ni
    parameter_i_lst_a = parameter_i_lst_0_a[(parameter_i_lst_0_a.max(axis=1)<bound)&
                                      (parameter_i_lst_0_a.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_a.sum(axis=1)!=0)] #用于计算theta_0
    
    ##2.Uniform the parameter of ln_dwork
    ln_dwork_3lst = np.array([parameter_i_lst_0_c[:,-1],parameter_i_lst_0_l[:,-1],parameter_i_lst_0_a[:,-1]])
    #ln_dwork_lst = ln_dwork_3lst.mean(axis=0)
    ln_dwork_lst = ln_dwork_3lst[0]
    # ln_dwork_3lst_abs = np.abs(ln_dwork_3lst)
    # index = np.argmin(ln_dwork_3lst_abs,axis=0)
    # ln_dwork_lst = []
    # for i in range(ln_dwork_3lst.shape[1]):
    #     ln_dwork_lst.append(ln_dwork_3lst[:,i][index[i]])
    
    #3.Fix ln_work and update the rest parameters
    parameter_i_lst_c = []
    parameter_i_lst_l = []
    parameter_i_lst_a = []
    sb_c2 = []
    sb_l2 = []
    sb_a2 = []
    for j,i in enumerate(shuffle[:sample_size]):
    #commuting
        aa = Commuting_choice_ms[Commuting_choice_ms['iid']==i]
        safe_boundary_c = sb_c[j]
        parameter_i_c,Z = solve_IO_random_commuting_fixvar(aa,x_k_c[:-1],ln_dwork_lst[j],epsilon_c,i,safe_boundary_c)
        while ((parameter_i_c.max()>bound)or(parameter_i_c.min()<-bound)or(parameter_i_c.sum()==0)) and (safe_boundary_c>boundary_min):
            safe_boundary_c -= step
            parameter_i_c,Z = solve_IO_random_commuting_fixvar(aa,x_k_c[:-1],ln_dwork_lst[j],epsilon_c,i,safe_boundary_c)
        parameter_i_lst_c.append(parameter_i_c)
        sb_c2.append(safe_boundary_c)
        #lunch
        bb = Lunch_choice_ms[Lunch_choice_ms['iid']==i]
        safe_boundary_l = sb_l[j]
        parameter_i_l,Z = solve_IO_random_lunch_fixvar(bb,x_k_l[:-1],ln_dwork_lst[j],epsilon_l,i,safe_boundary_l)
        while ((parameter_i_l.max()>bound)or(parameter_i_l.min()<-bound)or(parameter_i_l.sum()==0)) and (safe_boundary_l>boundary_min):
            safe_boundary_l -= step
            parameter_i_l,Z = solve_IO_random_lunch_fixvar(bb,x_k_l[:-1],ln_dwork_lst[j],epsilon_l,i,safe_boundary_l)
        parameter_i_lst_l.append(parameter_i_l)
        sb_l2.append(safe_boundary_l)
        #Afterwork
        cc = Afterwork_choice_ms[Afterwork_choice_ms['iid']==i]
        safe_boundary_a = sb_a[j]
        parameter_i_a,Z = solve_IO_random_afterwork_fixvar(cc,x_k_a[:-1],ln_dwork_lst[j],epsilon_a,i,safe_boundary_a)
        while ((parameter_i_a.max()>bound)or(parameter_i_a.min()<-bound)or(parameter_i_a.sum()==0)) and (safe_boundary_a>boundary_min):
            safe_boundary_a -= step
            parameter_i_a,Z = solve_IO_random_afterwork_fixvar(cc,x_k_a[:-1],ln_dwork_lst[j],epsilon_a,i,safe_boundary_a)
        parameter_i_lst_a.append(parameter_i_a)
        sb_a2.append(safe_boundary_a)
    #commuting
    parameter_i_lst_0_c = np.array(parameter_i_lst_c)
    parameter_i_lst_c = parameter_i_lst_0_c[(parameter_i_lst_0_c.max(axis=1)<bound)&
                                      (parameter_i_lst_0_c.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_c.sum(axis=1)!=0)]
    #lunch
    parameter_i_lst_0_l = np.array(parameter_i_lst_l)
    parameter_i_lst_l = parameter_i_lst_0_l[(parameter_i_lst_0_l.max(axis=1)<bound)&
                                      (parameter_i_lst_0_l.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_l.sum(axis=1)!=0)]
    #Afterwork
    parameter_i_lst_0_a = np.array(parameter_i_lst_a) #用于算theta_ni
    parameter_i_lst_a = parameter_i_lst_0_a[(parameter_i_lst_0_a.max(axis=1)<bound)&
                                      (parameter_i_lst_0_a.min(axis=1)>-bound)&
                                      (parameter_i_lst_0_a.sum(axis=1)!=0)] #用于计算theta_0
    
    ##4.Calculate y_0
    y_k_c = parameter_i_lst_c.mean(axis=0)
    y_k_l = parameter_i_lst_l.mean(axis=0)
    y_k_a = parameter_i_lst_a.mean(axis=0)
    y_k = np.append(np.append(y_k_c[:-1],y_k_l[:-1]),y_k_a[:-1])
    ln_dwork = (y_k_c[-1]+y_k_l[-1]+y_k_a[-1])/3
    y_k = np.append(y_k,ln_dwork)
    
    return y_k, parameter_i_lst_0_c, parameter_i_lst_0_l,parameter_i_lst_0_a,sb_c2,sb_l2,sb_a2


#test One_iteration
x_k = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
y_k, parameter_i_lst_0_c, parameter_i_lst_0_l,parameter_i_lst_0_a = One_iteration(Commuting_choice_ms,Lunch_choice_ms,Afterwork_choice_ms,shuffle,epsilon_c,epsilon_l,epsilon_a, x_k, sample_size=1000,bound=30,boundary_max=3,boundary_min=1,step=0.4)

parameter_i_lst_c = parameter_i_lst_0_c[(parameter_i_lst_0_c.sum(axis=1)!=0)]
parameter_i_lst_c_good = parameter_i_lst_0_c[(parameter_i_lst_0_c.sum(axis=1)!=0)&(parameter_i_lst_0_c.max(axis=1)<30)&(parameter_i_lst_0_c.min(axis=1)>-30)] 
parameter_i_lst_l = parameter_i_lst_0_l[(parameter_i_lst_0_l.sum(axis=1)!=0)]
parameter_i_lst_a = parameter_i_lst_0_a[(parameter_i_lst_0_a.sum(axis=1)!=0)]  

check = pd.DataFrame(parameter_i_lst_c)
check.mean()




def whole_experiment_SR(Commuting_choice,Lunch_choice,Afterwork_choice,
                        x_0,sample_size,tal=1.8,gama=0.3,bound=30,
                        boundary_max=3,boundary_min=-3,step=0.5,plot=True):
    ##0.Data Processing
    np.random.seed(8521)
    shuffle = np.random.permutation(26149)+1
    from sklearn.preprocessing import MinMaxScaler
    ms = MinMaxScaler()
    Commuting_choice_ms = Commuting_choice.copy(deep=True)
    Commuting_choice_ms.iloc[:,3:-1] = ms.fit_transform(Commuting_choice_ms.iloc[:,3:-1].values)
    Lunch_choice_ms = Lunch_choice.copy(deep=True)
    Lunch_choice_ms.iloc[:,3:-1] = ms.fit_transform(Lunch_choice_ms.iloc[:,3:-1].values)
    Afterwork_choice_ms = Afterwork_choice.copy(deep=True)
    Afterwork_choice_ms.iloc[:,3:-1] = ms.fit_transform(Afterwork_choice_ms.iloc[:,3:-1].values)
    
    alter_num_c = int(Commuting_choice_ms.groupby('iid').agg({'hw_od':'count'}).mean().values)
    alter_num_l = int(Lunch_choice_ms.groupby('iid').agg({'hw_od':'count'}).mean().values)
    alter_num_a = int(Afterwork_choice_ms.groupby('iid').agg({'hw_od':'count'}).mean().values)

    ##1.initialization
    start_time = time.time()
    params_track_x = []
    params_track_y = []
    k=1
    beta_k = 1
    #iteration 1
    epsilon_c = np.random.gumbel(0,1,26149*alter_num_c).reshape(26149,alter_num_c)
    epsilon_l = np.random.gumbel(0,1,26149*alter_num_l).reshape(26149,alter_num_l)
    epsilon_a = np.random.gumbel(0,1,26149*alter_num_a).reshape(26149,alter_num_a)
    x_0 = np.array(x_0)
    y_0, parameter_i_lst_0_c, parameter_i_lst_0_l,parameter_i_lst_0_a,sb_c2,sb_l2,sb_a2 = One_iteration(Commuting_choice_ms,Lunch_choice_ms,Afterwork_choice_ms,shuffle,epsilon_c,epsilon_l,epsilon_a,x_0, sample_size,bound,boundary_max,boundary_min,step)
    params_track_x.append(x_0) #record x
    params_track_y.append(y_0) #record y
    #iteration 2
    epsilon_c = np.random.gumbel(0,1,26149*alter_num_c).reshape(26149,alter_num_c)
    epsilon_l = np.random.gumbel(0,1,26149*alter_num_l).reshape(26149,alter_num_l)
    epsilon_a = np.random.gumbel(0,1,26149*alter_num_a).reshape(26149,alter_num_a)
    x_1 = y_0
    y_1, parameter_i_lst_0_c, parameter_i_lst_0_l,parameter_i_lst_0_a,sb_c2,sb_l2,sb_a2 = One_iteration(Commuting_choice_ms,Lunch_choice_ms,Afterwork_choice_ms,shuffle,epsilon_c,epsilon_l,epsilon_a,x_1, sample_size,bound,boundary_max,boundary_min,step)
    params_track_x.append(x_1) #record x
    params_track_y.append(y_1) #record y
    change = (x_1-x_0)/(x_0+1e-8)

    ##3.Main iteration
    while np.sum(np.abs(change))>0.014:
        #calculate x_k_next
        epsilon_c = np.random.gumbel(0,1,26149*alter_num_c).reshape(26149,alter_num_c)
        epsilon_l = np.random.gumbel(0,1,26149*alter_num_l).reshape(26149,alter_num_l)
        epsilon_a = np.random.gumbel(0,1,26149*alter_num_a).reshape(26149,alter_num_a)
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
        y_k_next,parameter_i_lst_0_c,parameter_i_lst_0_l,parameter_i_lst_0_a,sb_c2,sb_l2,sb_a2 = One_iteration(Commuting_choice_ms,Lunch_choice_ms,Afterwork_choice_ms,shuffle,epsilon_c,epsilon_l,epsilon_a,x_k_next, sample_size,bound,boundary_max,boundary_min,step)
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
    theta_ni_c = pd.DataFrame(parameter_i_lst_0_c)
    theta_ni_l = pd.DataFrame(parameter_i_lst_0_l)
    theta_ni_a = pd.DataFrame(parameter_i_lst_0_a)
    end_time = time.time()
    print('model running time: %is'%int(end_time-start_time))
    print('Meet the coveragence standard at %i-th iteration'%k)  
    
    if plot==True:
        #commuting
        fig,ax = plt.subplots()
        theta_0[0].plot(ax = ax,label='t_commute')
        theta_0[1].plot(ax = ax,label='c_commute')
        theta_0[2].plot(ax = ax,label='M_commute')
        theta_0[3].plot(ax = ax,label='SDE_work')
        theta_0[4].plot(ax = ax,label='SDL_work')
        theta_0[5].plot(ax = ax,label='PL_work')
        theta_0[13].plot(ax = ax,label='ln_work')
        plt.legend()
        plt.title('Values of theta_0 in each iteration (commuting)')
        plt.xlabel('iteration_num')
        plt.ylabel('value')
        fig,ax = plt.subplots()
        theta_ni_c[(theta_ni_c[0]>-bound)&(theta_ni_c[0]<bound)][0].hist(ax = ax,bins=30,label='t_commute')
        theta_ni_c[(theta_ni_c[1]>-bound)&(theta_ni_c[1]<bound)][1].hist(ax = ax,bins=30,label='c_commute')
        theta_ni_c[(theta_ni_c[2]>-bound)&(theta_ni_c[2]<bound)][2].hist(ax = ax,bins=30,label='M_commute')
        theta_ni_c[(theta_ni_c[3]>-bound)&(theta_ni_c[3]<bound)][3].hist(ax = ax,bins=30,label='SDE_work')
        theta_ni_c[(theta_ni_c[4]>-bound)&(theta_ni_c[4]<bound)][4].hist(ax = ax,bins=30,label='SDL_work')
        theta_ni_c[(theta_ni_c[5]>-bound)&(theta_ni_c[5]<bound)][5].hist(ax = ax,bins=30,label='PL_work')
        theta_ni_c[(theta_ni_c[6]>-bound)&(theta_ni_c[6]<bound)][6].hist(ax = ax,bins=30,label='ln_work')
        plt.legend()
        plt.title('Distribution of theta_i after the final iteration (commuting)')
        plt.xlabel('value')
        plt.ylabel('frequency')
        #lunch
        fig,ax = plt.subplots()
        theta_0[6].plot(ax = ax,label='SDE_lunch')
        theta_0[7].plot(ax = ax,label='SDL_lunch')
        theta_0[8].plot(ax = ax,label='K_lunch1')
        theta_0[9].plot(ax = ax,label='K_lunch2')
        theta_0[10].plot(ax = ax,label='t_worklunch')
        theta_0[13].plot(ax = ax,label='ln_dwork')
        plt.legend()
        plt.title('Values of theta_0 in each iteration (lunch)')
        plt.xlabel('iteration_num')
        plt.ylabel('value')
        fig,ax = plt.subplots()
        theta_ni_l[(theta_ni_l[0]>-bound)&(theta_ni_l[0]<bound)][0].hist(ax = ax,bins=30,label='SDE_lunch')
        theta_ni_l[(theta_ni_l[1]>-bound)&(theta_ni_l[1]<bound)][1].hist(ax = ax,bins=30,label='SDL_lunch')
        theta_ni_l[(theta_ni_l[2]>-bound)&(theta_ni_l[2]<bound)][2].hist(ax = ax,bins=30,label='K_lunch1')
        theta_ni_l[(theta_ni_l[3]>-bound)&(theta_ni_l[3]<bound)][3].hist(ax = ax,bins=30,label='K_lunch2')
        theta_ni_l[(theta_ni_l[4]>-bound)&(theta_ni_l[4]<bound)][4].hist(ax = ax,bins=30,label='t_worklunch')
        theta_ni_l[(theta_ni_l[5]>-bound)&(theta_ni_l[5]<bound)][5].hist(ax = ax,bins=30,label='ln_dwork')
        plt.legend()
        plt.title('Distribution of theta_i after the final iteration (commuting)')
        plt.xlabel('value')
        plt.ylabel('frequency')
        #afterwork
        fig,ax = plt.subplots()
        theta_0[11].plot(ax = ax,label='ln_dafterwork')
        theta_0[12].plot(ax = ax,label='ln_dwork*ln_afterwork')
        theta_0[13].plot(ax = ax,label='ln_dwork')
        plt.legend()
        plt.title('Values of theta_0 in each iteration (afterwork)')
        plt.xlabel('iteration_num')
        plt.ylabel('value')
        fig,ax = plt.subplots()
        theta_ni_a[(theta_ni_a[0]>-bound)&(theta_ni_a[0]<bound)][0].hist(ax = ax,bins=30,label='ln_dafterwork')
        theta_ni_a[(theta_ni_a[1]>-bound)&(theta_ni_a[1]<bound)][1].hist(ax = ax,bins=30,label='ln_dwork*ln_afterwork')
        theta_ni_a[(theta_ni_a[2]>-bound)&(theta_ni_a[2]<bound)][2].hist(ax = ax,bins=30,label='ln_dwork')
        plt.legend()
        plt.title('Distribution of theta_i after the final iteration (afterwork)')
        plt.xlabel('value')
        plt.ylabel('frequency')
    
    
    ##5.Outputs
    theta_ni_c['iid'] = shuffle[:sample_size]
    theta_ni_c['safe_boundary'] = sb_c2
    theta_ni_l['iid'] = shuffle[:sample_size]
    theta_ni_l['safe_boundary'] = sb_l2
    theta_ni_a['iid'] = shuffle[:sample_size]
    theta_ni_a['safe_boundary'] = sb_a2
    return theta_0, theta_ni_c, theta_ni_l, theta_ni_a
        

#Run the model
x_0 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
theta_0,theta_ni_c,theta_ni_l,theta_ni_a = whole_experiment_SR(Commuting_choice,Lunch_choice,Afterwork_choice,x_0,
                        sample_size=1000,tal=1.8,gama=0.3,bound=30,
                        boundary_max=1,boundary_min=1,step=0.5,plot=True)     

num =100000
aa = []
for i in range(num):
    epsilon = np.random.gumbel(0,1)
    aa.append(np.random.gumbel(0,1)-epsilon)
aa = np.array(aa)
aa.sort()
print(aa[int(num*0.25)],aa[int(num*0.75)])


# theta_0.to_csv('Random utility theta_0 1K.csv',index=False)   
# theta_ni_c.to_csv('Random utility theta_ni_c 1K.csv',index=False)
# theta_ni_l.to_csv('Random utility theta_ni_l 1K.csv',index=False)    
# theta_ni_a.to_csv('Random utility theta_ni_a 1K.csv',index=False)    

bound=30
# theta_0.iloc[-1]

theta_ni_c_good = theta_ni_c[(theta_ni_c.iloc[:,:-1].max(axis=1)<bound)&
                                      (theta_ni_c.iloc[:,:-1].min(axis=1)>-bound)&
                                      (theta_ni_c.iloc[:,:-1].sum(axis=1)!=0)]

theta_ni_l_good = theta_ni_l[(theta_ni_l.iloc[:,:-1].max(axis=1)<bound)&
                                      (theta_ni_l.iloc[:,:-1].min(axis=1)>-bound)&
                                      (theta_ni_l.iloc[:,:-1].sum(axis=1)!=0)]
# theta_ni_a_good = theta_ni_a[(theta_ni_a.max(axis=1)<bound)&
#                                       (theta_ni_a.min(axis=1)>-bound)&
#                                       (theta_ni_a.sum(axis=1)!=0)]

# theta_ni_c[theta_ni_c.sum(axis=1)==0].shape


# input necessary data
test2 = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\2.初步模型试验\Commuting_choice_0507.csv")
np.random.seed(8521)
shuffle = np.random.permutation(26149)+1
theta_ni = theta_ni_c
theta_ni['iid'] = shuffle[:100]
theta_ni = theta_ni[(theta_ni.iloc[:,:-1].sum(axis=1)!=0)]


from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
test2_ms = test2.copy()
test2_ms.iloc[:,3:-1] = ms.fit_transform(test2_ms.iloc[:,3:-1].values)


# the whole dataset
data_check = pd.DataFrame(theta_ni.values,columns=['p_t_commute','p_c_commute','p_M_commute','p_SDE_work','p_SDL_work','p_PL_work','p_ln_dwork','iid','safe_boundary'])
data_check2 = pd.merge(test2_ms,data_check,on='iid')

data_check2['Utility'] = data_check2['t_commute']*data_check2['p_t_commute']+data_check2['c_commute']*data_check2['p_c_commute']+data_check2['M_commute2']*data_check2['p_M_commute']+data_check2['SDE_work']*data_check2['p_SDE_work']+data_check2['SDL_work']*data_check2['p_SDL_work']+data_check2['PL_work']*data_check2['p_PL_work']+data_check2['ln_dwork']*data_check2['p_ln_dwork']


data_check2['Utility_MNL'] = test2['t_commute']*-0.053395+test2['c_commute']*-0.054025+test2['M_commute2']*0.35701+test2['SDE_work']*-0.035530+test2['SDL_work']*-0.035146+test2['PL_work']*0+test2['ln_dwork']*15.473

data_check3 = data_check2[['iid','alternative','chosen','Utility','Utility_MNL']]
data_check3['iid_index'] = data_check2['iid']
data_check3 = data_check3.set_index('iid_index')


data_check3['prob_IO'] = 999
aa = pd.Series()
for iid in data_check3['iid'].unique():
    utility = data_check3[data_check3['iid']==iid]['Utility']
    utility = (utility-utility.min())/(utility.max()-utility.min())*40
    exp = np.exp(utility)
    exp_sum = np.exp(utility).sum()
    prob = exp/exp_sum
    data_check3.loc[iid,'prob_IO']=prob.values    

data_check3['prob_MNL'] = 999
aa = pd.Series()
for iid in data_check3['iid'].unique():
    exp = np.exp(data_check3[data_check3['iid']==iid]['Utility_MNL'])
    exp_sum = np.exp(data_check3[data_check3['iid']==iid]['Utility_MNL']).sum()
    prob = exp/exp_sum
    data_check3.loc[iid,'prob_MNL']=prob.values



check2 =  data_check3[data_check3['iid']==31] 

check2['prob_IO'] = 999
check2['Scaled_U'] = 999
aa = pd.Series()
for iid in check2['iid'].unique():
    utility = check2[check2['iid']==iid]['Utility']
    utility = (utility-utility.min())/(utility.max()-utility.min())*20
    exp = np.exp(utility)
    exp_sum = np.exp(utility).sum()
    prob = exp/exp_sum
    check2.loc[iid,'prob_IO']=prob.values 
    check2.loc[iid,'Scaled_U']=utility.values 

#aggregated prediction
compare = pd.DataFrame(index=data_check3['alternative'].unique())
compare['observed'] = data_check3.groupby('alternative').agg({'chosen':'sum'})
compare['prediction_IO'] = data_check3.groupby('alternative').agg({'prob_IO':'sum'})
compare['prediction_MNL'] = data_check3.groupby('alternative').agg({'prob_MNL':'sum'})

compare[['observed','prediction_MNL']].plot.bar(width=0.6)
compare[['observed','prediction_IO']].plot.bar(width=0.6)

def predict_accruarcy(compare,observed,prediction,total):
    accuracy = compare[[observed,prediction]].min(axis=1)
    accuracy = accuracy.sum()/total
    return accuracy

predict_accruarcy(compare,'observed','prediction_IO',len(theta_ni_c))

#dis-aggregated prediction

for_matrix = pd.DataFrame(index=data_check3['iid'].unique())
for_matrix['observed'] = pd.Series(data_check3[data_check3['chosen']==True]['alternative'].values,index=for_matrix.index)
aa = data_check3[data_check3.groupby('iid')['Utility'].rank(method='first',ascending=False)==1]['alternative']
for_matrix['prediction_IO'] = pd.Series(aa.values,index=for_matrix.index)
aa = data_check3[data_check3.groupby('iid')['Utility_MNL'].rank(method='first',ascending=False)==1]['alternative']
for_matrix['prediction_MNL'] = pd.Series(aa.values,index=for_matrix.index)
for_matrix['num'] = 1

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(for_matrix['observed'],for_matrix['prediction_IO'])
labels_name = compare.index

def plot_confusion_matrix(cm,labels_name,title):
    cm = cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm,interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    plt.clim(0,0.8)
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local,labels_name,rotation=90)
    plt.yticks(num_local,labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
plot_confusion_matrix(cm,labels_name,'Confusion Matrix (IO)')
(for_matrix['observed']==for_matrix['prediction_IO']).sum()/len(theta_ni_c)

compare['prediction_MNL_direct'] = for_matrix.groupby('prediction_MNL')['num'].count()
compare['prediction_IO_direct'] = for_matrix.groupby('prediction_IO')['num'].count()


compare[['observed','prediction_IO_direct']].plot.bar(width=0.6)
predict_accruarcy(compare,'observed','prediction_IO_direct',len(theta_ni_c))




''''''''''''''''''''''''''''''''
'''Result Analysis'''
''''''''''''''''''''''''''''''''
###################
#import parameters#
###################
theta_0 =  pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_0.csv")
theta_ni_c = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_c.csv")
theta_ni_l = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_l.csv")
theta_ni_a = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_a.csv")
np.random.seed(8521)
shuffle = np.random.permutation(26149)+1

#######
#Means#
#######
theta_0.rename(columns={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'10':10,'11':11,'12':12,'13':13,'14':14},inplace=True)
theta_ni_c.rename(columns={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6},inplace=True)
theta_ni_l.rename(columns={'0':0,'1':1,'2':2,'3':3,'4':4,'5':5},inplace=True)
theta_ni_a.rename(columns={'0':0,'1':1,'2':2},inplace=True)
bound=20
bins = np.linspace(-30,30,60)

print(theta_ni_c[(theta_ni_c[0]>-bound)&(theta_ni_c[0]<bound)&(theta_ni_c[0]!=0)][0].mean())
print(theta_ni_c[(theta_ni_c[1]>-bound)&(theta_ni_c[1]<bound)&(theta_ni_c[1]!=0)][1].mean())
print(theta_ni_c[(theta_ni_c[2]>-bound)&(theta_ni_c[2]<bound)&(theta_ni_c[2]!=0)][2].mean())
print(theta_ni_c[(theta_ni_c[3]>-bound)&(theta_ni_c[3]<bound)&(theta_ni_c[3]!=0)][3].mean())
print(theta_ni_c[(theta_ni_c[4]>-bound)&(theta_ni_c[4]<bound)&(theta_ni_c[4]!=0)][4].mean())
print(theta_ni_c[(theta_ni_c[5]>-bound)&(theta_ni_c[5]<bound)&(theta_ni_c[5]!=0)][5].mean())
print(theta_ni_c[(theta_ni_c[6]>-bound)&(theta_ni_c[6]<bound)&(theta_ni_c[6]!=0)][6].mean())

print(theta_ni_l[(theta_ni_l[0]>-bound)&(theta_ni_l[0]<bound)&(theta_ni_l[0]!=0)][0].mean())
print(theta_ni_l[(theta_ni_l[1]>-bound)&(theta_ni_l[1]<bound)&(theta_ni_l[1]!=0)][1].mean())
print(theta_ni_l[(theta_ni_l[2]>-bound)&(theta_ni_l[2]<bound)&(theta_ni_l[2]!=0)][2].mean())
print(theta_ni_l[(theta_ni_l[3]>-bound)&(theta_ni_l[3]<bound)&(theta_ni_l[3]!=0)][3].mean())
print(theta_ni_l[(theta_ni_l[4]>-bound)&(theta_ni_l[4]<bound)&(theta_ni_l[4]!=0)][4].mean())
print(theta_ni_l[(theta_ni_l[5]>-bound)&(theta_ni_l[5]<bound)&(theta_ni_l[5]!=0)][5].mean())

print(theta_ni_a[(theta_ni_a[0]>-bound)&(theta_ni_a[0]<bound)&(theta_ni_a[0]!=0)][0].mean())
print(theta_ni_a[(theta_ni_a[1]>-bound)&(theta_ni_a[1]<bound)&(theta_ni_a[1]!=0)][1].mean())
print(theta_ni_a[(theta_ni_a[2]>-bound)&(theta_ni_a[2]<bound)&(theta_ni_a[2]!=0)][2].mean())


#######
#Plots#
#######
#commuting
bound=30
bins = np.linspace(-30,30,60)
fig,ax = plt.subplots()
theta_0[0].plot(ax = ax,label='t_commute')
theta_0[1].plot(ax = ax,label='c_commute')
theta_0[2].plot(ax = ax,label='m_commute')
theta_0[3].plot(ax = ax,label='e_work')
theta_0[4].plot(ax = ax,label='l_work')
theta_0[5].plot(ax = ax,label='pl_work')
theta_0[13].plot(ax = ax,label='dur_work',color='#e377c2')
plt.legend()
plt.title('Fixed-point prior in each iteration')
plt.xlabel('Number of iteration')
plt.ylabel('Value')
plt.ylim([-10,10])
plt.savefig('C:/Users/MSI-PC/Desktop/Fig2-a.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')

fig,ax = plt.subplots()
theta_ni_c[(theta_ni_c[0]>-bound)&(theta_ni_c[0]<bound)][0].hist(ax = ax,bins=bins,label='t_commute',density=True)
theta_ni_c[(theta_ni_c[1]>-bound)&(theta_ni_c[1]<bound)][1].hist(ax = ax,bins=bins,label='c_commute',density=True)
theta_ni_c[(theta_ni_c[2]>-bound)&(theta_ni_c[2]<bound)][2].hist(ax = ax,bins=bins,label='m_commute',density=True)
theta_ni_c[(theta_ni_c[3]>-bound)&(theta_ni_c[3]<bound)][3].hist(ax = ax,bins=bins,label='e_work',density=True)
theta_ni_c[(theta_ni_c[4]>-bound)&(theta_ni_c[4]<bound)][4].hist(ax = ax,bins=bins,label='l_work',density=True)
theta_ni_c[(theta_ni_c[5]>-bound)&(theta_ni_c[5]<bound)][5].hist(ax = ax,bins=bins,label='pl_work',density=True)
theta_ni_c[(theta_ni_c[6]>-bound)&(theta_ni_c[6]<bound)][6].hist(ax = ax,bins=bins,label='dur_work',density=True,color='#e377c2')
plt.legend()
plt.title('Distribution of estimated parameters')
plt.xlabel('Value')
plt.ylabel('Density')
plt.ylim([0,1])
plt.xlim([-30,30])
plt.grid(linestyle='--')
plt.savefig('C:/Users/MSI-PC/Desktop/Fig2-d.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')

#lunch
fig,ax = plt.subplots()
theta_0[6].plot(ax = ax,label='e_lunch')
theta_0[7].plot(ax = ax,label='l_lunch')
theta_0[8].plot(ax = ax,label='des1_lunch')
theta_0[9].plot(ax = ax,label='des2_lunch')
theta_0[10].plot(ax = ax,label='t_work_lunch')
theta_0[13].plot(ax = ax,label='dur_work',color='#e377c2')
plt.legend(loc='upper right')
plt.title('Fixed-point prior in each iteration')
plt.xlabel('Number of iteration')
plt.ylabel('Value')
plt.ylim([-10,10])
plt.savefig('C:/Users/MSI-PC/Desktop/Fig2-b.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')

fig,ax = plt.subplots()
theta_ni_l[(theta_ni_l[0]>-bound)&(theta_ni_l[0]<bound)&(theta_ni_l[0]!=0)][0].hist(ax = ax,bins=bins,label='e_lunch',density=True)
theta_ni_l[(theta_ni_l[1]>-bound)&(theta_ni_l[1]<bound)&(theta_ni_l[1]!=0)][1].hist(ax = ax,bins=bins,label='l_lunch',density=True)
theta_ni_l[(theta_ni_l[2]>-bound)&(theta_ni_l[2]<bound)&(theta_ni_l[2]!=0)][2].hist(ax = ax,bins=bins,label='des1_lunch',density=True)
theta_ni_l[(theta_ni_l[3]>-bound)&(theta_ni_l[3]<bound)&(theta_ni_l[3]!=0)][3].hist(ax = ax,bins=bins,label='des2_lunch',density=True)
theta_ni_l[(theta_ni_l[4]>-bound)&(theta_ni_l[4]<bound)&(theta_ni_l[4]!=0)][4].hist(ax = ax,bins=bins,label='t_work_lunch',density=True)
theta_ni_l[(theta_ni_l[5]>-bound)&(theta_ni_l[5]<bound)&(theta_ni_l[5]!=0)][5].hist(ax = ax,bins=bins,label='dur_work',density=True,color='#e377c2')
plt.legend()
plt.title('Distribution of estimated parameters')
plt.xlabel('Value')
plt.ylabel('Density')
plt.ylim([0,1])
plt.xlim([-30,30])
plt.grid(linestyle='--')
plt.savefig('C:/Users/MSI-PC/Desktop/Fig2-e.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')
        
#afterwork
fig,ax = plt.subplots()
theta_0[11].plot(ax = ax,label='dur_afterwork')
theta_0[12].plot(ax = ax,label='dur_inter')
theta_0[13].plot(ax = ax,label='dur_work',color='#e377c2')
plt.legend(loc='lower right')
plt.title('Fixed-point prior in each iteration')
plt.xlabel('Number of iteration')
plt.ylabel('Value')
plt.ylim([-10,10])
plt.savefig('C:/Users/MSI-PC/Desktop/Fig2-c.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')

fig,ax = plt.subplots()
theta_ni_a[(theta_ni_a[0]>-bound)&(theta_ni_a[0]<bound)][0].hist(ax = ax,bins=bins,label='dur_afterwork',density=True)
theta_ni_a[(theta_ni_a[1]>-bound)&(theta_ni_a[1]<bound)][1].hist(ax = ax,bins=bins,label='dur_inter',density=True)
theta_ni_a[(theta_ni_a[2]>-bound)&(theta_ni_a[2]<bound)][2].hist(ax = ax,bins=bins,label='dur_work',density=True,color='#e377c2')
plt.legend()
plt.title('Distribution of estimated parameters')
plt.xlabel('Value')
plt.ylabel('Density')
plt.ylim([0,1])
plt.xlim([-30,30])
plt.grid(linestyle='--')
plt.savefig('C:/Users/MSI-PC/Desktop/Fig2-f.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')


####################################
#consistency with mixed logit model#
####################################
theta_0 =  pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_0.csv")
theta_ni_c = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_c.csv")
theta_ni_l = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_l.csv")
theta_ni_a = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_a.csv")

np.random.seed(8521)
shuffle = np.random.permutation(26149)+1

#Fig.3
MXL_commute = np.array([[-8.852,3.068],[-4.212,5.578],[-2.124,4.716],
                        [-10.913,7.496],[-5.233,2.777],[-5.342,4.207],[36.327,9.004]])
MNL_commute = [-8.877,-1.005,0.374,-6.933,-5.324,0.335,22.365]

MXL_lunch = np.array([[-1.234,1.641],[-2.068,1.273],[-2.719,1.063],
                        [0,0],[-2.776,1.616],[12.129,4.207]])
MNL_lunch = [-0.840,-1.802,-1.927,0,-0.429,1.421]

MXL_afterwork = np.array([[8.415,8.092],[16.450,14.638],[2.908,2.968]])
MNL_afterwork = [6.398,11.892,2.277]

theta_ni_c_good = theta_ni_c[(theta_ni_c.iloc[:,:-1].max(axis=1)<30)&(theta_ni_c.iloc[:,:-1].min(axis=1)>-30)
                  &(theta_ni_c.iloc[:,:-1].sum(axis=1)!=0)]


fig,ax = plt.subplots(1,3,figsize=(20,5))
#t_commute
ax[0].axvline(x=MNL_commute[0], linestyle='-',linewidth=2,alpha=0.8,label='MNL')
aa = pd.Series(MXL_commute[0,0]+np.random.randn(50000)*MXL_commute[0,1])
aa.plot.kde(ax = ax[0],alpha=0.85,color='orange',
                              label='MXL',linewidth=2)
theta_ni_c_good['0'].plot.kde(ax = ax[0],alpha=0.85,
                              label='AMXL',color='red',linewidth=2)
ax[0].set_title('Probability density function',fontsize=18)
ax[0].grid(linestyle='--')
ax[0].legend()
ax[0].set_xlabel('Value',fontsize=12)
ax[0].set_ylabel('Density',fontsize=12)
ax[0].set_xlim([-30,30])
ax[0].set_ylim([0,0.4])
#SDE_early
ax[1].axvline(x=MNL_commute[3], linestyle='-',linewidth=2,alpha=0.8,label='MNL')
aa = pd.Series(MXL_commute[3,0]+np.random.randn(50000)*MXL_commute[3,1])
aa.plot.kde(ax = ax[1],alpha=0.85,color='orange',
                              label='MXL',linewidth=2)
theta_ni_c_good['3'].plot.kde(ax = ax[1],alpha=0.85,
                              label='AMXL',color='red',linewidth=2)
ax[1].set_title('Probability density function',fontsize=18)
ax[1].grid(linestyle='--')
ax[1].legend()
ax[1].set_xlabel('Value',fontsize=12)
ax[1].set_ylabel('Density',fontsize=12)
ax[1].set_xlim([-30,30])
ax[1].set_ylim([0,0.4])
#ln_dwork
aa = pd.Series(MXL_commute[6,0]+np.random.randn(50000)*MXL_commute[6,1])
aa.plot.kde(ax = ax[2],alpha=0.85,
                              label='MXL-commute',linewidth=2)
aa = pd.Series(MXL_lunch[5,0]+np.random.randn(50000)*MXL_lunch[5,1])
aa.plot.kde(ax = ax[2],alpha=0.85,color='orange',
                              label='MXL-lunch',linewidth=2)
aa = pd.Series(MXL_afterwork[2,0]+np.random.randn(50000)*MXL_afterwork[2,1])
aa.plot.kde(ax = ax[2],alpha=0.85,color='green',
                              label='MXL-afterwork',linewidth=2)

theta_ni_c_good['6'].plot.kde(ax = ax[2],alpha=0.85,
                              label='AMXL',color='red',linewidth=2)
ax[2].set_title('Probability density function',fontsize=18)
ax[2].grid(linestyle='--')
ax[2].legend()
ax[2].set_xlabel('Value',fontsize=12)
ax[2].set_ylabel('Density',fontsize=12)
ax[2].set_xlim([-10,50])
ax[2].set_ylim([0,0.4])
plt.savefig('C:/Users/MSI-PC/Desktop/Fig3_raw.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')






#commute
parameter_lst = ['t_commute','c_commute','M_commute','SDE_work','SDL_work','PL_work','ln_work']
MXL_commute = np.array([[-8.852,3.068],[-4.212,5.578],[-2.124,4.716],
                        [-10.913,7.496],[-5.233,2.777],[-5.342,4.207],[36.327,27.004]])
MNL_commute = [-8.877,-1.005,0.374,-6.933,-5.324,0.335,22.365]
bins = np.linspace(-30,30,60)



theta_ni_c_good = theta_ni_c[(theta_ni_c.iloc[:,:-1].max(axis=1)<30)&(theta_ni_c.iloc[:,:-1].min(axis=1)>-30)
                  &(theta_ni_c.iloc[:,:-1].sum(axis=1)!=0)]



fig,ax = plt.subplots(2,4,figsize=(26.6,10))
for i in range(MXL_commute.shape[0]):
    theta_ni_c_good['MXL'] = MXL_commute[i,0]+np.random.randn(len(theta_ni_c_good))*MXL_commute[i,1]
    theta_ni_c_good['MNL'] = MNL_commute[i]
    theta_ni_c_good['MXL'].hist(ax = ax[i//4,i%4],bins=bins,alpha=0.85,
                                label='MXL',color='red',density=True)
    theta_ni_c_good['MNL'].hist(ax = ax[i//4,i%4],bins=bins,alpha=0.85,
                                label='MNL',color='orange',density=True)
    theta_ni_c_good['%i'%i].hist(ax = ax[i//4,i%4],bins=bins,alpha=0.85,
                              label='IO',density=True)
    ax[i//4,i%4].axvline(x=0, color='black', linestyle='--',linewidth=2,alpha=0.7)
    ax[i//4,i%4].set_title('%s'%parameter_lst[i],fontsize=18)
    ax[i//4,i%4].grid(False)
    ax[i//4,i%4].legend()
    
plt.subplots_adjust(wspace=0.15, hspace=0.2)           
plt.xlim([-30,30])


#lunch
parameter_lst = ['SDE_lunch','SDL_lunch','K_lunch1 (inside CBD)','K_lunch2 (outside CBD)','t_worklunch','ln_work']
MXL_lunch = np.array([[-1.234,1.641],[-2.068,1.273],[-2.719,1.063],
                        [0,0],[-2.776,1.616],[12.129,4.207]])
MNL_lunch = [-0.840,-1.802,-1.927,0,-0.429,1.421]
bins = np.linspace(-30,30,60)

theta_ni_l_good = theta_ni_l[(theta_ni_l.iloc[:,:-1].max(axis=1)<30)&(theta_ni_l.iloc[:,:-1].min(axis=1)>-30)
                  &(theta_ni_l.iloc[:,:-1].sum(axis=1)!=0)]


fig,ax = plt.subplots(2,3,figsize=(20,10))
for i in range(MXL_lunch.shape[0]):
    theta_ni_l_good['MXL'] = MXL_lunch[i,0]+np.random.randn(len(theta_ni_l_good))*MXL_lunch[i,1]
    theta_ni_l_good['MNL'] = MNL_lunch[i]
    theta_ni_l_good['MXL'].hist(ax = ax[i//3,i%3],bins=bins,alpha=0.85,
                                label='MXL',color='red',density=True)
    theta_ni_l_good['MNL'].hist(ax = ax[i//3,i%3],bins=bins,alpha=0.85,
                                label='MNL',color='orange',density=True)
    theta_ni_l_good['%i'%i].hist(ax = ax[i//3,i%3],bins=bins,alpha=0.85,
                              label='IO',density=True)
    ax[i//3,i%3].axvline(x=0, color='black', linestyle='--',linewidth=2,alpha=0.7)
    ax[i//3,i%3].set_title('%s'%parameter_lst[i],fontsize=18)
    ax[i//3,i%3].grid(False)
    ax[i//3,i%3].legend()
    
plt.subplots_adjust(wspace=0.15, hspace=0.2)           
plt.xlim([-30,30])


#afterwork
parameter_lst = ['ln_dafterwork','ln_dwork*ln_dafterwork','ln_work']
MXL_afterwork = np.array([[8.415,8.092],[16.450,14.638],[2.908,2.968]])
MNL_afterwork = [6.398,11.892,2.277]
bins = np.linspace(-30,30,60)

theta_ni_a_good = theta_ni_a[(theta_ni_a.iloc[:,:-1].max(axis=1)<30)&(theta_ni_a.iloc[:,:-1].min(axis=1)>-30)
                  &(theta_ni_a.iloc[:,:-1].sum(axis=1)!=0)]

fig,ax = plt.subplots(2,3,figsize=(20,10))
for i in range(MXL_afterwork.shape[0]):
    theta_ni_a_good['MXL'] = MXL_afterwork[i,0]+np.random.randn(len(theta_ni_a_good))*MXL_afterwork[i,1]
    theta_ni_a_good['MNL'] = MNL_afterwork[i]
    theta_ni_a_good['MXL'].hist(ax = ax[i//3,i%3],bins=bins,alpha=0.85,
                                label='MXL',color='red',density=True)
    theta_ni_a_good['MNL'].hist(ax = ax[i//3,i%3],bins=bins,alpha=0.85,
                                label='MNL',color='orange',density=True)
    theta_ni_a_good['%i'%i].hist(ax = ax[i//3,i%3],bins=bins,alpha=0.85,
                              label='IO',density=True)
    ax[i//3,i%3].axvline(x=0, color='black', linestyle='--',linewidth=2,alpha=0.7)
    ax[i//3,i%3].set_title('%s'%parameter_lst[i],fontsize=18)
    ax[i//3,i%3].grid(False)
    ax[i//3,i%3].legend()
    
plt.subplots_adjust(wspace=0.15, hspace=0.2)           
plt.xlim([-30,30])


#shared parameter
bins = np.linspace(-10,50,60)
fig,ax = plt.subplots()
theta_ni_c_good['MXL'].hist(ax = ax,bins=bins,label='MXL_commute',density=True,alpha=0.85)
theta_ni_l_good['MXL'].hist(ax = ax,bins=bins,label='MXL_lunch',density=True,alpha=0.85)
theta_ni_a_good['MXL'].hist(ax = ax,bins=bins,label='MXL_afterwork',density=True,alpha=0.85)
theta_ni_c_good['6'].hist(ax = ax,bins=bins,label='IO',density=True,color='#e377c2',alpha=0.85)
plt.grid(False)
plt.legend()
plt.title('Distribution of shared parameter: ln_dwork')
plt.xlabel('value')
plt.ylabel('density')
plt.ylim([0,0.4])
plt.xlim([-10,50])


###########################
#check the spatial feature#
###########################
data = pd.DataFrame(theta_ni_c.values,columns=['t_commute','c_commute','M_commute','SDE_work','SDL_work','PL_work','ln_dwork','iid'])
id_info = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\1.离散选择模型构建\0507_choice_generation.csv",encoding='ANSI')[['iid','home_lng','home_lat','work_lng','work_lat']]
data = pd.merge(data,id_info,on='iid')


data = data[(data.iloc[:,:-5].max(axis=1)<30)&
            (data.iloc[:,:-5].min(axis=1)>-30)&
            (data.iloc[:,:-5].sum(axis=1)!=0)]




import geopandas as gpd
from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(data.home_lng,data.home_lat)]
data = gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)
jiedao = gpd.read_file(r"D:\Works\basic_data\jiedao.shp")[['geometry','CODE']]
jiedao_stats = gpd.sjoin(jiedao,data,op='contains')
jiedao_stats = jiedao_stats.groupby('CODE').agg({'iid':'count','t_commute':'mean','c_commute':'mean','M_commute':'mean','SDE_work':'mean','SDL_work':'mean','PL_work':'mean','ln_dwork':'mean'}).reset_index()

jiedao_stats.to_csv('jiedao_stats_wholeday_commuting.csv',index=False)

#correlation analysis
jiedao_factor = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\jiedao level parameter correlation analysis (basic).csv")

jiedao_stat =pd.read_csv(r"C:\Users\MSI-PC\Desktop\jiedao_stats_wholeday_commuting.csv")
jiedao = pd.merge(jiedao_stat,jiedao_factor,on='CODE')
jiedao.fillna(0,inplace=True)
jiedao['builtyear'] = jiedao['builtyear'].replace(0,1970)
jiedao['price'] = jiedao['price'].replace(0,16629)
jiedao = jiedao[jiedao['iid']>=10]

import scipy.stats as stats
print(stats.pearsonr(jiedao['SDE_work'],jiedao['distance_to_CBD']))
print(stats.pearsonr(jiedao['SDE_work'],jiedao['migrant_proportion']))
print(stats.pearsonr(jiedao['SDE_work'],jiedao['price']))
print(stats.pearsonr(jiedao['SDE_work'],jiedao['population_density']))
print(stats.pearsonr(jiedao['SDE_work'],jiedao['road_density']))
print(stats.pearsonr(jiedao['SDE_work'],jiedao['metro_coverage']))



#################
#Make prediction#
#################

# input necessary data
theta_ni_c = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_c.csv")
theta_ni_c.rename(columns={'0':'p_t_commute','1':'p_c_commute','2':'p_M_commute','3':'p_SDE_work','4':'p_SDL_work','5':'p_PL_work','6':'p_ln_dwork'},inplace=True)
theta_ni_l = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_l.csv")
theta_ni_l.rename(columns={'0':'p_SDE_lunch','1':'p_SDL_lunch','2':'p_K_lunch1','3':'p_K_lunch2','4':'p_t_worklunch','5':'p_ln_dwork'},inplace=True)
theta_ni_a = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Model_Results\Random utility theta_ni_a.csv")
theta_ni_a.rename(columns={'0':'p_ln_afterwork','1':'p_ln_dwork*ln_afterwork','2':'p_ln_dwork'},inplace=True)
Commuting_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Commuting_choice_0507.csv")
Lunch_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Lunch_choice_0507.csv")
Afterwork_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Afterwork_choice_0507.csv")
np.random.seed(8521)
shuffle = np.random.permutation(26149)+1


theta_ni_c = theta_ni_c[(theta_ni_c.iloc[:,:-1].sum(axis=1)!=0)]
#(theta_ni_c.iloc[:,:-1].max(axis=1)<30)&(theta_ni_c.iloc[:,:-1].min(axis=1)>-30)&
theta_ni_l = theta_ni_l[(theta_ni_l.iloc[:,:-1].sum(axis=1)!=0)]
theta_ni_a = theta_ni_a[(theta_ni_a.iloc[:,:-1].sum(axis=1)!=0)]
theta_ni_c = theta_ni_c[(theta_ni_c['iid'].isin(theta_ni_l['iid']))&(theta_ni_c['iid'].isin(theta_ni_a['iid']))]
theta_ni_l = theta_ni_l[(theta_ni_l['iid'].isin(theta_ni_c['iid']))&(theta_ni_c['iid'].isin(theta_ni_a['iid']))]
theta_ni_a = theta_ni_a[(theta_ni_a['iid'].isin(theta_ni_c['iid']))&(theta_ni_c['iid'].isin(theta_ni_l['iid']))]

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
Commuting_choice_ms = Commuting_choice.copy()
Commuting_choice_ms.iloc[:,3:-1] = ms.fit_transform(Commuting_choice_ms.iloc[:,3:-1].values)
Lunch_choice_ms = Lunch_choice.copy()
Lunch_choice_ms.iloc[:,3:-1] = ms.fit_transform(Lunch_choice_ms.iloc[:,3:-1].values)
Afterwork_choice_ms = Afterwork_choice.copy()
Afterwork_choice_ms.iloc[:,3:-1] = ms.fit_transform(Afterwork_choice_ms.iloc[:,3:-1].values)

Commuting_choice_ms = Commuting_choice_ms[['t_commute','c_commute','M_commute2','SDE_work','SDL_work','PL_work','ln_dwork','iid','hw_od','alternative','chosen']]
Commuting_choice_ms['alternative'] = Commuting_choice_ms['alternative'].map({'6:30-7:00,Driving':'Driving,6:30-7:00',
                                        '7:00-7:30,Driving':'Driving,7:00-7:30',
                                        '7:30-8:00,Driving':'Driving,7:30-8:00',
                                        '8:00-8:30,Driving':'Driving,8:00-8:30',
                                        '8:30-9:00,Driving':'Driving,8:30-9:00',
                                        '9:00-9:30,Driving':'Driving,9:00-9:30',
                                        '9:30-10:00,Driving':'Driving,9:30-10:00',
                                        '6:30-7:00,Transit':'Transit,6:30-7:00',
                                        '7:00-7:30,Transit':'Transit,7:00-7:30',
                                        '7:30-8:00,Transit':'Transit,7:30-8:00',
                                        '8:00-8:30,Transit':'Transit,8:00-8:30',
                                        '8:30-9:00,Transit':'Transit,8:30-9:00',
                                        '9:00-9:30,Transit':'Transit,9:00-9:30',
                                        '9:30-10:00,Transit':'Transit,9:30-10:00'})
Lunch_choice_ms = Lunch_choice_ms[['SDE_lunch','SDL_lunch','K_lunch1','K_lunch2','t_worklunch','ln_dwork','iid','hw_od','alternative','chosen']]
Afterwork_choice_ms = Afterwork_choice_ms[['ln_dafterwork','ln_dwork*ln_afterwork','ln_dwork','iid','hw_od','alternative','chosen']]

MNL_commute = [-8.877,-1.005,0.374,-6.933,-5.324,0.335,22.365]
MNL_lunch = [-0.840,-1.802,-1.927,0,-0.429/0.1,1.421/-0.1]
MNL_afterwork = [6.398/-0.05,11.892/0.1,2.277/-0.01]
alternative_c = np.array(['Driving,6:30-7:00','Driving,7:00-7:30','Driving,7:30-8:00',
                          'Driving,8:00-8:30','Driving,8:30-9:00','Driving,9:00-9:30',
                          'Driving,9:30-10:00','Transit,6:30-7:00','Transit,7:00-7:30',
                          'Transit,7:30-8:00','Transit,8:00-8:30','Transit,8:30-9:00',
                          'Transit,9:00-9:30','Transit,9:30-10:00'])
alternative_l = Lunch_choice_ms['alternative'].unique()
alternative_a = Afterwork_choice_ms['alternative'].unique()



def predict_accruarcy(compare,observed,prediction,total):
    accuracy = compare[[observed,prediction]].min(axis=1)
    accuracy = accuracy.sum()/total
    return accuracy

def plot_confusion_matrix(cm,labels_name,title):
    fig = plt.figure()
    column_sum = cm.sum(axis=1)[:,np.newaxis]
    column_sum[column_sum==0]=1
    cm = cm.astype('float')/column_sum
    plt.imshow(cm,interpolation='nearest')
    plt.title(title)
    plt.clim(0,0.8)
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local,labels_name,rotation=90)
    plt.yticks(num_local,labels_name)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.grid(False)
    plt.savefig('C:/Users/MSI-PC/Desktop/%s.jpg'%title,
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')

def choice_prediction(theta_ni_c,MNL_commute,Commuting_choice_ms,alternative):
    data_check = pd.merge(theta_ni_c,Commuting_choice_ms,on='iid')
    #calculate utility
    data_check['Utility_IO'] = 0
    data_check['Utility_MNL'] = 0
    for i in range(len(MNL_commute)):
        data_check['Utility_IO']+=data_check.iloc[:,i]*data_check.iloc[:,i+len(MNL_commute)+1]
        data_check['Utility_MNL']+=MNL_commute[i]*data_check.iloc[:,i+len(MNL_commute)+1]
    data_check = data_check[['iid','alternative','chosen','Utility_IO','Utility_MNL']]
    data_check['iid_index'] = data_check['iid']
    data_check = data_check.set_index('iid_index')
    #calculate prob
    data_check['prob_IO'] = 999
    aa = pd.Series()
    for iid in data_check['iid'].unique():
        utility = data_check[data_check['iid']==iid]['Utility_IO']
        utility = (utility - utility.min())/(utility.max()-utility.min()) * 30
        exp = np.exp(utility)
        exp_sum = np.exp(utility).sum()
        prob = exp/exp_sum
        data_check.loc[iid,'prob_IO']=prob.values    
    data_check['prob_MNL'] = 999
    aa = pd.Series()
    for iid in data_check['iid'].unique():
        utility = data_check[data_check['iid']==iid]['Utility_MNL']
        exp = np.exp(utility)
        exp_sum = np.exp(utility).sum()
        prob = exp/exp_sum
        data_check.loc[iid,'prob_MNL']=prob.values
    #disaggregated prediction
    compare = pd.DataFrame(index=alternative)
    for_matrix = pd.DataFrame(index=data_check['iid'].unique())
    for_matrix['observed'] = pd.Series(data_check[data_check['chosen']==True]['alternative'].values,index=for_matrix.index)
    aa = data_check[data_check.groupby('iid')['Utility_IO'].rank(method='first',ascending=False)==1]['alternative']
    for_matrix['prediction_IO'] = pd.Series(aa.values,index=for_matrix.index)
    aa = data_check[data_check.groupby('iid')['Utility_MNL'].rank(method='first',ascending=False)==1]['alternative']
    for_matrix['prediction_MNL'] = pd.Series(aa.values,index=for_matrix.index)
    for_matrix['num'] = 1
    from sklearn.metrics import confusion_matrix
    labels_name = compare.index
    cm = confusion_matrix(for_matrix['observed'],for_matrix['prediction_MNL'])
    #plot
    plot_confusion_matrix(cm,labels_name,'Confusion Matrix (MNL)')
    accuracy_MNL = (for_matrix['observed']==for_matrix['prediction_MNL']).sum()/len(data_check['iid'].unique())
    print('Individual-level accuracy of MNL prediction: %.4f'%accuracy_MNL)
    cm = confusion_matrix(for_matrix['observed'],for_matrix['prediction_IO'])
    #plot
    plot_confusion_matrix(cm,labels_name,'Confusion Matrix (AMXL)')
    accuracy_IO = (for_matrix['observed']==for_matrix['prediction_IO']).sum()/len(data_check['iid'].unique())
    print('Individual-level accuracy of IO prediction: %.4f'%accuracy_IO)
    #aggregated prediction
    compare['observed'] = data_check.groupby('alternative').agg({'chosen':'sum'})
    compare['prediction_IO'] = data_check.groupby('alternative').agg({'prob_IO':'sum'})
    compare['prediction_MNL'] = data_check.groupby('alternative').agg({'prob_MNL':'sum'})
    compare['prediction_IO'] = for_matrix.groupby('prediction_IO')['num'].count()
    #compare[['observed','prediction_IO','prediction_MNL']].plot.bar(width=0.75)
    compare.fillna(0,inplace=True)
    #plot
    compare[['observed','prediction_MNL']].plot.bar(width=0.6,title='Comparison at the aggregated level (MNL)')
    plt.savefig('C:/Users/MSI-PC/Desktop/AL_MNL.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')
    #plot
    compare[['observed','prediction_IO']].plot.bar(width=0.6,title='Comparison at the aggregated level (AMXL)')
    plt.savefig('C:/Users/MSI-PC/Desktop/AL_AMXL.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')
    accuracy = predict_accruarcy(compare,'observed','prediction_MNL',len(for_matrix))
    print('Aggregated level accuracy of MNL prediction: %.4f'%accuracy)
    accuracy = predict_accruarcy(compare,'observed','prediction_IO',len(for_matrix))
    print('Aggregated level accuracy of IO prediction: %.4f'%accuracy)
    return for_matrix

#commute
prediction_c = choice_prediction(theta_ni_c,MNL_commute,Commuting_choice_ms,alternative_c)
#lunch
prediction_l = choice_prediction(theta_ni_l,MNL_lunch,Lunch_choice_ms,alternative_l)
#afterwork
prediction_a = choice_prediction(theta_ni_a,MNL_afterwork,Afterwork_choice_ms,alternative_a)


#whole-day schedule
prediction_whole = pd.DataFrame(index=prediction_c.index)
prediction_whole['observed'] = prediction_c['observed']+' + '+prediction_l['observed']+' + '+prediction_a['observed']
prediction_whole['MNL_prediction'] = prediction_c['prediction_MNL']+' + '+prediction_l['prediction_MNL']+' + '+prediction_a['prediction_MNL']
prediction_whole['IO_prediction'] = prediction_c['prediction_IO']+' + '+prediction_l['prediction_IO']+' + '+prediction_a['prediction_IO']

correct_prediction = prediction_whole['observed']==prediction_whole['MNL_prediction']
correct_prediction.sum()/len(prediction_whole)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(prediction_whole['observed'],prediction_whole['IO_prediction'])
aa = pd.DataFrame()
aa['observed'] = cm.sum(axis=1)
aa['prediction'] = cm.sum(axis=0)
aa['correct'] = aa.min(axis=1)
aa['correct'].sum()/aa['observed'].sum()



plot_confusion_matrix(cm,[],'Confusion Matrix (IO)')
plt.savefig('C:/Users/MSI-PC/Desktop/Confusion Matrix (IO).jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')


#########################
#commute 0514 prediction#
#########################   
# input necessary data
def choice_prediction(theta_ni_c,MNL_commute,Commuting_choice_ms,alternative):
    data_check = pd.merge(theta_ni_c,Commuting_choice_ms,on='iid')
    #calculate utility
    data_check['Utility_IO'] = 0
    data_check['Utility_MNL'] = 0
    for i in range(len(MNL_commute)):
        data_check['Utility_IO']+=data_check.iloc[:,i]*data_check.iloc[:,i+len(MNL_commute)+1]
        data_check['Utility_MNL']+=MNL_commute[i]*data_check.iloc[:,i+len(MNL_commute)+1]
    data_check = data_check[['iid','alternative','chosen','Utility_IO','Utility_MNL']]
    data_check['iid_index'] = data_check['iid']
    data_check = data_check.set_index('iid_index')
    #calculate prob
    data_check['prob_IO'] = 999
    aa = pd.Series()
    for iid in data_check['iid'].unique():
        utility = data_check[data_check['iid']==iid]['Utility_IO']
        utility = (utility - utility.min())/(utility.max()-utility.min()) * 30
        exp = np.exp(utility)
        exp_sum = np.exp(utility).sum()
        prob = exp/exp_sum
        data_check.loc[iid,'prob_IO']=prob.values    
    data_check['prob_MNL'] = 999
    aa = pd.Series()
    for iid in data_check['iid'].unique():
        utility = data_check[data_check['iid']==iid]['Utility_MNL']
        exp = np.exp(utility)
        exp_sum = np.exp(utility).sum()
        prob = exp/exp_sum
        data_check.loc[iid,'prob_MNL']=prob.values
    #disaggregated prediction
    compare = pd.DataFrame(index=alternative)
    for_matrix = pd.DataFrame(index=data_check['iid'].unique())
    for_matrix['observed'] = pd.Series(data_check[data_check['chosen']==True]['alternative'].values,index=for_matrix.index)
    aa = data_check[data_check.groupby('iid')['Utility_IO'].rank(method='first',ascending=False)==1]['alternative']
    for_matrix['prediction_IO'] = pd.Series(aa.values,index=for_matrix.index)
    aa = data_check[data_check.groupby('iid')['Utility_MNL'].rank(method='first',ascending=False)==1]['alternative']
    for_matrix['prediction_MNL'] = pd.Series(aa.values,index=for_matrix.index)
    for_matrix['num'] = 1
    from sklearn.metrics import confusion_matrix
    labels_name = compare.index
    cm = confusion_matrix(for_matrix['observed'],for_matrix['prediction_MNL'])
    #plot
    plot_confusion_matrix(cm,labels_name,'Confusion Matrix (MNL)')
    accuracy_MNL = (for_matrix['observed']==for_matrix['prediction_MNL']).sum()/len(data_check['iid'].unique())
    print('Individual-level accuracy of MNL prediction: %.4f'%accuracy_MNL)
    cm = confusion_matrix(for_matrix['observed'],for_matrix['prediction_IO'])
    #plot
    plot_confusion_matrix(cm,labels_name,'Confusion Matrix (AMXL)')
    accuracy_IO = (for_matrix['observed']==for_matrix['prediction_IO']).sum()/len(data_check['iid'].unique())
    print('Individual-level accuracy of IO prediction: %.4f'%accuracy_IO)
    #aggregated prediction
    compare['observed'] = data_check.groupby('alternative').agg({'chosen':'sum'})
    compare['prediction_IO'] = data_check.groupby('alternative').agg({'prob_IO':'sum'})
    compare['prediction_MNL'] = data_check.groupby('alternative').agg({'prob_MNL':'sum'})
    #compare['prediction_IO'] = for_matrix.groupby('prediction_IO')['num'].count()
    #compare[['observed','prediction_IO','prediction_MNL']].plot.bar(width=0.75)
    compare.fillna(0,inplace=True)
    #plot
    compare[['observed','prediction_MNL']].plot.bar(width=0.6,title='Comparison at the aggregated level (MNL)')
    plt.savefig('C:/Users/MSI-PC/Desktop/AL_MNL.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')
    #plot
    compare[['observed','prediction_IO']].plot.bar(width=0.6,title='Comparison at the aggregated level (AMXL)')
    plt.savefig('C:/Users/MSI-PC/Desktop/AL_AMXL.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')
    accuracy = predict_accruarcy(compare,'observed','prediction_MNL',len(for_matrix))
    print('Aggregated level accuracy of MNL prediction: %.4f'%accuracy)
    accuracy = predict_accruarcy(compare,'observed','prediction_IO',len(for_matrix))
    print('Aggregated level accuracy of IO prediction: %.4f'%accuracy)
    return for_matrix




Commuting_choice_14 = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Commuting_choice_0514.csv")
Lunch_choice_14 = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Lunch_choice_0514.csv")
Afterwork_choice_14 = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Afterwork_choice_0514.csv")

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
Commuting_choice_ms = Commuting_choice_14.copy()
ms.fit(Commuting_choice.iloc[:,3:-1].values)
Commuting_choice_ms.iloc[:,3:-1] = ms.transform(Commuting_choice_ms.iloc[:,3:-1].values)
Lunch_choice_ms = Lunch_choice_14.copy()
ms.fit(Lunch_choice.iloc[:,3:-1].values)
Lunch_choice_ms.iloc[:,3:-1] = ms.transform(Lunch_choice_ms.iloc[:,3:-1].values)
Afterwork_choice_ms = Afterwork_choice_14.copy()
ms.fit(Afterwork_choice.iloc[:,3:-1].values)
Afterwork_choice_ms.iloc[:,3:-1] = ms.transform(Afterwork_choice_ms.iloc[:,3:-1].values)

Commuting_choice_ms = Commuting_choice_ms[['t_commute','c_commute','M_commute2','SDE_work','SDL_work','PL_work','ln_dwork','iid','hw_od','alternative','chosen']]
Commuting_choice_ms['alternative'] = Commuting_choice_ms['alternative'].map({'6:30-7:00,Driving':'Driving,6:30-7:00',
                                        '7:00-7:30,Driving':'Driving,7:00-7:30',
                                        '7:30-8:00,Driving':'Driving,7:30-8:00',
                                        '8:00-8:30,Driving':'Driving,8:00-8:30',
                                        '8:30-9:00,Driving':'Driving,8:30-9:00',
                                        '9:00-9:30,Driving':'Driving,9:00-9:30',
                                        '9:30-10:00,Driving':'Driving,9:30-10:00',
                                        '6:30-7:00,Transit':'Transit,6:30-7:00',
                                        '7:00-7:30,Transit':'Transit,7:00-7:30',
                                        '7:30-8:00,Transit':'Transit,7:30-8:00',
                                        '8:00-8:30,Transit':'Transit,8:00-8:30',
                                        '8:30-9:00,Transit':'Transit,8:30-9:00',
                                        '9:00-9:30,Transit':'Transit,9:00-9:30',
                                        '9:30-10:00,Transit':'Transit,9:30-10:00'})
Lunch_choice_ms = Lunch_choice_ms[['SDE_lunch','SDL_lunch','K_lunch1','K_lunch2','t_worklunch','ln_dwork','iid','hw_od','alternative','chosen']]
Afterwork_choice_ms = Afterwork_choice_ms[['ln_dafterwork','ln_dwork*ln_afterwork','ln_dwork','iid','hw_od','alternative','chosen']]

#commute
prediction_c = choice_prediction(theta_ni_c,MNL_commute,Commuting_choice_ms,alternative_c)
#lunch
prediction_l = choice_prediction(theta_ni_l,MNL_lunch,Lunch_choice_ms,alternative_l)
#afterwork
prediction_a = choice_prediction(theta_ni_a,MNL_afterwork,Afterwork_choice_ms,alternative_a)

#whole-day schedule
prediction_whole = pd.DataFrame(index=prediction_c.index)
prediction_whole['observed'] = prediction_c['observed']+' + '+prediction_l['observed']+' + '+prediction_a['observed']
prediction_whole['MNL_prediction'] = prediction_c['prediction_MNL']+' + '+prediction_l['prediction_MNL']+' + '+prediction_a['prediction_MNL']
prediction_whole['IO_prediction'] = prediction_c['prediction_IO']+' + '+prediction_l['prediction_IO']+' + '+prediction_a['prediction_IO']

correct_prediction = prediction_whole['observed']==prediction_whole['IO_prediction']
correct_prediction.sum()/len(prediction_whole)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(prediction_whole['observed'],prediction_whole['MNL_prediction'])
aa = pd.DataFrame()
aa['observed'] = cm.sum(axis=1)
aa['prediction'] = cm.sum(axis=0)
aa['correct'] = aa[['observed','prediction']].min(axis=1)
aa['correct'].sum()/aa['prediction'].sum()



#####################################
##### Scenario 1 Commuting time #####
#####################################
# Define function
def scenario_prediction(theta_ni_c,MNL_coef,Commuting_choice_ms,alternative):
    data_check = pd.merge(theta_ni_c,Commuting_choice_ms,on='iid')
    #calculate utility
    data_check['Utility_IO'] = 0
    data_check['Utility_MNL'] = 0
    for i in range(len(MNL_coef)):
        data_check['Utility_IO']+=data_check.iloc[:,i]*data_check.iloc[:,i+len(MNL_coef)+1]
        data_check['Utility_MNL']+=MNL_coef[i]*data_check.iloc[:,i+len(MNL_coef)+1]
    data_check = data_check[['iid','alternative','chosen','Utility_IO','Utility_MNL']]
    data_check['iid_index'] = data_check['iid']
    data_check = data_check.set_index('iid_index')
    #calculate prob
    data_check['prob_IO'] = 999
    for iid in data_check['iid'].unique():
        utility = data_check[data_check['iid']==iid]['Utility_IO']
        utility = (utility - utility.min())/(utility.max()-utility.min()) * 30
        exp = np.exp(utility)
        exp_sum = np.exp(utility).sum()
        prob = exp/exp_sum
        data_check.loc[iid,'prob_IO']=prob.values    
    data_check['prob_MNL'] = 999
    for iid in data_check['iid'].unique():
        utility = data_check[data_check['iid']==iid]['Utility_MNL']
        exp = np.exp(utility)
        exp_sum = np.exp(utility).sum()
        prob = exp/exp_sum
        data_check.loc[iid,'prob_MNL']=prob.values
    #disaggregated prediction
    compare = pd.DataFrame(index=alternative)
    for_matrix = pd.DataFrame(index=data_check['iid'].unique())
    for_matrix['observed'] = pd.Series(data_check[data_check['chosen']==True]['alternative'].values,index=for_matrix.index)
    aa = data_check[data_check.groupby('iid')['Utility_IO'].rank(method='first',ascending=False)==1]['alternative']
    for_matrix['prediction_IO'] = pd.Series(aa.values,index=for_matrix.index)
    aa = data_check[data_check.groupby('iid')['Utility_MNL'].rank(method='first',ascending=False)==1]['alternative']
    for_matrix['prediction_MNL'] = pd.Series(aa.values,index=for_matrix.index)
    for_matrix['num'] = 1
    #aggregated prediction
    compare['observed'] = data_check.groupby('alternative').agg({'chosen':'sum'})
    compare['prediction_IO'] = data_check.groupby('alternative').agg({'prob_IO':'sum'})
    compare['prediction_MNL'] = data_check.groupby('alternative').agg({'prob_MNL':'sum'})
    #compare['prediction_IO'] = for_matrix.groupby('prediction_IO')['num'].count()
    return compare,data_check

def scenario1(record):
    if record['alternative'] in ['7:30-8:00,Driving','8:00-8:30,Driving',
                                 '8:30-9:00,Driving','9:00-9:30,Driving']:
        time = record['t_commute']*0.8
    else:
        time = record['t_commute']
    return time

def result_comparison(scenario,benchmark):
    compare = pd.DataFrame()
    compare['benchmark_IO'] = benchmark['prediction_IO']
    compare['benchmark_MNL'] = benchmark['prediction_MNL']
    compare['scenario_IO'] = scenario['prediction_IO']
    compare['scenario_MNL'] = scenario['prediction_MNL']
    return compare

# Data Processing
Commuting_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Commuting_choice_0507.csv")

Commuting_choice_BM = Commuting_choice.copy()
Commuting_choice_S1 = Commuting_choice.copy()
Commuting_choice_S1['t_commute'] = Commuting_choice_S1.apply(scenario1,axis=1)

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(Commuting_choice.iloc[:,3:-1].values)
Commuting_choice_BM.iloc[:,3:-1] = ms.transform(Commuting_choice_BM.iloc[:,3:-1].values)
Commuting_choice_S1.iloc[:,3:-1] = ms.transform(Commuting_choice_S1.iloc[:,3:-1].values)

Commuting_choice_S1 = Commuting_choice_S1[['t_commute','c_commute','M_commute2','SDE_work','SDL_work','PL_work','ln_dwork','iid','hw_od','alternative','chosen']]
Commuting_choice_S1['alternative'] = Commuting_choice_S1['alternative'].map({'6:30-7:00,Driving':'Driving,6:30-7:00',
                                        '7:00-7:30,Driving':'Driving,7:00-7:30',
                                        '7:30-8:00,Driving':'Driving,7:30-8:00',
                                        '8:00-8:30,Driving':'Driving,8:00-8:30',
                                        '8:30-9:00,Driving':'Driving,8:30-9:00',
                                        '9:00-9:30,Driving':'Driving,9:00-9:30',
                                        '9:30-10:00,Driving':'Driving,9:30-10:00',
                                        '6:30-7:00,Transit':'Transit,6:30-7:00',
                                        '7:00-7:30,Transit':'Transit,7:00-7:30',
                                        '7:30-8:00,Transit':'Transit,7:30-8:00',
                                        '8:00-8:30,Transit':'Transit,8:00-8:30',
                                        '8:30-9:00,Transit':'Transit,8:30-9:00',
                                        '9:00-9:30,Transit':'Transit,9:00-9:30',
                                        '9:30-10:00,Transit':'Transit,9:30-10:00'})

Commuting_choice_BM = Commuting_choice_BM[['t_commute','c_commute','M_commute2','SDE_work','SDL_work','PL_work','ln_dwork','iid','hw_od','alternative','chosen']]
Commuting_choice_BM['alternative'] = Commuting_choice_BM['alternative'].map({'6:30-7:00,Driving':'Driving,6:30-7:00',
                                        '7:00-7:30,Driving':'Driving,7:00-7:30',
                                        '7:30-8:00,Driving':'Driving,7:30-8:00',
                                        '8:00-8:30,Driving':'Driving,8:00-8:30',
                                        '8:30-9:00,Driving':'Driving,8:30-9:00',
                                        '9:00-9:30,Driving':'Driving,9:00-9:30',
                                        '9:30-10:00,Driving':'Driving,9:30-10:00',
                                        '6:30-7:00,Transit':'Transit,6:30-7:00',
                                        '7:00-7:30,Transit':'Transit,7:00-7:30',
                                        '7:30-8:00,Transit':'Transit,7:30-8:00',
                                        '8:00-8:30,Transit':'Transit,8:00-8:30',
                                        '8:30-9:00,Transit':'Transit,8:30-9:00',
                                        '9:00-9:30,Transit':'Transit,9:00-9:30',
                                        '9:30-10:00,Transit':'Transit,9:30-10:00'})

MNL_commute = [-8.877,-1.005,0.374,-6.933,-5.324,0.335,22.365]
alternative_c = np.array(['Driving,6:30-7:00','Driving,7:00-7:30','Driving,7:30-8:00',
                          'Driving,8:00-8:30','Driving,8:30-9:00','Driving,9:00-9:30',
                          'Driving,9:30-10:00','Transit,6:30-7:00','Transit,7:00-7:30',
                          'Transit,7:30-8:00','Transit,8:00-8:30','Transit,8:30-9:00',
                          'Transit,9:00-9:30','Transit,9:30-10:00'])
theta_ni_c_good = theta_ni_c[theta_ni_c['p_t_commute']<0]


#Simulation
benchmark,data_BM = scenario_prediction(theta_ni_c_good,MNL_commute,Commuting_choice_BM,alternative_c)
scenario1,data_s1 = scenario_prediction(theta_ni_c_good,MNL_commute,Commuting_choice_S1,alternative_c)


compare = result_comparison(scenario1,benchmark)

compare['MNL'] = (compare['scenario_MNL'] - compare['benchmark_MNL'])/len(theta_ni_c_good)
compare['AMXL'] = (compare['scenario_IO'] - compare['benchmark_IO'])/len(theta_ni_c_good)

from matplotlib import ticker
fig,ax = plt.subplots(1,figsize=(6,3.5))
compare[['MNL','AMXL']].plot.bar(width=0.6,ylim=[-0.03,0.03],ax=ax)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.axhline(y=0, color='grey', linestyle='-',linewidth=2,alpha=0.7)
plt.title('Choice shift predicted by MNL and AMXL')
plt.savefig('C:/Users/MSI-PC/Desktop/S1-1.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')



#########################################
##### Scenario 2 Delay of departure #####
#########################################
Afterwork_choice_BM = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Afterwork_choice_0507.csv")
Afterwork_choice_S2_1 = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Afterwork_choice_0507_s2_1.csv")
Afterwork_choice_S2_2 = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Afterwork_choice_0507_s2_2.csv")

from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
ms.fit(Afterwork_choice_BM.iloc[:,3:-1].values)
Afterwork_choice_BM.iloc[:,3:-1] = ms.transform(Afterwork_choice_BM.iloc[:,3:-1].values)
Afterwork_choice_S2_1.iloc[:,3:-1] = ms.transform(Afterwork_choice_S2_1.iloc[:,3:-1].values)
Afterwork_choice_S2_2.iloc[:,3:-1] = ms.transform(Afterwork_choice_S2_2.iloc[:,3:-1].values)

Afterwork_choice_BM = Afterwork_choice_BM[['ln_dafterwork','ln_dwork*ln_afterwork','ln_dwork','iid','hw_od','alternative','chosen']]
Afterwork_choice_S2_1 = Afterwork_choice_S2_1[['ln_dafterwork','ln_dwork*ln_afterwork','ln_dwork','iid','hw_od','alternative','chosen']]
Afterwork_choice_S2_2 = Afterwork_choice_S2_2[['ln_dafterwork','ln_dwork*ln_afterwork','ln_dwork','iid','hw_od','alternative','chosen']]

MNL_afterwork = [6.398,11.892,2.277]
theta_ni_a_good = theta_ni_a[(theta_ni_a['p_ln_afterwork']>0)&(theta_ni_a['p_ln_dwork*ln_afterwork']>0)&(theta_ni_a['p_ln_dwork']>0)]

alternative_a = Afterwork_choice_BM['alternative'].unique()

benchmark,data_BM = scenario_prediction(theta_ni_a,MNL_afterwork,Afterwork_choice_BM,alternative_a)
scenario2_1,data_S2_1 = scenario_prediction(theta_ni_a,MNL_afterwork,Afterwork_choice_S2_1,alternative_a)
scenario2_2,data_S2_2 = scenario_prediction(theta_ni_a,MNL_afterwork,Afterwork_choice_S2_2,alternative_a)

compare = pd.DataFrame(index=benchmark.index)
compare['Benchmark'] = benchmark['prediction_IO']/len(theta_ni_a)
compare['Postpone_1h'] = scenario2_1['prediction_IO']/len(theta_ni_a)
compare['Postpone_2h'] = scenario2_2['prediction_IO']/len(theta_ni_a)

from matplotlib import ticker
fig,ax = plt.subplots(1,figsize=(6,3.5))
compare[['Benchmark','Postpone_1h','Postpone_2h']].plot.bar(width=0.6,ax=ax)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=1))
plt.axhline(y=0, color='grey', linestyle='-',linewidth=2,alpha=0.7)
plt.title('Time to leave work predicted by AMXL')
plt.savefig('C:/Users/MSI-PC/Desktop/S2-1.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')



data_BM['Utility_IO_2'] = 999
for iid in data_BM['iid'].unique():
    utility = data_BM[data_BM['iid']==iid]['Utility_IO']
    utility = (utility - utility.min())/(utility.max()-utility.min()) * 30
    data_BM.loc[iid,'Utility_IO_2']=utility.values  

data_S2_1['Utility_IO_2'] = 999
for iid in data_S2_1['iid'].unique():
    utility = data_S2_1[data_S2_1['iid']==iid]['Utility_IO']
    utility = (utility - utility.min())/(utility.max()-utility.min()) * 30
    data_S2_1.loc[iid,'Utility_IO_2']=utility.values

data_S2_2['Utility_IO_2'] = 999
for iid in data_S2_2['iid'].unique():
    utility = data_S2_2[data_S2_2['iid']==iid]['Utility_IO']
    utility = (utility - utility.min())/(utility.max()-utility.min()) * 30
    data_S2_2.loc[iid,'Utility_IO_2']=utility.values

#CDF
import statsmodels.api as sm
data = data_BM.groupby('iid')['Utility_MNL'].max()
sample = np.array(data)
ecdf = sm.distributions.ECDF(sample)
x = np.linspace(0, sample.max(),100)
y = ecdf(x)

data = data_S2_1.groupby('iid')['Utility_MNL'].max()
sample = np.array(data)
ecdf = sm.distributions.ECDF(sample)
x_1 = np.linspace(0, sample.max(),100)
y_1 = ecdf(x_1)

data = data_S2_2.groupby('iid')['Utility_MNL'].max()
sample = np.array(data)
ecdf = sm.distributions.ECDF(sample)
x_2 = np.linspace(0, sample.max(),100)
y_2 = ecdf(x_2)

fig,ax = plt.subplots(figsize=(5,3.5))
ax.plot(x, y, linewidth = 2,label='Benchmark')
ax.plot(x_1, y_1, linewidth = 2,label='Postpone_1h')
ax.plot(x_2, y_2, linewidth = 2,label='Postpone_2h')
ax.set_title('Cumulative density function of individuals’ utility')
ax.set_xlabel('Utility of the chosen alternative')
ax.set_ylabel('Cumulative density')
plt.legend(loc='upper left')
plt.savefig('C:/Users/MSI-PC/Desktop/S2-2.jpg',
            dpi=300,
            bbox_inches = 'tight',
            facecolor = 'w',
            edgecolor = 'w')

#######################################
##### Scenario 3 Lunch incentives #####
#######################################



# Data Processing
Lunch_choice = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\Dataset\Lunch_choice_0507.csv")
id_info = pd.read_csv(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\1.离散选择模型构建\0507_choice_generation.csv",encoding='ANSI')[['iid','home_lng','home_lat','work_lng','work_lat']]

Lunch_choice_ms = Lunch_choice.copy()
from sklearn.preprocessing import MinMaxScaler
ms = MinMaxScaler()
Lunch_choice_ms.iloc[:,3:-1] = ms.fit_transform(Lunch_choice_ms.iloc[:,3:-1].values)

MNL_lunch = [-0.840,-1.802,-1.927,0,-0.429,1.421]


data = pd.merge(theta_ni_l,id_info,on='iid')

import geopandas as gpd
from shapely.geometry import Point
geometry = [Point(xy) for xy in zip(data.work_lng,data.work_lat)]
data = gpd.GeoDataFrame(data, crs="EPSG:4326", geometry=geometry)
Block = gpd.read_file(r"C:\Users\MSI-PC\Desktop\陆家嘴新一轮个体数据研究\3.完整模型_第一版\GIS_file\80TAZ_WGS1984.shp")[['geometry','TAZID','备注','TAZ_lng','TAZ_lat']]
#Block_join里面记录了每个commuter的work_TAZ以及偏好参数
Block_join = gpd.sjoin(Block,data,op='contains')
Block_join = Block_join[['p_SDE_lunch', 'p_SDL_lunch','p_K_lunch1', 'p_K_lunch2',
                         'p_t_worklunch', 'p_ln_dwork', 'iid', 'TAZID','TAZ_lng','TAZ_lat']]
Block_join = Block_join[(Block_join['p_t_worklunch']<0)&(Block_join['p_K_lunch1']<0)]

#restaurant 位置
Lng = Block[Block['TAZID']==54]['TAZ_lng'].values
Lat = Block[Block['TAZID']==54]['TAZ_lat'].values

from math import radians, cos, sin, asin, sqrt
def geodistance(record):  
    lng1 = Lng
    lat1 = Lat
    lng2 = record['TAZ_lng']
    lat2 = record['TAZ_lat']
    lng1,lat1,lng2,lat2 = map(radians,[float(lng1), float(lat1), float(lng2), float(lat2)])
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2 
    distance=2*asin(sqrt(a))*6371*1000
    distance=round(distance,2)
    return distance

max_value = Lunch_choice['t_worklunch'].max()
Block_join['distance'] = Block_join.apply(geodistance,axis=1)/1/60/max_value


def with_coupon(record):
    distance = record['distance']
    c = distance*record['p_t_worklunch']+(-1)*record['p_K_lunch1']
    return c

Block_join['coupon'] = Block_join.apply(with_coupon,axis=1)

aa = Block_join.groupby('TAZID').agg({'iid':'count','coupon':'sum'}).reset_index()



from gurobipy import *
m = Model()
x = m.addVars(80,lb=0 ,ub=1 , vtype=GRB.INTEGER, name='x') #add decision variables

coupon = aa['coupon']
num = aa['iid']

m.addConstr(x[53]==0)

m.addConstr(x[0]*num[0]+x[1]*num[1]+x[2]*num[2]+x[3]*num[3]+x[4]*num[4]+x[5]*num[5]+x[6]*num[6]+x[7]*num[7]+x[8]*num[8]+x[9]*num[9]+x[10]*num[10]+x[11]*num[11]+x[12]*num[12]+x[13]*num[13]+x[14]*num[14]+x[15]*num[15]+x[16]*num[16]+x[17]*num[17]+x[18]*num[18]+x[19]*num[19]+x[20]*num[20]+x[21]*num[21]+x[22]*num[22]+x[23]*num[23]+x[24]*num[24]+x[25]*num[25]+x[26]*num[26]+x[27]*num[27]+x[28]*num[28]+x[29]*num[29]+x[30]*num[30]+x[31]*num[31]+x[32]*num[32]+x[33]*num[33]+x[34]*num[34]+x[35]*num[35]+x[36]*num[36]+x[37]*num[37]+x[38]*num[38]+x[39]*num[39]+x[40]*num[40]+x[41]*num[41]+x[42]*num[42]+x[43]*num[43]+x[44]*num[44]+x[45]*num[45]+x[46]*num[46]+x[47]*num[47]+x[48]*num[48]+x[49]*num[49]+x[50]*num[50]+x[51]*num[51]+x[52]*num[52]+x[53]*num[53]+x[54]*num[54]+x[55]*num[55]+x[56]*num[56]+x[57]*num[57]+x[58]*num[58]+x[59]*num[59]+x[60]*num[60]+x[61]*num[61]+x[62]*num[62]+x[63]*num[63]+x[64]*num[64]+x[65]*num[65]+x[66]*num[66]+x[67]*num[67]+x[68]*num[68]+x[69]*num[69]+x[70]*num[70]+x[71]*num[71]+x[72]*num[72]+x[73]*num[73]+x[74]*num[74]+x[75]*num[75]+x[76]*num[76]+x[77]*num[77]+x[78]*num[78]+x[79]*num[79]<= 5000)

m.addConstr(x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]+x[13]+x[14]+x[15]+x[16]+x[17]+x[18]+x[19]+x[20]+x[21]+x[22]+x[23]+x[24]+x[25]+x[26]+x[27]+x[28]+x[29]+x[30]+x[31]+x[32]+x[33]+x[34]+x[35]+x[36]+x[37]+x[38]+x[39]+x[40]+x[41]+x[42]+x[43]+x[44]+x[45]+x[46]+x[47]+x[48]+x[49]+x[50]+x[51]+x[52]+x[53]+x[54]+x[55]+x[56]+x[57]+x[58]+x[59]+x[60]+x[61]+x[62]+x[63]+x[64]+x[65]+x[66]+x[67]+x[68]+x[69]+x[70]+x[71]+x[72]+x[73]+x[74]+x[75]+x[76]+x[77]+x[78]+x[79]<= 20)

m.setObjective(quicksum(x[j] * coupon[j] for j in range(80)) , GRB.MAXIMIZE)

m.update()
m.Params.LogToConsole = 0
m.optimize()
variables = np.array(m.getAttr('X', m.getVars()))
Z = m.ObjVal

np.argwhere(variables==1)+1




























