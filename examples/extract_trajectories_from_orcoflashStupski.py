# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:20:43 2025

@author: mayc06
"""
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pynumdiff
import utils
from utils import unwrap_angle


def get_trajlist_from_behdata(csv,trajstart,trajstop,windspeed=40):
    
    '''
    A function to easily grab a list of trajectories from behavior data 
    from the van Breugel lab wind tunnels (Stupski & van Breugel 2024).
    
    trajstart and trajstop are in terms of 'time stamp', in ms, where 0 is
    aligned to Flash_bool, and sample rate = 10 ms/frame
    trajstart = -500
    trajstop = 1000
    windspeed = ambient windspeed in cm/s
    
    Returns: 
        A list of trajectory dataframes with new computed variables
    
    '''
    ## Import the data.
    data = pd.read_csv(csv)
    if 'ground speed' not in data:
        data['ground speed']=(data['xvel']**2+data['yvel']**2)**0.5
    
    ## Select time-segmented trajectories.
    data_2 = data[data['time stamp']>trajstart]
    data_3 = data_2[data_2['time stamp']<=trajstop]
    data_3.reset_index()
    
    n_traj = sum(data_3['Flash_bool'])
    startlocs = [0] + list(np.where(np.diff(data_3['time stamp'])<0)[0]+1)
    endlocs = list(np.where(np.diff(data_3['time stamp'])<0)[0]) + [len(data_3['time stamp'])]
    
    ## Compute variables of interest.
    traj_list = []
    new_index = pd.Index(np.linspace(trajstart+10.0,trajstop,num=int((trajstop-trajstart)/10)))
    for n in range(n_traj):
        traj = data_3.iloc[startlocs[n]:endlocs[n]+1,:].copy()
        traj['windspeed'] = np.ones_like(traj['time stamp'])*windspeed
        a = unwrap_angle(traj['heading'].values, 5)
        _, da = pynumdiff.finite_difference.first_order(a,0.01)
        traj['ang vel'] = da  # units per sec if you did the dt param correctly
        traj['abs ang vel'] = abs(da)
        traj['heading_unwrapped'] = a
        g = traj['ground speed'].copy()
        smoothg, dg = pynumdiff.linear_model.savgoldiff(g,10,[3,10,10])
        traj['ground speed smooth dot'] = dg.copy()
        o = np.abs(traj['heading'])
        traj['up-down-cross']=0 
        traj['up-down-cross'] = np.where( o<=np.pi/4, 1, 
                                         traj['up-down-cross'])
        traj['up-down-cross'] = np.where( o>3*np.pi/4, -1, 
                                         traj['up-down-cross'])
        traj['w_over_g'] = (traj['windspeed']/100) / g
        traj = traj.set_index('time stamp')
        traj = traj.reindex(new_index)
        traj_list.append(traj)
    
    return traj_list


def get_distribution(traj_list,variable,start,end,plotting,bins=30):
    '''
    Parameters
    ----------
    traj_list : list
        List of trajectory dataframes.
    variable : string
        Column label in traj_list dfs for desired data, e.g. 'heading'.
    start and end : int
        Start and end indices over which to compile data.
    plotting : bool
        Toggle plot of distribution.

    Returns
    -------
    Curated dataset used to plot histogram.

    '''
    out = np.empty((len(traj_list),traj_list[0][start:end].shape[0]))
    for i in range(len(traj_list)):
        out[i,:] = traj_list[i][variable][start:end]
        
    if plotting:
        if bins is None:
            bins=np.linspace(np.min(out),np.max(out),100)
        fig,ax = plt.subplots();
        ax.hist(np.reshape(out,out.shape[0]*out.shape[1]),bins);
        ax.set_title(variable + ', range: '+str(start)+':'+str(end));
        #ax.set_xlim(0,5);
    
    return out


def plot_log_probability(datalist,bins):
    '''
    This plots multiple PMF (probability mass function), not PDF, on a log-
    scale y-axis.
    
    Parameters
    ----------
    datalist : list of arrays
        Each list item is an output from get_distribution with certain 
        start/end indices. (e.g. -200:10, aka baseline data)
    bins : sequence
        Sets the bins (use np.linspace) for the histogram.

    Returns
    -------
    None.

    '''
    
    fig,ax = plt.subplots();
    for r in range(len(datalist)):
        data = datalist[r]
        d = np.histogram(data[np.isfinite(data)],bins=bins)
        ax.plot(d[1][1:],d[0]/np.sum(d[0]));

    
def mean_log_probability(data,bins):
    '''
    This will iterate over 2D trajectory data to compute the PMF for 
    each trajectory, and then the mean and SE of the set.

    Parameters
    ----------
    data : array
        Output from get_distribution. Rows are trajectories and columns are
        timepoints.
    bins : sequence
        Defines the bins for the histograms.

    Returns
    -------
    m : float array
        Mean of PMF of the trajectories.
    sem : float array
        SE of trajectory PMFs.

    '''
    
    n_traj = data.shape[0]
    pmfs = np.empty((n_traj,len(bins)-1))
    
    for traj in range(n_traj):
        pmfs[traj,:],_ = np.histogram(data[traj,np.isfinite(data[traj,:])],bins)
        pmfs[traj,:] = pmfs[traj,:]/np.nansum(pmfs[traj,:])
    
    m = np.nanmean(pmfs,axis=0)
    se = np.nanstd(pmfs,axis=0)/np.sqrt(n_traj)
    
    return m, se


def sep_by_value(traj_list,variable,start,end,threshold):
    '''
    This function will take a trajectory list and sort the trajectories into 
    two new lists based on the mean value of the specified variable and its 
    relationship to the threshold.

    Parameters
    ----------
    traj_list : list
        List of trajectory dataframes.
    variable : string
        Column label in traj_list dfs for desired data, e.g. 'heading'.
    start and end : int
        Start and end indices over which to compile data.
    threshold : float
        Value on which to sort the data 

    Returns
    -------
    below : list
        List of trajectory dataframes whose mean value of varibale during 
        the time period start->end is BELOW the threshold.
    above : list
        List of trajectory dataframes whose mean value of variable during 
        the time period start->end MEETS OR SURPASSES the threshold.

    '''
    below = []
    above = []
    for i in range(len(traj_list)):
        testdata = traj_list[i][variable].loc[start:end]
        if np.nanmean(testdata)<threshold:
            below.append(traj_list[i])
        else: above.append(traj_list[i])
        
    return below, above


def compute_angular_dispersion(traj_list,start,end,plotting=False,ax=None):
    '''
    
    '''
    end+=10
    angdisps = np.empty((len(traj_list),int(np.ceil((end-start)/10))))
    for i in range(len(traj_list)):
        angdisps[i,:] = np.abs(traj_list[i]['heading_unwrapped'].loc[start:end]-traj_list[i]['heading_unwrapped'].loc[start])
        # angdisps[i,:] = traj_list[i]['heading_unwrapped'].loc[start:end]-traj_list[i]['heading_unwrapped'].loc[start]
    
    m = np.nanmean(angdisps,axis=0)
    se = np.nanstd(angdisps,axis=0)/np.sqrt(len(traj_list))
    
    if plotting:
        if ax is None:
            fig,ax = plt.subplots();
        ax.plot(np.linspace(start,end,int(np.ceil((end-start)/10))),np.nanmean(angdisps,axis=0));
        ax.fill_between(np.linspace(start,end,int(np.ceil((end-start)/10))),
                        m-se,m+se,alpha=0.5)
    
    return angdisps


    
## Example run scripts.

traj_list40 = get_trajlist_from_behdata('OrcoCsChrimson_laminar_wind_merged.csv',-500.0,2000.0,windspeed=40.0)




# all_wvg150to340 = np.vstack([w4_wvg150to340,w15_wvg150to340,w40_wvg150to340])
# all_success810to1000 = np.vstack([w4_success810to1000,w15_success810to1000,w40_success810to1000])
# fig,ax = plt.subplots();\
# ax.scatter(np.max(all_wvg150to340,axis=1),np.nanmean(all_success810to1000,axis=1),alpha=0.3);\
# ax.set_xscale('log');\
# plt.savefig('all_meanSuccess810to1000_vs_all_maxWvg150to340.svg')

# fig,ax = plt.subplots();\
# ax.scatter(np.max(all_wvg150to340,axis=1),abs(np.nanmean(all_head810to1000,axis=1)),alpha=0.3);\
# ax.set_xscale('log');\
# plt.savefig('all_head810to1000_vs_all_wvg150to340.svg')


# fig,ax = plt.subplots();\
# ax.plot(traj_list40[0]['heading']);\
# ax.plot(np.abs(traj_list40[0]['ang vel']));


# decels = []
# phi0s = []
# for traj in traj_list100:
#     decel = min(traj['ground speed'][0.0:250.0]) - traj['ground speed'][-50.0:0.0].mean()
#     deceltime = traj['ground speed'][0.0:250.0].idxmin() / 1000   # time to min groundspeed from opto on in seconds
#     decels.append(decel/deceltime)
#     phi0s.append(traj['heading'][-50.0:0.0].mean())


# fig,ax = plt.subplots(figsize=(5,5));\
# ax.scatter(phi0s,decels,s=5);\
# ax.set_ylim(-12.5,2);\
# ax.invert_yaxis();
# plt.savefig('Stupski-orco100_phi0s-vs-decels.svg')

# fig,ax = plt.subplots(figsize=(5,5));
# for traj in traj_list100:
#     ax.plot(traj['ground speed'],'k',linewidth=0.1)
    
    
# count=0
# fig,ax = plt.subplots(figsize=(5,5));
# for traj in traj_list20:
#     print(count)
#     a = traj['ground speed']
#     smootha,da = pynumdiff.linear_model.savgoldiff(a,10,[3,10,10])
#     ax.plot(traj.index,da,'k',linewidth=0.1)
#     count+=1
