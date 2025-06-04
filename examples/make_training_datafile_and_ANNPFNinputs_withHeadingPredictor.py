# -*- coding: utf-8 -*-
"""
Created on Wed May 21 11:57:12 2025

Streamlined code workflow for simulating flight trajectories with no-wind MPC,
then running it through the heading predictor model, then simulating again
using the originally-computed no-wind control values and the new heading values.

@author: mayc06
"""

import sys
import os

sys.path.append(os.path.join(os.path.pardir, 'util'))


import numpy as np
import pandas as pd
import math
from math import ceil, floor
import scipy
from scipy.io import savemat
import matplotlib as mpl
import matplotlib.pyplot as plt

from PFN_model import PFN, BuildSSInitial
from keras.models import load_model
from pybounds import Simulator#, SlidingEmpiricalObservabilityMatrix, FisherObservability, SlidingFisherObservability, ObservabilityMatrixImage, colorline
from pybounds_inputs import f,h
from heading_predictor_functions import augment_with_time_delay_embedding, predict_heading_from_fly_trajectory

save = bool(int(input('Do you want to save your stimulus and PFN activity dataframes? 1 for yes, 0 for no\n')))
stimulus = pd.DataFrame()
PFNdf = pd.DataFrame()

if save:
    stimfilename = input('Type the name of the stimulus file (no filetype extension) and press enter\n')
    # stimfilename = "n10rand_lesscontrol_varG_stims-streamlined.csv"
    stimfilename = stimfilename+'.csv'
    stimfile = open(stimfilename, 'a')
    
    PFNfilename = input('Type the name of the PFN activity file (no filetype extension) and press enter\n')
    PFNfilename = PFNfilename+'.csv'
    # PFNfilename = "ANN-PFNinputs_n10rand_lesscontrol_varG-streamlined.csv"
    PFNfile = open(PFNfilename, 'a')

headingPredictor = load_model('model_CEM_all-angle-rotate.keras')

plt.style.use('default')


## MPC simulator setup.
state_names = ['x',  # x position [m]
               'y',  # y position [m]
               'z',  # z position (altitude) [m]
               'v_para',  # parallel velocity [m/s]
               'v_perp',  # perpendicular velocity [m/s]
               'phi', # heading [rad]
               'phi_dot',  # angular velocity [rad/s]
               'w',  # ambient wind speed [m/s]
               'zeta',  # ambient wind angle [rad]
               'm', # mass [kg]
               'I',  # inertia [kg*m^2]
               'C_para',  # parallel damping [N*s/m]
               'C_perp',  # perpendicular damping [N*s/m]
               'C_phi',  # rotational damping [N·m/rad/s]
               'km1',  # parallel motor calibration coefficient
               'km2',  # offset motor calibration coefficient
               'km3',  # perpendicular motor calibration coefficient
               'km4',  # rotational motor calibration coefficient
               ]

input_names = ['u_para',  # parallel thrust force [N]
               'u_perp',  # perpendicular thrust force [N]
               'u_phi'  # torque [N*m]
              ]

measurement_names = ['phi', 'psi', 'gamma', 'a', 'g', 'r']


## Heading Predictor setup.
predictor_input_names = [
    'groundspeed',
    'groundspeed_angle',
    'airspeed',
    'airspeed_angle',
    'thrust',
    'thrust_angle',
]

predictor_output_names = ['heading_angle_x', 'heading_angle_y']

time_window = 4

time_augmentation_kwargs = {
    "time_window": time_window,
    "input_names": predictor_input_names,
    "output_names": predictor_output_names,
    "direction": "backward"
}

## Trajectory setup.
# Parameters in SI units
m = 0.25e-6  # [kg]
I = 5.2e-13  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.02369
# I = 4.971e-12  # [N*m*s^2] yaw mass moment of inertia: 10.1242/jeb.038778
C_phi = 27.36e-12  # [N*m*s] yaw damping: 10.1242/jeb.038778
C_para = m / 0.170  # [N*s/m] calculate using the mass and time constant reported in 10.1242/jeb.098665
C_perp = C_para  # assume same as C_para

# Scale Parameters
m = m * 1e6  # [mg]
I = I * 1e6 * (1e3) ** 2  # [mg*mm/s^2 * mm*s^2]
C_phi = C_phi * 1e6 * (1e3) ** 2  # [mg*mm/s^2 *m*s]
C_para = C_para * 1e6  # [mg/s]
C_perp = C_perp * 1e6  # [mg/s]

# Set time
fs = 500   # 500 for Mayetal2025
dt = 1 / fs
T = 0.1
tsim = np.arange(0.0, T + 0.9*dt, dt)

# make up a trajectory
v_para_0 = [0.125,0.25,1.0]
v_para_0 = [0.25]
v_para_dot_0 = 2.5/int(np.floor(len(tsim)/2))/dt

v_perp_0 = 0.0
v_perp_dot_0 = 0.0

deltaPhi = np.linspace(-np.pi,np.pi,num=26)
deltaPhia = np.array([deltaPhi[i*2] for i in range(7)])
deltaPhib = np.array([deltaPhi[i*2 + 13] for i in range(7)])

deltaPhi = np.append(deltaPhia,deltaPhib)
phi_0 = np.linspace(-np.pi,np.pi,num=19)
phi_0[0]=-np.pi+0.02
phi_0[9]=0.02
phi_0[-1]=np.pi-0.02
a=phi_0[7]
b=phi_0[11]
phi_0 = [phi_0[i*3] for i in range(6)]
addition = [a,b]                            
for i in addition: phi_0.append(i)
phi_0.sort()

#w_0 = [0.1,0.5,2.0]
w_0 = [0]

zeta_0 = np.linspace(-np.pi+0.04,np.pi+0.04,10)
zeta_0 = zeta_0[:9]
zeta_0 = np.hstack((zeta_0[:4],zeta_0[5:]))

elev = [0.3,1.0,2.0]

n_traj = len(elev)*len(deltaPhi)*len(zeta_0)*len(w_0)*len(phi_0)*len(v_para_0)
count = 0
print("Number of trajectories:",n_traj)

# for v in v_perp_0:                          # 3
for v in v_para_0:
    # v_perp = v_perp_0*np.ones_like(tsim) + v_perp_dot_0*tsim
    
    m = 0
    for el in elev:                         # 3
        # alt = el*np.ones_like(tsim)
        
        p = 0
        for wsp in w_0:                     # 3
            # w = wsp*np.ones_like(tsim)
        
            k = 0    
            for z in zeta_0:                # 10 or 8
                # zeta = z*np.ones_like(tsim)
            
                i = 0
                for d in deltaPhi:          # 26 or 14
                # for vdot in v_para_dot_0:
                    l = 0
                    for s in phi_0:         # 19 or 7 or 2
                    # for vdot in v_para_dot_0:
                        # vpara_start = v*np.ones_like(tsim) #+ v_para_dot_0*tsim
                        # accel = v_para_dot_0*np.square(tsim[0:ceil(len(tsim)/2)]) #v_para_dot_0*np.square(...)
                        # v_para = vpara_start - np.append(accel,np.flip(accel[:-1]))
                        # # v_para = v_para_0*np.ones_like(tsim)                          # no decel
                        
                        # phi_start = s*np.ones_like(tsim)
                        # angaccel = (d)/(1+np.exp(-100*(tsim-0.052)))
                        # phi = phi_start + angaccel                
                        # # phi = s*np.ones_like(tsim) + (10*d)*tsim                      # constant turn
                        # # phi = s*np.ones_like(tsim)                                      # no turn
                        
                        vpara_start = v_para_0[0]*np.ones_like(tsim)
                        accel = v_para_dot_0*np.square(tsim[0:ceil(len(tsim)/2)])
                        v_para = vpara_start - np.append(accel,np.flip(accel[:-1]))
                        v_perp = v_perp_0*np.ones_like(tsim) + v_perp_dot_0*tsim
                        phi_start = phi_0[0]*np.ones_like(tsim)
                        angaccel = (deltaPhi[0])/(1+np.exp(-100*(tsim-0.052)))
                        phi = phi_start + angaccel
                        w = w_0[0]*np.ones_like(tsim)
                        zeta = zeta_0[0]*np.ones_like(tsim)
                        alt = elev[0]*np.ones_like(tsim)
                        
                        setpoint = {'x': 0.0 * np.ones_like(tsim),
                                    'y': 0.0 * np.ones_like(tsim),
                                    'z': alt,
                                    'v_para': v_para,
                                    'v_perp': v_perp,
                                    'phi': phi,
                                    'phi_dot': 0.0*np.ones_like(tsim),
                                    'w': 0.0*np.ones_like(tsim),                    # no wind!
                                    'zeta': zeta,
                                    'm': m * np.ones_like(tsim),
                                    'I': I * np.ones_like(tsim),
                                    'C_para': C_para * np.ones_like(tsim),
                                    'C_perp': C_perp * np.ones_like(tsim),
                                    'C_phi': C_phi * np.ones_like(tsim),
                                    'km1': 1.0 * np.ones_like(tsim),
                                    'km2': 0.0 * np.ones_like(tsim),
                                    'km3': 1.0 * np.ones_like(tsim),
                                    'km4': 1.0 * np.ones_like(tsim),
                                   }
                        
                        
                        simulator = Simulator(f, h, dt=dt, state_names=state_names, input_names=input_names, measurement_names=measurement_names)
                        # Update the simulator set-point
                        simulator.update_dict(setpoint, name='setpoint')
                        # Define cost function: penalize the squared error between parallel & perpendicular velocity and heading
                        cost = ((simulator.model.x['v_para'] - simulator.model.tvp['v_para_set']) ** 2 +
                                (simulator.model.x['v_perp'] - simulator.model.tvp['v_perp_set']) ** 2 +
                                (simulator.model.x['phi'] - simulator.model.tvp['phi_set']) ** 2)
                        
                        # Set cost function
                        simulator.mpc.set_objective(mterm=cost, lterm=cost)
                        # Set input penalty: make this small for accurate state following
                        simulator.mpc.set_rterm(u_para=1e-6, u_perp=1e-6, u_phi=1e-6)
                        # Run model predictive control with no wind (check that 'w' setpoint is zeros!) to get control inputs ('u_sim')
                        t_sim1, x_sim1, u_sim, y_sim1 = simulator.simulate(x0=None, mpc=True, return_full_output=True)
                        
                        # Now run simulator without MPC, and add wind back in:
                        t_sim, x_sim, u_sim2, y_sim = simulator.simulate(x0={'w':w,'zeta':zeta}, mpc=False, u=u_sim, return_full_output=True)
                        # Preprocess and feed necessary variables into heading predictor.
                        trajec = pd.DataFrame()
                        trajec['groundspeed'] = y_sim['g']
                        trajec['groundspeed_angle'] = y_sim['psi']+y_sim['phi']
                        trajec['thrust'] = np.sqrt(u_sim['u_para']**2 + u_sim['u_perp']**2)
                        trajec['thrust_angle'] = u_sim['u_phi']
                        trajec['airspeed'] = y_sim['a']
                        trajec['airspeed_angle'] = y_sim['gamma']+y_sim['phi']
                        
                        
                        #####~~~~NEED TO DOWNSAMPLE~~~~#####
                        
                        heading_angle_predicted = predict_heading_from_fly_trajectory(trajec, 
                                                                                      24,
                                                                                      augment_with_time_delay_embedding, 
                                                                                      headingPredictor, smooth=True,
                                                                                      **time_augmentation_kwargs)                    
                        
                        # if np.mod(count,5)==0:
                        #     fig, ax = plt.subplots();\
                        #     ax.plot(sim_data.v_para);\
                        #     ax.plot(v_para);
                        
                        
                        stimulus['time'] = tsim
                        stimulus['obj_id'] = count
                        stimulus['heading'] = np.round(heading_angle_predicted,6)
                        stimulus['course_dir'] = np.round(y_sim['phi'],6)
                        stimulus['airspeed'] = np.round(y_sim['a'],6)
                        stimulus['gamma'] = np.round(y_sim['gamma'],6)
                        stimulus['gspd'] = np.round(y_sim['r'],6)
                        stimulus['psi'] = np.round(y_sim['psi'],6)
                        stimulus['zeta'] = np.round(x_sim['zeta'],6)
                        stimulus['wspd'] = np.round(x_sim['w'],6)
                        stimulus['altitude'] = np.round(x_sim['z'],6)
                        stimulus['fspd'] = np.round(y_sim['g'],6)
                        
                        # To run Nehal's code on the output you also need these:
                        stimulus['xpos'] = np.round(x_sim['x'],6)
                        stimulus['ypos'] = np.round(x_sim['y'],6)
                        # stimulus['xvel'] = np.round(sim_data['xvel'],6)
                        # stimulus['yvel'] = np.round(sim_data['yvel'],6)
                        # stimulus['gamma_x'] = np.round(sim_data['gamma_x'],6)
                        # stimulus['gamma_y'] = np.round(sim_data['gamma_y'],6)
                        # stimulus['phi_x'] = np.round(sim_data['phi_x'],6)
                        # stimulus['phi_y'] = np.round(sim_data['phi_y'],6)
                        # stimulus['psi_x'] = np.round(sim_data['psi_x'],6)
                        # stimulus['psi_y'] = np.round(sim_data['psi_y'],6)
                    
                        
                        
                        if save and count == 0:
                            stimulus.to_csv(stimfile)
                        elif save and count!=0:
                            stimulus.to_csv(stimfile,header=False)
                        
                        print(p,k,i,"initphi",np.round(s,2),
                              "zeta",np.round(z,3),
                              "wspd",wsp,"elev",el,"g",v)                        
                        
                        
                        ## Model PFN activity during trajectory
                        pfnmodel = PFN()
                        
                        p_d = pfnmodel.model_param['PFNd']
                        p_v = pfnmodel.model_param['PFNv']
                        p_pc = pfnmodel.model_param['PFNpc']
                        p_a = pfnmodel.model_param['PFNa']
                        AF = [-stimulus['gamma'].copy().iloc[0], 100*stimulus['airspeed'].copy().iloc[0]]
                        OF = [-stimulus['psi'].copy().iloc[0], 100*stimulus['gspd'].copy().iloc[0]]   # "gspd" is actually of speed
                        bump = [-stimulus['heading'].copy().iloc[0], -stimulus['gamma'].copy().iloc[0], 100*stimulus['airspeed'].copy().iloc[0]]
                        initcond = BuildSSInitial(p_d, p_v, p_pc, p_a, AF, OF, bump)
                        
                        pfnmodel.run(tsim=np.array(stimulus['time']),
                                     phi=-np.array(stimulus['heading']),
                                     a=100*np.array(stimulus['airspeed']),
                                     gamma=-np.array(stimulus['gamma']),
                                     g=100*np.array(stimulus['gspd']),
                                     psi=-np.array(stimulus['psi']),
                                     initcond=initcond)
                        
                        inputs = {}
                            
                        neurons = list(pfnmodel.heatmap.keys())
                        labels = {'EPG':['EPG_c1','EPG_c2','EPG_c3','EPG_c4','EPG_c5','EPG_c6','EPG_c7','EPG_c8'],
                                  'PFNdL_pb':['PFNd_c1','PFNd_c2','PFNd_c3','PFNd_c4','PFNd_c5','PFNd_c6','PFNd_c7','PFNd_c8'],
                                  'PFNdR_pb':['PFNd_c9','PFNd_c10','PFNd_c11','PFNd_c12','PFNd_c13','PFNd_c14','PFNd_c15','PFNd_c16'],
                                  'PFNvL_pb':['PFNv_c1','PFNv_c2','PFNv_c3','PFNv_c4','PFNv_c5','PFNv_c6','PFNv_c7','PFNv_c8'],
                                  'PFNvR_pb':['PFNv_c9','PFNv_c10','PFNv_c11','PFNv_c12','PFNv_c13','PFNv_c14','PFNv_c15','PFNv_c16'],
                                  'PFNpcL_pb':['PFNpc_c1','PFNpc_c2','PFNpc_c3','PFNpc_c4','PFNpc_c5','PFNpc_c6','PFNpc_c7','PFNpc_c8'],
                                  'PFNpcR_pb':['PFNpc_c9','PFNpc_c10','PFNpc_c11','PFNpc_c12','PFNpc_c13','PFNpc_c14','PFNpc_c15','PFNpc_c16'],
                                  'PFNaL_pb':['PFNa_c1','PFNa_c2','PFNa_c3','PFNa_c4','PFNa_c5','PFNa_c6','PFNa_c7','PFNa_c8'],
                                  'PFNaR_pb':['PFNa_c9','PFNa_c10','PFNa_c11','PFNa_c12','PFNa_c13','PFNa_c14','PFNa_c15','PFNa_c16']   
                                 }
                        for celltype in neurons:
                            j = 0
                            for label in labels[celltype]:
                                inputs[label] = pfnmodel.heatmap[celltype][:,j]
                                j+=1
                        
                        angle_map = np.arange(-np.pi,np.pi,step=np.pi/4)
                        n_angles = angle_map.shape[0]
                        n_sim = stimulus['time'].shape[0]
                        
                        inputs['wind_c1'] = np.empty((n_sim,))
                        inputs['wind_c2'] = np.empty((n_sim,))
                        inputs['wind_c3'] = np.empty((n_sim,))
                        inputs['wind_c4'] = np.empty((n_sim,))
                        inputs['wind_c5'] = np.empty((n_sim,))
                        inputs['wind_c6'] = np.empty((n_sim,))
                        inputs['wind_c7'] = np.empty((n_sim,))
                        inputs['wind_c8'] = np.empty((n_sim,))
                        
                        for pt in range(n_sim):
                            wind = 0.5 + 0.5 * np.cos(angle_map - stimulus['zeta'].iloc[pt] - np.pi)
                            inputs['wind_c1'][pt]=wind[0]
                            inputs['wind_c2'][pt]=wind[1]
                            inputs['wind_c3'][pt]=wind[2]
                            inputs['wind_c4'][pt]=wind[3]
                            inputs['wind_c5'][pt]=wind[4]
                            inputs['wind_c6'][pt]=wind[5]
                            inputs['wind_c7'][pt]=wind[6]
                            inputs['wind_c8'][pt]=wind[7]
                        
                        inputs['obj_id'] = count
                        
                        print(str(count+1)+' out of '+str(n_traj))
                        
                        PFNdf = pd.DataFrame(inputs)
                        if save and i == 0:
                            PFNdf.to_csv(PFNfile)
                        elif save and i!=0:
                            PFNdf.to_csv(PFNfile,header=False)
                        
                        count+=1
                        l+=1
                        
                    i+=1
                k+=1
            p+=1
        m+=1
            
if save:
    stimfile.close()
    PFNfile.close()