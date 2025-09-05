import numpy as np
import matplotlib.pyplot as plt
#import figurefirst as fifi
import utils

from utils import wrapToPi


class PFN:
    """ Model of PFN neural responses to air speed/direction & optic flow speed/direction.
        Will generate sinusoidal bumps for the compass/EB and in the PB and FB (for PFNs).
        Takes parameters previously determined from fitting for time-courses
        of PFN calcium intensity and bump position in the PB.
    """

    def __init__(self, PFNd_params=None, PFNv_params=None, PFNpc_params=None, PFNa_params=None):
        """ Initialize the model.

            PFNd_params:
            PFNv_params:
            PFNpc_params:
            PFNa_params:
        """

        # Set parameters
        if PFNd_params is None:
            PFNd_params = np.array([[0.5, 1.0, 1.4, 0.88, 0.4, 1.23, 0.0, 1.0],
                                    [0.4, 1.0, 1.4, 0.38, 1.0, 1.35, 0.0, 0.0]]) #0.4

        if PFNv_params is None:
            PFNv_params = np.array([[0.0, 1.0, -1.74, 0.88, 0.4, 1.23, 0.0, 1.0], # v params [0,0] = 0.5 fitted
                                    [0.4, 1.0, -1.74, 0.38, 1.0, 1.35, 0.0, 0.0]]) #0.4

        #if PFNpc_params is None:
        #    PFNpc_params = np.array([[1.24, 0.01, 0.97, 2.6, 0.0, 1.77, -0.3, 1.0],
        #                             [0.01, 1.00, 2.10, 15.0, 0.6, 2.00, 0.0, 1.0]]) # pc params [1,0] = 0.01 fitted
        
        if PFNpc_params is None:
            PFNpc_params = np.array([[1.16, 0.01, 0.97, 2.6, 0.0, 1.77, -0.3, 1.0],
                                     [1.0, 0.01, 0.2, 2.6, 0.0, 6, -0.3, 1.0],
                                     [0.01, 1.0, 2.1, 15.0, 0.6, 2.0, 0.0, 1.0],   # pc params [2,0] = 0.01 fitted
                                     [0.0, 1.0, 2.1, 15.0, 0.6, 2.0, 0.0, 0.0]]) 

        if PFNa_params is None:
            PFNa_params = np.array([[1.34, 0.57, 1.12, 0.48, 0.22, -0.24, 1.57, 1.0],
                                    [0.2, -0.3, 1.12, 0.48, 1.0, -0.24, 1.57, 0.0]]) # a params [1,0] = 0.2 fitted

        # Store model parameters for each neuron type
        self.model_param = {'PFNd': PFNd_params,
                            'PFNv': PFNv_params,
                            'PFNpc': PFNpc_params,
                            'PFNa': PFNa_params}

        # Set inputs
        self.tsim = np.array(0.0)
        self.phi = np.array(0.0)
        self.gamma = np.array(0.0)
        self.a = np.array(0.0)
        self.psi = np.array(0.0)
        self.g = np.array(0.0)
        self.initcond = np.zeros((9,2))

        # Outputs
        self.res = {}
        self.heatmap = {}
        self.allpts = {}
        self.new_inits = np.zeros((9,2))

    def run(self, tsim=None, phi=None, a=None, gamma=None, g=None, psi=None, initcond=None, unwrap=False):
        """ Set inputs & run.

        tsim: time [s]
        phi: heading angle [rad]
        a: air speed [m/s]
        gamma: air speed direction [rad]
        g: ground speed [m/s]
        psi: ground speed  direction [rad]

        """

        # Set inputs
        self.tsim = tsim
        self.phi = phi #np.zeros_like(phi)
        self.gamma = gamma
        self.a = a
        self.psi = psi
        self.g = g

        # Run model
        self.stims = {'t': self.tsim,
                      'heading': self.phi,
                      'Atheta': self.gamma,
                      'Amag': self.a,
                      'Otheta': self.psi,
                      'Omag': self.g,
                      }

        # Set initial conditions
        if initcond is not None:
            # print('not None')
            self.initcond = initcond
        else:
            print('No initials')

        # Run model
        self.res, self.heatmap, self.new_inits = FBmodel_obs(self.stims,
                                                         self.model_param['PFNd'],
                                                         self.model_param['PFNv'],
                                                         self.model_param['PFNpc'],
                                                         self.model_param['PFNa'],
                                                         self.initcond
                                                        )

        # Unwrap bump position
        if unwrap:
            wrap_data = ['PFNd_bumpFB', 'PFNv_bumpFB', 'PFNpc_bumpFB', 'PFNa_bumpFB']
            for w in wrap_data:
                self.res[w] = np.unwrap(self.res[w])

    def runstep(self, tsim=None, phi=None, a=None, gamma=None, g=None, psi=None, unwrap=False, start=None, win=None, initcond=None):
        """ Set inputs & run FBmodel_obs for a portion of the model.

        tsim: time [s]
        phi: heading angle [rad]
        a: air speed [m/s]
        gamma: air speed direction [rad]
        g: ground speed [m/s]
        psi: ground speed  direction [rad]
        start_fin: start and end frames for window

        """

        # Set inputs
        self.tsim = tsim
        self.phi = phi
        self.gamma = gamma
        self.a = a
        self.psi = psi
        self.g = g
        self.start = start
        self.win = win
        self.initcond = initcond

        # Run model
        self.stims = {'t': self.tsim[self.start:self.start+self.win],
                      'heading': self.phi[self.start:self.start+self.win],
                      'Atheta': self.gamma[self.start:self.start+self.win],
                      'Amag': self.a[self.start:self.start+self.win],
                      'Otheta': self.psi[self.start:self.start+self.win],
                      'Omag': self.g[self.start:self.start+self.win],
                     }

        self.res,self.new_inits = FBmodel_obs(self.stims,
                                              self.model_param['PFNd'],
                                              self.model_param['PFNv'],
                                              self.model_param['PFNpc'],
                                              self.model_param['PFNa'],
                                              #self.initcond
                                             )

        # Unwrap bump position
        if unwrap:
            wrap_data = ['PFNd_bumpFB', 'PFNv_bumpFB', 'PFNpc_bumpFB', 'PFNa_bumpFB']
            for w in wrap_data:
                self.res[w] = np.unwrap(self.res[w])
                
                
    def plot_stimulus(self, fig_size=1):
        stim_keys = list(self.stims.keys())
        n_k = len(stim_keys)

        fig, ax = plt.subplots(n_k-1,1, figsize=( fig_size * 2, fig_size * 1.5 * n_k), sharex=True, dpi=100)
        for n, k in enumerate(stim_keys[1:]):
            ax[n].plot(self.stims['t'], self.stims[k])
            ax[n].set_ylabel(k)
            ax[n].set_yticks((min(self.stims[k])-5,max(self.stims[k]+5)))
        ax[0].set_ylim(-np.pi,np.pi)
        ax[1].set_ylim(-np.pi,np.pi)
        ax[3].set_ylim(-np.pi,np.pi)

    def plot_bump(self, fig_size=1):
        """ Plot bump position & amplitude over time for each neuron type.
        """

        keys = list(self.res.keys())
        n_key = len(keys)

        n_col = 4
        n_row = np.ceil(n_key / n_col).astype(int)

        plot_order = np.hstack((np.arange(1, n_key, step=1), np.array(0)))

        fig, ax = plt.subplots(n_row, n_col, figsize=(fig_size * 3 * n_col, fig_size * 2 * n_row), sharex=True)
        for n, k in enumerate(plot_order):
            ax.flat[n].plot(self.stims['t'], self.res[keys[k]].T)
            ax.flat[n].set_title(keys[k])

        for a in ax.flat:
            a.set_xticks(np.arange(0, self.stims['t'][-1], step=10))
            
        for y in [4,5,6,7,8,9,10,11,24,25,26,27,28]:
            ax.flat[y].set_yticks(np.arange(-4, 4.1, step=1))
        for y2 in [0,1,2,3,12,13,14,15,16,17,18,19,20,21,22,23]:
            ax.flat[y2].set_yticks(np.arange(-0.5, 3.1, step = 1))

        fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.6, hspace=0.4)

    def plot_heatmap(self, fig_size=7, cmap='jet'):
        """ Plot bump position & amplitude as a heatmap over time for each neuron type.
        """

        fig, ax = plt.subplots(1, 9, figsize=(fig_size, 1.2 * fig_size), sharex=True, sharey=True, dpi=100)

        pos = ax[0].imshow(self.heatmap['EPG'], vmin=0, vmax=1.2, cmap=cmap)
        ax[1].imshow(self.heatmap['PFNdL_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[2].imshow(self.heatmap['PFNdR_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[3].imshow(self.heatmap['PFNvL_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[4].imshow(self.heatmap['PFNvR_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[5].imshow(self.heatmap['PFNpcL_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[6].imshow(self.heatmap['PFNpcR_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[7].imshow(self.heatmap['PFNaL_pb'], vmin=0, vmax=1.2, cmap=cmap)
        ax[8].imshow(self.heatmap['PFNaR_pb'], vmin=0, vmax=1.2, cmap=cmap)
        
        ax[0].set_title('EPG')
        fig.colorbar(pos,ax=ax[0])
        ax[1].set_title('PFNd PB')
        ax[3].set_title('PFNv PB')
        ax[5].set_title('PFNpc PB')
        ax[7].set_title('PFNa PB')


        for a in ax.flat:
            a.set_aspect(0.5)
            a.set_xlim(0, 16)
            #fifi.mpl_functions.adjust_spines(a, [])
        # a.set_tick_params(left=None, right=None, top=None, bottom=None)


def FBmodel_obs(gust_res, p_PFNd, p_PFNv, p_PFNpc, p_PFNa, initcond=None):
    # like FBmodel_observ() in Matlab
    # initcond is a dict with the following keys:
    # dL, dR, vL, vR, pcL, pcR, aL, aR
    # each key references a 1x2 (tuple? array?) with col=0 AF and col=1 OF initial values

    p_bump = 2

    heading = gust_res['heading']
    thva = gust_res['Atheta']
    spda = gust_res['Amag']
    thvo = gust_res['Otheta']
    spdo = gust_res['Omag']
    t = gust_res['t']

    if initcond is None:
        print('it is None')
        AFinput = np.array([thva[0],spda[0]])
        OFinput = np.array([thvo[0],spdo[0]])
        bumpinput = np.array([heading[0],thva[0],spda[0]])
        initcond = BuildSSInitial(p_PFNd, p_PFNv,
                                  p_PFNpc, p_PFNa,
                                  AFinput,
                                  OFinput,
                                  bumpinput
                                 )

    #print(initcond)
    Rinputs = np.vstack((thva, spda, thvo, spdo, t))
    Linputs = np.vstack((-thva, spda, -thvo, spdo, t))
    
    # If you want to make PFNv unresponsive to airflow, run the following line:
    #p_PFNv[0,0] = 0
    
    new = np.zeros((18, len(t)))

    PFNd_amp = np.zeros((2, len(t)))
    PFNd_amp[0, :],new[0,:],new[1,:] = PFNd_integ(p_PFNd, Linputs,initcond[0,:])
    PFNd_amp[1, :],new[2,:],new[3,:] = PFNd_integ(p_PFNd, Rinputs,initcond[1,:])

    PFNv_amp = np.zeros((2, len(t)))
    PFNv_amp[0, :],new[4,:],new[5,:] = PFNd_integ(p_PFNv, Linputs,initcond[2,:])
    PFNv_amp[1, :],new[6,:],new[7,:] = PFNd_integ(p_PFNv, Rinputs,initcond[3,:])

    PFNpc_amp = np.zeros((2, len(t)))
    PFNpc_amp[0, :],new[8,:],new[9,:] = PFNpc_integ2(p_PFNpc, Linputs,initcond[4,:])
    PFNpc_amp[1, :],new[10,:],new[11,:] = PFNpc_integ2(p_PFNpc, Rinputs,initcond[5,:])

    PFNa_amp = np.zeros((2, len(t)))
    PFNa_amp[0, :],new[12,:],new[13,:] = PFNa_integ(p_PFNa, Linputs,initcond[6,:])
    PFNa_amp[1, :],new[14,:],new[15,:] = PFNa_integ(p_PFNa, Rinputs,initcond[7,:])

    # Bump movement code.
    # This will allow us to test the information in the system if bump movement
    # is driven by heading, airflow direction, both, or neither.
    ha = np.zeros_like(spda)
    for x in range(len(ha)):
        if spda[x]>0: ha[x]=1
    
    # If airflow drives bump movement, use bumpmdl_de(); if not, use np.zeros().
    # bumppos = bumpmdl_de(p_bump, thva, t, ha,initcond[8,0]) # comment out for noAF
    bumppos = np.zeros((len(t),))                             # comment in for noAF
    
    #print(bumppos)
    new[16,:] = bumppos.copy()
    new[17,:] = bumppos.copy()
    
    ## If heading drives bump movement: (otherwise, comment out)
    bumppos = bumppos - heading             # bumppos in radians
    
    PFNd_bump = np.zeros((2, len(t)))
#    PFNd_bump[0, :] = wrapToPi(bumppos + np.pi/4)
#    PFNd_bump[1, :] = wrapToPi(bumppos - np.pi/4)
    
    PFNd_bump[0, :] = wrapToPi(bumppos)
    PFNd_bump[1, :] = wrapToPi(bumppos)

    PFNv_bump = np.zeros((2, len(t)))
#    PFNv_bump[0, :] = wrapToPi(bumppos + np.pi/4)
#    PFNv_bump[1, :] = wrapToPi(bumppos - np.pi/4)
    
    PFNv_bump[0, :] = wrapToPi(bumppos)
    PFNv_bump[1, :] = wrapToPi(bumppos)    

    PFNpc_bump = np.zeros((2, len(t)))
    # If PFNpc bump moves like other PFN bumps, use:
    #bumppos_pc = bumppos.copy()
    # If PFNpc bump doesn't move with heading, use one of the next two lines:
    #bumppos_pc = new[16,:]
    #bumppos_pc = bumpmdl_de(p_bump,thva,t,ha,initcond[8,0])
    # If PFNpc bump doesn't move with airflow, use:
    #bumppos_pc = -heading
    # If PFNpc bump doesn't move with airflow or heading, use:
    bumppos_pc = np.zeros_like(bumppos)
#    PFNpc_bump[0, :] = wrapToPi(bumppos_pc + np.pi/4)
#    PFNpc_bump[1, :] = wrapToPi(bumppos_pc - np.pi/4)
    
    PFNpc_bump[0, :] = wrapToPi(bumppos_pc)
    PFNpc_bump[1, :] = wrapToPi(bumppos_pc)    

    PFNa_bump = np.zeros((2, len(t)))
#    PFNa_bump[0, :] = bumppos + np.pi/4
#    PFNa_bump[1, :] = bumppos - np.pi/4
    
    PFNa_bump[0, :] = bumppos
    PFNa_bump[1, :] = bumppos
    
    for i,v in enumerate(wrapToPi(thva)):
        if spda[i]>0:
            if v>0.78 or v<-2.35:
                PFNa_bump[0, i] = PFNa_bump[0, i]+np.pi
            elif v<-0.78 or v>2.35:
                PFNa_bump[1, i] = PFNa_bump[1, i]+np.pi
    PFNa_bump[0, :] = wrapToPi(PFNa_bump[0, :])
    PFNa_bump[1, :] = wrapToPi(PFNa_bump[1, :])

    res = {'bump': wrapToPi(bumppos),
           'PFNd_amp': PFNd_amp,
           'PFNv_amp': PFNv_amp,
           'PFNpc_amp': PFNpc_amp,
           'PFNa_amp': PFNa_amp,
           'PFNd_bumpL': PFNd_bump[0,:],
           'PFNd_bumpR': PFNd_bump[1,:],
           'PFNv_bumpL': PFNv_bump[0,:],
           'PFNv_bumpR': PFNv_bump[1,:],
           'PFNpc_bumpL': PFNpc_bump[0,:],
           'PFNpc_bumpR': PFNpc_bump[1,:],
           'PFNa_bumpL': PFNa_bump[0,:],
           'PFNa_bumpR': PFNa_bump[1,:],
           'PFNd_ampL':  PFNd_amp[0, :],
           'PFNd_ampR': PFNd_amp[1, :],
           'PFNv_ampL': PFNv_amp[0, :],
           'PFNv_ampR': PFNv_amp[1, :],
           'PFNpc_ampL': PFNpc_amp[0, :],
           'PFNpc_ampR': PFNpc_amp[1, :],
           'PFNa_ampL': PFNa_amp[0, :],
           'PFNa_ampR': PFNa_amp[1, :],
           }

    PFNd_bumpFB, PFNd_ampFB = sumPFNvecs(PFNd_bump, PFNd_amp)
    PFNv_bumpFB, PFNv_ampFB = sumPFNvecs(PFNv_bump, PFNv_amp)
    PFNpc_bumpFB, PFNpc_ampFB = sumPFNvecs(PFNpc_bump, PFNpc_amp)
    PFNa_bumpFB, PFNa_ampFB = sumPFNvecs(PFNa_bump, PFNa_amp)
    
    res['PFNd_ampFB'] = PFNd_ampFB
    res['PFNv_ampFB'] = PFNv_ampFB
    res['PFNpc_ampFB'] = PFNpc_ampFB
    res['PFNa_ampFB'] = PFNa_ampFB
    res['PFNd_bumpFB'] = PFNd_bumpFB
    res['PFNv_bumpFB'] = PFNv_bumpFB
    res['PFNpc_bumpFB'] = PFNpc_bumpFB
    res['PFNa_bumpFB'] = PFNa_bumpFB
    
    new = new[:,-1]
    new_inits = new.reshape((9,2))
    
    # Heatmaps
    #angle_map = np.arange(-135, 181, step=45)
    angle_map = np.arange(-np.pi,np.pi,step=np.pi/4)
    n_angles = angle_map.shape[0]
    n_sim = t.shape[0]

    EPG = np.zeros((n_angles, n_sim))
    PFNdL_pb = np.zeros((n_angles, n_sim))
    PFNdR_pb = np.zeros((n_angles, n_sim))
    PFNvL_pb = np.zeros((n_angles, n_sim))
    PFNvR_pb = np.zeros((n_angles, n_sim))
    PFNpcL_pb = np.zeros((n_angles, n_sim))
    PFNpcR_pb = np.zeros((n_angles, n_sim))
    PFNaL_pb = np.zeros((n_angles, n_sim))
    PFNaR_pb = np.zeros((n_angles, n_sim))

    for i in range(len(t)):
        EPG[:, i] = 0.5 + 0.5 * np.cos(angle_map - bumppos[i])
        PFNdL_pb[:, i] = (PFNd_amp[0, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNd_bumpL'][i]))
        PFNdR_pb[:, i] = (PFNd_amp[1, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNd_bumpR'][i]))
        PFNvL_pb[:, i] = (PFNv_amp[0, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNv_bumpL'][i]))
        PFNvR_pb[:, i] = (PFNv_amp[1, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNv_bumpR'][i]))
        PFNpcL_pb[:, i] = (PFNpc_amp[0, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNpc_bumpL'][i]))
        PFNpcR_pb[:, i] = (PFNpc_amp[1, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNpc_bumpR'][i]))
        PFNaL_pb[:, i] = (PFNa_amp[0, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNa_bumpL'][i]))
        PFNaR_pb[:, i] = (PFNa_amp[1, i] + 0.5) * (0.5 + 0.5 * np.cos(angle_map - res['PFNa_bumpR'][i]))
    
    heatmap = {}
    heatmap['EPG'] = np.vstack((EPG,EPG[0, :])).T

    heatmap['PFNdL_pb'] = np.vstack((PFNdL_pb,PFNdL_pb[0, :])).T
    heatmap['PFNdR_pb'] = np.vstack((PFNdR_pb,PFNdR_pb[0, :])).T
    heatmap['PFNvL_pb'] = np.vstack((PFNvL_pb,PFNvL_pb[0, :])).T
    heatmap['PFNvR_pb'] = np.vstack((PFNvR_pb,PFNvR_pb[0, :])).T
    heatmap['PFNpcL_pb'] = np.vstack((PFNpcL_pb,PFNpcL_pb[0, :])).T
    heatmap['PFNpcR_pb'] = np.vstack((PFNpcR_pb,PFNpcR_pb[0, :])).T
    heatmap['PFNaL_pb'] = np.vstack((PFNaL_pb,PFNaL_pb[0, :])).T
    heatmap['PFNaR_pb'] = np.vstack((PFNaR_pb,PFNaR_pb[0, :])).T
    
    return res, heatmap, new_inits


def circshift(arr, shift): # don't need this, just np.roll?
    """
    Circularly shift the elements of the array by the specified shift amount.

    Parameters:
        arr (array-like): The input array.
        shift (int): The number of positions by which elements are shifted.

    Returns:
        np.ndarray: The circularly shifted array.
    """
    n = len(arr)
    shift = shift % n  # Ensure shift is within the range [0, n)
    return np.concatenate((arr[-shift:], arr[:-shift]))


def bumpmdl_de(params, thetavec, t, h, initcond):
    # output radians
    tau = 2#np.atleast_2d(params)[0]
    res = np.zeros_like(t)
    #res[0] = thetavec[0]*h[0]
    res[0] = initcond

    for i in range(len(t) - 1):
        coeff = -(utils.wrapToPi(np.array(thetavec[i] * h[i])))
        res[i + 1] = res[i] + (coeff - res[i]) * ((t[i + 1] - t[i]) / tau)

    return res


def A_response_de(params, inputs, initcond):
    # This function generates a vector of intensity values over time for one
    # half of the PB innervated by PFNa, to a sequence of single-modality experience (either AF or OF).

    # Setup with arguments.
    # inputs are sensory info
    thetavec = inputs[0, :]  # radians, (+) is ipsi to the PB half
    speedvec = np.fmax(np.zeros_like(inputs[1,:]),inputs[1, :])  # cm/s, zero-floored
    t = inputs[2, :]  # seconds

    # Parameters
    a = params[0]  # like Amp
    c = params[1]  # coeff for second cosine term
    prefdir = params[2]  # in rads; parameter formerly known as theta0
    b = params[3]  # like offset
    r = params[4]  # offset to arrive at steady-state
    d = params[5]  # amplitude of steady-state driven by direction tuning
    tau = params[6]  # time constant
    flip = params[7]  # for rising OF response =0 vs. falling AF response !=0

    #% Set anchoring values for numerical solution: C is like max value,
    # T is full tau expression, res(1) is the initial result
    # (assumed to be the steady state if the first timestep inputs were held constant).
    C = a * (1 - np.exp(-speedvec)) * (np.cos(thetavec - prefdir) ** 2 + c * np.cos(thetavec - prefdir + np.pi) + b)
    T = tau
    ratio = r + d * np.cos(thetavec - prefdir)
    res = np.zeros_like(t)
    #res[0] = C[0]
    res[0] = initcond

    # Handle direction of response (rise/decay).
    if flip != 0:  # if AF response curve
        res[0] = C[0]-initcond
        ratio = 1 - ratio

    # Calculate numerical solution using the given inputs and parameters.
    for i in range(len(t) - 1):
        dt = (t[i + 1] - t[i])
        res[i + 1] = (res[i] + ((ratio[i]) * C[i] - res[i]) * (dt / (T)))

    r = res
    
    if flip != 0:  # if AF response curve
        # res = np.maximum(0, C - res)
        res = np.abs(C - res)
        r = C-r

    return res, r


def D_response_de(params, inputs, initcond):
    # This function generates a vector of intensity values over time for one
    # half of the PB innervated by PFNd, to a sequence of single-modality experience (either AF or OF).

    # Setup with arguments.
    # inputs are sensory info
    thetavec = inputs[0, :]  # radians, (+) is ipsi to the PB half
    speedvec = np.fmax(np.zeros_like(inputs[1,:]),inputs[1, :])  # cm/s, zero-floored
    t = inputs[2, :]  # seconds

    # Parameters
    a = params[0]  # like Amp
    c = params[1]  # speed coefficient for max, =1 for non-speed-tuned PFNs
    prefdir = params[2]  # in rads; parameter formerly known as theta0
    b = params[3]  # like offset
    ratio = abs(params[4])  # ratio of steady-state to max amp; =0 for steady-state=0;
    # ratio=1 for PFNd OF
    tau = params[5]  # time constant (offset)
    tauslope = params[6]  # speed coefficient for tau
    flip = params[7]  # for rising OF response =0 vs. falling AF response !=0

    # Set anchoring values for numerical solution: C is like max value, T is
    # full tau expression, res(1) is the initial result (assumed to be the
    # steady state if the first timestep inputs were held constant).
    C = a * (1 - np.exp(c * -speedvec)) * (np.cos(thetavec - prefdir) + b)
    T = tau + tauslope * np.exp(speedvec / 100)
    res = np.zeros_like(t)
    #res[0] = C[0]
    res[0] = initcond

    if flip != 0:  # if AF response curve
        res[0] = C[0]-initcond
        ratio = 1 - ratio

    # Calculate numerical solution using the given inputs and parameters.
    for i in range(len(t) - 1):
        dt = (t[i + 1] - t[i])
        res[i + 1] = (res[i] + ((ratio) * C[i] - res[i]) * (dt / (T[i])))

    r = res
    
    # Handle negative values for AF response curves.
    if flip != 0:  # if AF response curve
        res = np.maximum(0, C - res)
        # res = np.abs(C - res)
        r = C-r

    return res, r


def PC_response_de(params, inputs, initcond):
    # This function generates a vector of intensity values over time for one
    # half of the PB to a sequence of single-modality experience (either AF or
    # OF). It can handle PFNpc and PFNa offset responses if given the correct
    # parameters (see 'script_20240404.m')

    # 'params' is a 2x8 matrix where first row is for onset/stimulus responses 
    # and second row is for offset responses.
    
    # REFERENCE: on_off_de.m in MatLab

    # Setup with arguments.
    # inputs are sensory info
    thetavec = inputs[0, :]  # radians, (+) is ipsi to the PB half
    speedvec = np.fmax(np.zeros_like(inputs[1,:]),inputs[1, :])  # cm/s, zero-floored
    t = inputs[2, :]  # seconds
    
    # Get stimulus decrease periods ("offset") and change speedvec to produce the desired results
    dspd = np.diff(speedvec)
    offs = np.zeros_like(t)
    for i in range(len(speedvec)-1):
        if dspd[i] !=0:
            speedvec[i+1] = speedvec[i]+abs(dspd[i])
        else:
            speedvec[i+1] = speedvec[i]
            
        if dspd[i]<0:
            offs[i+1] = 1
        elif dspd[i]==0:
            offs[i+1] = offs[i]
        else:
            offs[i+1] = 0
    
    # Parameters
    # a is like Amp
    a = np.ones_like(t)*params[0,0]
    a[np.nonzero(offs)[0]] = params[1,0]
    # c is speed coefficient for max, =1 for non-speed-tuned PFNs
    c = np.ones_like(t)*params[0,1]
    c[np.nonzero(offs)[0]] = params[1,1]
    # prefdir is in rads; parameter formerly known as theta0
    prefdir = np.ones_like(t)*params[0,2]
    prefdir[np.nonzero(offs)[0]] = params[1,2]    
    # b is the tuning curve offset/shift term
    b = np.ones_like(t)*params[0,3]
    b[np.nonzero(offs)[0]] = params[1,3]
    # ratio is of steady-state to max amp; =0 for steady-state=0; ratio=1 for PFNd OF
    ratio = np.ones_like(t)*params[0,4]
    ratio[np.nonzero(offs)[0]] = params[1,4]
    # tau is time constant with no dependency
    tau = np.ones_like(t)*params[0,5]
    tau[np.nonzero(offs)[0]] = params[1,5]
    # tauslope is speed coefficient for tau
    tauslope = np.ones_like(t)*params[0,6]
    tauslope[np.nonzero(offs)[0]] = params[1,6]
    # flip is to indicate whether activity is rising OF response =0 vs. falling AF response !=0
    flip = np.ones_like(t)*params[0,7]
    flip[np.nonzero(offs)[0]] = params[1,7]

    # Set anchoring values for numerical solution: C is like max value, T is
    # full tau expression, res[0] is the initial result (assumed to be the
    # steady state if the first timestep inputs were held constant).
    C = np.multiply(a, np.multiply((1 - np.exp(np.multiply(c, -speedvec))), (np.cos(thetavec - prefdir) + b)))
    T = tau + np.multiply(tauslope, np.exp(speedvec / 100))
    res = np.zeros_like(t)

    res[0] = initcond

    # Handle direction of response (rise/decay).
    for i in range(len(t)):
        if flip[i] != 0:  # if AF response curve
            res[i] = C[i]-initcond
            ratio[i] = 1 - ratio[i]

    # Calculate numerical solution using the given inputs and parameters.
    for i in range(len(t) - 1):
        dt = (t[i + 1] - t[i])
        res[i + 1] = (res[i] + ((ratio[i]) * C[i] - res[i]) * (dt / (T[i])))

    r = res
    
    # Handle negative values for AF response curves.
    for i in range(len(t)):
        if flip[i] != 0:
            #res[i] = np.abs(C[i] - res[i])
            res[i] = C[i]-res[i]
            r[i] = r[i]

    return res, r


def sumPFNvecs(PFN_bumps, PFN_amps):
    # assumes PFN_bumps is already pi/4-rad shifted.
    adj = PFN_amps[0, :] * np.cos(PFN_bumps[0, :]) + PFN_amps[1, :] * np.cos(PFN_bumps[1, :])
    opp = PFN_amps[0, :] * np.sin(PFN_bumps[0, :]) + PFN_amps[1, :] * np.sin(PFN_bumps[1, :])
    va = np.arctan2(opp, adj)
    vm = np.sqrt(opp ** 2 + adj ** 2)

    return va, vm


def PFNd_integ(params, inputs, initcond):
    # params = array where first row is AF, second row is OF
    # inputs = array where:
    # [ AF directions;
    #   AF speeds;
    #   OF directions;
    #   OF speeds;
    #   time ]
    
    if len(initcond.shape)==1:
        AF, a = D_response_de(params[0, :], np.vstack((inputs[0:2, :], inputs[4, :])), initcond[0])
        OF, o = D_response_de(params[1, :], inputs[2:5, :], initcond[1])
    else:
        AF, a = D_response_de(params[0, :], np.vstack((inputs[0:2, :], inputs[4, :])), initcond[0,:])
        OF, o = D_response_de(params[1, :], inputs[2:5, :], initcond[1,:])

    res = AF + OF                     # comment out for noAF or noOF
    # res = OF                            # comment in for noAF
    # res = AF                          # comment in for noOF

    return res, a, o


def PFNa_integ(params, inputs, initcond):
    # params = array where first row is AF, second row is OF
    # inputs = array where:
    # [ AF directions;
    #   AF speeds;
    #   OF directions;
    #   OF speeds;
    #   time ]

    if len(initcond.shape)==1:
        AF, a = A_response_de(params[0, :], np.vstack((inputs[0:2, :], inputs[4, :])), initcond[0])
        OF, o = A_response_de(params[1, :], inputs[2:5, :], initcond[1])
    else:
        AF, a = A_response_de(params[0, :], np.vstack((inputs[0:2, :], inputs[4, :])), initcond[0,:])
        OF, o = A_response_de(params[1, :], inputs[2:5, :], initcond[1,:])

    # Instantiate result
    res = np.zeros(inputs.shape[1])

    # Where OF response exists, insert into result
    OFinds = np.where(OF)[0]            # comment out for noOF
    res[OFinds] = OF[OFinds]            # comment out for noOF

    # Where AF or AFOF response exists, insert into result (overwriting OF
    # except where OF is alone).
    AFinds = np.where(AF)[0]          # comment out for noAF
    res[AFinds] = AF[AFinds]          # comment out for noAF

    return res, a, o


def PFNpc_integ2(params, inputs, initcond):
    
    if len(initcond.shape)==1:
        AF, a = PC_response_de(params[0:2, :], np.vstack((inputs[0:2, :], inputs[4, :])), initcond[0])
        OF, o = PC_response_de(params[2:, :], inputs[2:5, :], initcond[1])
    else:
        AF, a = PC_response_de(params[0:2, :], np.vstack((inputs[0:2, :], inputs[4, :])), initcond[0,:])
        OF, o = PC_response_de(params[2:, :], inputs[2:5, :], initcond[1,:])
    
    res = np.zeros(inputs.shape[1])

    OF_inds = np.where(OF)[0]           # comment out for noOF
    res[OF_inds] = OF[OF_inds]          # comment out for noOF

    AF_inds = np.where(AF)[0]         # comment out for noAF
    res[AF_inds] = AF[AF_inds]        # comment out for noAF

    return res, a, o


def SteadyStateInitial_amp(params,inputs,isPFNa=False):
    # params is the parameter vector for one half of the PB to one stimulus modality, e.g. p_PFNd[0,:]
    # inputs is a vector of len=2, the starting direction (0) and speed (1) for the modality of interest
    # returns the steady-state value of the amplitude model for the given inputs
    if isPFNa:
        C = params[0] * (1 - np.exp(-inputs[1])) * (np.cos(inputs[0] - params[2]) ** 2 + params[1] * np.cos(inputs[0] - params[2] + np.pi) + params[3])
        ratio = params[4] + params[5] * np.cos(inputs[0] - params[2])
        ss = ratio * C
    else:
        C = params[0] * (1 - np.exp(params[1] * -inputs[1])) * (np.cos(inputs[0] - params[2]) + params[3])
        ss = params[4] * C
        
    return ss


def SteadyStateInitial_bump(inputs):
    ## This is a holdover from when bump position is sensitive to airflow direction
    ## inputs is a vector of len=3, for heading direction, airflow direction, and airspeed
    #h = -inputs[0]
    if inputs[2]!=0:
        ss = -inputs[1]
    else:
        ss = 0
        
    ss=0  # set to zero if no influence of airflow on bump position. (Heading added in later in the model)
    
    return ss


def BuildSSInitial(p_PFNd, p_PFNv, p_PFNpc, p_PFNa, input_AF, input_OF, bump_inputs):

    # if you don't want PFNv responses to airflow, run the following line:
    #p_PFNv[0,0] = 0
    #p_PFNpc[1,0] = 0 # no pc response to optic flow
    
    Raf = input_AF
    Laf = [-(input_AF[0]),input_AF[1]]
    Rof = input_OF
    Lof = [-(input_OF[0]),input_OF[1]]
    
    ssinit = np.zeros((9,2))
    ssinit[0,0] = SteadyStateInitial_amp(p_PFNd[0,:],Laf) # comment out for noAF
    ssinit[0,1] = SteadyStateInitial_amp(p_PFNd[1,:],Lof) # comment out for noOF
    ssinit[1,0] = SteadyStateInitial_amp(p_PFNd[0,:],Raf) # comment out for noAF
    ssinit[1,1] = SteadyStateInitial_amp(p_PFNd[1,:],Rof) # comment out for noOF
    ssinit[2,0] = SteadyStateInitial_amp(p_PFNv[0,:],Laf) # comment out for noAF
    ssinit[2,1] = SteadyStateInitial_amp(p_PFNv[1,:],Lof) # comment out for noOF
    ssinit[3,0] = SteadyStateInitial_amp(p_PFNv[0,:],Raf) # comment out for noAF
    ssinit[3,1] = SteadyStateInitial_amp(p_PFNv[1,:],Rof) # comment out for noOF
    ssinit[4,0] = SteadyStateInitial_amp(p_PFNpc[0,:],Laf) # comment out for noAF
    ssinit[4,1] = SteadyStateInitial_amp(p_PFNpc[1,:],Lof) # comment out for noOF
    ssinit[5,0] = SteadyStateInitial_amp(p_PFNpc[0,:],Raf) # comment out for noAF
    ssinit[5,1] = SteadyStateInitial_amp(p_PFNpc[1,:],Rof) # comment out for noOF
    ssinit[6,0] = SteadyStateInitial_amp(p_PFNa[0,:],Laf,True) # comment out for noAF
    ssinit[6,1] = SteadyStateInitial_amp(p_PFNa[1,:],Lof,True) # comment out for noOF
    ssinit[7,0] = SteadyStateInitial_amp(p_PFNa[0,:],Raf,True) # comment out for noAF
    ssinit[7,1] = SteadyStateInitial_amp(p_PFNa[1,:],Rof,True) # comment out for noOF
    ssinit[8,0] = SteadyStateInitial_bump(bump_inputs)
    ssinit[8,1] = SteadyStateInitial_bump(bump_inputs)
    
    return ssinit
