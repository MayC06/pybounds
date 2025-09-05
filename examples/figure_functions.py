import numpy as np
import scipy
import pandas as pd
from fractions import Fraction
import matplotlib.pyplot as plt
from matplotlib import cm, colors, gridspec
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figurefirst as fifi
import fly_plot_lib_plot as fpl
import utils

#from fisher import FisherObservability


class LatexStates:
    """Holds LaTex format corresponding to set symbolic variables.
    """
    def __init__(self):
        self.dict = {'v_para': r'$v_{\parallel}$',
                     'v_perp': r'$v_{\perp}$',
                     'phi': r'$\phi$',
                     'phidot': r'$\dot{\phi}$',
                     'w': r'$w$',
                     'zeta': r'$\zeta$',
                     'I': r'$I$',
                     'm': r'$m$',
                     'C_para': r'$C_{\parallel}$',
                     'C_perp': r'$C_{\perp}$',
                     'C_phi': r'$C_{\phi}$',
                     'km1': r'$k_{m_1}$',
                     'km2': r'$k_{m_2}$',
                     'km3': r'$k_{m_3}$',
                     'km4': r'$k_{m_4}$',
                     'd': r'$d$',
                     'psi': r'$\psi$',
                     'gamma': r'$\gamma$',
                     'alpha': r'$\alpha$',
                     'of': r'$\frac{g}{d}$',
                     'gdot': r'$\dot{g}$',}

    def convert_to_latex(self, list_of_strings, remove_dollar_signs=False):
        """ Loop through list of strings and if any match the dict, then swap in LaTex symbol.
        """

        if isinstance(list_of_strings, str):  # if single string is given instead of list
            list_of_strings = [list_of_strings]
            string_flag = True
        else:
            string_flag = False

        list_of_strings = list_of_strings.copy()
        for n, s in enumerate(list_of_strings):  # each string in list
            for k in self.dict.keys():  # check each key in Latex dict
                if s == k:  # string contains key
                    # print(s, ',', self.dict[k])
                    list_of_strings[n] = self.dict[k]   # replace string with LaTex
                    if remove_dollar_signs:
                        list_of_strings[n] = list_of_strings[n].replace('$', '')

        if string_flag:
            list_of_strings = list_of_strings[0]

        return list_of_strings


class ObservabilityMatrixImage:
    """ Display an image of an observability matrix.
    """

    def __init__(self, O, R=None, state_names='x', sensor_names='y', vmax_percentile=100, vmin_ratio=0.0, cmap='bwr'):
        """ Initialize.
        """

        # Plotting parameters
        self.vmax_percentile = vmax_percentile
        self.vmin_ratio = vmin_ratio
        self.cmap = cmap
        self.crange = None
        self.fig = None
        self.ax = None
        self.cbar = None

        # Get O
        self.n_measurement, self.n_state = O.shape
        if isinstance(O, pd.DataFrame):  # data-frame
            self.O = O.copy() # O in matrix form

            # Default state names based on data-frame columns
            self.state_names_default = list(O.columns)

            # Default sensor names based on data-frame 'sensor' index
            sensor_names_all = list(np.unique(O.index.get_level_values('sensor')))
            self.sensor_names_default = list(O.index.get_level_values('sensor')[0:len(sensor_names_all)])

        else:  # numpy matrix
            raise TypeError('n-sensor must be an integer value when O is given as a numpy matrix')

        self.n_sensor = len(self.sensor_names_default)  # number of sensors
        self.n_time_step = int(self.n_measurement / self.n_sensor)  # number of time-steps

        # Calculate Observability Gramian
        self.W = self.O.values.T @ self.O.values

        # Calculate Fisher Information Matrix & inverse
        self.FO = FisherObservability(self.O, R, beta=1e-9, epsilon=1e-6, binary_diagonal=False)
        self.F = self.FO.F.copy()
        self.F_pinv = self.FO.F_pinv
        self.F_pinv_norm = utils.log_scale_with_negatives(self.F_pinv, epsilon=2, inverse=True)
        # self.F_pinv = np.eye(self.n_state)
        # np.fill_diagonal(self.F_pinv, self.FO.error_covariance.values)
        # # self.F_pinv = np.nan_to_num(self.F_pinv, nan=-np.nanmax(self.F_pinv))
        # self.F_pinv = np.nan_to_num(self.F_pinv, nan=0.0)

        # Set state names
        if state_names is not None:
            if len(state_names) == self.n_state:
                self.state_names = state_names.copy()
            elif len(state_names) == 1:
                self.state_names = ['$' + state_names[0] + '_{' + str(n) + '}$' for n in range(1, self.n_state + 1)]
            else:
                raise TypeError('state_names must be of length n or length 1')
        else:
            self.state_names = self.state_names_default.copy()

        # Convert to Latex
        LatexConverter = LatexStates()
        self.state_names = LatexConverter.convert_to_latex(self.state_names)

        # Set sensor & measurement names
        if sensor_names is not None:
            if len(sensor_names) == self.n_sensor:
                self.sensor_names = sensor_names.copy()
                self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)
                self.measurement_names = []
                for w in range(self.n_time_step):
                    for p in range(self.n_sensor):
                        m = '$' + self.sensor_names[p] + ',_{' + 'k=' + str(w) + '}$'
                        self.measurement_names.append(m)

            elif len(sensor_names) == 1:
                self.sensor_names = [sensor_names[0] + '_{' + str(n) + '}$' for n in range(1, self.n_sensor + 1)]
                self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)
                self.measurement_names = []
                for w in range(self.n_time_step):
                    for p in range(self.n_sensor):
                        m = '$' + sensor_names[0] + '_{' + str(p) + ',k=' + str(w) + '}$'
                        self.measurement_names.append(m)
            else:
                raise TypeError('sensor_names must be of length p or length 1')

        else:
            self.sensor_names = self.sensor_names_default.copy()
            self.sensor_names = LatexConverter.convert_to_latex(self.sensor_names, remove_dollar_signs=True)
            self.measurement_names = []
            for w in range(self.n_time_step):
                for p in range(self.n_sensor):
                    m = '$' + self.sensor_names[p] + '_{' + ',k=' + str(w) + '}$'
                    self.measurement_names.append(m)

    def plot(self, vmax_percentile=100, vmin_ratio=0.0,  vmax_override=None, cmap='bwr', grid=True, scale=1.0, dpi=150, ax=None):
        """ Plot the observability matrix.
        """

        # Plot properties
        self.vmax_percentile = vmax_percentile
        self.vmin_ratio = vmin_ratio
        self.cmap = cmap

        if vmax_override is None:
            self.crange = np.percentile(np.abs(self.O), self.vmax_percentile)
        else:
            self.crange = vmax_override

        # Display O
        O_disp = self.O.values
        # O_disp = np.nan_to_num(np.sign(O_disp) * np.log(np.abs(O_disp)), nan=0.0)
        for n in range(self.n_state):
            for m in range(self.n_measurement):
                oval = O_disp[m, n]
                if (np.abs(oval) < (self.vmin_ratio * self.crange)) and (np.abs(oval) > 1e-6):
                    O_disp[m, n] = self.vmin_ratio*self.crange*np.sign(oval)

        # Plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(0.3*self.n_state*scale, 0.3*self.n_measurement*scale), dpi=dpi)
        else:
            fig = None

        O_data = ax.imshow(O_disp, vmin=-self.crange, vmax=self.crange, cmap=self.cmap)
        ax.grid(visible=False)

        ax.set_xlim(-0.5, self.n_state - 0.5)
        ax.set_ylim(self.n_measurement - 0.5, -0.5)

        ax.set_xticks(np.arange(0, self.n_state))
        ax.set_yticks(np.arange(0, self.n_measurement))

        ax.set_xlabel('States', fontsize=10, fontweight='bold')
        ax.set_ylabel('Measurements', fontsize=10, fontweight='bold')

        ax.set_xticklabels(self.state_names)
        ax.set_yticklabels(self.measurement_names)

        ax.tick_params(axis='x', which='major', labelsize=7, pad=-1.0)
        ax.tick_params(axis='y', which='major', labelsize=7, pad=-0.0, left=False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.set_label_position('top')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Draw grid
        if grid:
            grid_color = [0.8, 0.8, 0.8, 1.0]
            grid_lw = 1.0
            for n in np.arange(-0.5, self.n_measurement + 1.5):
                ax.axhline(y=n, color=grid_color, linewidth=grid_lw)
            for n in np.arange(-0.5, self.n_state + 1.5):
                ax.axvline(x=n, color=grid_color, linewidth=grid_lw)

        # Make colorbar
        axins = inset_axes(ax, width='100%', height=0.1, loc='lower left',
                           bbox_to_anchor=(0.0, -1.0*(1.0 / self.n_measurement), 1, 1), bbox_transform=ax.transAxes,
                           borderpad=0)

        cbar = plt.colorbar(O_data, cax=axins, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('matrix values', fontsize=9, fontweight='bold', rotation=0)

        # Store figure & axis
        self.fig = fig
        self.ax = ax
        self.cbar = cbar

    def plot_fisher(self, inverse=False, diagonal=False, vmax_percentile=100, vmin_ratio=0.0,  vmax_override=None, cmap=None, grid=True, scale=1.0, dpi=150):
        """ Plot the fisher information matrix.
        """

        if cmap is None:
            cmap = self.cmap

        # Plot properties
        self.vmax_percentile = vmax_percentile
        self.vmin_ratio = vmin_ratio
        self.cmap = cmap

        # Display F
        if inverse:
            # F_disp = np.abs(self.F_pinv_norm)
            F_disp = utils.log_scale_with_negatives(self.F_pinv, epsilon=2, inverse=True)
            F_disp = np.abs(F_disp)
            # F_disp = 1 / np.log(self.F_pinv.copy() + 2)
            if not diagonal:
                for r in range(F_disp.shape[0]):
                    for c in range(F_disp.shape[1]):
                        if r != c:
                            F_disp[c, r] = np.nan
            # print(pd.DataFrame(F_disp))
            # F_disp = np.log(F_disp)
        else:
            F_disp = self.F.copy()

        # Remove large values
        if vmax_override is None:
            vmax = np.nanpercentile(np.abs(F_disp), self.vmax_percentile)
            self.crange = (-vmax, vmax)
            if inverse:
                self.crange = (0, vmax)
                # self.crange = (-vmax, vmax)
        else:
            self.crange = (-vmax_override, vmax_override)

        for r in range(self.n_state):
            for c in range(self.n_state):
                fval = F_disp[r, c]
                if (np.abs(fval) < (self.vmin_ratio * self.crange[1])) and (np.abs(fval) > 1e-6):
                    F_disp[r, c] = self.vmin_ratio*self.crange[1]*np.sign(fval)

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(0.3*self.n_state*scale, 0.3*self.n_state*scale), dpi=dpi)
        O_data = ax.imshow(F_disp, vmin=self.crange[0], vmax=self.crange[1], cmap=cmap)
        ax.grid(visible=False)

        ax.set_xlim(-0.5, self.n_state - 0.5)
        ax.set_ylim(self.n_state - 0.5, -0.5)

        ax.set_xticks(np.arange(0, self.n_state))
        ax.set_yticks(np.arange(0, self.n_state))

        ax.set_xlabel('States', fontsize=10, fontweight='bold')
        ax.set_ylabel('States', fontsize=10, fontweight='bold')

        ax.set_xticklabels(self.state_names)
        ax.set_yticklabels(self.state_names)

        ax.tick_params(axis='x', which='major', labelsize=8, pad=-1.0)
        ax.tick_params(axis='y', which='major', labelsize=8, pad=-0.0, left=False)
        ax.tick_params(axis='x', which='both', top=False, labeltop=True, bottom=False, labelbottom=False)
        ax.xaxis.set_label_position('top')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')

        # Draw grid
        if grid:
            grid_color = [0.8, 0.8, 0.8, 1.0]
            grid_lw = 1.0
            for n in np.arange(-0.5, self.n_state + 1.5):
                ax.axhline(y=n, color=grid_color, linewidth=grid_lw)
            for n in np.arange(-0.5, self.n_state + 1.5):
                ax.axvline(x=n, color=grid_color, linewidth=grid_lw)

        # Make colorbar
        axins = inset_axes(ax, width='100%', height=0.1, loc='lower left',
                           bbox_to_anchor=(0.0, -1.0*(1.0 / self.n_state), 1, 1), bbox_transform=ax.transAxes,
                           borderpad=0)

        cbar = fig.colorbar(O_data, cax=axins, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('matrix values', fontsize=9, fontweight='bold', rotation=0)

        # Store figure & axis
        self.fig = fig
        self.ax = ax
        self.cbar = cbar


def make_color_map(color_list=None, color_proportions=None):
    """ Make a colormap from a list of colors.
    """

    if color_list is None:
        color_list = ['white', 'deepskyblue', 'mediumblue', 'yellow', 'orange', 'red', 'darkred']

    if color_proportions is None:
        color_proportions = np.linspace(0.01, 1, len(color_list) - 1)

    v = np.hstack((np.array(0.0), color_proportions))
    l = list(zip(v, color_list))
    cmap = LinearSegmentedColormap.from_list('rg', l, N=256)

    return cmap


def add_colorbar(fig, ax, data, cmap=None, label=None, ticks=None):
    offset_x = 0.017
    offset_y = 0.08

    cb_width = 0.75 * ax.get_position().width
    cb_height = 0.05 * ax.get_position().height

    cnorm = colors.Normalize(vmin=data.min(), vmax=data.max())
    cbax = fig.add_axes([offset_x + ax.get_position().x0, ax.get_position().y0 - offset_y, cb_width, cb_height])
    cb = fig.colorbar(cm.ScalarMappable(norm=cnorm, cmap=cmap), cax=cbax, orientation='horizontal')
    cb.ax.tick_params(labelsize=7, direction='in')
    cbax.yaxis.set_ticks_position('left')
    cb.set_label(label, labelpad=0, size=8)
    cb.ax.set_xticks(np.round(np.linspace(data.min(), data.max(), 5), 2))


def image_from_xyz(x, y, z=None, bins=100, sigma=None):
    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)
    z_min = np.min(z)
    z_max = np.max(z)

    hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[x_min - 1, x_max + 1], [y_min - 1, y_max + 1]])

    if z is None:  # use point density as color
        I = hist.T
    else:  # use z value as color
        I = np.zeros_like(hist)
        for bx in range(xedges.shape[0] - 1):
            x1 = xedges[bx]
            x2 = xedges[bx + 1]
            for by in range(yedges.shape[0] - 1):
                y1 = yedges[by]
                y2 = yedges[by + 1]
                xy_bin = (x >= x1) & (x < x2) & (y >= y1) & (y < y2)
                color_bin = z[xy_bin]
                if color_bin.shape[0] < 1:
                    I[by, bx] = z_min
                else:
                    I[by, bx] = np.mean(color_bin)

    if sigma is not None:
        I = scipy.ndimage.gaussian_filter(I, sigma=sigma, mode='reflect')

    return I


def plot_pulse_trajectory(df, cvar, data_range=None, arrow_size=0.008, nskip=1, dpi=100, figsize=8, cmap=None):
    # Get trajectory data
    time = df.time.values - 0.1
    x = df.xpos.values
    y = df.ypos.values
    phi = utils.wrapToPi(df.phi.values)
    phidot = df.phidot.values
    cvar_raw = df.cvar_raw.values

    # Pulse time
    startI = np.argmin(np.abs(time - 0.0))
    endI = np.argmin(np.abs(time - 0.675))
    pulse = np.arange(startI, endI)
    time_pulse = time[pulse]
    phi_pulse = phi[pulse]
    phidot_pulse = phidot[pulse]

    # Get data in range
    if data_range is None:
        data_range = (0, time.shape[0])

    index = np.arange(data_range[0], data_range[-1], 1)

    time = time[index]
    x = x[index]
    y = y[index]
    phi = phi[index]
    phidot = phidot[index]
    obsv = cvar[index]
    cvar_raw = cvar_raw[index]

    # Make figure
    fig, ax = plt.subplots(2, 2, figsize=(figsize, 0.4 * figsize), dpi=dpi,
                           gridspec_kw={
                               'width_ratios': [1.5, 1],
                               'height_ratios': [1, 1],
                               'wspace': 0.4,
                               'hspace': 0.4}
                           )

    # Plot pulse trajectory
    cmap_pulse = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cmap_pulse = ListedColormap(cmap_pulse)
    color = 0 * np.ones_like(time)
    color[pulse] = 1.0
    color = color[index]

    plot_trajectory(x, y, phi,
                    color=color,
                    ax=ax[0, 0],
                    nskip=nskip,
                    size_radius=arrow_size,
                    colormap=cmap_pulse)

    fifi.mpl_functions.adjust_spines(ax[0, 0], [])

    # Plot color trajectory
    ax[1, 0].set_title('Observability', fontsize=8)

    if cmap is None:
        crange = 0.1
        cmap = cm.get_cmap('RdPu')
        cmap = cmap(np.linspace(crange, 1, 100))
        cmap = ListedColormap(cmap)

    color = obsv.copy()
    cnorm = (np.nanmin(color), np.nanmax(color))
    # cnorm = (0.04, 0.09)
    plot_trajectory(x, y, phi,
                    color=color,
                    ax=ax[1, 0],
                    nskip=nskip,
                    size_radius=arrow_size,
                    reverse=True,
                    colormap=cmap,
                    colornorm=cnorm)

    fifi.mpl_functions.adjust_spines(ax[1, 0], [])

    # Colorbar
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cax = fig.add_axes([ax[1, 0].get_position().x1 - 0.25, ax[1, 0].get_position().y0 - 0.05,
                        0.4 * ax[1, 0].get_position().width, 0.075 * ax[1, 0].get_position().height])
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),
                      cax=cax, orientation='horizontal', label='Observability level')
    cb.ax.tick_params(labelsize=6, direction='out')
    cb.set_label('Observability level', labelpad=0, size=7)
    cb.ax.set_xticks([0, 1])

    # Phi
    ax[0, 1].axhline(y=0, color='gray', linestyle='--', lw=0.5)
    ax[0, 1].plot(*circplot(time, phi), color='black')
    ax[0, 1].plot(*circplot(time_pulse, phi_pulse), color='red')
    ax[0, 1].set_ylabel('Course direction \n(rad)', fontsize=8)
    ax[0, 1].tick_params(axis='both', which='major', labelsize=7)
    ax[0, 1].set_xlim(time[0], time[-1])
    pi_yaxis(ax[0, 1], tickpispace=0.5, lim=(-np.pi, np.pi))
    fifi.mpl_functions.adjust_spines(ax[0, 1], ['left', 'bottom'], tick_length=3, linewidth=0.75)
    ax[0, 1].spines['bottom'].set_visible(False)
    ax[0, 1].tick_params(bottom=False, labelbottom=False)

    # Phidot
    ax[1, 1].axhline(y=0, color='gray', linestyle='--', lw=0.5)
    ax[1, 1].plot(time, np.abs(phidot), color='black')
    ax[1, 1].plot(time_pulse, np.abs(phidot_pulse), color='red')
    ax[1, 1].set_ylabel('Angular velocity \n(rad/s)', fontsize=8)
    ax[1, 1].set_xlabel('Time (s)', fontsize=8)
    ax[1, 1].tick_params(axis='both', which='major', labelsize=7)
    ax[1, 1].set_xlim(time[0], time[-1])

    cc = 'darkmagenta'
    ax_right = ax[1, 1].twinx()
    ax_right.plot(time, obsv, color=cc)
    ax_right.plot(time, cvar_raw, '.', color='dodgerblue', linewidth=1.0, markersize=2)
    ax_right.set_ylabel('Observability level', fontsize=8, color=cc)
    ax_right.tick_params(axis='both', which='major', labelsize=7)
    ax_right.tick_params(axis='y', direction='in', colors=cc)
    ax_right.spines['right'].set_color(cc)
    ax_right.set_xlim(time[0], time[-1])
    ax_right.spines[['top', 'bottom', 'left']].set_visible(False)
    ax_right.spines['right'].set_position(('data', 1.08))
    # ax_right.set_ylim(0.025, 0.125)

    # ax[1, 1].set_ylim(-5, 60)
    # ax[1, 1].set_ylim(bottom=-5)

    ax_right.set_ylim(bottom=-0.0)
    ax_right.set_ylim(top=1.0)

    ax[0, 1].set_xlim(left=-0.1)
    ax[1, 1].set_xlim(left=-0.1)

    ax[1, 1].set_ylim(0, 60)
    fifi.mpl_functions.adjust_spines(ax[1, 1], ['left', 'bottom'], tick_length=3, linewidth=0.75)
    # ax[1, 1].set_ylim(-0.1, 40)
    ax[1, 1].set_ylim(bottom=-0.1)

    fig.align_ylabels(ax[:, 1])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)


def plot_trajectory(xpos, ypos, phi, color=None, ax=None, size_radius=None, nskip=0,
                    colormap='bone_r', colornorm=None, edgecolor='none', reverse=False):
    if color is None:
        color = phi

    color = np.array(color)

    # Set size radius
    xymean = np.mean(np.abs(np.hstack((xpos, ypos))))
    if size_radius is None:  # auto set
        xymean = 0.21 * xymean
        if xymean < 0.0001:
            sz = np.array(0.01)
        else:
            sz = np.hstack((xymean, 1))
        size_radius = sz[sz > 0][0]
    else:
        if isinstance(size_radius, list):  # scale defualt by scalar in list
            xymean = size_radius[0] * xymean
            sz = np.hstack((xymean, 1))
            size_radius = sz[sz > 0][0]
        else:  # use directly
            size_radius = size_radius

    if colornorm is None:
        colornorm = [np.min(color), np.max(color)]

    if reverse:
        xpos = np.flip(xpos, axis=0)
        ypos = np.flip(ypos, axis=0)
        phi = np.flip(phi, axis=0)
        color = np.flip(color, axis=0)

    fpl.colorline_with_heading(ax, np.flip(xpos), np.flip(ypos), 
                               np.flip(color, axis=0), np.flip(phi),
                               nskip=nskip,
                               size_radius=size_radius,
                               deg=False,
                               colormap=colormap,
                               center_point_size=0.0001,
                               colornorm=colornorm,
                               show_centers=False,
                               size_angle=20,
                               alpha=1,
                               edgecolor=edgecolor)

    ax.set_aspect('equal')
    xrange = xpos.max() - xpos.min()
    xrange = np.max([xrange, 0.02])
    yrange = ypos.max() - ypos.min()
    yrange = np.max([yrange, 0.02])

    if yrange < (size_radius / 2):
        yrange = 10

    if xrange < (size_radius / 2):
        xrange = 10

    ax.set_xlim(xpos.min() - 0.2 * xrange, xpos.max() + 0.2 * xrange)
    ax.set_ylim(ypos.min() - 0.2 * yrange, ypos.max() + 0.2 * yrange)

    # fifi.mpl_functions.adjust_spines(ax, [])


def pi_yaxis(ax=0.5, tickpispace=0.5, lim=None, real_lim=None):
    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    lim = ax.get_ylim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = np.round(ticks / np.pi, 3)
    y0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for y in range(len(tickslabels)):
        tickslabels[y] = ('$' + str(Fraction(tickslabels[y])) + '\pi $')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[y0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tickslabels)

    if real_lim is None:
        real_lim = np.zeros(2)
        real_lim[0] = lim[0] - 0.4
        real_lim[1] = lim[1] + 0.4

    if lim is None:
        ax.set_ylim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_ylim(lim)

    ax.set_ylim(real_lim)


def pi_xaxis(ax, tickpispace=0.5, lim=None):
    if lim is None:
        ax.set_xlim(-1 * np.pi, 1 * np.pi)
    else:
        ax.set_xlim(lim)

    lim = ax.get_xlim()
    ticks = np.arange(lim[0], lim[1] + 0.01, tickpispace * np.pi)
    tickpi = ticks / np.pi
    x0 = abs(tickpi) < np.finfo(float).eps  # find 0 entry, if present

    tickslabels = tickpi.tolist()
    for x in range(len(tickslabels)):
        tickslabels[x] = ('$' + str(Fraction(tickslabels[x])) + '\pi$')

    tickslabels = np.asarray(tickslabels, dtype=object)
    tickslabels[x0] = '0'  # replace 0 entry with 0 (instead of 0*pi)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tickslabels)


def circplot(t, phi, jump=np.pi):
    """ Stitches t and phi to make unwrapped circular plot. """

    t = np.squeeze(t)
    phi = np.squeeze(phi)

    difference = np.abs(np.diff(phi, prepend=phi[0]))
    ind = np.squeeze(np.array(np.where(difference > jump)))

    phi_stiched = np.copy(phi)
    t_stiched = np.copy(t)
    for i in range(phi.size):
        if np.isin(i, ind):
            phi_stiched = np.concatenate((phi_stiched[0:i], [np.nan], phi_stiched[i + 1:None]))
            t_stiched = np.concatenate((t_stiched[0:i], [np.nan], t_stiched[i + 1:None]))

    return t_stiched, phi_stiched
