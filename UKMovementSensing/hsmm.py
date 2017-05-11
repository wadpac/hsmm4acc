from __future__ import print_function
from builtins import str
from builtins import zip
from builtins import range
import pyhsmm
import pyhsmm.basic.distributions as distributions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pybasicbayes.util.plot import plot_gaussian_2D
from pyhsmm.util.general import rle
from time import clock
import copy
import pandas as pd
from matplotlib.dates import date2num, AutoDateLocator

def train_hsmm(X_list, Nmax=10, nr_resamples=10, trunc=600, visualize=True,
               example_index=0, max_hamming=0.05):
    """
    Fit an Hidden Semi Markov Model on a list of sequences.

    Parameters
    ------
    X_list : list of Numpy arrays
        The sequences of shape (num_timesteps, num_channels)
    Nmax : int, optional
        Maximum number of states
    nr_resamples : int, optional
        Number of times the model is resampled on all data
    trunc : int, optional
        Maximum duration of a state, for optimization
    visualize : bool, optional
        Option to show plots during the process
    example_index : int, optional
        Which of the sequences to use as an example for plotting
    max_hamming : float, optional
        Terminates when hamming distance between consecutive state sequences
        is smaller then this number.

    Returns
    ------
    model : pyhsmm model
        The resampled model
    model_dists : list of distributions
        Observation distributions of the states for each sample step
    """
    dim = X_list[0].shape[1]
    model = initialize_model(Nmax, dim)
    model_dists = []
    prevstates = [np.zeros((X.shape[0])) for X in X_list]

    for X in X_list:
        model.add_data(X, trunc=trunc)
    converged = False
    for idx in range(nr_resamples):
        if not converged:
            t_start = clock()
            # model_copy = model.resample_and_copy()
            model.resample_model()
            t_end = clock()
            model_dists.append(copy.deepcopy(model.obs_distns))
            if (idx + 1) % 1 == 0 and visualize:
                print(idx)
                print('Resampled {} sequences in {:.1f} seconds'.format(len(X_list), t_end - t_start))
                print('Log likelihood: ', model.log_likelihood())

                model.plot_stateseq(example_index, draw=False)
                if(X_list[example_index].shape[1]>1):
                    plot_observations(X_list[example_index], 0, 1, model,
                                  model.stateseqs[example_index], Nmax)
                plt.draw()

            newstates = model.stateseqs
            hamdis = np.mean(
                [np.mean(a != b) for a, b in zip(prevstates, newstates)])

            prevstates = newstates
            print('Convergence: average Hamming distance is', hamdis)
            if hamdis < max_hamming:
                converged = True

    return model, model_dists


def read_data(filename, column_names):
    data = pd.read_csv(filename)
    X = data[column_names].as_matrix()
    return X


def iterate_hsmm_batch(X_list, model, current_states, trunc,
                       example_index=None, axis=None):
    # First time, the states need to be initalized:
    """

    Parameters
    ----------
    X_list : list of Numpy arrays
        The sequences of shape (num_timesteps, num_channels)
    model : pyhsmm model
        The HSMM model
    current_states : list of arrays
        The resulting statesequences of previous iteration
    trunc : int, optional
        Maximum duration of a state, for optimization
    example_index : int, optional
        Which of the sequences to use as an example for plotting
    axis : pyplot Axis
        axis to plot the example sequence

    Returns
    -------

    """
    if current_states is None:
        current_states = [np.zeros((X.shape[0])) for X in X_list]
        for X in X_list:
            model.add_data(X, trunc=trunc)
    else:
        for i, X in enumerate(X_list):
            model.add_data(X, stateseq=current_states[i], trunc=trunc)
    model.resample_model()
    newstates = model.stateseqs
    hamdis = [np.mean(a != b) for a, b in zip(current_states, newstates)]

    # Visualize
    if example_index is not None:
        model.plot_stateseq(example_index, ax=axis, draw=False)

    model.states_list = []
    return model, hamdis, newstates


def train_hsmm_all(filenames, column_names, batchsize=10, Nmax=10,
                   nr_resamples=10, trunc=600, visualize=True,
                   example_index=0, max_hamming=0.05):
    """
    Fit an Hidden Semi Markov Model a list of files, in batches.

    Parameters
    ------
    filenames : list of str
        List of paths to the csv files
    column_names : list of str
        Names of variables to use
    batchsize : int, optional
        Number of files to process in one batch
    Nmax : int, optional
        Maximum number of states
    nr_resamples : int, optional
        Number of times the model is resampled on all data
    trunc : int, optional
        Maximum duration of a state, for optimization
    visualize : bool, optional
        Option to show plots during the process
    example_index: int, optional
        Which of the sequences to use as an example for plotting
    max_hamming : float, optional
        Terminates when hamming distance between consecutive state sequences
        is smaller then this number.

    Returns
    ------
    model : pyhsmm model
        The resampled model
    model_dists : list of distributions
        Observation distributions of the states for each sample step
    """
    dim = len(column_names)
    model = initialize_model(Nmax, dim)
    if visualize:
        fig, axes = plt.subplots(nr_resamples, figsize=(15, 5))

    # Make batches and keep all states in memory to compute the hamming distance
    batches = [filenames[i:i + batchsize] for i in
               range(0, len(filenames), batchsize)]
    example_batch = np.floor(float(example_index) / len(filenames))
    example_index = example_index % batchsize
    print('Nr of batches: ' + str(len(batches)))
    states = [None for i in range(len(batches))]

    for idx in range(nr_resamples):
        t_start = clock()
        hamdis_list = []
        for i, batch in enumerate(batches):
            X_list = [read_data(filename, column_names) for filename in batch]
            visualize_index, axis = (example_index, axes[
                idx]) if i == example_batch and visualize else (None, None)
            model, hamdis_sub, newstates = iterate_hsmm_batch(X_list, model,
                                                              states[i],
                                                              trunc=trunc,
                                                              example_index=visualize_index,
                                                              axis=axis)
            states[i] = newstates
            hamdis_list.extend(hamdis_sub)
        t_end = clock()
        hamdis = np.mean(hamdis_list)

        if (idx + 1) % 1 == 0 and visualize:
            print(idx)
            print('Resampled all sequences in {:.1f} seconds'.format(
                t_end - t_start))
            print('Convergence: average Hamming distance is', hamdis)

        # Early convergence:
        if hamdis < max_hamming:
            return model
    return model


def initialize_model(Nmax, dim):
    """
    Initialize a HSMM model.

    Parameters
    ------
    Nmax : int, optional
        Maximum number of states
    dim : int
        The number of channels

    Returns
    ------
    model : pyhsmm model
        The initial model
    """
    obs_hypparams = {'mu_0': np.zeros(dim),  # mean of gaussians
                     'sigma_0': np.eye(dim),  # std of gaussians
                     'kappa_0': 0.3,
                     'nu_0': dim + 5
                     }

    # duration is going to be poisson, so prior is a gamma distribution
    # (params alpha beta)
    expected_lam = 12 * 30
    dur_hypparams = {'alpha_0': 2 * expected_lam,
                     'beta_0': 2}

    obs_distns = [distributions.Gaussian(**obs_hypparams)
                  for state in range(Nmax)]
    dur_distns = [
        distributions.PoissonDuration(
            **dur_hypparams) for state in range(
            Nmax)]
    model = pyhsmm.models.WeakLimitHDPHSMM(
        alpha=6., gamma=6.,  # priors
        init_state_concentration=6.,  # alpha0 for the initial state
        obs_distns=obs_distns,
        dur_distns=dur_distns)
    return model


def plot_observations(X, dim0, dim1, model, hidden_states, num_states):
    """
    Plots 2 dimensions of the data, with the gaussian observation distributions

    Parameters
    ----------
    X : Numpy array
        Observeration sequence of dimensions (num_timesteps, num_channels)
    dim0 : int
        First channel to plot
    dim1 : int
        Second channel to plot
    model : pyhsmm model
        The model to plot
    hidden_states : iteretable
        List with the states for each time step
    num_states : int
        Total number of states

    """
    fig, axes = plt.subplots(2, figsize=(10, 10))
    colormap, cmap = get_color_map(num_states)
    statecolors = [colormap[i] for i in hidden_states]
    axes[0].scatter(X[:, dim0], X[:, dim1], color='black', s=5)
    for i in range(num_states):
        plot_gaussian_2D(model.obs_distns[i].mu[[dim0, dim1],],
                         model.obs_distns[i].sigma[[dim0, dim1], :][:, [dim0, dim1]], color=colormap[i], ax=axes[0])

    axes[1].scatter(X[:, 0], X[:, 1], color=statecolors, s=5)


def get_color_map(num_states):
    colours = plt.cm.viridis(np.linspace(0, 1, num_states))
    colormap = {i: colours[i] for i in range(num_states)}
    cmap = LinearSegmentedColormap.from_list('name',
                                             list(colormap.values()),
                                             num_states)
    return colormap, cmap


def plot_boxplots(data, hidden_states):
    """
    Plot boxplots for all variables in the dataset, per state

    Parameters
    ------
    data : pandas DataFrame
        Data to plot
    hidden_states: iteretable
        the hidden states corresponding to the timesteps
    """
    column_names = data.columns
    figs, axes = plt.subplots(len(column_names), figsize=(15, 15))
    for j, var in enumerate(column_names):
        axes[j].set_title(var)
        vals = data[var]
        data_to_plot = []
        labels = []
        for i in set(hidden_states):
            mask = hidden_states == i
            if (sum(mask) > 0):
                labels.append(str(i))
                values = np.array(vals[mask])
                data_to_plot.append(values)
        axes[j].boxplot(data_to_plot, sym='', labels=labels)


def plot_perstate(data, hidden_states):
    '''
    Make, for each state, a plot of the data

    Parameters
    ----------
    data : pandas DataFrame
        Data to plot
    hidden_states: iteretable
        the hidden states corresponding to the timesteps
    '''
    num_states = max(hidden_states) + 1
    fig, axs = plt.subplots(
        num_states, sharex=True, sharey=True, figsize=(15, 15))
    colours = plt.cm.rainbow(np.linspace(0, 1, num_states))
    for i, (ax, colour) in enumerate(zip(axs, colours)):
        # Use fancy indexing to plot data in each state.
        data_to_plot = data.copy()
        data_to_plot[hidden_states != i] = 0
        data_to_plot.plot(ax=ax, legend=False)
        ax.set_title("{0}th hidden state".format(i))
        ax.grid(True)
    plt.legend(bbox_to_anchor=(0, -1, 1, 1), loc='lower center')
    plt.draw()



def plot_states_and_var(data, hidden_states, cmap=None, columns=None, by='Activity'):
    """
    Make  a plot of the data and the states

    Parameters
    ----------
    data : pandas DataFrame
        Data to plot
    hidden_states: iteretable
        the hidden states corresponding to the timesteps
    columns : list, optional
        Which columns to plot
    by : str
        The column to group on
    """
    fig, ax = plt.subplots(figsize=(15, 5))
    if columns is None:
        columns = data.columns
    df = data[columns].copy()
    stateseq = np.array(hidden_states)
    stateseq_norep, durations = rle(stateseq)
    datamin, datamax = np.array(df).min(), np.array(df).max()
    y = np.array(
        [datamin, datamax])
    maxstate = stateseq.max() + 1
    x = np.hstack(([0], durations.cumsum()[:-1], [len(df.index) - 1]))
    C = np.array(
        [[float(state) / maxstate] for state in stateseq_norep]).transpose()
    ax.set_xlim((min(x), max(x)))

    if cmap is None:
        num_states = max(hidden_states) + 1
        colormap, cmap = get_color_map(num_states)
    pc = ax.pcolorfast(x, y, C, vmin=0, vmax=1, alpha=0.3, cmap=cmap)
    plt.plot(df.as_matrix())
    locator = AutoDateLocator()
    locator.create_dummy_axis()
    num_index = pd.Index(df.index.map(date2num))
    ticks_num = locator.tick_values(min(df.index), max(df.index))
    ticks = [num_index.get_loc(t) for t in ticks_num]
    plt.xticks(ticks, df.index.strftime('%H:%M')[ticks], rotation='vertical')
    cb = plt.colorbar(pc)
    cb.set_ticks(np.arange(1./(2*cmap.N), 1, 1./cmap.N))
    cb.set_ticklabels(np.arange(0, cmap.N))
    # Plot the activities
    if by is not None:
        actseq = np.array(data[by])
        sca = ax.scatter(
            np.arange(len(hidden_states)), #data.index,
            np.ones_like(hidden_states) * datamax,
            c=actseq,
            edgecolors='none'
        )
    plt.draw()
    return fig, ax

def plot_heatmap(plotdata, horizontal_labels=None, vertical_labels=None, form='{:.4}'):
    """
    Plot a heat map with marked values
    Parameters
    ----------
    plotdata : numpy array
        data to plotdata
    horizontal_labels: list, optional
        horizontal labels
    vertical_labels : list, optional
        vertical labels
    form : str, optional
        format for the value labels
    """
    fig, ax = plt.subplots()
    colorplot = ax.pcolor(plotdata, cmap='coolwarm', )
    ax.set_xticks(np.arange(plotdata.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(plotdata.shape[0]) + 0.5, minor=False)
    if horizontal_labels is None:
        horizontal_labels = list(range(plotdata.shape[1]))
    ax.set_xticklabels(horizontal_labels, minor=False)
    if vertical_labels is None:
        vertical_labels = list(range(plotdata.shape[0]))
    ax.set_yticklabels(vertical_labels, minor=False)
    plt.colorbar(colorplot)
    for y in range(plotdata.shape[0]):
        for x in range(plotdata.shape[1]):
            plt.text(x + 0.5, y + 0.5, form.format(plotdata[y, x]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
