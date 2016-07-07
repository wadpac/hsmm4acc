import pyhsmm
import pyhsmm.basic.distributions as distributions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pybasicbayes.util.plot import plot_gaussian_2D
from time import clock
import copy

def train_hsmm(X_list, Nmax=10, nr_resamples=10, trunc=600, visualize=True, example_index=0):
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
    example_index: int, optional
        Which of the sequences to use as an example for plotting

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
    if visualize:
        fig, axes = plt.subplots(nr_resamples, figsize=(15,5))
    for X in X_list:
        model.add_data(X, trunc=trunc)
    for idx in xrange(nr_resamples):
        t_start = clock()
        #model_copy = model.resample_and_copy()
        model.resample_model()
        t_end = clock()
        model_dists.append(copy.deepcopy(model.obs_distns))
        if (idx + 1) % 1 == 0 and visualize:
            print(idx)
            print('Resampled {} sequences in {:.1f} seconds'.format(len(X_list), t_end-t_start))
            model.plot_stateseq(example_index, ax=axes[idx])
            plot_observations(X_list[example_index], 0, 1, model,
                              model.stateseqs[example_index], Nmax)
    return model, model_dists


def initialize_model(Nmax, dim):
    """
    Fit an Hidden Semi Markov Model on a list of sequences.

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
    fig, axes = plt.subplots(2, figsize=(10,10))
    colormap, cmap = get_color_map(num_states)
    statecolors = [colormap[i] for i in hidden_states]
    axes[0].scatter(X[:, dim0], X[:, dim1], color='black', s=5)
    for i in range(num_states):
        plot_gaussian_2D(model.obs_distns[i].mu[[dim0, dim1],], model.obs_distns[i].sigma[[dim0, dim1], :][:, [dim0, dim1]], color=colormap[i], ax=axes[0])

    axes[1].scatter(X[:,0], X[:,1], color=statecolors, s=5)

def get_color_map(num_states):
    colours = plt.cm.rainbow(np.linspace(0, 1, num_states))
    colormap = {i: colours[i] for i in range(num_states)}
    cmap = LinearSegmentedColormap.from_list('name',
                                             colormap.values(),
                                             num_states)
    return colormap, cmap

def plot_boxplots(data, hidden_states):
    '''
    Plot boxplots for all variables in the dataset, per state

    Parameters
    ------
    data : pandas DataFrame
        Data to plot
    hidden_states: iteretable
        the hidden states corresponding to the timesteps
    '''
    column_names = data.columns
    figs, axes = plt.subplots(len(column_names), figsize=(15, 15))
    for j, var in enumerate(column_names):
        axes[j].set_title(var)
        vals = data[var]
        data_to_plot = []
        labels = []
        for i in set(hidden_states):
            mask = hidden_states == i
            if(sum(mask) > 0):
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
    num_states = max(hidden_states)+1
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
    plt.show()


def plot_states_and_var(data, hidden_states, cmap=None, columns=None, by='Activity'):
    '''
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
    '''
    fig = plt.figure(figsize=(15, 5))
    fig, ax = plt.subplots(figsize=(15, 5))
    if columns is None:
        columns = data.columns
    for act in set(data[by]):
        for col in columns:
            dfa = data[col].copy()
            dfa[data[by] != act] = 0
            dfa.plot(label=str(col)+' - '+str(act), ax=ax)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3)
    if cmap is None:
        num_states = max(hidden_states)+1
        colormap, cmap = get_color_map(num_states)
    scale = np.array(data[columns]).max()
    sca = plt.scatter(
        data.index,
        np.ones_like(hidden_states) * scale,
        c=hidden_states,
        cmap=cmap,
        edgecolors='none')
    plt.colorbar(
        sca,
        ticks=np.arange(np.min(hidden_states),
                        np.max(hidden_states) + 1))


def plot_heatmap(plotdata, horizontal_labels=None, vertical_labels=None, form='{:.4}'):
    '''
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
    '''
    fig, ax = plt.subplots()
    colorplot = ax.pcolor(plotdata, cmap='coolwarm',)
    ax.set_xticks(np.arange(plotdata.shape[1])+0.5, minor=False)
    ax.set_yticks(np.arange(plotdata.shape[0])+0.5, minor=False)
    if horizontal_labels is None:
        horizontal_labels = range(plotdata.shape[1])
    ax.set_xticklabels(horizontal_labels, minor=False)
    if vertical_labels is None:
        vertical_labels = range(plotdata.shape[0])
    ax.set_yticklabels(vertical_labels, minor=False)
    plt.colorbar(colorplot)
    for y in range(plotdata.shape[0]):
        for x in range(plotdata.shape[1]):
            plt.text(x + 0.5, y + 0.5, form.format(plotdata[y, x]),
                     horizontalalignment='center',
                     verticalalignment='center',
                     )
