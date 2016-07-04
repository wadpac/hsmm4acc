import pyhsmm
import pyhsmm.basic.distributions as distributions
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def train_hsmm(X, Nmax=10, nr_resamples=100, visualize=True):
    dim = X.shape[1]
    model = initialize_model(Nmax, dim)
    model.add_data(X, trunc=600)
    for idx in xrange(nr_resamples):
        model.resample_model()
        if (idx + 1) % 10 == 0 and visualize:
            print(idx)
            model.plot()
    return model


def initialize_model(Nmax, dim):
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


def plot_boxplots(data, hidden_states):
    '''
    Plot boxplots for all variables in the dataset, per state

    Parameters
    ----------
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
    num_states = max(hidden_states)
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


def plot_states_and_var(data, hidden_states, columns=None, by='Activity'):
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
    byAct = data.groupby(by)
    fig, ax = plt.subplots(figsize=(15, 5))
    if columns is None:
        columns = data.columns
    for act, dfa in byAct:
        for col in columns:
            dfa[col].plot(label=col+' - '+act, ax=ax)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3)
    num_states = max(hidden_states)
    colours = plt.cm.rainbow(np.linspace(0, 1, num_states))
    colormap = {i: colours[i] for i in range(num_states)}
    cmap = LinearSegmentedColormap.from_list('name',
                                             colormap.values(),
                                             num_states)
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
