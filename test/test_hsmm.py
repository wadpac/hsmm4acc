import hsmm4acc.hsmm as hsmm
import numpy as np


def test_initialize_model():
    Nmax = 2
    dim = 3
    model = hsmm.initialize_model(Nmax, dim)
    assert len(model.obs_distns) == Nmax


def test_colormap():
    num_states = 5
    colormap, cmap = hsmm.get_color_map(num_states)
    assert len(colormap.keys()) == num_states


def test_train_hsmm_one():
    # np.random.seed(1)
    maxduration = 5
    X_list = create_dataset_many(
        dim=2, nrstates=5, size=1, maxduration=maxduration)
    nr_resamples = 10
    model = hsmm.train_hsmm(X_list, Nmax=10,
                                         nr_resamples=nr_resamples,
                                         trunc=maxduration, visualize=False,
                                         example_index=0, max_hamming=0.05)
    assert len(model.stateseqs) == len(X_list)


def create_dataset_many(dim, nrstates, size, maxduration):
    means = 10 * np.random.rand(nrstates, dim)
    X_list = []
    for i in range(size):
        X = np.zeros((0, dim))
        for j in range(nrstates):
            length = np.random.randint(1, maxduration)
            X = np.concatenate(
                (X, np.random.normal(means[j], 0.2, (length, dim))))
        X_list.append(X)
    return X_list


def test_train_hsmm_many():
    # np.random.seed(1)
    nrstates = 5
    maxduration = 3
    X_list = create_dataset_many(
        dim=2, size=10, maxduration=maxduration, nrstates=nrstates)
    # print([X for X in X_list])
    nr_resamples = 10
    model = hsmm.train_hsmm(
        X_list, Nmax=nrstates, nr_resamples=nr_resamples, trunc=maxduration, visualize=False,
        example_index=0, max_hamming=0.05)
    assert len(model.stateseqs) == len(X_list)
