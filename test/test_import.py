import UKMovementSensing.hsmm as hsmm

def test_empty():
    Nmax = 2
    dim = 3
    model = hsmm.initialize_model(Nmax, dim)
    assert len(model.obs_distns)==2
