from assignment1 import elastoplastic as ep
import numpy as np
import pytest

@pytest.fixture
def ElastoMat():
    E=1000
    E_t=100
    Y0=10
    H=E*E_t/(E-E_t)
    return ep.ElastoPlastic(E=E,H=H,Y0=Y0)

@pytest.fixture
def IsoMat():
    E=1000
    E_t=100
    Y0=10
    H=E*E_t/(E-E_t)
    return ep.IsotropicHardening(E=E,H=H,Y0=Y0)


@pytest.fixture
def KinMat():
    E=1000
    E_t=100
    Y0=10
    H=E*E_t/(E-E_t)
    return ep.KinematicHardening(E=E,H=H,Y0=Y0)


def test_compute_delta_sigma_trial(ElastoMat):
    delta_epsilon = 0.001
    found=ElastoMat.compute_delta_sigma_trial(delta_epsilon)
    known=1
    assert np.isclose(found,known)


def test_predict_sigma_trial(ElastoMat):
    delta_epsilon=0.001
    found=ElastoMat.predict_sigma_trial(delta_epsilon)
    known=1.0
    assert np.isclose(found,known)


def test_in_elastic_regime(ElastoMat):
    in_elastic=ElastoMat.in_elastic_regime(-1.0)
    in_plastic=ElastoMat.in_elastic_regime(1.0)
    assert in_elastic
    assert not in_plastic


def test_compute_delta_epsilon_p(IsoMat):
    phi_trial=-9.0
    found=IsoMat.compute_delta_epsilon_p(phi_trial)
    known=-0.0081
    assert np.isclose(found,known)


def test_compute_yield_stress(IsoMat):
    IsoMat.epsilon_p_n=0.001
    found = IsoMat.compute_yield_stress()
    known = 10.111111111111
    assert np.isclose(found,known)


def test_compute_phi_trial_iso(IsoMat):
    sigma_trial=1.0
    Yn=10
    found=IsoMat.compute_phi_trial(sigma_trial,Yn)
    known=-9.0
    assert np.isclose(found,known)


def test_compute_sigma_n_plastic_iso(IsoMat):
    sigma_trial=10
    delta_epsilon_p=0.05
    found=IsoMat.compute_sigma_n_plastic(sigma_trial,delta_epsilon_p)
    known=-40
    assert np.isclose(found,known)


def test_update_step_elastic_iso(IsoMat):
    delta_epsilon=0.001
    IsoMat.update_step(delta_epsilon)
    found_sn=IsoMat.sigma_n
    found_epn=IsoMat.epsilon_p_n
    known_sn=1.0
    known_epn=0.0
    assert np.isclose(found_sn,known_sn)
    assert np.isclose(found_epn,known_epn)


def test_update_step_plastic_iso(IsoMat):
    delta_epsilon=0.001
    IsoMat.sigma_n=50
    IsoMat.epsilon_p_n=0.1
    IsoMat.update_step(delta_epsilon)
    found_sn=IsoMat.sigma_n
    found_epn=IsoMat.epsilon_p_n
    known_sn=24.1
    known_epn=0.1269
    assert np.isclose(found_sn,known_sn)
    assert np.isclose(found_epn,known_epn)


def test_start_experiment_iso(IsoMat):
    strain_list=[0.001,0.002,0.003]
    found = IsoMat.start_experiment(strain_list)
    assert isinstance(found,list)


def test_compute_eta_trial(KinMat):
    found=KinMat.compute_eta_trial(10,1)
    known=9
    assert np.isclose(found,known)


def test_compute_phi_trial_kin(KinMat):
    found=KinMat.compute_phi_trial(1)
    known=-9
    assert np.isclose(found,known)


def test_compute_sigma_n_plastic_kin(KinMat):
    found=KinMat.compute_sigma_n_plastic(100,10,0.001)
    known=99
    assert np.isclose(found,known)


def test_update_step_elastic_kin(KinMat):
    delta_epsilon=0.001
    KinMat.update_step(delta_epsilon)
    found_sn=KinMat.sigma_n
    found_epn=KinMat.epsilon_p_n
    known_sn=1.0
    known_epn=0.0
    assert np.isclose(found_sn,known_sn)
    assert np.isclose(found_epn,known_epn)


def test_update_step_plastic_kin(KinMat):
    delta_epsilon=0.001
    KinMat.sigma_n=50
    KinMat.epsilon_p_n=0.1
    KinMat.update_step(delta_epsilon)
    found_sn=KinMat.sigma_n
    found_epn=KinMat.epsilon_p_n
    known_sn=14.09999999
    known_epn=0.1369
    assert np.isclose(found_sn,known_sn)
    assert np.isclose(found_epn,known_epn)


def test_start_experiment_kin(KinMat):
    strain_list=[0.001,0.002,0.003]
    found = KinMat.start_experiment(strain_list)
    assert isinstance(found,list)
