import numpy as np
from scipy.optimize import differential_evolution
from skneuromsi.neural import Cuppini2017
from skneuromsi.neural import Paredes2022

## TEMPORAL DATA

temporal_dis = (
    np.array(
        [
            -0.7,
            -0.5,
            -0.3,
            -0.25,
            -0.2,
            -0.15,
            -0.1,
            -0.08,
            -0.05,
            -0.02,
            -0.01,
            0.0,
            0.01,
            0.02,
            0.05,
            0.08,
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.5,
            0.7,
        ]
    )
    * 1000
)

temporal_causes_data = np.array(
    [
        0.03154762,
        0.03214286,
        0.07619048,
        0.19699793,
        0.30345238,
        0.50193093,
        0.72754579,
        0.82582418,
        0.90882784,
        0.93468864,
        0.97271062,
        0.97710623,
        0.95398352,
        0.94500916,
        0.94871795,
        0.94787546,
        0.92052078,
        0.87787546,
        0.75857143,
        0.51714286,
        0.33121212,
        0.05779221,
        0.02836439,
    ]
)

temporal_causes_data_short = temporal_causes_data[6:-4]


def temporal_cuppini2017_causes_job(a_onset, a_tau, v_tau, m_tau, ff_weight, cm_weight):

    v_onset = 110

    model = Cuppini2017(
        neurons=10,
        position_range=(0, 10),
        position_res=1,
        time_range=(0, 325),
        tau=(a_tau, v_tau, m_tau),
    )

    res = model.run(
        feedforward_weight=ff_weight,
        cross_modal_weight=cm_weight,
        noise=False,
        causes_kind="prob",
        causes_dim="time",
        auditory_stim_n=1,
        visual_stim_n=1,
        auditory_duration=6,
        visual_duration=6,
        auditory_onset=a_onset,
        visual_onset=v_onset,
        causes_peak_threshold=0.15,
    )
    prob_causes = res.causes_

    return prob_causes


def temporal_cuppini2017_causes_cost(theta):
    v_onset = 110
    causes = []
    for a_onset in v_onset + temporal_dis[6:-4]:
        prob_causes_per_a_onset = temporal_cuppini2017_causes_job(
            a_onset, theta[0], theta[1], theta[2], theta[3], theta[4]
        )
        causes.append(prob_causes_per_a_onset)

    model_data = np.array(causes)
    exp_data = temporal_causes_data_short

    cost = np.sum(np.square(np.divide(exp_data - model_data, exp_data)))

    return cost


bounds = [(0.1, 75), (0.1, 75), (0.1, 75), (0.01, 100), (0.01, 25)]
cuppini2017_temporal_causes_fit_res = differential_evolution(
    temporal_cuppini2017_causes_cost,
    bounds,
    disp=True,
    updating="deferred",
    workers=28,
    polish=False,
    seed=111,
)

pars = cuppini2017_temporal_causes_fit_res.x
np.save("Cuppini_2017_temporal_fit_res_pars.npy", pars)
