import pymc3 as pm
import numpy as np
import theano.tensor as tt
import theano 

class HMMStates(pm.Categorical):
    def __init__(self, p_transition, init_prob, actions, n_states, *args, **kwargs):
        super(pm.Categorical, self).__init__(*args, **kwargs)
        self.p_transition = p_transition
        self.init_prob = init_prob
        self.actions = actions
        self.k = n_states
        self.mode = tt.cast(0,dtype='int64')

    def logp(self, x):
        p_init = self.init_prob
        acts = self.actions[:-1]
        p_tr = self.p_transition[acts, x[:-1]]

        # the logp of the initial state 
        initial_state_logp = pm.Categorical.dist(p_init).logp(x[0])

        # the logp of the rest of the states.
        x_i = x[1:]
        ou_like = pm.Categorical.dist(p_tr).logp(x_i)
        transition_logp = tt.sum(ou_like)
        return initial_state_logp + transition_logp
    
class TruncatedNormalEmissionsAR_k(pm.Continuous):
    def __init__(self, states, mu_r, sigma_r, mu_d, sigma_d, mu_init, sigma_init, k, nu_r, nu_d, nu_init, actions, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.states = states
        self.sigma_r = sigma_r
        self.mu_r = mu_r
        self.mu_d = mu_d
        self.sigma_d = sigma_d
        self.mu_init = mu_init
        self.sigma_init = sigma_init
        self.k = k
        self.nu_r = nu_r
        self.nu_d = nu_d
        self.nu_init = nu_init
        self.indices_0 = [i for i, x in enumerate(actions[:-1]) if x == 0]
        self.indices_1 = [i for i, x in enumerate(actions[:-1]) if x != 0]
        self.actions_r = np.array([x for x in actions[:-1] if x != 0])

    def logp(self, x):
        """
        x: observations
        """
        states = self.states
        mu_r = self.mu_r[states]
        sigma_r = self.sigma_r[states]
        sigma_d = self.sigma_d[states]
        mu_d = self.mu_d[states]
        mu_init = self.mu_init[states[0]]
        sigma_init = self.sigma_init[states[0]]
        k = self.k[self.actions_r-1]
        nu_r = self.nu_r[states]
        nu_d = self.nu_d[states]
        nu_init = self.nu_init[states[0]]

        prev_x = x[:-1]
        cur_x = x[1:]
        x_det = cur_x[self.indices_0]
        x_rep = cur_x[self.indices_1]
        prev_x_det = prev_x[self.indices_0]
        prev_x_rep = prev_x[self.indices_1]
        cur_mu_r = mu_r[1:]
        cur_sigma_r = sigma_r[1:]
        cur_nu_r = nu_r[1:]
        cur_mu_d = mu_d[1:]
        cur_sigma_d = sigma_d[1:]
        cur_nu_d = nu_d[1:]
        mu_rep = cur_mu_r[self.indices_1]
        mu_det = cur_mu_d[self.indices_0]
        sigma_det = cur_sigma_d[self.indices_0]
        sigma_rep = cur_sigma_r[self.indices_1]
        nu_rep = cur_nu_r[self.indices_1]
        nu_det = cur_nu_d[self.indices_0]
        
        delta_det = x_det - prev_x_det
        
        DetStudentT = pm.Bound(pm.StudentT, upper=-prev_x_det)
        NegativeStudentT = pm.Bound(pm.StudentT, upper=0.0)         
        det_like = tt.sum(DetStudentT.dist(mu=mu_det, sigma=sigma_det, nu=nu_det).logp(delta_det))
        rep_like = tt.sum(NegativeStudentT.dist(mu=k*prev_x_rep + mu_rep, sigma=sigma_rep, nu=nu_rep).logp(x_rep))
        boundary_like = NegativeStudentT.dist(mu=mu_init, sigma=sigma_init, nu=nu_init).logp(x[0])
        return det_like + rep_like + boundary_like