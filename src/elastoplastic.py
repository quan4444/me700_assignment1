import numpy as np
from typing import List


class ElastoPlastic:
    def __init__(self,E,H,Y0):
        self.E=E
        self.H=H
        self.Y0=Y0
        self.sigma_n=0
        self.epsilon_p_n=0

    def compute_delta_sigma_trial(self,delta_epsilon):
        return self.E*delta_epsilon
    
    def predict_sigma_trial(self,delta_epsilon):
        delta_sigma_trial=self.compute_delta_sigma_trial(delta_epsilon)
        return self.sigma_n+delta_sigma_trial
    
    def in_elastic_regime(self,phi_trial):
        if phi_trial<=0:
            return True
        else:
            False

    def compute_delta_epsilon_p(self,phi_trial):
        return phi_trial/(self.E+self.H)


class IsotropicHardening(ElastoPlastic):

    def compute_yield_stress(self):
        return self.Y0+self.H*self.epsilon_p_n

    def compute_phi_trial(self,sigma_trial,Yn):
        return np.abs(sigma_trial)-Yn
    
    def compute_sigma_n_plastic(self,sigma_trial,delta_epsilon_p):
        return sigma_trial - np.sign(sigma_trial)*self.E*delta_epsilon_p

    def update_step(self,delta_epsilon):
        Yn = self.compute_yield_stress()

        # predictor step
        sigma_trial=self.predict_sigma_trial(delta_epsilon)
        phi_trial = self.compute_phi_trial(sigma_trial,Yn)

        if self.in_elastic_regime(phi_trial):
            self.sigma_n=sigma_trial
        else:
            delta_epsilon_p = self.compute_delta_epsilon_p(phi_trial)
            self.sigma_n = self.compute_sigma_n_plastic(sigma_trial,delta_epsilon_p)
            self.epsilon_p_n+=delta_epsilon_p

    def start_experiment(self,strain_list:List[float]):
        stress_list=[]
        for step_ind,cur_strain in enumerate(strain_list):
            
            if step_ind==0:
                stress_list.append(self.sigma_n)
                continue

            delta_epsilon = cur_strain - strain_list[step_ind-1]
            if np.isclose(delta_epsilon,0):
                stress_list.append(self.sigma_n)
                continue

            self.update_step(delta_epsilon)
            stress_list.append(self.sigma_n)
        return stress_list


class KinematicHardening(ElastoPlastic):

    def __init__(self,E,H,Y0):
        super().__init__(E,H,Y0)
        self.alpha_n=0

    def compute_eta_trial(self,sigma_trial,alpha_trial):
        return sigma_trial-alpha_trial
    
    def compute_phi_trial(self,eta_trial):
        return np.abs(eta_trial)-self.Y0
    
    def compute_sigma_n_plastic(self,sigma_trial,eta_trial,delta_epsilon_p):
        return sigma_trial - np.sign(eta_trial)*self.E*delta_epsilon_p

    def update_step(self,delta_epsilon):

        # predictor step
        sigma_trial=self.predict_sigma_trial(delta_epsilon)
        alpha_trial = self.alpha_n
        eta_trial = self.compute_eta_trial(sigma_trial,alpha_trial)
        phi_trial = self.compute_phi_trial(eta_trial)

        if self.in_elastic_regime(phi_trial):
            self.sigma_n=sigma_trial
        else:
            delta_epsilon_p = self.compute_delta_epsilon_p(phi_trial)
            self.sigma_n = self.compute_sigma_n_plastic(sigma_trial,eta_trial,delta_epsilon_p)
            self.alpha_n+=np.sign(eta_trial)*self.H*delta_epsilon_p
            self.epsilon_p_n+=delta_epsilon_p

    def start_experiment(self,strain_list:List[float]):
        stress_list=[]
        for step_ind,cur_strain in enumerate(strain_list):

            if step_ind==0:
                stress_list.append(self.sigma_n)
                continue

            delta_epsilon = cur_strain - strain_list[step_ind-1]
            if np.isclose(delta_epsilon,0):
                stress_list.append(self.sigma_n)
                continue

            self.update_step(delta_epsilon)
            stress_list.append(self.sigma_n)
        return stress_list