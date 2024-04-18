import time
import casadi as ca
import numpy as np
import math as m
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.transforms as tf
import pdb
from itertools import product
import copy
from random import sample
from infrastructure.utils import flatten, unflatten_duals
from scipy.optimize import lsq_linear
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.linalg import clarkson_woodruff_transform as sketch

class SMPC_MMPreds():

    def __init__(self, routes, ev, 
                N            =  6,
                V_MIN        = -1.,       #Speed, acceleration constraints
                V_MAX        = 10.0, 
                A_MIN        = -5.0,
                A_MAX        =  2.0,
                TIGHTENING   =  2.4, #2.6
                EV_NOISE_STD    =  [0.001, 0.001],
                TV_NOISE_STD    =[[0.01, 0.02]]*5,
                Q = 1.,       # cost for measuring progress: -Q*s_{t+1}. #was 1.
                R = 1.,       # cost for penalizing large input rate: (u_{t+1}-u_t).T@R@(u_{t+1}-u_t) #was 1.5
                offline_mode=True,
                solver="ipopt",
                reduced_mode=False,
                open_loop = False,
                eval_mode = False
                ):
        self.routes=routes
        self.ev=ev
        self.N=N
        self.V_MIN=V_MIN
        self.V_MAX=V_MAX
        self.A_MAX=A_MAX
        self.A_MIN=A_MIN
        
        if not reduced_mode:
            self.N_TV=len(TV_NOISE_STD)
            self.N_modes=[2,2,2,4,4]
        else:
            # Only 3 TVs
            self.N_TV=3
            self.N_modes=[2,2,4]
        
        # Maps a mode, say 10, to the modes of the TVs, like (0,1,1,3,3)
        self.mode_map = dict(enumerate(product(*[range(self.N_modes[k]) for k in range(self.N_TV)])))
        self.tight=TIGHTENING
        self.ev_n_std=EV_NOISE_STD
        self.tv_n_std=TV_NOISE_STD

        self.Q = ca.diag(Q)
        self.R = ca.diag(R)

        self.A=ev.A
        self.B=ev.B
        
        self.Atv=self.A
        self.Btv=self.B

        self.vars_kept = None
        self.constr_kept = None
        self.offline=offline_mode
        self.open_loop = open_loop
   
        
        p_opts = {'expand': False, 'print_time':0, 'verbose' :False, 'error_on_fail':0}
        s_opts = {'print_level': 1,'tol':1e-4,'max_wall_time': 60.}
        if eval_mode:
            s_opts.update({'max_wall_time': 10.,'constr_viol_tol':1e-2})

        s_opts_grb = {'OutputFlag': 0, 'PSDTol' : 1e-2,
                       'FeasibilityTol' : 1e-2, 
                       'BarConvTol':1e-2, 
                       'BarQCPConvTol':1e-2,
                       'LogToConsole': 0}
        p_opts_grb = {'expand': True,'error_on_fail':0, 'verbose':False, 'ad_weight':0}

        self.solver=solver
        
        if self.solver=="ipopt":
            self.opti=ca.Opti()
            self.opti.solver("ipopt", p_opts, s_opts)
        else:
            self.opti=ca.Opti("conic")
            self.opti.solver("gurobi", p_opts_grb, s_opts_grb)


        def _flatten2ca(xs):
            if type(xs) == type([]):
                for x in xs:
                    yield from _flatten2ca(x)
            else:
                yield ca.vec(xs)


       
        self.params = []
        
        self.z_curr=self.opti.parameter(2)
        self.u_prev=self.opti.parameter(1)
        
        self.params+=[self.z_curr, self.u_prev]
        
        self.z_lin=self.opti.parameter(2,self.N+1)
        self.x_pos=self.opti.parameter(2,self.N+1)    
        self.dpos =[self.opti.parameter(2,1) for _ in range(self.N)]

        self.params+=[ca.vec(self.z_lin), ca.vec(self.x_pos), ca.vec(ca.horzcat(*self.dpos))]

        self.l1_lmbd=1000*0.01

        self.z_tv_curr=[self.opti.parameter(2) for _ in range(self.N_TV)]
        self.u_tvs=[[self.opti.parameter(self.N,1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        self.pos_tvs=[[self.opti.parameter(2,self.N+1) for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        self.dpos_tvs=[[[self.opti.parameter(2,1) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]
        self.Qs=[[[self.opti.parameter(2,2) for _ in range(self.N)] for _ in range(self.N_modes[k])] for k in range(self.N_TV)]

        self.params+=[self.z_tv_curr, self.u_tvs, self.pos_tvs, self.dpos_tvs, self.Qs]

        
        if not self.offline:
            self.gain_keep=[[self.opti.parameter(self.N-1,1) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
            self.constr_keep=[[self.opti.parameter(self.N-1,1) for j in range(len(self.mode_map))] for k in range(self.N_TV)]

        # self.params +=[self.gain_keep, self.constr_keep]

        self.params = ca.vertcat(*_flatten2ca(self.params))  
        
        
        self.policy=self._return_policy_class()
        self._add_constraints_and_cost()
        
        self._update_ev_initial_condition(np.array([20., 10.]), 0.)
        self._update_ev_preds(np.ones((2,self.N+1)), 50*np.ones((2,self.N+1)), [np.ones((2,1))]*self.N)

        self._update_tv_initial_condition([np.array([0., 0.])]*self.N_TV)
        self._update_tv_preds([[np.zeros((self.N,1))]*self.N_modes[k] for k in range(self.N_TV)], [[np.zeros((2,self.N+1))]*self.N_modes[k] for k in range(self.N_TV)], 
                              [[[np.ones((2,1))]*self.N]*self.N_modes[k] for k in range(self.N_TV)], [[[np.eye(2)]*self.N]*self.N_modes[k] for k in range(self.N_TV)])
        
        if not self.offline: 
            self._update_gain_and_constr_keeps()            
        self.solve(first_solve=True)

    def _return_policy_class(self):

        """
        EV Affine disturbance feedback + TV state feedback policies from https://arxiv.org/abs/2109.09792
        """ 
    
        h0=self.opti.variable(1)
    
        # Uncomment next line for disturbance feedback when using Gurobi. 
        # Runs slow with Ipopt (default)
        if not self.offline:
            M=[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)]
            if self.open_loop:
                K=[[[ ca.DM(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            else:
                K=[[[ca.if_else(self.gain_keep[k][j][t], self.opti.variable(1,2), ca.DM(1,2), True) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            h=[self.opti.variable(1) for t in range(self.N-1)]

            
        else:
            M=[[ca.DM(1, 2) for n in range(t)] for t in range(self.N)]
            if self.open_loop:
                K=[[[ ca.DM(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            else:
                K=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
            h=[self.opti.variable(1) for t in range(self.N-1)]
            
            self.gain_l1=[[[self.opti.variable(1,2) for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]

        h_stack=ca.vertcat(h0,*[h[t] for t in range(self.N-1)])
        M_stack=ca.vertcat(*[ca.horzcat(*[M[t][n] for n in range(t)], ca.DM(1,2*(self.N-t))) for t in range(self.N)])
        K_stack=[[ca.diagcat(ca.DM(1,2),*[K[k][j][t] for t in range(self.N-1)]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 
        
        self.vars_pol = ca.vertcat(h_stack, ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(K[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)]))
        if self.offline:
            self.vars_epi = ca.vertcat(*[ca.vertcat(*[ca.vertcat(*[ca.vec(self.gain_l1[k][j][t]) for t in range(self.N-1)]) for j in range(self.N_modes[k])]) for k in range(self.N_TV)])
        self.vars_ws, self.vars_epi_ws  = None, None 
        return h_stack,M_stack,K_stack

        
    def _get_ATV_TV_dynamics(self):
        """
        Constructs system matrices such that for mode j and for TV k,
        O_t=T_tv@o_{t|t}+c_tv+E_tv@N_t
        where
        O_t=[o_{t|t}, o_{t+1|t},...,o_{t+N|t}].T, (TV state predictions)
        N_t=[n_{t|t}, n_{t+1|t},...,n_{t+N-1|t}].T,  (TV process noise sequence)
        o_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 

        T_tv=[[ca.DM(2*(self.N+1), 2) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        TB_tv=[[ca.DM(2*(self.N+1), self.N) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        c_tv=[[ca.DM(2*(self.N+1), 1) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        E_tv=[[ca.DM(2*(self.N+1),self.N*2) for j in range(self.N_modes[k])] for k in range(self.N_TV)]

        u_tvs=self.u_tvs

        for k in range(self.N_TV):
            E=ca.diag(self.tv_n_std[k])
            for j in range(self.N_modes[k]):
                for t in range(self.N+1):
                    if t==0:
                        T_tv[k][j][:2,:]=ca.DM.eye(2)
                    else:
                        T_tv[k][j][t*2:(t+1)*2,:]=self.Atv@T_tv[k][j][(t-1)*2:t*2,:]
                        TB_tv[k][j][t*2:(t+1)*2,:]=self.Atv@TB_tv[k][j][(t-1)*2:t*2,:]
                        TB_tv[k][j][t*2:(t+1)*2,t-1:t]=self.Btv
                        E_tv[k][j][t*2:(t+1)*2,:]=self.Atv@E_tv[k][j][(t-1)*2:t*2,:]    
                        E_tv[k][j][t*2:(t+1)*2,(t-1)*2:t*2]=E

                c_tv[k][j]=TB_tv[k][j]@u_tvs[k][j]             

        return T_tv, c_tv, E_tv


    def _get_LTV_EV_dynamics(self):
        """
        Constructs system matrices such for EV,
        X_t=A_pred@x_{t|t}+B_pred@U_t+E_pred@W_t
        where
        X_t=[x_{t|t}, x_{t+1|t},...,x_{t+N|t}].T, (EV state predictions)
        U_t=[u_{t|t}, u_{t+1|t},...,u_{t+N-1|t}].T, (EV control sequence)
        W_t=[w_{t|t}, w_{t+1|t},...,w_{t+N-1|t}].T,  (EV process noise sequence)
        x_{i|t}= state prediction of kth vehicle at time step i, given current time t
        """ 
            
        E=ca.diag(self.ev_n_std)
                
        A_pred=ca.DM(2*(self.N+1), 2)
        B_pred=ca.DM(2*(self.N+1),self.N)
        E_pred=ca.DM(2*(self.N+1),self.N*2)
        
        A_pred[:2,:]=ca.DM.eye(2)
        
        for t in range(1,self.N+1):
                A_pred[t*2:(t+1)*2,:]=self.A@A_pred[(t-1)*2:t*2,:]
                
                B_pred[t*2:(t+1)*2,:]=self.A@B_pred[(t-1)*2:t*2,:]
                B_pred[t*2:(t+1)*2,t-1]=self.B
                
                E_pred[t*2:(t+1)*2,:]=self.A@E_pred[(t-1)*2:t*2,:]
                E_pred[t*2:(t+1)*2,(t-1)*2:t*2]=E
                
        
        return A_pred,B_pred,E_pred
        
    def _add_constraints_and_cost(self):
        """
        Constructs obstacle avoidance, state-input constraints for Stochastic MPC, based on https://arxiv.org/abs/2109.09792
        """   

        
        [A,B,E]=self._get_LTV_EV_dynamics()
        [T_tv,c_tv,E_tv]=self._get_ATV_TV_dynamics()
        [h,M,K]=self.policy

        self.nom_z_tv=[[T_tv[k][j]@self.z_tv_curr[k]+c_tv[k][j]  for j in range(self.N_modes[k])] for k in range(self.N_TV)]
        
        
        cost = 0
        self.opti.subject_to(self.opti.bounded(self.V_MIN, A[[t*2+1 for t in range(1,self.N+1)],:]@self.z_curr+B[[t*2+1 for t in range(1,self.N+1)],:]@h, self.V_MAX))
        self.opti.subject_to(self.opti.bounded(self.A_MIN, h, self.A_MAX))

        

        
        nom_z=A@self.z_curr+B@h
        nom_s=ca.vec(nom_z.reshape((2,-1))[0,:])
        nom_z_diff=ca.vec(ca.diff(nom_z.reshape((2,-1)),1,1))

        cost+=-1.*self.Q*ca.sum1(nom_s) + 3.5*self.Q*nom_z_diff.T@nom_z_diff# penalizes slow progress (was -2.5, 2)
        cost+=self.R*ca.diff(ca.vertcat(self.u_prev,h),1,0).T@ca.diff(ca.vertcat(self.u_prev,h),1,0) # penalizes large input rates

        
        if self.offline:
            self.lin_ineq_l1 = []
            self.ca_ineq = []
            self.l1_constr=[[[ [] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
            self.ca_constr=[[[ [] for t in range(self.N-1)] for j in range(len(self.mode_map))] for k in range(self.N_TV)]

        for k in range(self.N_TV):
            for j in range(len(self.mode_map)):
                m=self.mode_map[j][k]
                cost+=0.1*ca.trace(K[k][m]@E_tv[k][m][:2*self.N,:]@E_tv[k][m][:2*self.N,:].T@K[k][m].T)

                for t in range(1, self.N):  # position at time-step 1 not a function of decision variables 
                    # Linearised obstacle avoidance constraints

                    # EV position projection onto obstacle ellipse
                    
                    oa_ref=self.pos_tvs[k][m][:,t]
                    oa_ref+=(self.x_pos[:,0]-self.pos_tvs[k][m][:,t])/((self.x_pos[:,0]-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,0]-self.pos_tvs[k][m][:,t]))**(0.5)
                    # Coefficient of random variables in affine chance constraint
                    z=self.tight*((oa_ref-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(ca.horzcat(self.dpos[t-1]@(B[2*t,:]@M+E[2*t,:]),*[self.dpos[t-1]@B[2*t,:]@K[l][self.mode_map[j][l]]@E_tv[l][self.mode_map[j][l]][:2*self.N,:]-int(l==k)*self.dpos_tvs[k][m][t-1]@E_tv[k][m][2*t,:] for l in range(self.N_TV)])))
                    
                    # constant term in affine chance constraint
                    y=(oa_ref-self.pos_tvs[k][m][:,t]).T@self.Qs[k][m][t-1]@(self.x_pos[:,t]-oa_ref+self.dpos[t-1]*(A[2*t,:]@self.z_curr+B[2*t,:]@h-self.z_lin[0,t]))
                    
                    
                    if self.solver=="ipopt":
                        # norm_2(z)<=y
                        if self.offline:
                            self.ca_constr[k][j][t-1]+=[z@z.T<=y**2, 0<=y]
                            self.opti.subject_to(self.ca_constr[k][j][t-1][0])
                            self.opti.subject_to(self.ca_constr[k][j][t-1][1])

                            self.ca_ineq.append(ca.horzcat(z,y))
                              
                            if len(self.l1_constr[k][m][t-1])==0:
                                self.l1_constr[k][m][t-1]+=[K[k][m][t,2*t:2*(t+1)]<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=K[k][m][t,2*t:2*(t+1)]]
                                self.opti.subject_to(self.l1_constr[k][m][t-1][0])
                                self.opti.subject_to(self.l1_constr[k][m][t-1][1])

                                self.lin_ineq_l1+=[self.l1_constr[k][m][t-1][0]]
                                cost+=self.l1_lmbd*ca.sum1(ca.vec(self.gain_l1[k][m][t-1]))
                        else:
                            soc_constr=ca.vertcat(y,y**2-z@z.T)
                            soc_switch=ca.if_else(self.constr_keep[k][j][t-1], soc_constr, ca.DM(*soc_constr.shape), True)
                            self.opti.subject_to(soc_switch>=0)

                    else:
                        # Use for SOCP solvers: SCS and Gurobi
                        soc_constr=ca.soc(z,y)
                        if self.offline:
                            self.ca_ineq.append(ca.horzcat(z,y))
                            if len(self.l1_constr[k][m][t-1])==0:
                                self.l1_constr[k][m][t-1]+=[K[k][m][t,2*t:2*(t+1)]<=self.gain_l1[k][m][t-1], -self.gain_l1[k][m][t-1]<=K[k][m][t,2*t:2*(t+1)]]
                                self.opti.subject_to(self.l1_constr[k][m][t-1][0])
                                self.opti.subject_to(self.l1_constr[k][m][t-1][1])

                                self.lin_ineq_l1+=[self.l1_constr[k][m][t-1][0]]
                                cost+=self.l1_lmbd*ca.sum1(ca.vec(self.gain_l1[k][m][t-1]))
                                self.opti.subject_to(soc_constr>0)
                        else:
                            soc_switch=ca.if_else(self.constr_keep[k][j][t-1], soc_constr, ca.DM(*soc_constr.shape), True)
                            self.opti.subject_to(soc_switch>0)

        self.opti.minimize( cost ) 

        if self.offline:
            self.lin_ineq_constr =[]
            self.lin_ineq_constr+=[A[t*2+1,:]@self.z_curr+B[t*2+1,:]@h<=self.V_MAX  for t in range(1,self.N+1)]
            self.lin_ineq_constr+=[-A[t*2+1,:]@self.z_curr-B[t*2+1,:]@h<=-self.V_MIN  for t in range(1,self.N+1)]
            self.lin_ineq_constr+=[h[t]<=self.A_MAX for t in range(self.N)]
            self.lin_ineq_constr+=[-h[t]<=-self.A_MIN for t in range(self.N)]
            self.f_l_i_c = ca.Function("lin_ineq", [self.vars_pol,self.params], self.lin_ineq_constr)

            # F\theta < =f
            self.F, self.f = ca.jacobian(ca.vertcat(*self.f_l_i_c(self.vars_pol,self.params)),self.vars_pol), ca.vertcat(*self.f_l_i_c(ca.DM(*self.vars_pol.shape),self.params))
            
            self.f_l_i_l1 =ca.Function("l1_ineq", [self.vars_pol,self.vars_epi],self.lin_ineq_l1)
            # L\theta <= psi
            self.L = ca.jacobian(ca.vertcat(*self.f_l_i_l1(self.vars_pol,self.vars_epi)), self.vars_pol)
        
            self.f_ca_i = ca.Function("ca_ineq", [self.vars_pol, self.params], self.ca_ineq)

            # C\theta + c \in K_1 x K_2 x .................
            self.C =  (ca.jacobian(ca_constr, self.vars_pol) for ca_constr in self.f_ca_i(self.vars_pol, self.params))
            self.c = self.f_ca_i(ca.DM(*self.vars_pol.shape), self.params)
            
            self.f_cost = ca.Function("cost", [self.vars_pol, self.vars_epi, self.params], [cost])

            self.Q, self.p = ca.hessian(self.f_cost(self.vars_pol, self.vars_epi,self.params), self.vars_pol)
            self.d         = self.f_cost(ca.DM(*self.vars_pol.shape),ca.DM(*self.vars_epi.shape),self.params)

    def solve(self,first_solve=False):
        try:            
            sol = self.opti.solve()
            # Collect Optimal solution.
            u_control  = sol.value(self.policy[0][0])
            h_opt      = sol.value(self.policy[0]).squeeze()
            M_opt      = sol.value(self.policy[1])
            K_opt      = [[sol.value(self.policy[2][k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)]
            nom_z_tv   = [[sol.value(self.nom_z_tv[k][j]) for j in range(self.N_modes[k])] for k in range(self.N_TV)] 

            if self.offline and not first_solve:
                self.vars_ws , self.vars_epi_ws = sol.value(self.vars_pol), sol.value(self.vars_epi)

            if self.offline and self.solver=='ipopt':
                l1_duals=[[[[sol.value(self.opti.dual(self.l1_constr[k][j][t][0]))] for t in range(self.N-1)] for j in range(self.N_modes[k])] for k in range(self.N_TV)]
                ca_duals=[[[[sol.value(self.opti.dual(self.ca_constr[k][j][t][0]))] for t in range(self.N-1)] for j in range(len(self.mode_map))] for k in range(self.N_TV)]
        
            is_opt     = True
        except:
            if self.offline:
                self.vars_ws , self.vars_epi_ws = None, None  

            infeas_status = ['Infeasible_Problem_Detected'] if self.solver=="ipopt" else ["INF_OR_UNBD"]
            if self.opti.stats()['return_status'] not in infeas_status:
              # Suboptimal solution (e.g. timed out)
                u_control=self.opti.debug.value(self.policy[0][0])
            else:
                u_control  = self.u_backup
            
            is_opt = False

        t_proc_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_proc'))
        t_wall_sum = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall'))

        solve_time = sum(value for key, value in self.opti.stats().items() if key.startswith('t_wall_solver')) if self.solver == 'grb' else t_wall_sum
        
        sol_dict = {}
        sol_dict['u_control']  = u_control  # control input to apply based on solution
        sol_dict['optimal']    = is_opt      # whether the solution is optimal or not
        if is_opt:
                sol_dict['h_opt']=h_opt
                sol_dict['M_opt']=M_opt
                sol_dict['K_opt']=K_opt
                sol_dict['nom_z_tv']=nom_z_tv

                if self.offline and self.solver == 'ipopt':
                    sol_dict['l1_duals']=l1_duals
                    sol_dict['ca_duals']=ca_duals
                
                
        sol_dict['solve_time'] = solve_time  # how long the solver took in seconds
        sol_dict['t_wall_sum'] = t_wall_sum
        sol_dict['t_proc_sum'] = t_proc_sum
        sol_dict['vars'] = self.vars_kept
        sol_dict['constr'] = self.constr_kept


        return sol_dict

    def update(self, update_dict):
        self._update_ev_initial_condition(*[update_dict[key] for key in ['x0', 'u_prev']] )
        self._update_tv_initial_condition(*[update_dict[key] for key in ['o0']] )
        self._update_ev_preds(update_dict['z_lin'], update_dict['x_pos'], update_dict['dpos'])
        self._update_tv_preds(update_dict['u_tvs'], update_dict['o_glob'],
                                update_dict['droutes'], update_dict['Qs'])
        
        if not self.offline:
            if 'l1_duals' in update_dict.keys():
                if 'canon_prob' in update_dict.keys():
                    # print('smpcfr.py: Calling set_canon_form_mats,,,')
                    self._set_canon_form_mats(update_dict['canon_prob'])
                    # print('smpcfr.py: Finished set_canon_form_mats,,,')
                self._update_gain_and_constr_keeps(*[update_dict[key] for key in ['l1_duals', 'ca_duals']])
                # print('smpcfr.py: Finished update_gain_and_constr_keeps,,,')
            else:
                self._update_gain_and_constr_keeps()
        elif self.offline and (self.vars_ws is not None) and (self.vars_epi_ws is not None):
            # print('warm starting'.center(80,'#'))
            self.opti.set_initial(self.vars_pol,self.vars_ws)
            self.opti.set_initial(self.vars_epi,self.vars_epi_ws)

    def _update_ev_initial_condition(self, x0, u_prev):
        self.opti.set_value(self.z_curr, x0)
        self.opti.set_value(self.u_prev, u_prev)

        self.u_backup=self.A_MIN
                  
    def _update_tv_initial_condition(self, x_tv0):
        for k in range(self.N_TV):
            self.opti.set_value(self.z_tv_curr[k], x_tv0[k])

    def _update_ev_preds(self, z_lin, x_pos, dpos):
        
        self.opti.set_value(self.z_lin, z_lin)
        self.opti.set_value(self.x_pos, x_pos)
        for  t in range(self.N):
            self.opti.set_value(self.dpos[t],dpos[t])
    
    def _update_tv_preds(self, u_tvs, pos_tvs, dpos_tvs, Qs):

        for k in range(self.N_TV):
            for j in range(self.N_modes[k]):
                self.opti.set_value(self.pos_tvs[k][j], pos_tvs[k][j])
                self.opti.set_value(self.u_tvs[k][j], u_tvs[k][j].reshape((-1,1)))
                for  t in range(self.N):
                    self.opti.set_value(self.dpos_tvs[k][j][t],dpos_tvs[k][j][t])
                    self.opti.set_value(self.Qs[k][j][t],Qs[k][j][t])


    def _set_canon_form_mats(self, canon_prob):
        self.Q, self.p, self.d = canon_prob["Q"],canon_prob["p"], canon_prob["d"]
        self.F, self.f = canon_prob["F"], canon_prob["f"]
        self.L = canon_prob["L"]
        self.C, self.c = canon_prob["C"], canon_prob["c"]

    
    def _get_canon_form_mats(self):
        '''
        Returns dictionary containing canonical form of the problem
        '''
        canon_prob ={}
        canon_prob.update({"Q":self.opti.value(self.Q), "p":self.opti.value(self.p), "d":self.opti.value(self.d), 
                           "F":self.opti.value(self.F), "f":self.opti.value(self.f), "L":self.opti.value(self.L), 
                        "C":[self.opti.value(ca.vertcat(C)) for C in self.C], "c":[self.opti.value(ca.vertcat(c)) for c in self.c]}) 
        return canon_prob


    def _update_gain_and_constr_keeps(self, l1_duals=None, ca_duals=None):
        '''
        l1_dual (tertiary pred) classes:
            0: g in int(K_s) 
            1: g == 0
            2: g == l1_dual_lmbd
        '''

        constr_kept=0
        vars_kept=0
        vars_seen=set()

        for k in range(self.N_TV):
            for m in range(len(self.mode_map)):
                j=self.mode_map[m][k]
                for  t in range(self.N-1):
                    
                    if l1_duals is not None:
                        if not (k,j,t) in vars_seen:
                            gain_keep = 1 #Only do constraint screening
                            vars_kept+=gain_keep*2
                            vars_seen.add((k,j,t))
                        constr_keep =int(np.linalg.norm(ca_duals[k][m][t][0])>1e-3)
                        if t <= 1: 
                            constr_keep = 1
                        constr_kept+=constr_keep
                    else:
                        if not (k,j,t) in vars_seen:
                            gain_keep = 1
                            vars_kept+=gain_keep*2
                            vars_seen.add((k,j,t))
                        constr_keep = 1     
                        constr_kept+=constr_keep
                       
                    self.opti.set_value(self.gain_keep[k][j][t],gain_keep)
                    self.opti.set_value(self.constr_keep[k][m][t],constr_keep)

        if l1_duals:
            self.vars_kept = vars_kept
            self.constr_kept = constr_kept
            
        print("vars: ", vars_kept, " constr: ",constr_kept)