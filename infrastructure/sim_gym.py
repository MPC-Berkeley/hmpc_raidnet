import gymnasium as gym
from gymnasium.spaces import Box, Tuple, Discrete, MultiDiscrete, Dict
from infrastructure.sim import Simulator
from infrastructure.vehicle import Vehicle
from infrastructure.smpcfr import SMPC_MMPreds
import numpy as np
import random
from infrastructure.utils import flatten


import matplotlib.pyplot as plt
import matplotlib.transforms as tf
from celluloid import Camera
from IPython.display import HTML
import pdb


class TrafficEnv(gym.Env):
    def __init__(self, reduced_mode=False, N = 10,use_ttc=True, env_mode=1, solver ='ipopt'):
        '''
        Parameters:
        reduced_mode = if true, use 3 TVs instead of 5
        N            = MPC prediction horizon
        use_ttc      = if true, compute ttc for graph encoding 
        env_mode     =  0: use expert pol only, 1: use both expert and hierarchical, 2: evaluation mode (use specified solver for both expert and hiearchical)
        '''
        super(TrafficEnv, self).__init__()
        # self.ev_noise_std=[0.1,0.01]
        # self.tv_noise_std=[0.1, 0.1]

        self.ev_noise_std=[0.01,0.1] if reduced_mode else [0.05, 0.2]
        self.tv_noise_std=[0.1, 0.1]

        # in reduced mode, only 3 (at max) TVs are spawned. One each from W, S, E 
        self.reduced_mode=reduced_mode
        self.N = N #SMPC prediction horizon length
        self.env_mode = env_mode
        self.solver = solver

        ev=Vehicle(role='EV', dt = 0.2 if reduced_mode else 0.5, cl=5, noise_std=self.ev_noise_std)

        if not self.reduced_mode:
            self.max_tvs=5
            vehicles=[Vehicle(role='TV', dt = 0.2 if reduced_mode else 0.5, cl=0, state=np.array([8., 8.], dtype=np.float32), noise_std=self.tv_noise_std)]
            vehicles+=[Vehicle(role='TV', dt = 0.2 if reduced_mode else 0.5, cl=8*(i==0)+4*(i==1)+1*(i==2) +6*(i==3), state=np.array([9.*(1-i%2), 7. + 1.*int(i>1)], dtype=np.float32), noise_std=self.tv_noise_std) for i in range(4)]
        else:
            self.max_tvs=3
            vehicles=[Vehicle(role='TV', dt = 0.2 if reduced_mode else 0.5, cl=0, state=np.array([8., 8.], dtype=np.float32), noise_std=self.tv_noise_std)]
            vehicles+=[Vehicle(role='TV', dt = 0.2 if reduced_mode else 0.5, cl=8*(i==0)+6*(i==1), state=np.array([0., 7.+ 1.*int(i>1)], dtype=np.float32), noise_std=self.tv_noise_std) for i in range(2)]

        
        tv_n_stds=[v.noise_std*np.array([6, 6]) for v in vehicles]
        vehicles.append(ev)
        self.Sim=Simulator(vehicles, reduced_mode=self.reduced_mode,eval_mode=True if self.env_mode==2 else False)

        match self.env_mode:
            case 0:
                self.smpc=SMPC_MMPreds(self.Sim.routes, ev, N=self.N, EV_NOISE_STD=self.ev_noise_std, TV_NOISE_STD=tv_n_stds, offline_mode=True, solver='ipopt',reduced_mode=self.reduced_mode)
                self.smpc_on=SMPC_MMPreds(self.Sim.routes, ev, N=self.N, EV_NOISE_STD=self.ev_noise_std, TV_NOISE_STD=tv_n_stds, offline_mode=True, solver='ipopt',reduced_mode=self.reduced_mode)
            case 1:
                self.smpc=SMPC_MMPreds(self.Sim.routes, ev, N=self.N, EV_NOISE_STD=self.ev_noise_std, TV_NOISE_STD=tv_n_stds, offline_mode=True, reduced_mode=self.reduced_mode)
                self.smpc_on=SMPC_MMPreds(self.Sim.routes, ev, N=self.N, EV_NOISE_STD=self.ev_noise_std, TV_NOISE_STD=tv_n_stds, offline_mode=False, reduced_mode=self.reduced_mode)
            case 2:
                self.smpc=SMPC_MMPreds(self.Sim.routes, ev, N=self.N, EV_NOISE_STD=self.ev_noise_std, TV_NOISE_STD=tv_n_stds, offline_mode=True, solver = self.solver, reduced_mode=self.reduced_mode,eval_mode=True)
                self.smpc_on=SMPC_MMPreds(self.Sim.routes, ev, N=self.N, EV_NOISE_STD=self.ev_noise_std, TV_NOISE_STD=tv_n_stds, offline_mode=False, solver=self.solver, reduced_mode=self.reduced_mode)

        self.Sim.set_MPC_N(self.smpc.N)
        
        
        # Define the action and observation spaces
        # Replace these with appropriate values based on your system
        self.action_space = Dict({ "l1_duals":Tuple([Tuple([Tuple([Box(low=np.zeros(2), high=self.smpc.l1_lmbd*np.ones(2)) for t in range(self.smpc.N-1)]) for j in range(self.smpc.N_modes[k])]) for k in range(self.smpc.N_TV)]),
                                   "ca_duals":Tuple([Tuple([Tuple([Box(low=np.zeros(1), high=1e5*np.ones(1)) for t in range(self.smpc.N-1)]) for j in range(len(self.smpc.mode_map))]) for k in range(self.smpc.N_TV)]) })
        
        self.classes = {0 : "both duals uninformative",
                        1 : "l1 dual is 0 or 10", 
                        2: "over 1/10 ca duals are positive",
                        3 : "both informative"}
        self.info={}
        if use_ttc:
            if not self.reduced_mode:
                self.observation_space = Dict({ "x0"      : Box(low=np.array([0.,-.1]), high=np.array([110., 10.]), dtype=np.float32),
                                                "u_prev"  : Box(low=-2.0, high=2.0),
                                                "ttc"     : Box(low=np.array([0., 0., 0., 0., 0., 0.]), high=np.array([10., 10., 10., 10., 10., 10.])),
                                                "ev_route": Discrete(2),
                                                "o0"      : Tuple((Box(low=np.array([-15.,-.1]), high=np.array([111., 10.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([62., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([62., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([111., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([111., 8.])))),
                                                "mmpreds" : MultiDiscrete([4,3,3,5,5])}) # extra mode for dummy vehicles 
            else:
                self.observation_space = Dict({ "x0"      : Box(low=np.array([0.,-.1]), high=np.array([110., 10.]), dtype=np.float32),
                                                "u_prev"  : Box(low=-2.0, high=2.0),
                                                "ttc"     : Box(low=np.array([0., 0., 0., 0. ]), high=np.array([10., 10., 10., 10.])),
                                                "ev_route": Discrete(2),
                                                "o0"      : Tuple((Box(low=np.array([-15.,-.1]), high=np.array([111., 10.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([62., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([111., 8.])))),
                                                "mmpreds" : MultiDiscrete([4,3,5])}) # extra mode for dummy vehicles 
        else:
            if not self.reduced_mode:
                self.observation_space = Dict({ "x0"      : Box(low=np.array([0.,-.1]), high=np.array([120., 10.]), dtype=np.float32),
                                                "u_prev"  : Box(low=-2.0, high=2.0),
                                                "ev_route": Discrete(2),
                                                "o0"      : Tuple((Box(low=np.array([-15.,-.1]), high=np.array([111., 10.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([62., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([62., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([111., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([111., 8.])))),
                                                "mmpreds" : MultiDiscrete([4,3,3,5,5])}) # extra mode for dummy vehicles 
            else:
                self.observation_space = Dict({ "x0"      : Box(low=np.array([0.,-.1]), high=np.array([111., 10.]), dtype=np.float32),
                                                "u_prev"  : Box(low=-2.0, high=2.0),
                                                "ev_route": Discrete(2),
                                                "o0"      : Tuple((Box(low=np.array([-15.,-.1]), high=np.array([111., 10.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([62., 8.])),
                                                                    Box(low=np.array([-15.,-.1]), high=np.array([111., 8.])))),
                                                "mmpreds" : MultiDiscrete([4,3,5])}) # extra mode for dummy vehicles 

    def reset(self, seed=None):
        # Reset the simulator and return the initial observation
        if seed:
            random.seed(seed); 
        sample=random.sample

        ev=Vehicle(role='EV', dt = 0.2 if self.reduced_mode else 0.5, cl=sample([0,5],1)[0], noise_std=self.ev_noise_std)
        
        # randomly sample which tvs to spawn out of 5 possibilities
        num_tvs=sample(range(2,self.max_tvs),1)[0]
        # num_tvs=sample(range(3,self.max_tvs+1),1)[0]
        tv_spawns=sample(range(self.max_tvs),num_tvs)

        vehicles=[]
        for i in range(self.max_tvs):
            if i in tv_spawns:
                if i==0:
                    _cl=sample(self.Sim.modes["W"],1)[0]
                    vehicles.append(Vehicle(role='TV', dt = 0.2 if self.reduced_mode else 0.5, cl=_cl, state=np.array([8., 8.], dtype=np.float32), noise_std=self.tv_noise_std))
                elif (i==1 or i==2 and not self.reduced_mode) or (i==1 and self.reduced_mode):
                    _cl=sample(self.Sim.modes["S"],1)[0]
                    vehicles.append(Vehicle(role='TV', dt = 0.2 if self.reduced_mode else 0.5, cl=_cl, state=np.array([9*(i%2)*int(not self.reduced_mode), 7.], dtype=np.float32), noise_std=self.tv_noise_std))
                elif (i==3 or i==4 and not self.reduced_mode) or (i==2 and self.reduced_mode):
                    _cl=sample(self.Sim.modes["E"],1)[0]
                    vehicles.append(Vehicle(role='TV', dt = 0.2 if self.reduced_mode else 0.5, cl=_cl, state=np.array([9*(i%2)*int(not self.reduced_mode), 8.], dtype=np.float32), noise_std=self.tv_noise_std))
            else:
                if i==0:
                    _cl=self.Sim.modes["W"][0]
                    vehicles.append(Vehicle(role='dummy', dt = 0.2 if self.reduced_mode else 0.5, cl=_cl, state=np.array([-15., 0.], dtype=np.float32), noise_std=self.tv_noise_std))
                elif (i==1 or i==2 and not self.reduced_mode) or (i==1 and self.reduced_mode):
                    _cl=self.Sim.modes["S"][0]
                    vehicles.append(Vehicle(role='dummy', dt = 0.2 if self.reduced_mode else 0.5, cl=_cl, state=np.array([-15., 0.], dtype=np.float32), noise_std=self.tv_noise_std))
                elif (i==3 or i==4 and not self.reduced_mode) or (i==2 and self.reduced_mode):
                    _cl=self.Sim.modes["E"][0]
                    vehicles.append(Vehicle(role='dummy', dt = 0.2 if self.reduced_mode else 0.5, cl=_cl, state=np.array([-15., 0.], dtype=np.float32), noise_std=self.tv_noise_std))
        vehicles.append(ev)

        self.Sim=Simulator(vehicles, reduced_mode=self.reduced_mode,eval_mode=True if self.env_mode==2 else False)
        
        self.Sim.set_MPC_N(self.smpc.N)

        observation = self.get_observation()  # Define a method to get the observation from the Simulator

        self.info={}
        return observation, self.info
    
    

    def step(self, action=None):
        env_mode = self.env_mode
        # Take a step in the environment based on the given action
        # Update the simulator with the action
        update_dict=self.Sim.get_update_dict()
        info={}
        # done = False
        infeas=False
        infeas_on=False

        match env_mode: #collect_exp_traj mode
            case 0:
                control=None #Means run expert
                self.smpc.update(update_dict)
                sol=self.smpc.solve()
                if sol["optimal"]:
                    info.update({"l1_duals":sol["l1_duals"], "ca_duals":sol["ca_duals"]})
                    dual_class = 0
                    l1_duals_vec = np.fromiter(flatten(info["l1_duals"]),float)
                    ca_duals_vec = np.fromiter(flatten(info["ca_duals"]),float)
                    l1_dual_active = (1-int(np.all(l1_duals_vec<(self.smpc.l1_lmbd-1e-3)*np.ones(l1_duals_vec.shape[0])))) or (1-int(np.all(l1_duals_vec>1e-3*np.ones(l1_duals_vec.shape[0]))))
                    ca_duals_active = np.sum(ca_duals_vec>1e-3*np.ones(ca_duals_vec.shape[0]))/ca_duals_vec.shape[0]

                    if l1_dual_active == 1:
                        if ca_duals_active > 0.05 :
                            dual_class = 3
                        else:
                            dual_class = 1
                    elif ca_duals_active > 0.05 :
                        dual_class = 2
                     

                    info.update({"solve_time":sol["solve_time"], "dual_class": dual_class}) #save the solve time for the expert (full SMPC)
                    self.info=info
                else:
                    print("infeasible!")
                    # done = True
                    infeas=True

                control=sol["u_control"]
            
            case 1: #dagger mode
                control=None #Means run expert
                self.smpc.update(update_dict)
                sol=self.smpc.solve()
                if sol["optimal"]:
                    info.update({"l1_duals":sol["l1_duals"], "ca_duals":sol["ca_duals"]})
                    dual_class = 0
                    l1_duals_vec = np.fromiter(flatten(info["l1_duals"]),float)
                    ca_duals_vec = np.fromiter(flatten(info["ca_duals"]),float)
                    l1_dual_active = (1-int(np.all(l1_duals_vec<(self.smpc.l1_lmbd-1e-3)*np.ones(l1_duals_vec.shape[0])))) or (1-int(np.all(l1_duals_vec>1e-3*np.ones(l1_duals_vec.shape[0]))))
                    ca_duals_active = np.sum(ca_duals_vec>1e-3*np.ones(ca_duals_vec.shape[0]))/ca_duals_vec.shape[0] #percentage of ca_duals active

                    if l1_dual_active == 1:
                        if ca_duals_active > 0.05 :
                            dual_class = 3
                        else:
                            dual_class = 1
                    elif ca_duals_active > 0.05 :
                        dual_class = 2
                     

                    info.update({"solve_time":sol["solve_time"], "dual_class": dual_class}) #save the solve time for the expert (full SMPC)
                    self.info=info
                else:
                    print("Expert Infeasible!")
                    infeas=True

                control=sol["u_control"]

                if action is not None and not infeas:
                    update_dict.update({"l1_duals": action[0], "ca_duals":action[1], "canon_prob": self.smpc._get_canon_form_mats()})
                    self.smpc_on.update(update_dict)
                    if not infeas:
                        sol_on=self.smpc_on.solve()
                        if sol_on["optimal"]:
                            info.update({"vars_kept":sol_on["vars"], "const_kept":sol_on["constr"]})
                            info.update({"solve_time":sol_on["solve_time"]}) #save the solve time for the reduced SMPC
                            self.info=info                
                        else:
                            print("Reduced SMPC Infeasible!")
                            infeas_on = True
                        control = sol_on["u_control"] 
            case 2: #eval mode
                if action is None:
                    control = None
                    #Running the expert for collecting expert data
                    self.smpc.update(update_dict)
                    sol=self.smpc.solve()
                    if sol["optimal"]:
                        info.update({"l1_duals":[], "ca_duals":[]})
                        info.update({"solve_time":sol["solve_time"], "dual_class":0}) #save the solve time for the reduced SMPC
                        self.info=info
                        info.update({"t_wall_sum":sol["t_wall_sum"], "t_proc_sum":sol["t_proc_sum"]})
                    else:
                        print("infeasible!")
                        infeas=True
                    control=sol["u_control"]
                else:
                    control = None
                    update_dict.update({"l1_duals": action[0], "ca_duals":action[1]})
                    self.smpc_on.update(update_dict)
                    sol_on=self.smpc_on.solve()
                    if sol_on["optimal"]:
                        info.update({"l1_duals":[], "ca_duals":[]}) #Will not be used in evaluation mode
                        info.update({"vars_kept":sol_on["vars"], "const_kept":sol_on["constr"]})
                        info.update({"solve_time":sol_on["solve_time"], "dual_class":0}) #save the solve time for the reduced SMPC
                        info.update({"t_wall_sum":sol_on["t_wall_sum"], "t_proc_sum":sol_on["t_proc_sum"]})
                        info.update({'dual_recovery_time': self.smpc_on.dual_recovery_time if hasattr(self.smpc_on,'dual_recovery_time') else None})
                        self.info=info                
                    else:
                        print("infeasible!")
                        infeas_on = True
                    control = sol_on["u_control"]

        if not infeas and not infeas_on:
                self.Sim.step(control,NN_pred = action if self.env_mode == 2 else None)
        else:
            # self.Sim.step(-self.Sim.ev.a_max)
            self.Sim.step(NN_pred = action if self.env_mode == 2 else None)
        
        # Get the new observation, reward, done flag, and info from the simulator
        observation = self.get_observation()  # Define a method to get the observation from the Simulator
        reward = 0  # Define your reward calculation based on the simulator state

        # if not done:
        done = self.Sim.done()  # Use your done method to check if the episode is done
        
        discard = self.Sim.collision() #checks if any vehicle in the simulator has crashed
        if discard:
            print("Collision!")
            print([(v.cl, v.role) for v in self.Sim.vehicles])
            
        #Logging for evaluation 
        self.info['discard'] = discard
        if env_mode == 1.:
            self.info['infeas'] = infeas
        else:
            self.info['infeas'] = infeas or infeas_on


        return observation, reward, done, False, self.info
    
    def render(self, mode='human'):
        # Implement rendering if needed
        fig, ax= plt.subplots()
        camera = Camera(fig)

        for  i in range(self.Sim.t):
            self.Sim.draw_intersection(ax, i)
            camera.snap()

        animation = camera.animate(repeat = True, repeat_delay = 500)
        # HTML(animation.to_html5_video())
    
        return animation
        
    def close(self):
        # Implement any necessary cleanup
        pass
    
    def get_observation(self):
        # Define a method to extract the observation from the simulator
        obs={}
        t=self.Sim.t
        obs.update({"x0": self.Sim.ev.traj[:,t], "u_prev":self.Sim.ev.u[t-1], "ttc": np.array(self.Sim.get_ittc()),
                    "ev_route": self.Sim.ev.cl % 2, 
                    "o0": [np.array(self.Sim.vehicles[i].traj[:,t], dtype=np.float32) for i in range(len(self.Sim.vehicles)-1)]})
        mm_preds=[]

        for i in range(len(self.Sim.vehicles)-1):
            v=self.Sim.vehicles[i]
            if v.role=="dummy":
                mm_preds.append(0)
            else:
                if self.Sim.sources[v.cl]=="W":
                    if v.traj[0,t]<=51.+3.:
                        mm_preds.append(1)
                    else:
                        mm_preds.append(2 if v.cl==0 else 3)
                elif self.Sim.sources[v.cl]=="S":
                    if v.traj[0,t]<=self.Sim.routes_pose[4][-1,-1]+3:
                        mm_preds.append(1)
                    else:
                        if v.cl==4:
                            mm_preds.append(1)
                        else:
                            mm_preds.append(2)
                elif self.Sim.sources[v.cl]=="E":
                    if v.traj[0,t]<=self.Sim.routes_pose[2][-1,-1]+3:
                        mm_preds.append(1)
                    else:
                        if v.cl==2:
                            mm_preds.append(1)
                        elif v.cl==1:
                            mm_preds.append(2)
                        elif v.cl==6:
                            mm_preds.append(3)
                        else:
                            mm_preds.append(4)
        
        obs["mmpreds"]=np.array(mm_preds, dtype=np.float32)
        return obs

                        
                




