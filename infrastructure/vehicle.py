
import numpy as np
from random import sample
import copy
import polytope as pc


class Vehicle():
    def __init__(self,
                 role="TV",
                 cl=1,
                 dt=0.2,
                 state=np.array([0., 8.]),
                 noise_std=[0.01, 0.01],
                 T_max=1000,
                 A_max=5
                 ):
        self.role=role
        self.cl=cl
        self.dt=dt
        self.noise_std=noise_std
        self.a_max=A_max
        self.t_h=1.
        self.s_0=8.
        self.T_max=T_max
        self.traj=np.zeros((2, T_max+1))
        self.traj_glob=np.zeros((3,T_max+1))
        self.u=np.zeros(T_max)
        self.traj[:,0]=state
        self.t=0

        self.A=np.array([[1., self.dt], [0., 1.]])
        self.B=np.array([0.5*self.dt**2,self.dt])
        self.veh_dims= np.array([2.9, 1.7])
        self.S=np.diag(self.veh_dims**(-1.0))
        self.vB=pc.box2poly([[-self.veh_dims[0], self.veh_dims[0]],[-self.veh_dims[1],self.veh_dims[1]]])


        

    def step(self, control):
        rng=np.random.default_rng(self.t)
        next_state=self.A@self.traj[:,self.t]+self.B*control\
                              +int(self.role=="TV")*np.array([rng.normal(0,self.noise_std[0]), rng.normal(0,self.noise_std[1])])
        self.traj[:,self.t+1]=np.array(next_state).squeeze()
        self.traj[1,self.t+1]=max(self.traj[1,self.t+1], -0.05)
        self.u[self.t]=control
        self.t+=1

    def strip_list(self,x):
        '''
        Assumes x is a scalar value wrapped in a list
        '''
        if isinstance(x,list):
            return self.strip_list(x[0])
        else:
            return x
        
    def clip_vel_acc(self, state, a,verbose=False):
        curr_vel = state[1]
        next_vel = max(self.A[1,:]@state+self.B[1]*self.strip_list(a), -0.1)
        eff_a    = (next_vel-curr_vel)/self.dt
        if verbose:
            print(f'v_cur: {curr_vel}, v_next: {next_vel}, a: {a}, a_eff: {eff_a}')
        return self.strip_list(eff_a)

    
    def get_next(self,state, control): 
        return self.A@state+self.B*control
        
    def reset_vehicle(self, init, cl):
        # self.traj=np.zeros((2, self.T_max+1))
        self.traj[:,self.t]=copy.copy(init)
        self.cl=copy.copy(cl)
        


    def idm(self, v_des=8, dv= 0, ds=1e5):
        
        if v_des<0.1 or ds <=0.5:
            a_idm=-self.a_max
        else:
            a_idm=0.6*self.a_max*(1-(self.traj[1,self.t]/v_des)**4 -1.5*((self.s_0+np.abs(self.traj[1,self.t])*(self.t_h+dv/2/self.a_max))/ds**2))
        return np.clip(a_idm,-self.a_max, self.a_max)