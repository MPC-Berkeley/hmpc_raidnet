import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.transforms as tf
import pdb
from random import sample
import copy
import polytope as pc
from infrastructure.utils import unflatten_duals
from pytope import Polytope
from itertools import product

class Simulator():
    def __init__(self,
                vehicles, 
                T_FINAL     = 1000,
                reduced_mode=False,
                viz_preds=True,
                eval_mode=False
                ):
        
        self.reduced_mode=reduced_mode
        self._make_lanes() 
        self.vehicles=vehicles

        self.N_TV=0
        self.tvs=[]
        self.dummys=[]
        self.tv_idxs=[]
        self.mm_preds=[]
        self.ev_sols =[]
        self.viz_preds=viz_preds
        self.eval_mode = eval_mode
        if not reduced_mode:
            self.N_modes=[2,2,2,4,4]
        else:
            # Only 3 TVs
            self.N_modes=[2,2,4]

        if eval_mode:
            self.NN_preds = []
            # Maps a mode, say 10, to the modes of the TVs, like (0,1,1,3,3)
            self.mode_map = dict(enumerate(product(*[range(self.N_modes[k]) for k in range(len(self.N_modes))])))
            self.inv_mode_map = [[[m for m in range(len(self.mode_map)) if self.mode_map[m][k]==j] for j in range(self.N_modes[k])] for k in range(len(self.N_modes))]

        for i,v in enumerate(self.vehicles):
            if v.role=="TV":
                self.N_TV+=1
                self.tvs.append(v)
                self.tv_idxs.append(i)
            elif v.role=="dummy":
                self.dummys.append(v)
            else:
                self.ev=v
        self.t=0
        self.T=T_FINAL
    
    def _get_idm_params(self, v, cl, v_, cl_, verbose = False):
        '''
        Our IDM-based Interaction Engine
        '''
        v_des=self.routes[cl](v[0]+2.0)[-1]

        dv   =0.0 
        ds   =1e5

        psi=self.routes[cl](v[0])[2]
            
        for i, (vh, clh) in enumerate(zip(v_,cl_)):

            if clh in self.modes[self.sources[cl]]:

                
                vh_s=vh[0]
                vh_psi=self.routes[clh](vh[0])[2]

                if vh_s-v[0]>=0. and np.abs(np.cos(psi-vh_psi))>=0.3:
                    ds=max(vh_s-v[0]-6.0,0.01)
                    vh_psi=self.routes[clh](vh[0])[2]
                    dv=v[1]-vh[1]*np.cos(psi-vh_psi)+6.
                    if verbose:
                        print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #1: Keep going')
                    break
            
            if self.sinks[cl]==self.sinks[clh]:
                
                vh_pos=self.routes[clh](vh[0]-.0)[:2].reshape((-1,1))
                vh_s=self._g2f(vh_pos,cl)
                v_s_on_vh = self._g2f(self.routes[cl](v[0]-.0)[:2].reshape((-1,1)),clh)
                vh_psi=self.routes[clh](vh[0])[2]

                if np.abs(np.sin(float(2*psi)))<=1e-3 and vh_s-v[0]>=0. and vh_s-v[0]<=30. and self._check_out_inter(cl,v[0]):
                    if (vh[1]>=1 and vh_s-v[0]<=18) and (1<=(v_s_on_vh - vh[0]) <= 25):
                        ds=0.01
                        dv=8
                        if verbose:
                            print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #2: Yield')
                    else:
                        ds=max(vh_s-v[0]-6.,0.01) 
                        dv=v[1]-vh[1]*np.cos(psi-vh_psi)
                        dv+= 5. if 2*float(psi)%np.pi==0 else -5.

                        if verbose:
                            print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #3: Keep Going. Vehicle ahead')
                            
                        
                        if vh[1]<0 and np.abs(np.sin(2*psi))>=0.001: #if vh is hestitating, just go
                            ds = 1e5
                            dv = 0.0

                elif vh_s-v[0]-6.0 < 0. and vh_s - v[0]>=.0 and np.abs(np.cos(psi-vh_psi))>=0.3:
                    ds=max(vh_s-v[0]-6.,0.01) 
                    dv=v[1]-vh[1]*np.cos(psi-vh_psi)
                    dv+= 5. if 2*float(psi)%np.pi==0 else -5.

                    if verbose:
                        print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #4: Keep Going. Vehicle ahead')
                else:
                    if verbose:
                        print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Keep Going')
            else:

                vh_pos=self.routes[clh](vh[0]-.0)[:2].reshape((-1,1))
                vh_s=self._g2f(vh_pos,cl)
                vh_psi=self.routes[clh](vh[0]+0.)[2]

                p2p1=-self.routes[cl](v[0]+0.)[:2].reshape((-1,1))+vh_pos
                d1 = self.droutes[cl](v[0]+0.)[:2].reshape((-1,1))*20.      #30 m lookahead
                d2 = self.droutes[clh](vh[0]-.0)[:2].reshape((-1,1))*20.
                d1cd2, pcd2, pcd1  =ca.det(ca.horzcat(d1,d2)), ca.det(ca.horzcat(p2p1, d2)), ca.det(ca.horzcat(p2p1, d1))

                if int(d1cd2 > 0 or d1cd2 < 0)==1:
                    t, u = pcd2/d1cd2, pcd1/d1cd2
                    if bool(0.<=t<=1.) and bool(0.<=u<=1.):
                        if np.abs(np.sin(float(psi)))<=1e-3 and vh_s-v[0]>=0.5 and np.linalg.norm(p2p1)<=30. and self._check_out_inter(cl,v[0]):
                            # if vh[1]>=2.:
                            ds=0.01
                            dv=8.
                            if verbose:
                                print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #5: Yield')
                            
                        elif vh_s-v[0]-6.0<0. and vh_s-v[0]>=0.5 and np.linalg.norm(p2p1)<=20. and not self._check_out_inter(clh,vh[0]):
                            ds=max(vh_s-v[0]-6.0,0.01)
                            dv=v[1]-vh[1]*np.cos(psi-vh_psi)
                            dv+=6. if np.abs(np.sin(float(psi)))<=1e-3 else 2.
                            if verbose:
                                    print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #7: Keep Going. Vehicle ahead ')

                            if vh[1]<0 and np.abs(np.sin(psi))>=0.001: #if vh is hestitating, just go
                                ds = 1e5
                                dv = 0.0
                        else:
                            if verbose:
                                print(f'Vehicle #{i} (EV: {i==len(cl_)-1}):=   Branch #8: Keep Going')       

        if verbose:
            print(f'v_des, dv, ds: {v_des}, {dv}, {ds}')   
            print('++++++++++++++++++')      
        return v_des, dv, ds
            
            
    def done(self):
        #Reached the pre-defined time = limit of the simulator
        done = self.t==self.T or self.routes_pose[self.ev.cl][-1,-1]-self.ev.traj[0,self.ev.t]<=0.1

        if not done:
            self._TV_gen()
        else:
            print(f"EV reached {self.ev.traj[0,self.ev.t]}")

        return done
    
    def collision(self):
        #A collision occured during simulation
        return self._check_collision()
    
    def set_MPC_N(self, N):
        self.N=N
    
    def _TV_gen(self):
        #checks if TVs have reached destination and respawns TVs accordingly
        for i,v in enumerate(self.tvs):
            if self.routes_pose[v.cl][-1,-1]-v.traj[0,v.t]<=0.01:
                print("Resetting TV: {}".format(i+1))
                print("Reached {}".format(v.traj[0,v.t]))
                new_cl=sample(self.modes[self.sources[v.cl]],1)[0]


                if v.cl!=2 and v.cl!=4:
                    next_init_close=False
                    for vh in self.tvs:
                        if vh!=v and self.sources[v.cl]==self.sources[vh.cl] and np.abs(vh.traj[0,vh.t])<=6.:
                            next_init_close=True

                    init=np.array([-2.0,7.0]) if next_init_close else np.array([6.,6.0])
                else:
                    init=copy.copy(v.traj[:,v.t])
                v.reset_vehicle(init, new_cl)
    
    def _check_collision(self):
        ev_S=Polytope(self.ev.vB.A, self.ev.vB.b)
        psi=self.routes[self.ev.cl](self.ev.traj[0,self.ev.t])[-1]
        Rev=np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi), np.cos(psi)]]).squeeze()
        ev_S=Rev*ev_S+self.routes[self.ev.cl](self.ev.traj[0,self.ev.t])[:2]
        ev_S=pc.Polytope(ev_S.A, ev_S.b)
        for i, v in enumerate(self.tvs):
            tv_S=Polytope(v.vB.A, v.vB.b)
            psi=self.routes[v.cl](v.traj[0,v.t])[-1]
            Rtv=np.array([[np.cos(psi), -np.sin(psi)],[np.sin(psi), np.cos(psi)]]).squeeze()
            tv_S=Rtv*tv_S+self.routes[v.cl](v.traj[0,v.t])[:2]
            tv_S=pc.Polytope(tv_S.A, tv_S.b)
            tv_ev= tv_S.intersect(ev_S)
            if not pc.is_empty(tv_ev):
                print(f"EV Collided with TV {self.tv_idxs[i]} ({self.vehicles[self.tv_idxs[i]].cl}) at position {self.ev.traj[0, self.ev.t]}")
                return True
        
        return False



    def step(self, u_ev=None, verbose=False,NN_pred=None):

        for ind, v in enumerate(self.vehicles):
            if v != self.ev:
                v.traj_glob[:,v.t]=np.array(self.routes[v.cl](v.traj[0,v.t])[:3]).squeeze()
                if v.role!="dummy":
                    if verbose:
                        print(f'TV{ind+1}')
                    idx_=set(self.tv_idxs)-set([ind])
                    v_ =[self.vehicles[k].traj[:,v.t] for k in idx_] + [self.ev.traj[:,v.t]]
                    cl_=[self.vehicles[k].cl for k in idx_] + [self.ev.cl]
                    v_des, dv, ds= self._get_idm_params(v.traj[:,v.t], v.cl, v_, cl_, verbose)
                    v.step(v.clip_vel_acc(v.traj[:,v.t],v.idm(v_des, dv, ds)))
                else:
                    v.step(0.)

        self.ev.traj_glob[:,self.ev.t]=np.array(self.routes[self.ev.cl](self.ev.traj[0,self.ev.t])[:3]).squeeze()
        if not u_ev:
            v_ =[self.vehicles[k].traj[:,v.t] for k in self.tv_idxs]
            cl_=[v.cl for v in self.tvs]
            v_des, dv, ds= self._get_idm_params(self.ev.traj[:,self.ev.t], self.ev.cl, v_, cl_,verbose)
            self.ev.step(self.ev.clip_vel_acc(self.ev.traj[:,self.ev.t],self.ev.idm(v_des, dv, ds)))
        else:
            self.ev.step(u_ev)
        self.t+=1

        if self.eval_mode:
            self.NN_preds.append(NN_pred)

    
    def get_update_dict(self, u_opt=None):

        z_lin, x_pos, dpos, mm_o_glob, mm_u_tvs, mm_routes, mm_droutes, mm_Qs =self._get_preds(u_opt)
        u_prev=self.ev.u[self.ev.t-1] if self.ev.t>0 else 0.

        update_dict={'x0': self.ev.traj[:,self.ev.t], 'u_prev': u_prev ,
                     'o0': [v.traj[:,v.t] for v in self.vehicles if v!=self.ev], 'o_glob': mm_o_glob, 
                     'routes': mm_routes, 'droutes': mm_droutes, 'Qs' : mm_Qs,
                     'z_lin': z_lin, 'x_pos':x_pos,  'dpos': dpos, 'u_tvs': mm_u_tvs }
        
        self.mm_preds.append(mm_o_glob)

        self.ev_sols.append(x_pos)

        return update_dict
    
    def get_ittc(self):
        self.ittc=[]

        for ind, v in enumerate(self.vehicles):
            if v != self.ev:
                if v.role!="dummy":
                    idx_=set(self.tv_idxs)-set([ind])
                    v_ =[self.vehicles[k].traj[:,v.t] for k in idx_]
                    cl_=[self.vehicles[k].cl for k in idx_]
                    v_des, dv, ds= self._get_idm_params(v.traj[:,v.t], v.cl, v_, cl_)
                    self.ittc.append(float(np.clip(dv/ds,0.05, 10.)))
                else:
                    self.ittc.append(0.05)

        self.ittc=[0.05]+self.ittc
        return self.ittc
    
    def _check_out_inter(self,cl,s):
        return (self.sources[cl]=="E" and (s <= self.routes_pose[2][-1,-1]+0.5))\
                 or (self.sources[cl]=="S" and (s <= self.routes_pose[4][-1,-1]+0.5))\
                 or (self.sources[cl]=="W" and (s <= 51.+0.5))
        
    def _get_preds(self, u_opt):
        '''
        Getting EV predictions from previous MPC solution.
        This is used for linearizing the collision avoidance constraints
        '''
        N=self.N
        
        
        x=self.ev.traj[:,self.ev.t].reshape((-1,1))+np.zeros((2,N+1))
    
        x_glob=self.routes[self.ev.cl](x[0,0])[:2].reshape((-1,1))+np.zeros((2, N+1))
        dx_glob=[ca.DM(2,1) for _ in range(N)]
        o=[v.traj[:,v.t].reshape((-1,1))+np.zeros((2,self.N+1)) for v in self.vehicles if v!=self.ev]
        o_glob=[self.routes[v.cl](v.traj[0,v.t])[:2].reshape((-1,1))+np.zeros((2,N+1)) for v in self.vehicles if v!=self.ev]
        u_tvs=[np.zeros((1,N)) for v in self.vehicles if v!=self.ev]
        tv_list=self.tv_idxs
        do_glob = [[ca.DM(2,1) for _ in range(N)] for v in self.vehicles if v!=self.ev]
        Qs = [[np.identity(2) for _ in range(N)] for v in self.vehicles if v!=self.ev]
        iSev=np.linalg.inv(self.ev.S)
        iSev[-1,-1]+=0.3
        Sev=np.linalg.inv(iSev)

        
        
        for t in range(N):
            if u_opt is None:
                v_ =[o[i][:,t] for i in tv_list]
                cl_=[v.cl for v in self.tvs]
                v_des, dv, ds= self._get_idm_params(x[:,t], self.ev.cl, v_, cl_)
                a=self.ev.clip_vel_acc(x[:,t], self.ev.idm(v_des, dv, ds))
                
            else:
                a=u_opt[t]
                
   
            x[:,t+1]=self.ev.get_next(x[:,t+1], a)
            x_glob[:,t+1]=self.routes[self.ev.cl](x[0,t+1])[:2]
            dx_glob[t]=self.droutes[self.ev.cl](x[0,t+1])[:2]
            psi= self.routes[self.ev.cl](x[0,t+1])[2]
            Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
            
            for i in tv_list:
                idx_=set(tv_list)-set([i])
                v_ =[o[k][:,t] for k in idx_] + [x[:,t]]
   
                cl_=[self.vehicles[k].cl for k in idx_] + [self.ev.cl]

                v_des, dv, ds= self._get_idm_params(o[i][:,t], self.vehicles[i].cl, v_, cl_)
                a=self.vehicles[i].clip_vel_acc(o[i][:,t],self.vehicles[i].idm(v_des, dv, ds))
                u_tvs[i][0,t]=a
                o[i][:,t+1]=self.vehicles[i].get_next(o[i][:,t], a)
                o_glob[i][:,t+1]=self.routes[self.vehicles[i].cl](o[i][0,t+1])[:2]
                do_glob[i][t]=self.droutes[self.vehicles[i].cl](o[i][0,t+1])[:2]
                psi=self.routes[self.vehicles[i].cl](o[i][0,t+1])[2]
                Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T

                mat=Rev@iSev@Rtv.T@self.vehicles[i].S@self.vehicles[i].S@Rtv@iSev@Rev.T
                E, V =np.linalg.eigh(mat)
                S=np.diag((E**(-0.5)+1.0)**(-2))
                Qs[i][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)
                
        mm_o      = [[copy.deepcopy(o[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.vehicles) if v!=self.ev]
        mm_o_glob = [[copy.deepcopy(o_glob[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.vehicles) if v!=self.ev]
        mm_u_tvs  = [[copy.deepcopy(u_tvs[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.vehicles) if v!=self.ev]
        mm_Qs     = [[copy.deepcopy(Qs[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.vehicles) if v!=self.ev]
        mm_routes = [[copy.copy(self.routes[v.cl]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.vehicles) if v!=self.ev]
        mm_droutes = [[copy.deepcopy(do_glob[i]) for _ in range(self.n_modes[i])] for i,v in enumerate(self.vehicles) if v!=self.ev]

        for i in tv_list:
            modes=set(self.modes[self.sources[self.vehicles[i].cl]])-set([self.vehicles[i].cl])
            
            for t in range(N):
                psi= self.routes[self.ev.cl](x[0,t+1])[2]
                Rev=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                
                if  (self.sources[self.vehicles[i].cl]=="E" and self.vehicles[i].traj[0,self.t]<=self.routes_pose[2][-1,-1]+3.)\
                 or (self.sources[self.vehicles[i].cl]=="S" and self.vehicles[i].traj[0,self.t]<=self.routes_pose[4][-1,-1]+3.)\
                 or (self.sources[self.vehicles[i].cl]=="W" and self.vehicles[i].traj[0,self.t]<=51.+3.):
                    for j in modes:
                        n=self.modes[self.sources[self.vehicles[i].cl]].index(j)
                        
                        if t==0:
                            mm_routes[i][n]=self.routes[j]
                        idx_=set(tv_list)-set([i])
                        v_ =[o[k][:,t] for k in idx_]+[x[:,t]]
                        cl_=[self.vehicles[k].cl for k in idx_]+[self.ev.cl]

                        v_des, dv, ds= self._get_idm_params(mm_o[i][n][:,t], j, v_, cl_)
                        a=self.vehicles[i].clip_vel_acc(mm_o[i][n][:,t],self.vehicles[i].idm(v_des, dv, ds))

                        mm_o[i][n][:,t+1]=self.vehicles[i].get_next(mm_o[i][n][:,t], a)
                        mm_o_glob[i][n][:,t+1]=self.routes[j](mm_o[i][n][0,t+1])[:2]
                        mm_droutes[i][n][t]=self.droutes[j](mm_o[i][n][0,t+1])[:2]
                        mm_u_tvs[i][n][0,t]=a
                        psi=self.routes[j](mm_o[i][n][0,t+1])[2]
                        Rtv=np.array([[np.cos(psi), np.sin(psi)],[-np.sin(psi), np.cos(psi)]]).squeeze().T
                        mat=Rev@iSev@Rtv.T@self.vehicles[i].S@self.vehicles[i].S@Rtv@iSev@Rev.T 
                        E, V =np.linalg.eigh(mat)
                        S=np.diag((E**(-0.5)+1.0)**(-2))
                        mm_Qs[i][n][t]=Sev@Rev.T@V@S@V.T@Rev@Sev if t <=4 else (1/5**2)*np.eye(2)

        return x, x_glob, dx_glob, mm_o_glob, mm_u_tvs, mm_routes, mm_droutes, mm_Qs 
        
    
    def _make_lanes(self):
        #                         
        # lane numbering:= 0:W->E, 1:E->W, 2:E->W (slow), 3:S->N, 4:S->N (slow),    (straights)
        #                  5:W->N, 6:E->S,                                          (lefts)
        #                  7:E->N, 8:S->E                                           (rights) 
        self.modes={'E':[1,2,6,7], 'W':[0,5], 'S':[4,8]}
        self.sources={0:'W', 1:'E', 2: 'E', 3:'S', 4:'S', 5:'W', 6:'E', 7:'E', 8:'S'}
        self.sinks  ={0:'E', 1:'W', 2: 'W', 3:'N', 4:'N', 5:'N', 6:'S', 7:'N', 8:'E'}

        if self.reduced_mode:
            self.n_modes  = [2,  2,      4]
        else:
            #                W-, S-, S-, E-, E- 
            self.n_modes  = [2,  2,  2,  4,  4]
        
        def _make_ca_fun(s, x, y, psi, v):

            x_ca= ca.interpolant("f2gx", "linear", [s], x)
            y_ca= ca.interpolant("f2gy", "linear", [s], y)
            psi_ca= ca.interpolant("f2gpsi", "linear", [s], psi)
            v_ca= ca.interpolant("f2gv", "linear", [s], v)
            s_sym=ca.MX.sym("s",1)

            glob_fun=ca.Function("fx",[s_sym], [ca.vertcat(x_ca(s_sym), y_ca(s_sym), psi_ca(s_sym), v_ca(s_sym))])

            return glob_fun
        
        def _make_jac_fun(pos_fun):
            s_sym=ca.MX.sym("s",1)
            pos_jac=ca.jacobian(pos_fun(s_sym), s_sym)
            return ca.Function("pos_jac",[s_sym], [pos_jac])

        self.routes_pose=[]
        self.droutes=[]

        straights_x=[]
        s=np.array([0,110])
        xs=np.array((-50,60))
        ys=np.array((0,0))
        psis=np.array((np.pi, np.pi))
        vs=np.array([7.5, 9.0])
        r_fun=_make_ca_fun(s,xs,ys, 0.*psis, vs)
        straights_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

   
        r_fun=_make_ca_fun(s,xs[::-1],ys+15., psis, vs)
        straights_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))


        s=np.array([0,29])
        xs=np.array((60,31))
        ys=np.array((15,15))
        vs=np.array([7.0,0.0])
        r_fun=_make_ca_fun(s,xs,ys, -psis, vs)
        straights_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        straights_y=[]

        s=np.array([0,80])
        xs=np.array((23.5,23.5))
        ys=np.array((-40,40))
        vs=np.array([7.0,7.0])
        r_fun=_make_ca_fun(s,xs,ys, 0.5*psis, vs)
        straights_y.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        s=np.array([0,32.5])
        xs=np.array((23.5,23.5))
        ys=np.array((-40,-7.5))
        vs=np.array([6.0,0.0])
        r_fun=_make_ca_fun(s,xs,ys, 0.5*psis, vs)
        straights_y.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))
       

        straights = straights_x+straights_y


        lefts_x=[]

        thet=np.linspace(0,np.pi/2)


        x_f= lambda t :  8.5 + 15*np.sin(t)
        y_f= lambda t :  15 - 15*np.cos(t)
        s=np.hstack((np.array([0, 58.5]), 58.501 + 15.*thet, np.array([58.502+15.*np.pi/2, 58.5 + 15.*np.pi/2+25])))
        vs=np.hstack((np.array([8., 6.]), 6. + 0.*thet, np.array([6., 8.])))
        x_l=np.hstack((np.array([-50.,7.5]),x_f(thet), np.array([23.5, 23.5])))
        y_l=np.hstack((np.array([0.,0.]), y_f(thet), np.array([15, 40.])))

        psis=np.hstack((np.array([0.0, 0.]),  thet, 0.5*np.array([np.pi, np.pi])))
        r_fun=_make_ca_fun(s, x_l, y_l, psis, vs)
        lefts_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))


        x_f= lambda t :  31. - 22.5*np.sin(t)
        y_f= lambda t :  -7.5 + 22.5*np.cos(t)
        s=np.hstack((np.array([0, 29.]), 29.001 + 22.5*thet, np.array([29.002+22.5*np.pi/2, 29.0 + 22.5*np.pi/2+32.5])))
        x_l=np.hstack((np.array([60.,31.]), x_f(thet), np.array([8.5, 8.5])))
        y_l=np.hstack((np.array([15.,15.]), y_f(thet), np.array([-7.5, -40.])))
        psis=np.hstack((-np.array([np.pi, np.pi]), -np.pi+thet, -0.5*np.array([np.pi, np.pi])))
        vs=np.hstack((np.array([8, 6.]), 6. + 0.*thet, np.array([6., 7])))
        r_fun=_make_ca_fun(s, x_l, y_l, psis, vs)
        lefts_x.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        lefts=lefts_x

        rights=[] 


        x_f= lambda t :  31. - 7.5*np.sin(t)
        y_f= lambda t :  22.5 - 7.5*np.cos(t)
        s=np.hstack((np.array([0, 29.]), 29.001 + 7.5*thet, np.array([29.002+7.5*np.pi/2, 29.0 + 7.5*np.pi/2+17.5])))
        x_l=np.hstack((np.array([60.,31.]), x_f(thet), np.array([23.5, 23.5])))
        y_l=np.hstack((np.array([15.,15.]), y_f(thet), np.array([22.5, 40.])))
        psis=np.hstack((np.array([np.pi, np.pi]), np.pi-thet, 0.5*np.array([np.pi, np.pi])))
        vs=np.hstack((np.array([6, 3.]), 3. + 0.*thet, np.array([3., 8])))
        r_fun=_make_ca_fun(s, x_l, y_l, psis,vs)
        rights.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))


        x_f= lambda t :  31. - 7.5*np.cos(t)
        y_f= lambda t :  -7.5 + 7.5*np.sin(t)
        s=np.hstack((np.array([0, 32.5]), 32.501 + 7.5*thet, np.array([32.502+7.5*np.pi/2, 32.5 + 7.5*np.pi/2+29.])))
        x_l=np.hstack((np.array([23.5,23.5]), x_f(thet), np.array([31., 60.])))
        y_l=np.hstack((np.array([-40.,-7.5]), y_f(thet), np.array([0., 0.])))
        psis=np.hstack((0.5*np.array([np.pi, np.pi]), thet[::-1], 0.*np.array([np.pi, np.pi])))
        vs=np.hstack((np.array([7, 3.]), 3. + 0.*thet, np.array([3., 7])))
        r_fun=_make_ca_fun(s, x_l, y_l, psis,vs)
        rights.append(r_fun)
        self.droutes.append(_make_jac_fun(r_fun))
        s_r=np.linspace(s[0], s[-1])
        r_p=np.array([r_fun(s_r[i])[:3] for i in range(s_r.shape[0])]).squeeze()
        self.routes_pose.append(np.vstack((r_p.T,s_r.reshape((1,-1)))))

        self.routes=straights+lefts+rights

    def _g2f(self, pos, cl):
        idx=np.argmin(np.linalg.norm(self.routes_pose[cl][:2,:]-pos.reshape((-1,1)), axis=0))
        return self.routes_pose[cl][-1,idx]

    
    def draw_intersection(self, ax, i):
        pred_both = False
        alpha_mm = None
        _tf = lambda x: tf.Affine2D().rotate(x[-1]).translate(x[0], x[1])
        if self.eval_mode :
            if self.NN_preds[i] is not None:
                l1_dual, ca_dual = self.NN_preds[i]
                ca_dual_mw = []
                for k in range(len(ca_dual)):
                    ca_dual_k=[]
                    for j in range(self.N_modes[k]):          
                        ca_duals_j = np.zeros(self.N-1)
                        for t in range(self.N-1):
                            for m in self.inv_mode_map[k][j]:
                                ca_duals_j[t]+=ca_dual[k][m][t]
                        ca_dual_k.append(ca_duals_j)
                    ca_dual_mw.append(ca_dual_k)
            else:
                l1_dual, ca_dual = None, None

        #Map Boundaries and roads
        ax.add_patch(Rectangle((-50, -7.5),110,30,linewidth=1,edgecolor='darkgrey', fc='darkgrey',fill=True, alpha=0.7))
        ax.add_patch(Rectangle((1, -40),30,80,linewidth=1,edgecolor='darkgrey', fc='darkgrey',fill=True, alpha=0.7))
        ax.plot([-50, 1], [22.5, 22.5], color='k', lw=2)
        ax.plot([-50, 1], [-7.5, -7.5], color='k', lw=2)
        ax.plot([31, 60], [22.5, 22.5], color='k', lw=2)
        ax.plot([31, 60], [-7.5, -7.5], color='k', lw=2)
        ax.plot([1, 1], [22.5, 40], color='k', lw=2)
        ax.plot([31, 31], [22.5, 40], color='k', lw=2)
        ax.plot([1, 1], [-7.5, -40], color='k', lw=2)
        ax.plot([31, 31], [-7.5, -40], color='k', lw=2)
        
        for r in self.routes_pose:
            ax.plot(r[0,:], r[1,:], color='w', linewidth= 1.2, linestyle = (0, (5,10)))

        #Drawing Vehicles
        v_color= {"EV": "g", "TV":"b","dummy":"w"}
        v_alpha= {"EV": 1, "TV":0.5,"dummy":0.}
        v_shapes=[]
        if self.viz_preds:
            #Ego predictions
            ax.plot(np.array(self.ev_sols[i][0,:]).squeeze(), np.array(self.ev_sols[i][1,:]).squeeze(), color="g", lw = 2, alpha=0.6 )

        for k, v in enumerate(self.vehicles):
            if v.role!="dummy":
                v_pos=v.traj_glob[:,i]
                # if v.role =="TV":
                #     print(v_pos)
                v_shapes.append(Rectangle((0.-3.0,0.-1.8),6.,3.6,linewidth=1., ec='k' if v.role =="TV" else v_color[v.role], fc=v_color[v.role], alpha = v_alpha[v.role]))
                v_shapes[-1].set_transform(_tf(v_pos)+ax.transData)
                ax.add_patch(v_shapes[-1])

                if self.viz_preds and k in self.tv_idxs:
                    tv_mm_preds=self.mm_preds[i][k]

                    for m, pred in enumerate(tv_mm_preds):
                        #TV predictions 
                        if self.eval_mode and l1_dual is not None:
                            l1_d = l1_dual[k][m][0::2] + l1_dual[k][m][1::2]
                            if pred_both:
                                l1_d = np.ones_like(l1_d)
                            ca_d = ca_dual_mw[k][m]

                            horizon_set = set(range(self.N-1))
                            l1_dual_iac = set(np.argwhere(np.array(l1_d) == 0).flatten())
                            ca_dual_iac = set(np.argwhere(ca_d == 0).flatten())

                            pred_magenta = np.array(pred)[:,list(l1_dual_iac.intersection(ca_dual_iac))]
                            pred_orange  = np.array(pred)[:,list(horizon_set.difference(ca_dual_iac) - horizon_set.difference(l1_dual_iac))]
                            if pred_both:
                                pred_yellow  = np.array(pred)[:,list(horizon_set.difference(l1_dual_iac) - horizon_set.difference(ca_dual_iac))]
                                pred_red     = np.array(pred)[:,list((horizon_set - l1_dual_iac).intersection(horizon_set - ca_dual_iac))]
                                            
                            ax.scatter(np.array(pred_magenta[0,:]).squeeze(), np.array(pred_magenta[1,:]).squeeze(), color="m", alpha=alpha_mm, edgecolors='k',linewidths=1.)
                            if pred_both:
                                ax.scatter(np.array(pred_yellow[0,:]).squeeze(), np.array(pred_yellow[1,:]).squeeze(), color="y", alpha=alpha_mm, edgecolors='k',linewidths=1.)
                            ax.scatter(np.array(pred_orange[0,:]).squeeze(), np.array(pred_orange[1,:]).squeeze(), color="orange", alpha=alpha_mm, edgecolors='k',linewidths=1.)
                            if pred_both:
                                ax.scatter(np.array(pred_red[0,:]).squeeze(), np.array(pred_red[1,:]).squeeze(), color="r", alpha=alpha_mm,edgecolors='k',linewidths=1.)
                        else:
                            ax.scatter(np.array(pred[0,:]).squeeze(), np.array(pred[1,:]).squeeze(), color="orange", alpha=alpha_mm,edgecolors='k',linewidths=1.)

        #Print EV States
        ev_legend = Rectangle((-48,-15),6.,3.6,linewidth=1., ec='green', fc='green')
        ax.add_patch(ev_legend)
        ax.text(-40,-15,f's: {self.vehicles[-1].traj[0,i].round(1)}, vel: {self.vehicles[-1].traj[1,i].round(1)}')

        if self.eval_mode:
            if pred_both:
                circle_red = Circle((-45, -20+1.5), radius=2, color='red', alpha = 0.4)
                ax.add_patch(circle_red)
                ax.text(-40,-20,f': vars. + constr. removed')
                circle_o = Circle((-45, -25+1.5), radius=2, color='orange', alpha = 0.4)
                ax.add_patch(circle_o)
                ax.text(-40,-25,f': constr. active')
                circle_y = Circle((-45, -30+1.5), radius=2, color='yellow', alpha = 0.4)
                ax.add_patch(circle_y)
                ax.text(-40,-30,f': vars. removed')
                circle_m = Circle((-45, -35+1.5), radius=2, color='m', alpha = 0.4)
                ax.add_patch(circle_m)
                ax.text(-40,-35,f': Both removed')
            else:
                circle_o = Circle((-45, -20+1.5), radius=2, color='orange', alpha = 0.4)
                ax.add_patch(circle_o)
                ax.text(-40,-20,f': Constr. Imposed')
                if l1_dual is not None:
                    circle_m = Circle((-45, -25+1.5), radius=2, color='m', alpha = 0.4)
                    ax.add_patch(circle_m)
                    ax.text(-40,-25,f': Constr. Removed')

        ax.set_xlim(-50,60)
        ax.set_ylim(-50,50)
        # ax.axis('equal')
