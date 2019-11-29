import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
from baseline_racer import BaselineRacer
from utils import to_airsim_vector, to_airsim_vectors
import airsimneurips as airsim
import argparse
import numpy as np
import time
import datetime
import math
import threading
import os
import random
# Use non interactive matplotlib backend
import matplotlib
#matplotlib.use("TkAgg")
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D #3d plotting

biggest_counter=0



class BaselineRacerGTP(BaselineRacer):
    def __init__(self, drone_names, drone_i, drone_params,use_vel_constraints=False,race_tier=1):
        super().__init__(drone_name=drone_names[drone_i], viz_traj=True)
        self.level_name = "Undefined"
        self.race_tier = "Undefined"
        self.drone_names = drone_names
        self.drone_i = drone_i
        self.drone_params = drone_params
        self.use_vel_constraints = use_vel_constraints

        self.stationary_obj_poses = []
        self.gates_complete = 0
        self.switch_log = False
        self.logfile = None

        self.scene_objs = self.airsim_client.simListSceneObjects()
        self.scene_obj_poses = self.get_all_scene_obj_poses()
        self.race_tier = race_tier

        self.observation_space = [52,0]
        self.action_space = [4,0]
        self.reward = 0
        self.action_high = 1
        self.action_low = -1
        self.max_vel = 40

        self.prev_gate = 0
        self.prev_time = 0
        self.prev_coll = 0

        self.time = 0
        self.last_gate_change_time = 0
        self.collision_penalty = 0

        self.DQ = False
        self.other_DQ = False

        self.seq_features = []
        self.seq_ptr = 0

        print("Test Racer ready!")

    def re_init_attrs(self):
        self.stationary_obj_poses = []
        self.gates_complete = 0
        self.scene_objs = self.airsim_client.simListSceneObjects()
        self.scene_obj_poses = self.get_all_scene_obj_poses()
        self.reward = 0
        self.DQ = False
        self.other_DQ = False
        self.prev_gate = 0
        self.prev_time = 0
        self.prev_coll = 0
        self.time = 0
        self.last_gate_change_time = 0
        self.collision_penalty = 0
        self.seq_features = []
        self.seq_ptr = 0


    def restart_race(self):
        self.reset_race()
        time.sleep(1)
        self.re_init_attrs()
        self.start_race(self.race_tier)
        # self.initialize_drone()
        self.takeoff_with_moveOnSpline()
        time.sleep(0.0)  # give opponent a little advantage



    def get_all_scene_obj_poses(self):
        l = []
        for obj in self.scene_objs:
            l.append(self.airsim_client.simGetObjectPose(obj))
        return l


    def get_next_gate_poses(self):
        
        gate_nums = [i for i in range(self.gates_complete+1,min(self.gates_complete+3,len(self.gate_poses_ground_truth)+1))]
        mypos = self.airsim_client.simGetVehiclePose(self.drone_names[self.drone_i]).position
        
        features = []
        for gate_num in gate_nums:
            gate_pose = self.gate_poses_ground_truth[gate_num-1]
            g_pos = gate_pose.position
            g_or = gate_pose.orientation
            features.extend([g_pos.x_val - mypos.x_val, g_pos.y_val- mypos.y_val, g_pos.z_val - mypos.z_val, g_or.x_val, g_or.y_val, g_or.z_val, g_or.w_val])

        if(len(features) == 7):
            prev_gate = self.gate_poses_ground_truth[self.gates_complete-1].position
            prev_g = np.array([prev_gate.x_val,prev_gate.y_val,prev_gate.z_val])
            cur_gate = self.gate_poses_ground_truth[self.gates_complete].position
            cur_g = np.array([cur_gate.x_val,cur_gate.y_val,cur_gate.z_val])
            
            dir_vec = cur_g - prev_g
            dir_vec = dir_vec / np.linalg.norm(dir_vec)

            new_gate = 0.02*dir_vec + cur_g
            features.extend(list(new_gate)+features[-4:])


        if self.gates_complete == len(self.gate_poses_ground_truth):
            features = [0]*14

        features = np.array(features)


        features = np.reshape(features,(2*7,))
        return features


    def dist(self, mypos, otherpos):
    	diff = (mypos.x_val-otherpos.x_val) ** 2
    	diff += (mypos.y_val-otherpos.y_val) ** 2
    	diff += (mypos.z_val-otherpos.z_val) ** 2
    	diff = math.sqrt(diff)
    	return diff

    def get_nearest_n_object(self,n = 5):

    	distances = []
    	mypos = self.airsim_client.simGetVehiclePose(self.drone_names[self.drone_i]).position
    	for obj in self.scene_obj_poses:
    		otherpos  = obj.position
    		distances.append([self.dist(mypos, otherpos), otherpos.x_val - mypos.x_val, otherpos.y_val- mypos.y_val, otherpos.z_val - mypos.z_val])
    	distances.sort()

    	dist_ = [i[1:] for i in distances[:n]]
    	dist_ = np.array(dist_)
    	dist_= np.reshape(dist_,(n*3,))
    	return dist_

   
    def file_name(self):
        # Training
        # qualification

        l=os.listdir("/home/iiitd/GameOfDrone/AirSim_qualification/AirSimExe/Saved/Logs/RaceLogs")
        l.sort()
        # print("Log file [Breaks every month]",l[-1])

        print(self.logfile)
        print(l[-1])

        self.logfile = l[-1]
        self.switch_log = False
        


    def read_file(self):
        open_file = open("/home/iiitd/GameOfDrone/AirSim_qualification/AirSimExe/Saved/Logs/RaceLogs/"+self.logfile,'r')
        print("Reading file")
        linecount = 0
        while(True):
            time.sleep(0.005)
            for line in open_file:
                linecount += 1
                if("time" in line):
                    try:
                        self.time = int(line.split()[2])
                    except ValueError:
                        print("Invalid time in line", linecount)
                    except IndexError:
                        print("Invalid time in line (incomplete line)", linecount, line)

                if "penalty" in line:
                    if "drone_1" in line:
                        try:
                            self.collision_penalty = int(line.split()[-1])
                        except ValueError:
                            print("Invalid penalty in line", linecount) 

                if "disqualified" in line:
                    if "drone_1" in line:
                        self.DQ = True
                    else:
                        self.other_DQ = True

                if "gates_passed" in line:
                    if "drone_1" in line:
                        try:
                            g = int(line.split(" ")[-1])
                            if(g != self.gates_complete):
                                print("gate passed ", g)
                                self.last_gate_change_time = self.time
                            self.gates_complete = g

                        except ValueError:
                            print("Invalid Literal",line)
                    else:
                        pass #print(int(line.split(" ")[-1]))

                elif "signature" in line:
                    self.switch_log = True
                    break

            if self.switch_log:
                return
            

    def read_file_thread(self):
        self.file_name()
        threading.Thread(target=self.read_file).start()


    def construct_feature_vector(self):
        drone_state  = self.airsim_client.getMultirotorState()
        position = drone_state.kinematics_estimated.position.to_numpy_array() # Not part of feature vector
        orientation = drone_state.kinematics_estimated.orientation.to_numpy_array()
        linear_velocity = drone_state.kinematics_estimated.linear_velocity.to_numpy_array()
        angular_velocity = drone_state.kinematics_estimated.angular_velocity.to_numpy_array()
        linear_acc = drone_state.kinematics_estimated.linear_acceleration.to_numpy_array()
        angular_acc = drone_state.kinematics_estimated.angular_acceleration.to_numpy_array()

        other_drone_pose = self.airsim_client.simGetObjectPose(self.drone_names[1-self.drone_i])

        while math.isnan(other_drone_pose.position.x_val):
            other_drone_pose = self.airsim_client.simGetObjectPose(self.drone_names[1-self.drone_i])

        other_orientation = other_drone_pose.orientation.to_numpy_array()
        other_position_rel = other_drone_pose.position.to_numpy_array() - position
       
        closest_objects = self.get_nearest_n_object()
        nxt_gates = self.get_next_gate_poses()

        #We can pass our z in absolute value as well, to keep track of ground
        feature_vector = np.r_[orientation,linear_velocity,angular_velocity,linear_acc,angular_acc,other_orientation,other_position_rel, closest_objects, nxt_gates]
        
        feature_vector = feature_vector / np.linalg.norm(feature_vector)

        feature_vector[np.isnan(feature_vector)] = 0.

        # feature_vector = np.reshape(feature_vector,(1,52))

        return feature_vector


    #action : [vx, vy, vz, abs(v)]
    def calculate_velocity(self, action = [0, 0, 0, 0],stall=False):
        nxt_gate = self.gate_poses_ground_truth[self.gates_complete].position
        g = np.array([nxt_gate.x_val,nxt_gate.y_val,nxt_gate.z_val])

        other_pos = self.airsim_client.simGetObjectPose(self.drone_names[1-self.drone_i]).position
        o = np.array([other_pos.x_val, other_pos.y_val, other_pos.z_val])


        mypos = self.airsim_client.simGetVehiclePose(self.drone_names[self.drone_i]).position
        m = np.array([mypos.x_val, mypos.y_val, mypos.z_val])

        diff = (g - m) # +(o-m)

        noise = 0
        diff = diff / np.linalg.norm(diff)

        del_v = action[:3] / np.linalg.norm(action[:3])


        diff = (3.5+8.5*(action[3]-0.2))*(diff+0.15*del_v)*(1-int(stall))

        action[3] = 0
        return diff[0],diff[1],diff[2]
        

    def update_and_plan(self,delta_v=[0.,0.,0.,1.],stall=False):
        global biggest_counter 
        burst = 0.055
        biggest_counter+=1
        if(biggest_counter%30==0):
            print("action delta:",delta_v[3])
        
        # After last gate hover
        if self.gates_complete == len(self.gate_poses_ground_truth):
            print("reseting after completing gates", self.gates_complete, len(self.gate_poses_ground_truth))
            self.airsim_client.moveByVelocityAsync(0,0,0,1000,vehicle_name=self.drone_name)  
            next_state = self.construct_feature_vector()    
            self.add_feature(next_state)
            seq_state = self.construct_sequential_feature()
            reward = 5 
            print("Completion reward",5)
            return seq_state,reward,True,"finished debug"
            
        vel = self.calculate_velocity(delta_v,stall)

        assert len(delta_v)==4, "delta_v should have 4 components"

        self.airsim_client.moveByVelocityAsync(vel[0],vel[1],vel[2],burst,vehicle_name=self.drone_name)
        time.sleep(burst-0.005)

        next_state = self.construct_feature_vector()
        
        self.add_feature(next_state)

        # print(next_state)
        done = False
        if(self.DQ):
            print("self dq")
            done = True
        elif(self.other_DQ):
            print("other dq")
            done = True
        else:
            done = False

        reward = self.check_reward()
        
        seq_state = self.construct_sequential_feature()

        return seq_state, reward, done, "debug" # next_state,reward,done,"debug string"

    def add_feature(self,next_state):
        if len(self.seq_features) < 4:
            self.seq_features.append(next_state)
            return

        self.seq_features[0] = self.seq_features[1]
        self.seq_features[1] = self.seq_features[2]
        self.seq_features[2] = self.seq_features[3]
        self.seq_features[3] = next_state

    def construct_sequential_feature(self):
        seq_len = 4
        feature = np.zeros((seq_len,52))

        for i in range(seq_len):
            for j in range(52):
                feature[i,j] = feature[i,j] + self.seq_features[min(i,len(self.seq_features)-1)][j]

        return feature


    '''
    TODO : Vary Time Penalty based on relative drone ordering
    '''
    def check_reward(self):
        time_penalty = self.time - self.prev_time
        collision_penalty = self.collision_penalty - self.prev_coll
        gates_passed = self.gates_complete - self.prev_gate
        rew = gates_passed*10 - collision_penalty/1000 - time_penalty/1000
        self.prev_time = self.time
        self.prev_coll = self.collision_penalty
        self.prev_gate = self.gates_complete
        if(self.DQ):
            rew -= 300
        if(self.other_DQ):
            rew += 20

        # print(gates_passed, time_penalty, collision_penalty)

        return rew  


    def sim_Pause(self):
        self.airsim_client.simPause(is_paused=False)

    def sim_Play(self):
        self.airsim_client.simPause(is_paused=True)


    def step(self,action,stall = False):
        if stall:
            self.last_gate_change_time = self.time

        if self.time - self.last_gate_change_time >= 40000:
            self.add_feature(self.construct_feature_vector())
            return self.construct_sequential_feature(),0,True,"Stuck"

        return self.update_and_plan(action,stall)

    def reset(self):
        self.restart_race()
        self.read_file_thread()

        self.add_feature(self.construct_feature_vector())

        return self.construct_sequential_feature()

    def cold_start(self):
        self.start_race(self.race_tier)
        self.initialize_drone()
        self.takeoff_with_moveOnSpline()
        self.get_ground_truth_gate_poses()
        self.read_file_thread()
        assert self.airsim_client.isApiControlEnabled(vehicle_name=self.drone_name)==True
        
    def run(self):

        self.start_race(self.race_tier)
        self.initialize_drone()
        self.takeoff_with_moveOnSpline()
        time.sleep(0.0)  # give opponent a little advantage

        self.get_ground_truth_gate_poses()
        self.read_file_thread()

        while True:

            while self.airsim_client.isApiControlEnabled(vehicle_name=self.drone_name):
                ns,r,done,_ = self.update_and_plan()
                # time.sleep(0.05)
                if done: 
                    print("broken")
                    break

            print("Restarting")
            self.restart_race()
            self.read_file_thread()
            print("GO!!!")

        print("ended")



def main(args):
    drone_names = ["drone_1", "drone_2"]
    drone_params = [
        {"r_safe": 0.5,
         "r_coll": 0.5,
         "v_max": 80.0,
         "a_max": 40.0},
        {"r_safe": 0.4,
         "r_coll": 0.3,
         "v_max": 20.0,
         "a_max": 10.0}]

    # set good map-specific conditions
    if (args.level_name == "Soccer_Field_Easy"):
        pass
    elif (args.level_name == "Soccer_Field_Medium"):
        drone_params[0]["v_max"] = 60.0
        drone_params[0]["a_max"] = 35.0
    elif (args.level_name == "ZhangJiaJie_Medium"):
        drone_params[0]["v_max"] = 60.0
        drone_params[0]["a_max"] = 35.0
    elif (args.level_name == "Building99_Hard"):
        drone_params[0]["v_max"] = 10.0
        drone_params[0]["a_max"] = 30.0
    elif (args.level_name == "Qualifier_Tier_1"):
        pass
    elif (args.level_name == "Qualifier_Tier_2"):
        pass
    elif (args.level_name == "Qualifier_Tier_3"):
        pass

    # ensure you have generated the neurips planning settings file by running python generate_settings_file.py
    baseline_racer_gtp = BaselineRacerGTP(
        drone_names=drone_names,
        drone_i=0,  # index of the first drone
        drone_params=drone_params,
        use_vel_constraints=args.vel_constraints,
        race_tier=args.race_tier)

    baseline_racer_gtp.level_name = args.level_name
    baseline_racer_gtp.race_tier = args.race_tier

    baseline_racer_gtp.load_level(args.level_name)
    
    baseline_racer_gtp.run()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--vel_constraints', dest='vel_constraints', action='store_true', default=False)
    parser.add_argument('--level_name', type=str, choices=["Soccer_Field_Easy", "Soccer_Field_Medium", "ZhangJiaJie_Medium", "Building99_Hard", 
        "Qualifier_Tier_1", "Qualifier_Tier_2", "Qualifier_Tier_3"], default="ZhangJiaJie_Medium")
    parser.add_argument('--enable_viz_traj', dest='viz_traj', action='store_true', default=False)
    parser.add_argument('--race_tier', type=int, choices=[1,2,3], default=1)
    args = parser.parse_args()
    main(args)