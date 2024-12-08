# Copyright 2019 Nathan Jay and Noga Rotman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
from gymnasium.envs.registration import register
import numpy as np
import heapq
import time
import datetime
import random
import json
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
from common import sender_obs, config, logger
from common.simple_arg_parse import arg_or_default
from graph_plot import plot
from tqdm import tqdm
import math

log = logger.get_logger(__name__)

MAX_CWND = 5000
MIN_CWND = 4

MAX_RATE = 1000
MIN_RATE = 40

REWARD_SCALE = 0.001

MAX_STEPS = 400

EVENT_TYPE_SEND = 'S'
EVENT_TYPE_ACK = 'A'

BYTES_PER_PACKET = 1500

LATENCY_PENALTY = 1.0
LOSS_PENALTY = 1.0

USE_LATENCY_NOISE = False
MAX_LATENCY_NOISE = 1.1

HISTORY_LENGTH = 10
# Environment parameter ranges
MIN_BW, MAX_BW = (100, 500)
MIN_LAT, MAX_LAT = (0.05, 0.5)
MIN_QUEUE, MAX_QUEUE = (0, 8)
MIN_LOSS, MAX_LOSS = (0.0, 0.05)

USE_CWND = False
START_TIME = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") # Used for logging and plotting

class Link():

    def __init__(self, bandwidth, delay, queue_size, loss_rate):
        """
        Initializes a Link with the given parameters.

        Args:
            bandwidth (float): The bandwidth of the network link in packets per second (between 100, 500).
            delay (float): The latency of the network link in seconds (0.05, 0.5).
            queue_size (int): The length of the buffer queue in packets (2 - 2981).
            loss_rate (float): The loss rate of the network link as a decimal (0 - 0.05).
        """
        self.bw = float(bandwidth)
        self.dl = delay
        self.lr = loss_rate
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0
        self.max_queue_delay = queue_size / self.bw
        # TODO: Add queue and marking threshold to the link
        self.queue_size = queue_size    #TODO: Keep track of how the queue is changing

    def get_cur_queue_delay(self, event_time):
        return max(0.0, self.queue_delay - (event_time - self.queue_delay_update_time))

    def get_cur_latency(self, event_time):
        return self.dl + self.get_cur_queue_delay(event_time)

    def packet_enters_link(self, event_time):
        if (random.random() < self.lr):
            return False
        self.queue_delay = self.get_cur_queue_delay(event_time)
        self.queue_delay_update_time = event_time
        extra_delay = 1.0 / self.bw
        #print("Extra delay: %f, Current delay: %f, Max delay: %f" % (extra_delay, self.queue_delay, self.max_queue_delay))
        if extra_delay + self.queue_delay > self.max_queue_delay:
            #print("\tDrop!")
            return False
        self.queue_delay += extra_delay
        #print("\tNew delay = %f" % self.queue_delay)
        return True

    def print_debug(self):
        print("Link:")
        print("Bandwidth: %f" % self.bw)
        print("Delay: %f" % self.dl)
        print("Queue Delay: %f" % self.queue_delay)
        print("Max Queue Delay: %f" % self.max_queue_delay)
        print("One Packet Queue Delay: %f" % (1.0 / self.bw))
        print("Queue Length: %f" % (self.queue_delay * self.bw))

    def reset(self):
        self.queue_delay = 0.0
        self.queue_delay_update_time = 0.0

class Network():
    
    def __init__(self, senders, links):
        self.q = []
        self.cur_time = 0.0
        self.senders = senders
        self.links = links
        self.queue_initial_packets()

    def queue_initial_packets(self):
        for sender in self.senders:
            sender.register_network(self)
            sender.reset_obs()
            heapq.heappush(self.q, (1.0 / sender.rate, sender, EVENT_TYPE_SEND, 0, 0.0, False)) 

    def reset(self):
        self.cur_time = 0.0
        self.q = []
        [link.reset() for link in self.links]
        [sender.reset() for sender in self.senders]
        self.queue_initial_packets()

    def get_cur_time(self):
        return self.cur_time

    def run_for_dur(self, dur, reward_fun=None, reward_coefficients=None):
        end_time = self.cur_time + dur
        for sender in self.senders:
            sender.reset_obs()

        while self.cur_time < end_time:
            event_time, sender, event_type, next_hop, cur_latency, dropped = heapq.heappop(self.q)
            #print("Got event %s, to link %d, latency %f at time %f" % (event_type, next_hop, cur_latency, event_time))
            self.cur_time = event_time
            new_event_time = event_time
            new_event_type = event_type
            new_next_hop = next_hop
            new_latency = cur_latency
            new_dropped = dropped
            push_new_event = False

            if event_type == EVENT_TYPE_ACK:
                if next_hop == len(sender.path):
                    if dropped:
                        sender.on_packet_lost()
                        #print("Packet lost at time %f" % self.cur_time)
                    else:
                        sender.on_packet_acked(cur_latency)
                        #print("Packet acked at time %f" % self.cur_time)
                else:
                    new_next_hop = next_hop + 1
                    link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                    if USE_LATENCY_NOISE:
                        link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                    new_latency += link_latency
                    new_event_time += link_latency
                    push_new_event = True
            if event_type == EVENT_TYPE_SEND:
                if next_hop == 0:
                    #print("Packet sent at time %f" % self.cur_time)
                    if sender.can_send_packet():
                        sender.on_packet_sent()
                        push_new_event = True
                    heapq.heappush(self.q, (self.cur_time + (1.0 / sender.rate), sender, EVENT_TYPE_SEND, 0, 0.0, False))
                
                else:
                    push_new_event = True

                if next_hop == sender.dest:
                    new_event_type = EVENT_TYPE_ACK
                new_next_hop = next_hop + 1
                
                link_latency = sender.path[next_hop].get_cur_latency(self.cur_time)
                if USE_LATENCY_NOISE:
                    link_latency *= random.uniform(1.0, MAX_LATENCY_NOISE)
                new_latency += link_latency
                new_event_time += link_latency
                new_dropped = not sender.path[next_hop].packet_enters_link(self.cur_time)
                   
            if push_new_event:
                heapq.heappush(self.q, (new_event_time, sender, new_event_type, new_next_hop, new_latency, new_dropped))

        sender_mi = self.senders[0].get_run_data()
        throughput = sender_mi.get("recv rate")
        latency = sender_mi.get("avg latency")
        loss = sender_mi.get("loss ratio")
        bw_cutoff = self.links[0].bw * 0.8
        lat_cutoff = 2.0 * self.links[0].dl * 1.5
        loss_cutoff = 2.0 * self.links[0].lr * 1.5
        #print("thpt %f, bw %f" % (throughput, bw_cutoff))
        #reward = 0 if (loss > 0.1 or throughput < bw_cutoff or latency > lat_cutoff or loss > loss_cutoff) else 1 #
        
        # Super high throughput
        #reward = REWARD_SCALE * (20.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Very high thpt
        if reward_fun == "default" or reward_fun is None:
            reward = (10.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
            # reward = (10.0 * throughput/self.links[0].bw - 1e3 * latency/self.links[0].dl - 2e3 * loss)
        elif reward_fun == "low_latency":
            reward = (2.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        elif reward_fun == "high_throughput":
            reward = (20.0 * throughput / (8 * BYTES_PER_PACKET) - 1e3 * latency - 2e3 * loss)
        elif reward_fun == "custom":
            a, b, c = reward_coefficients
            reward = (a * throughput / (8 * BYTES_PER_PACKET) + b * latency + c * loss)
        elif reward_fun == "log":
            reward = 100 * math.log(throughput / (8 * BYTES_PER_PACKET)+1e-7) - 1e3 * latency - 2e3 * loss
        elif reward_fun == "log2":
            reward = 1000 * math.log(throughput / (8 * BYTES_PER_PACKET)+1e-7) - 1e3 * latency - 2e3 * loss 
    # High thpt
        #reward = REWARD_SCALE * (5.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        
        # Low latency
        #reward = REWARD_SCALE * (2.0 * throughput / RATE_OBS_SCALE - 1e3 * latency / LAT_OBS_SCALE - 2e3 * loss)
        #if reward > 857:
        #print("Reward = %f, thpt = %f, lat = %f, loss = %f" % (reward, throughput, latency, loss))
        
        #reward = (throughput / RATE_OBS_SCALE) * np.exp(-1 * (LATENCY_PENALTY * latency / LAT_OBS_SCALE + LOSS_PENALTY * loss))
        return reward * REWARD_SCALE

class Sender():
    
    def __init__(self, rate, path, dest, features, cwnd=25, history_len=HISTORY_LENGTH):
        self.id = Sender._get_next_id()
        self.starting_rate = rate
        self.rate = rate
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.bytes_in_flight = 0
        self.min_latency = None
        self.rtt_samples = []
        self.sample_time = []
        self.net = None
        self.path = path
        self.dest = dest
        self.history_len = history_len
        self.features = features
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)
        self.cwnd = cwnd

    _next_id = 1
    def _get_next_id():
        result = Sender._next_id
        Sender._next_id += 1
        return result

    def apply_rate_delta(self, delta):
        delta *= config.DELTA_SCALE # 0.025
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_rate(self.rate * (1.0 + delta))
        else:
            self.set_rate(self.rate / (1.0 - delta))

    def apply_cwnd_delta(self, delta):
        delta *= config.DELTA_SCALE
        #print("Applying delta %f" % delta)
        if delta >= 0.0:
            self.set_cwnd(self.cwnd * (1.0 + delta))
        else:
            self.set_cwnd(self.cwnd / (1.0 - delta))

    def can_send_packet(self):
        if USE_CWND:
            return int(self.bytes_in_flight) / BYTES_PER_PACKET < self.cwnd
        else:
            return True

    def register_network(self, net):
        self.net = net

    def on_packet_sent(self):
        self.sent += 1
        self.bytes_in_flight += BYTES_PER_PACKET

    def on_packet_acked(self, rtt):
        self.acked += 1
        self.rtt_samples.append(rtt)
        if (self.min_latency is None) or (rtt < self.min_latency):
            self.min_latency = rtt
        self.bytes_in_flight -= BYTES_PER_PACKET

    def on_packet_lost(self):
        self.lost += 1
        self.bytes_in_flight -= BYTES_PER_PACKET

    def set_rate(self, new_rate):
        self.rate = new_rate.item()
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.rate > MAX_RATE:
            self.rate = MAX_RATE
            # log.debug(f"->Rate {self.rate} too high, set to {MAX_RATE}")
        elif self.rate < MIN_RATE:
            self.rate = MIN_RATE
            # log.debug(f"->Rate {self.rate} too low, set to {MIN_RATE}")
        else:
            pass
            # log.debug(f"->Rate within bounds: {self.rate}")

    def set_cwnd(self, new_cwnd):
        self.cwnd = int(new_cwnd)
        #print("Attempt to set new rate to %f (min %f, max %f)" % (new_rate, MIN_RATE, MAX_RATE))
        if self.cwnd > MAX_CWND:
            self.cwnd = MAX_CWND
        if self.cwnd < MIN_CWND:
            self.cwnd = MIN_CWND

    def record_run(self):
        smi = self.get_run_data()
        self.history.step(smi)

    def get_obs(self):
        return self.history.as_array()

    def get_run_data(self):
        obs_end_time = self.net.get_cur_time()
        
        #obs_dur = obs_end_time - self.obs_start_time
        #print("Got %d acks in %f seconds" % (self.acked, obs_dur))
        #print("Sent %d packets in %f seconds" % (self.sent, obs_dur))
        #print("self.rate = %f" % self.rate)

        return sender_obs.SenderMonitorInterval(
            self.id,
            bytes_sent=self.sent * BYTES_PER_PACKET,
            bytes_acked=self.acked * BYTES_PER_PACKET,
            bytes_lost=self.lost * BYTES_PER_PACKET,
            send_start=self.obs_start_time,
            send_end=obs_end_time,
            recv_start=self.obs_start_time,
            recv_end=obs_end_time,
            rtt_samples=self.rtt_samples,
            packet_size=BYTES_PER_PACKET
        )

    def reset_obs(self):
        self.sent = 0
        self.acked = 0
        self.lost = 0
        self.rtt_samples = []
        self.obs_start_time = self.net.get_cur_time()

    def print_debug(self):
        print("Sender:")
        print("Obs: %s" % str(self.get_obs()))
        print("Rate: %f" % self.rate)
        print("Sent: %d" % self.sent)
        print("Acked: %d" % self.acked)
        print("Lost: %d" % self.lost)
        print("Min Latency: %s" % str(self.min_latency))

    def reset(self):
        #print("Resetting sender!")
        self.rate = self.starting_rate
        self.bytes_in_flight = 0
        self.min_latency = None
        self.reset_obs()
        self.history = sender_obs.SenderHistory(self.history_len,
                                                self.features, self.id)

class SimulatedNetworkEnv(gym.Env):
    """
    In this environment, the action space is changed from a multiplier of the previous rate 
    to an absolute rate in the range of [MIN=40, MAX=1000] .

    Version history:
    - v1: Mapped actions from [-1, 1] to [MIN, MAX] and shifted from using delta to using absolute value
    - v0: Initial implementation

    """
    def __init__(self,
                 history_len=arg_or_default("--history-len", default=10),
                 features=arg_or_default("--input-features",
                    default="sent latency inflation,"
                          + "latency ratio,"
                          + "send ratio"),
                 reward_fun = "default",
                 seed=0,
                 ACTION_SCALE=1,
                 max_steps=MAX_STEPS,
                 mode="training",
                 network_values: tuple = None,
                 reward_coefficients: tuple = None
                 ):
        self.viewer = None
        self.rand = None
        self.mode = mode
        self.network_values = network_values
        

        # Validate parameters
        assert reward_fun in ["default", "low_latency", "high_throughput", "custom", "log", "log2"], "Invalid reward function"
        assert mode in ["training", "testing"], "Invalid mode"
        if network_values and len(network_values) != 4:
            raise ValueError("Requires 4 network values: (bw, lat, queue, loss)")
        if reward_coefficients and len(reward_coefficients) != 3:
            raise ValueError(f"Custom reward function requires 3 reward coefficients (a, b, c) but got {len(reward_coefficients)} from {reward_coefficients}")
        if reward_coefficients is None and reward_fun == "custom":
            log.warning("Custom reward function received no reward coefficients (a, b, c). Using defaults (10, -1000, -2000)")
            reward_coefficients = (10, -1000, -2000)

        self.reward_coefficients = reward_coefficients
        

        # Introduce the seed parameter to the environment
        random.seed(seed)
        np.random.seed(seed)

        log.info(f"Initializing with reward function as {reward_fun} and history length of {history_len}. Seed: {seed}, action_scale: +/-{ACTION_SCALE}")
        self.reward_fun = reward_fun

        # Read the configuration from the config file
        config_file_path = os.path.join(currentdir, 'config.json')
        with open(config_file_path, 'r') as f:
            self.config = json.load(f)
        if self.config is None:
            log.warning("Config file not found or empty, using default parameters")
            self.min_bw, self.max_bw = (MIN_BW, MAX_BW) # Initial (100, 500)
            self.min_lat, self.max_lat = (MIN_LAT, MAX_LAT) #(0.05, 0.5)
            self.min_queue, self.max_queue = (MIN_QUEUE, MAX_QUEUE) #(0, 8)
            self.min_loss, self.max_loss = (MIN_LOSS, MAX_LOSS) #(0.0, 0.05)
        else:
            if mode not in self.config:
                # print(f"{self.config.keys()}")
                log.warning(f"Configuration type {mode} not found in config file. Available types: {self.config.keys()}")
            
            self.min_bw, self.max_bw = (self.config[mode]["MIN_BW"], self.config[mode]["MAX_BW"])
            self.min_lat, self.max_lat = (self.config[mode]["MIN_LAT"], self.config[mode]["MAX_LAT"])
            self.min_queue, self.max_queue = (self.config[mode]["MIN_QUEUE"], self.config[mode]["MAX_QUEUE"])
            self.min_loss, self.max_loss = (self.config[mode]["MIN_LOSS"], self.config[mode]["MAX_LOSS"])

        # If network values are provided (while in testing mode), use them instead of the config file
        if network_values is not None and mode == "testing":
            self.min_bw = self.max_bw = network_values[0]
            self.min_lat = self.max_lat = network_values[1]
            self.min_queue = self.max_queue = network_values[2]
            self.min_loss = self.max_loss = network_values[3]
        
        log.info(f"Using Bandwidth: {self.min_bw} - {self.max_bw}, Latency: {self.min_lat} - {self.max_lat}, Queue: {self.min_queue} - {self.max_queue}, Loss: {self.min_loss} - {self.max_loss}. {mode=}")

        if reward_fun == "custom":
            log.info(f"Using custom reward function with coefficients {reward_coefficients}")

        self.history_len = history_len
        # print("History length: %d" % history_len)
        self.features = features.split(",")
        # print("Features: %s" % str(self.features))

        self.links = None
        self.senders = None
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.run_dur = None
        self.run_period = 0.1
        self.steps_taken = 0
        self.max_steps = max_steps
        self.debug_thpt_changes = False
        self.last_thpt = None
        self.last_rate = None

        # self.ACTION_SCALE = 1e12
        if USE_CWND:
            self.action_space = spaces.Box(np.array([-ACTION_SCALE, -ACTION_SCALE]), np.array([ACTION_SCALE, ACTION_SCALE]), dtype=np.float32)
        else:
            self.action_space = spaces.Box(np.array([-ACTION_SCALE]), np.array([ACTION_SCALE]), dtype=np.float32)

        self.observation_space = None
        use_only_scale_free = True
        single_obs_min_vec = sender_obs.get_min_obs_vector(self.features)
        single_obs_max_vec = sender_obs.get_max_obs_vector(self.features)
        self.observation_space = spaces.Box(np.tile(single_obs_min_vec, self.history_len),
                                            np.tile(single_obs_max_vec, self.history_len),
                                            dtype=np.float32)

        self.reward_sum = 0.0
        self.reward_ewma = 0.0

        self.event_record = {"Events":[]}
        self.episodes_run = -1

    def seed(self, seed=None):
        self.rand, seed = seeding.np_random(seed)
        return [seed]

    def _get_all_sender_obs(self):
        sender_obs = self.senders[0].get_obs()
        sender_obs = np.array(sender_obs, dtype=np.float32).reshape(-1,)
        #print(sender_obs)
        return sender_obs

    def step(self, actions):
        #print("Actions: %s" % str(actions))
        #print(actions)
        # NB: Action space scalled to be between -1 and 1
        # This should be mapped to the appropriate range for each action (-1,1) -> (Min, Max)

        # TODO: Offset and scale actions. Eliminates ACTION_SCALE.
        # Mapping: (-1,1) -> (Min, Max)
        actions = MIN_RATE + (actions - (-1)) * (MAX_RATE - MIN_RATE) / 2.0
        # An agents actions between -1 and 1 are mapped to the range of rates between MIN_RATE and MAX_RATE
        # If this is used with the PCC-Uspace, the delta scale will map [-1, 1] to a multiplier for the previous rate
        # Decrease and increase will no longer be consistent.
        # TODO: Examine the trained model and see its output for different state inputs
         

        # actions =  np.array([self.ACTION_SCALE *actions]) # Scale actions
        # self.senders[0].apply_rate_delta(actions[0]) # NB: Shift from using delta to using absolute value
        # if USE_CWND:
        #     self.senders[0].apply_cwnd_delta(actions[1])
        # for i in range(0, 1):#len(actions)):
        #     #print("Updating rate for sender %d" % i)
        #     action = actions
        #     self.senders[i].apply_rate_delta(action[0])
        #     if USE_CWND:
        #         self.senders[i].apply_cwnd_delta(action[1])
        # #print("Running for %fs" % self.run_dur)
        self.senders[0].set_rate(actions) # NB: This is a breaking change that shifts from using delta to using absolute value
        reward = self.net.run_for_dur(self.run_dur, self.reward_fun, self.reward_coefficients)
        for sender in self.senders:
            sender.record_run()
        self.steps_taken += 1
        sender_obs = self._get_all_sender_obs()
        sender_mi = self.senders[0].get_run_data()
        event = {}
        event["Name"] = "Step"
        event["Time"] = self.steps_taken
        event["Reward"] = reward
        #event["Target Rate"] = sender_mi.target_rate
        event["Send Rate"] = sender_mi.get("send rate")
        event["Throughput"] = sender_mi.get("recv rate")
        event["Latency"] = sender_mi.get("avg latency")
        event["Loss Rate"] = sender_mi.get("loss ratio")
        event["Latency Inflation"] = sender_mi.get("sent latency inflation")
        event["Latency Ratio"] = sender_mi.get("latency ratio")
        event["Send Ratio"] = sender_mi.get("send ratio")
        event["bandwidth"] = self.bw*BYTES_PER_PACKET*8 # bits/s
        event["latency_init"] = self.lat
        event["queue_init"] = self.queue
        event["loss_init"] = self.loss
        #event["Cwnd"] = sender_mi.cwnd
        #event["Cwnd Used"] = sender_mi.cwnd_used
        self.event_record["Events"].append(event)
        if event["Latency"] > 0.0:
            self.run_dur = 0.5 * sender_mi.get("avg latency") # Half the RTT. Also in PCC-Uspace
        #print("Sender obs: %s" % sender_obs)

        should_stop = False

        self.reward_sum += reward
        terminated = truncated = (self.steps_taken >= self.max_steps or should_stop)
        info = event # Send collected stats back

       
        return sender_obs, reward, terminated, truncated, info

    def print_debug(self):
        print("---Link Debug---")
        for link in self.links:
            link.print_debug()
        print("---Sender Debug---")
        for sender in self.senders:
            sender.print_debug()

    def create_new_links_and_senders(self):
        bw    = random.uniform(self.min_bw, self.max_bw)
        lat   = random.uniform(self.min_lat, self.max_lat)
        queue = 1 + int(np.exp(random.uniform(self.min_queue, self.max_queue)))
        loss  = random.uniform(self.min_loss, self.max_loss)
        self.bw = bw # Record these values for each episode
        self.lat = lat
        self.queue = queue
        self.loss = loss
        #bw    = 200
        #lat   = 0.03
        #queue = 5
        #loss  = 0.00
        self.links = [Link(bw, lat, queue, loss), Link(bw, lat, queue, loss)]
        #self.senders = [Sender(0.3 * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        #self.senders = [Sender(random.uniform(0.2, 0.7) * bw, [self.links[0], self.links[1]], 0, self.history_len)]
        self.senders = [Sender(random.uniform(0.3, 1.5) * bw, [self.links[0], self.links[1]], 0, self.features, history_len=self.history_len)]
        self.run_dur = 3 * lat

    def reset(self, seed=None, options=None):
        super().reset()
        self.steps_taken = 0
        self.successful_runs = 0
        self.net.reset()
        self.create_new_links_and_senders()
        self.net = Network(self.senders, self.links)
        self.episodes_run += 1
        PLOT_EVERY = 100
        if self.episodes_run > 0 and self.episodes_run % PLOT_EVERY == 0:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"pcc_env_log_run_{self.episodes_run}_{timestamp}.json"
            self.dump_events_to_file(filename)
            plot(filename=filename, run_folder=START_TIME)
        self.event_record = {"Events":[]}
        self.net.run_for_dur(self.run_dur)
        self.net.run_for_dur(self.run_dur)
        self.reward_ewma *= 0.99
        self.reward_ewma += 0.01 * self.reward_sum
        log.debug(f"Reward: {self.reward_sum}, Ewma Reward: {self.reward_ewma}, Episode: {self.episodes_run}")
        self.reward_sum = 0.0
        return self._get_all_sender_obs(), {}

    def render(self, mode='human'):
        pass

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    def dump_events_to_file(self, filename):
        from numpyencoder import NumpyEncoder
        if not os.path.exists(f"logs/{START_TIME}"): 
            os.makedirs(f"logs/{START_TIME}")
        with open(f"logs/{START_TIME}/{filename}", 'w') as f:
            json.dump(self.event_record, f, indent=4, cls=NumpyEncoder)

register(id='PccNs-v1', entry_point='network_sim_v1:SimulatedNetworkEnv')
#env = SimulatedNetworkEnv()
#env.step([1.0])

def evaluate(env, n_eval_episodes=1, approach="random"):

    episode_rewards = []
    episode_lengths = []
    record = {"Events":[]}
    # Evaluate the model
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        # Keep track of events: data from info

        while not done:
            if approach == "random":
                action = env.action_space.sample()
            elif approach == "max":
                action = env.action_space.high
            elif approach == "min":
                action = env.action_space.low
            else:
                raise ValueError(f"Invalid approach: {approach}. Choose from ['random', 'max', 'min']")
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            if info:
                record["Events"].append(info)
                # Append the action taken
                record["Events"][-1]["Action"] = action[0] # Assumes only one action from the model
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    # Print results
    log.info(f"Mean reward over {n_eval_episodes} episodes: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f} {episode_rewards=}")
    return episode_rewards, record


def plot_eval(data, title, include_init=True):
    # print(f"Plotting {data=} with title: {title}")
    log.debug(f"Plotting using title: {title}")
    time_data = [float(event["Time"]) for event in data["Events"][1:]]
    rew_data = [float(event["Reward"]) for event in data["Events"][1:]]
    send_data = [float(event["Send Rate"])/1e6 for event in data["Events"][1:]]
    thpt_data = [float(event["Throughput"])/1e6 for event in data["Events"][1:]]
    latency_data = [float(event["Latency"]) for event in data["Events"][1:]]
    loss_data = [float(event["Loss Rate"]) for event in data["Events"][1:]]
    bandwidth_data = [float(event["bandwidth"])/1e6 for event in data["Events"][1:]]
    init_latency = [float(event["latency_init"]) for event in data["Events"][1:]]
    init_loss = [float(event["loss_init"]) for event in data["Events"][1:]]
    actions = [float(event["Action"]) for event in data["Events"][1:]]

    fig, axes = plt.subplots(5, figsize=(10, 12))
    # Add grid lines to plot
    for ax in axes:
        ax.grid(True)
    send_thpt_axis = axes[0]
    loss_axis = axes[1]
    latency_axis = axes[2]
    rew_axis= axes[3]
    action_axis = axes[4]

    

    send_thpt_axis.plot(time_data, send_data, "r", 
                        time_data, thpt_data, "b", 
                        time_data, bandwidth_data, "--g"),
    send_thpt_axis.set_ylabel("Rate (Mbps)")
    #Set the legend at bottom right corner of the graph
    send_thpt_axis.legend(('Send Rate', 'Throughput', 'Bandwidth'), loc='upper right')
    
    latency_axis.plot(time_data, latency_data, label="Current Latency")
    latency_axis.set_ylabel("Latency (seconds)")
    if include_init:
        latency_axis.plot(time_data, init_latency,"--g", label="Initial Latency")
        latency_axis.legend(loc='upper right')

    
    loss_axis.plot(time_data, loss_data, label="Current Loss Rate")
    loss_axis.set_ylabel("Loss Rate (normalized)")
    if include_init:
        loss_axis.plot(time_data, init_loss,"--g", label="Initial Loss Rate")
        loss_axis.legend(loc='upper right')

    rew_axis.plot(time_data, rew_data)
    rew_axis.set_ylabel("Reward")
    # Include average reward line
    rew_axis.axhline(y=np.mean(rew_data), color='blue', linestyle='-.', label="Average Reward")
    rew_axis.legend(loc = 'lower right')
    # rew_axis.set_xlabel("Monitor Interval (time steps)")

    action_axis.plot(time_data, actions)
    action_axis.set_ylabel("Action")
    action_axis.set_xlabel("Monitor Interval (time steps)")

    fig.suptitle(title)
    plt.show()

    if not os.path.exists("results"):
        os.makedirs("results")
    plt.savefig(f"results/{title}.pdf", dpi=300)
    print(f"Plot saved as results/{title}.pdf")
    plt.close()


if __name__ == "__main__":
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_checker import check_env
    import matplotlib.pyplot as plt
    import argparse

    # Create the environment
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-len", type=int, default=10)
    parser.add_argument("--reward-fun", type=str, default="custom")
    args = parser.parse_args()

    env = SimulatedNetworkEnv(reward_fun=args.reward_fun, history_len=args.history_len, 
                              mode="testing", network_values=(500, 0.05, 2, 0.0),
                              reward_coefficients=(10, 1e3, 2e3))
    env = Monitor(env)
    check_env(env) # Confirm that the environment is working
    env.reset()
    default_env = SimulatedNetworkEnv(reward_fun="custom") # Ensure logic is working for default values
    default_env.reset()
    # Take random actions until the environment terminates
    EPISODE_COUNT = 10 #400

    # #Plot number
    # PLOT_NO = 11
    
    done = False
    env.reset(seed=0)
    max_steps = env.unwrapped.max_steps
    timescale = np.linspace(1, max_steps, max_steps)

    episode_rewards, record = evaluate(env, n_eval_episodes=1, approach="random")
    plot_eval(record, "Evaluation_for_Random_actions")

    # Evaluate the env with max actions
    episode_rewards, record = evaluate(env, n_eval_episodes=1, approach="max")
    plot_eval(record, "Evaluation_for_Max_actions")

    # Evaluate the env with min actions
    episode_rewards, record = evaluate(env, n_eval_episodes=1, approach="min")
    plot_eval(record, "Evaluation_for_Min_actions")
