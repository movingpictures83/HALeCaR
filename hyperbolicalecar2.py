from disk_struct import Disk
from page_replacement_algorithm import page_replacement_algorithm
from priorityqueue import priorityqueue
from CacheLinkedList import CacheLinkedList
import time
import numpy as np
import random
import math
from collections import OrderedDict
# import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class CacheMetaData(object):
    def __init__(self):
        super(CacheMetaData, self).__setattr__("page", -1)
        super(CacheMetaData, self).__setattr__("index", -1)
        super(CacheMetaData, self).__setattr__("freq", -1)
        super(CacheMetaData, self).__setattr__("time", -1)

    def __setattr__(self, name, value):
        if not hasattr(self, name):
            raise AttributeError("Creating new attributes is not allowed!")
        super(CacheMetaData, self).__setattr__(name, value)


# sys.path.append(os.path.abspath("/home/giuseppe/))

## Keep a LRU list.
## Page hits:
##      Every time we get a page hit, mark the page and also move it to the MRU position
## Page faults:
##      Evict an unmark page with the probability proportional to its position in the LRU list.
class hyperbolicALeCar(page_replacement_algorithm):    

    def __init__(self, cache_size, learning_rate=0, initial_weight=0.5, discount_rate=1, visualize=0):
        self.N = int(cache_size)

        if self.N < 10:
            self.N = 10
        self.H = int(self.N * 0.25)

        print("BOEB")
        self.SampleSize = max(2, int(math.log(self.N)))

        self.IndexPointer = 0
        self.IndexArray = []
        self.ValueArray = []
        self.Map = {}

        print("NN %d" % self.N)

        i = 0
        while i < self.N:
            self.IndexArray.append(i)
            self.ValueArray.append(-2)
            i += 1


        #self.H = int(1 * self.N * int(param['history_size_multiple'])) if 'history_size_multiple' in param else self.N
        self.learning_rate = learning_rate
        #float(param['learning_rate']) if 'learning_rate' in param else 0
        # self.learning_rate = 0.1
        self.initial_weight = initial_weight
        #float(param['initial_weight']) if 'initial_weight' in param else 0.5
        self.discount_rate = discount_rate
        #float(param['discount_rate']) if 'discount_rate' in param else 1
        self.Visualization = visualize
        #'visualize' in param and bool(param['visualize'])
        # self.discount_rate = 0.005**(1/self.N)
        np.random.seed(123)

        self.Hist1 = OrderedDict()
        self.Hist2 = OrderedDict()

        self.log = False

        ## Accounting variables
        self.time = 0
        self.timer = 0
        self.W = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)

        self.X = []
        self.Y1 = []
        self.Y2 = []
        self.eTime = {}

        self.unique = {}
        self.unique_cnt = 0
        self.reused_block_count = 0
        self.page_entering_cache = {}
        self.unique_block_count = 0
        self.block_reused_duration = 0
        self.page_lifetime_cache = {}
        self.block_lifetime_duration = 0
        self.block_lifetime_durations = []
        self.pollution_dat_x = []
        self.pollution_dat_y = []
        self.pollution_dat_y_val = 0
        self.pollution_dat_y_sum = []
        self.pollution = 0

        ## Learning Rate adaptation variables
        self.reset_point = self.learning_rate
        # self.reset_point = 0.45
        self.seq_len = int(1 * self.N)
        ### PLot with self.seq_len = int( 2 * self.N )
        self.reset_seq_len = int(self.N)
        self.CacheHit = 0
        self.PreviousHR = 0.0
        self.NewHR = 0.0
        self.PreviousChangeInHR = 0.0
        self.NewChangeInHR = 0.0
        self.learning_rate = np.sqrt(2 * np.log(2) / self.N)
        self.reset_point = min(1, max(0.001, self.learning_rate))
        self.max_val = min(1, max(0.001, self.learning_rate))
        self.PreviousLR = 0.0
        # self.PreviousLR = 0.0
        self.NewLR = self.learning_rate
        self.learning_rates = []

        self.SampleHR = []

        self.SampleCIR = 0
        self.hitrate_negative_counter = 0
        self.hitrate_zero_counter = 0

        self.LRU = 0
        self.LFU = 1
        self.page_fault_count = 0

        self.info = {
            'lru_misses': 0,
            'lfu_misses': 0,
            'lru_count': 0,
            'lfu_count': 0,
        }

    def get_N(self):
        return self.N

    def __contains__(self, q):
        return q in self.Map

    def getWeights(self):
        return np.array([self.X, self.Y1, self.Y2, self.pollution_dat_x, self.pollution_dat_y]).T

    #         return np.array([self.pollution_dat_x,self.pollution_dat_y ]).T

    def getPollutions(self):
        return self.pollution_dat_y_sum

    def getLearningRates(self):
        return self.learning_rates

    def get_block_reused_duration(self):
        return self.block_reused_duration

    def get_block_lifetime_duration(self):
        for pg in self.LFUalg:
            self.block_lifetime_duration += self.time - self.page_lifetime_cache[pg]
            self.unique_block_count += 1
            self.block_lifetime_durations.append(self.time - self.page_lifetime_cache[pg])
        print("Unique no of blocks", self.unique_block_count)
        return self.block_lifetime_duration / float(self.unique_block_count)

    def get_block_lifetime_durations(self):
        return self.block_lifetime_durations

    def getStats(self):
        d = {}
        d['weights'] = np.array([self.X, self.Y1, self.Y2]).T
        d['pollution'] = np.array([self.pollution_dat_x, self.pollution_dat_y]).T
        return d

    def visualize(self, ax_w, ax_h, averaging_window_size):
        lbl = []
        if self.Visualization:
            X = np.array(self.X)
            Y1 = np.array(self.Y1)
            Y2 = np.array(self.Y2)
            ax_w.set_xlim(np.min(X), np.max(X))
            ax_h.set_xlim(np.min(X), np.max(X))

            ax_w.plot(X, Y1, 'y-', label='W_lru', linewidth=2)
            ax_w.plot(X, Y2, 'b-', label='W_lfu', linewidth=1)
            # ax_h.plot(self.pollution_dat_x,self.pollution_dat_y, 'g-', label='hoarding',linewidth=3)
            # ax_h.plot(self.pollution_dat_x,self.pollution_dat_y, 'k-', linewidth=3)

            ax_h.set_ylabel('Hoarding')
            ax_w.legend(loc=" upper right")
            ax_w.set_title('LeCaR')
            pollution_sums = self.getPollutions()
            temp = np.append(np.zeros(averaging_window_size), pollution_sums[:-averaging_window_size])
            pollutionrate = (pollution_sums - temp) / averaging_window_size

            ax_h.set_xlim(0, len(pollutionrate))

            ax_h.plot(range(len(pollutionrate)), pollutionrate, 'k-', linewidth=3)

        return lbl

    ##############################################################
    ## There was a page hit to 'page'. Update the data structures
    ##############################################################
    def pageHitUpdate(self, page):
        block = self.Map[page]
        block.time = self.time
        block.freq += 1

    ######################
    ## Get LFU or LFU page
    ## 0 lfu
    ## 1 lirs
    ######################
    def selectEvictPage(self, policy):
        f = None
        h = None

        Sample = random.sample(self.IndexArray, self.SampleSize)

        for pageIndex in Sample:
            pageNumber = self.ValueArray[pageIndex]
            if pageNumber == -2:
              print("BOE")
              print(pageIndex)
            
            pageBlock = self.Map[pageNumber]
            

            if f is None:
                f = pageBlock
                h = pageBlock
            else:
                if f.freq > pageBlock.freq:
                    f = pageBlock
                if float(h.freq) / float(self.time - h.time) > float(pageBlock.freq) / float(self.time - pageBlock.time):
                    h = pageBlock

        pageToEvit, policyUsed = None, None

        if f is h:
            pageToEvit, policyUsed = f, -1
        if policy == 0:
            pageToEvit, policyUsed = f, 0
        elif policy == 1:
            pageToEvit, policyUsed = h, 1

        return pageToEvit, policyUsed

    def evictPageAndAddToCache(self, index, old_page, new_page):

        new_page.index = index
        new_page.time = self.time
        new_page.freq += 1

        #print(index)

        if old_page != -1:
            assert self.ValueArray[index] == old_page

        self.ValueArray[index] = new_page.page

        if old_page in self.Map:

            del self.Map[old_page]

        assert -1 not in self.Map
        assert -2 not in self.Map

        self.Map[new_page.page] = new_page

    def getQ(self):
        lamb = 0.05
        return (1 - lamb) * self.W + lamb

    ############################################
    ## Choose a page based on the q distribution
    ############################################
    def chooseRandom(self):
        r = np.random.rand()
        if r < self.W[0]:
            return 0
        return 1

    def addToseparateHistory(self, poly, cacheevic):
        histevict = None
        assert len(self.Hist1) <= self.H
        assert len(self.Hist2) <= self.H

        # if (poly == 0) or (poly==-1 and np.random.rand() < 0.5):
        if poly == 0:
            if len(self.Hist1) == self.H:
                histevict = next(iter(self.Hist1))
                del self.Hist1[histevict]
                # if histevict.page is not None:
                #     del self.Hist1[histevict.page]
            self.Hist1[cacheevic.page] = cacheevic
            assert len(self.Hist1) >= 1
        elif poly == 1:
            if len(self.Hist2) == self.H:
                histevict = next(iter(self.Hist2))
                del self.Hist2[histevict]
                # if histevict.page is not None:
                #     del self.Hist2[histevict.page]
            self.Hist2[cacheevic.page] = cacheevic
            assert len(self.Hist2) >= 1

        #if histevict is not None:
        #del self.eTime[histevict]

    def updateInDeltaDirection(self, delta_LR):

        delta = 0
        delta_HR = 1

        if (delta_LR > 0 and self.NewChangeInHR > 0) or (delta_LR < 0 and self.NewChangeInHR < 0):
            delta = 1
        elif (delta_LR < 0 and self.NewChangeInHR > 0) or (delta_LR > 0 and self.NewChangeInHR < 0):
            delta = -1

        elif (delta_LR > 0 or delta_LR < 0) and self.NewChangeInHR == 0:
            delta_HR = 0

        return delta, delta_HR

    def updateInRandomDirection(self):

        if self.learning_rate >= 1:
            self.learning_rate = 0.9
            # print("After LR equal and Inside negative extreme")
        elif self.learning_rate <= 0.001:
            self.learning_rate = 0.005
        else:
            val = round(np.random.uniform(0.001, 0.1), 3)
            val = np.random.rand()
            increase = np.random.choice([True, False])

            if increase:
                self.learning_rate = min(self.learning_rate + abs(self.learning_rate * 0.25), 1)
            else:

                self.learning_rate = max(self.learning_rate - abs(self.learning_rate * 0.25), 0.001)

    def updateLearningRates(self):

        if self.time % (self.seq_len) == 0:

            self.NewHR = round(self.CacheHit / float(self.seq_len), 3)
            self.NewChangeInHR = round(self.NewHR - self.PreviousHR, 3)

            # self.updateSamples()
            delta_LR = round(self.NewLR, 3) - round(self.PreviousLR, 3)
            delta, delta_HR = self.updateInDeltaDirection(delta_LR)

            # if self.page_fault_count >=  self.N:
            #     self.page_fault_count  = 0
            #     if self.W[0] > 0.5 :
            #         self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)

            if delta > 0:
                self.learning_rate = min(self.learning_rate + abs(self.learning_rate * delta_LR), 1)

                # print("Inside positive update",delta_LR, self.NewChangeInHR,self.learning_rate)
                self.hitrate_negative_counter = 0
                self.hitrate_zero_counter = 0


            elif delta < 0:
                self.learning_rate = max(self.learning_rate - abs(self.learning_rate * delta_LR), 0.001)
                # print("Inside negative update", delta_LR, self.NewChangeInHR, self.learning_rate)
                self.hitrate_negative_counter = 0
                self.hitrate_zero_counter = 0


            elif delta == 0 and (self.NewChangeInHR <= 0):

                if (self.NewHR <= 0 and self.NewChangeInHR <= 0) or self.NewChangeInHR < 0:
                    self.hitrate_zero_counter += 1

                if self.NewChangeInHR < 0:
                    self.hitrate_negative_counter += 1

                if self.hitrate_zero_counter >= 10:

                    self.learning_rate = self.reset_point
                    self.timer = 0
                    # if self.hitrate_negative_counter >= 5:
                    #     self.hitrate_negative_counter = 0
                    #     if self.W[0] > 0.5:
                    #         self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)
                    # if self.W[0] > 0.5:
                    #     self.W = np.array([self.initial_weight,1-self.initial_weight], dtype=np.float32)

                    self.hitrate_zero_counter = 0
                    # self.hitrate_negative_counter = 0

                elif self.NewChangeInHR < 0:

                    # # if self.hitrate_negative_counter >= 5:
                    # #     self.hitrate_negative_counter = 0
                    # if self.W[0] > 0.5:
                    #     self.W = np.array([self.initial_weight, 1 - self.initial_weight], dtype=np.float32)
                    #
                    # else:
                    self.updateInRandomDirection()

            self.PreviousLR = self.NewLR
            self.NewLR = self.learning_rate
            self.PreviousHR = self.NewHR
            self.PreviousChangeInHR = self.NewChangeInHR
            self.CacheHit = 0

    ########################################################################################################################################
    ####REQUEST#############################################################################################################################
    ########################################################################################################################################
    def request(self, page):
        #print("request %d" % page)
        page_fault = False
        self.time = self.time + 1
        self.timer = self.timer + 1

        #####################
        ## Visualization data
        #####################
        if self.Visualization:
            self.X.append(self.time)
            self.Y1.append(self.W[0])
            self.Y2.append(self.W[1])

        self.updateLearningRates()

        ##########################
        ## Process page request
        ##########################
        if page in self.Map:
            #print("HIT")
            page_fault = False
            self.pageHitUpdate(page)
            self.CacheHit += 1
        else:
            #print("MISS")
            #####################################################
            ## Learning step: If there is a page fault in history
            #####################################################
            pageevict = None

            reward = np.array([0, 0], dtype=np.float32)

            if page in self.Hist1:
                #print("In H1")
                pageevict = self.Hist1[page]

                del self.Hist1[page]
                reward[0] = -1

            elif page in self.Hist2:
                #print("In H2")
                pageevict = self.Hist2[page]

                del self.Hist2[page]
                reward[1] = -1
            #else:
                #print("Total")
            #################
            ## Update Weights
            #################
            if pageevict is not None:
                assert pageevict.page == page
                assert pageevict.freq >= 1
                assert pageevict.time >= 0

                self.W = self.W * np.exp(self.learning_rate * reward)
                self.W = self.W / np.sum(self.W)

                if self.W[0] >= 0.99:
                    self.W = np.array([0.99, 0.01], dtype=np.float32)

                elif self.W[1] >= 0.99:
                    self.W = np.array([0.01, 0.99], dtype=np.float32)
            else:
                pageevict = CacheMetaData()
                pageevict.page = page

                pageevict.index = -1
                pageevict.freq = 0
                pageevict.time = -1

            ####################
            ## Remove from Cache
            ####################

            if self.IndexPointer == self.N:
                ################
                ## Choose Policy
                ################
                act = self.chooseRandom()
                cacheevict, poly = self.selectEvictPage(act)
                self.eTime[cacheevict] = self.time

                #################
                ## Remove from Cache and Add to history
                #################
                self.evictPageAndAddToCache(cacheevict.index, cacheevict.page, pageevict)
                self.unique_block_count += 1

                self.addToseparateHistory(poly, cacheevict)
            else:
                self.evictPageAndAddToCache(self.IndexPointer, -1, pageevict)
                self.IndexPointer += 1

            page_fault = True

        if page_fault:
            self.unique_cnt += 1
        self.unique[page] = self.unique_cnt

        self.learning_rates.append(self.learning_rate)
        return page_fault

    def get_list_labels(self):
        return ['L']

