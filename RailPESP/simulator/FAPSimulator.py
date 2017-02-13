import simpy
import random
import copy
from os import path
import re
from datetime import timedelta
import time
import numpy as np
import datetime
import pandas as pd
import utilities
import railml
import xml.etree
class SimulatedTrain(object):
    def __init__(self, railway, env, train_tt_record, trainStartTime, previous_train, trainID):
        #every train recieves a deep copy of the itinerary railMl details
        #every train will:
        #   Read: The departure/arrival details from the railML data
        #   Update: The even occurence time with the simulation time
        #   Return: A modifies version of the itinirary to the railway object
        self.utl = utilities.Utilities()
        self.env = env
        self.railway = railway
        # get a deep copy of the timetable for one train, update it with event time
        self.train_TT = train_tt_record
        self.trainStartTime = trainStartTime
        self.first_op_time = None
        self.previous_train = previous_train
        self.capacity = 2000
        self.payload = self.K_current = self.R_current = self.P_current =  self.PLF_current = 0
        self.WT_TOTAL = self.PLF_RAW_TOTAL = self.P_TOTAL = 0
        self.trainPart = self.train_TT #self.train_TT.find("timeTable").find("trainParts").find("trainPart")
        #if self.previous_train is None: # this is the first train
        #    self._code = self.trainPart.get("id")
        #    #dt = datetime.datetime.strptime(train_tt_record['OcpsTT']['ocpTT'][0]['times']['@departure'], '%H:%M:%S')
        #    #trainStartingTime = self.
                    #datetime.datetime.strptime(train_tt_record['OcpsTT']['ocpTT'][0]['times']['@departure'], '%H:%M:%S')
        #else:
        #    self._code = str(int(self.trainPart.get("id")) + 2)
        #    #self.train_TT['@id'] = self._code
        self._code = trainID
        self.trainPart.set("id", str(self._code))
            #dt = datetime.datetime.combine(datetime.date.today(), self.previous_train.first_op_time)  + timedelta(minutes= railway.headway())
        self.first_op_time = self.trainStartTime #self.utl.convert_datetime_to_time(dt)

        if self.previous_train is None:
            self.previouse_first_op_time = self.first_op_time
        else:
            self.previouse_first_op_time = self.previous_train.first_op_time

        self.start_min = self.utl.get_integer_time(self.first_op_time, trainStartTime)
		#start the run process every time we create new instant
        self.action = env.process(self.run())
    @property
    def ID(self):
        return self._id

    @property
    def Code(self):
        return self._code

    @property
    def Description(self):
        return self._desc

    def run(self):
        #init everything
        last_arr = last_dep = t0 = t1 = 0.0
        #initia the statistics object assigned to train trip (trip code = train code)
        self.railway.timetableStatsInfo.init_trip_stat_info(self.Code)
        #each train start in scheduled time
        until_start = self.start_min - self.env.now
        yield self.env.process(self.dwell_process(until_start))
        running = True
        while running:
            #loop through the train timetable OcpsTT dictionary:
            #for current_op in self.train_TT['OcpsTT']['ocpTT']:
            for current_op in self.trainPart.find('OcpsTT').findall('ocpTT'):
                # if its the source operation so it yields a departure event only
                current_op_duration = self.getRunningTime(current_op)
                current_op_delay = self.calc_sim_delay()
                if(current_op.get('ocpType') == 'begin'):
                    #print the name of the first station
                    ##################
                    #???print ('Train %s is now starting itinirary and departing station: %s at time: %d' %(self.Code,current_op['@ocpRef'],self.env.now))
                    # calculate the passenger waiting time for 
                    self.updateEventTime(current_op, self.env.now,'dep')
                    #self.register_train_first_departure(current_op['@ocpRef'])
                    #yield a runing process
                    yield self.env.process(self.running_process(current_op_duration, current_op_delay))
                else:
                    #???print ('Train %s arrives to station: %s @ time: %d' %(self.Code,current_op['@ocpRef'],self.env.now))
                    ##################
                    self.updateEventTime(current_op, self.env.now,'arr')
                    #self.register_train_arrival(current_op['@ocpRef'])
                    #Dwell process
                    yield self.env.process(self.dwell_process(self.getDwellTime(current_op)))
                    #???print ('Train %s departing from station: %s @ time: %d' %(self.Code,current_op['@ocpRef'],self.env.now))
                    ##################
                    self.updateEventTime(current_op, self.env.now,'dep')
                    #self.register_train_departure(current_op['@ocpRef'])
                    yield self.env.process(self.running_process(current_op_duration, current_op_delay))
                
                last_op = current_op
            running = False
    def register_train_first_departure(self, station):
        if(self.previouse_first_op_time == self.first_op_time): # first train
            t0 = self.trainStartTime
        else:
            t0 = self.previouse_first_op_time

        t1 = self.utl.get_time(self.env.now, self.trainStartTime,False)
        #psgr = pflw.get_number_of_passenger_by_interval(station, t0, t1)
        
        p, wt = self.PA(station,t0,t1)
        self.P_current = p
        self.P_TOTAL+=p
        #accepted passengers
        self.K_current = self.estimate_accepted_psgrs(station)
        self.payload+= self.K_current # update the train load with accepted passengers from the arrival process.
        #rejected passengers
        self.R_current = p -self.K_current
        self.railway.timetable_stats_info.record_trip_stat_info(self.Code,station,'DEP',t0,t1,p,wt,self.payload,0,0.0,self.K_current, self.R_current)
        self.WT_TOTAL+= wt
        #update station payload by rejected passengers
        sld = self.railway.StationsData.loc[station]['ocpPayload']
        sld+= self.R_current
        self.railway.StationsData.loc[station]['ocpPayload'] = sld
    '''With each train departure we calculate the passenger waiting time, train pyload'''
    def register_train_departure(self, station):
        #get the previous train departure time on the same station
        t0=0
        t1 = self.utl.get_time(self.env.now, self.trainStartTime)
        if self.previous_train is None:
            t0 = t1
        else:
            for item in self.previous_train.train_TT['OcpsTT']['ocpTT']:
                if (item['@ocpRef'] == station):
                    t0 = item['times']['@departure']
        self.payload+= self.K_current # update the train load with accepted passengers from the arrival process.

        self.railway.timetable_stats_info.record_trip_stat_info(self.Code,station,'DEP',t0,t1,0,0,self.payload,0,0,self.K_current, self.R_current)

    '''with each train arrival we recalculate train payload, PLF'''

    def get_train_leaving_psgr(self, station):
        L_total = plf_op =  K=0
        ocp_to = station
        i = len(self.railway.timetable_stats_info.TripStatInfos[self.Code].index)
        i = (i+1)/2 # to get station index as every station is recorded by 2 rows (dep, arr) excpet forst one only dep.
        for index, row in self.railway.timetable_stats_info.TripStatInfos[self.Code].iterrows():
            if row['OCPType'] == 'ARR':
                ocp_from = row['OcpRef']
                K = int(row['K'])
                L_total = K * self.railway.ODM(ocp_from, ocp_to)
                plf_op += L_total * i
                i-=1

        return L_total, plf_op
    def register_train_arrival(self, station):
        #leaving passengers and plf_raw
        l_i, plf_raw = self.get_train_leaving_psgr(station)
        self.PLF_current = math.ceil(plf_raw)
        self.PLF_RAW_TOTAL+=self.PLF_current
        l_i = math.ceil(l_i)
        #waiting passengers
        t0=0
        t1 = self.utl.get_time(self.env.now,self.trainStartTime, False)
        if self.previous_train is None:
            t0 = self.trainStartTime #get waiting passengers since simulation start
        else:
            for item in self.previous_train.train_TT ['OcpsTT']['ocpTT']:
                if (item['@ocpRef'] == station):
                    t0 = item['times']['@arrival']

        p, wt = self.PA(station,t0,t1)
        #if self.previous_train is None: 
         #   wt = 0 #for first train we assume passenger on all station will check the schedule before they arrive
        self.P_current = p
        self.P_TOTAL+=p
        #train payload
        self.payload-= l_i
        #accepted passengers
        self.K_current = self.estimate_accepted_psgrs(station)
        #rejected passengers
        self.R_current = p-self.K_current
        self.WT_TOTAL+=wt
        #self.payload+= self.K_current
        self.railway.timetable_stats_info.record_trip_stat_info(self.Code,station,'ARR',t0,t1,p,wt,self.payload,0,plf_raw,self.K_current, self.R_current)
        #update station payload by rejected passengers
        sld = self.railway.StationsData.loc[station]['ocpPayload']
        sld+= self.R_current
        self.railway.StationsData.loc[station]['ocpPayload'] = sld

    def estimate_accepted_psgrs(self, station):
        prev_R = self.railway.StationsData.loc[station]['ocpPayload']
        pot_psgr = self.payload + self.P_current + prev_R
        chance = 0.0
        if pot_psgr < self.capacity:
            chance = 1.0
        elif (self.capacity*1.5) > pot_psgr > self.capacity:
            chance = 0.8
        elif (self.capacity*2.3) > pot_psgr > (self.capacity * 1.5):
            chance = 0.6
        elif (self.capacity*2.3) > pot_psgr > (self.capacity * 1.5):
            chance = 0.1
        elif (pot_psgr >self.capacity*2.5):
            chance = 0.0
        
        return self.P_current * chance # return the accepted number of passengers
        
    def register_train_last_arrival(station, time):
        pass

    def PA(self, ocp, t0,t1):
        t0= self.utl.get_integer_time(t0,self.trainStartTime)
        t1 = self.utl.get_integer_time(t1, self.trainStartTime)
        p = psgr = wt = 0
        if t0==t1:
            psgr = self.railway.psgr_data.loc[t0][ocp]
            wt = 0

        for t in range(t0,t1):
            p = self.railway.psgr_data.loc[t][ocp]
            psgr+=p
            wt+= (p * (t1-t))
        return psgr, wt
    def updateEventTime(self, ocp, simu_env_now, type):
        #convert simu_env_now to hours and minutes
        now_hour = int(simu_env_now / 60)
        now_minute = int(simu_env_now % 60)
        ocp.find('times').set('scope',  "calculated")
        event_time = datetime.time.strftime((datetime.datetime.combine(datetime.date.today(), self.trainStartTime) 
                                                          + datetime.timedelta(hours=now_hour, minutes=now_minute, seconds=0)).time(),
                                                         '%H:%M:%S')
        if(type == 'arr'):
            ocp.find('times').set('arrival', event_time)
        else:
            ocp.find("times").set("departure", event_time)

    def getRunningTime(self, ocp):
        #run_time_min = self.utl.parse_time(ocp['sectionTT']['runTimes']['@minimalTime'])
        runTimes = ocp.find('sectionTT').find("runTimes")
        run_time_min =  self.utl.parse_time(runTimes.get("minimalTime"))
        #run_time_reserve = self.utl.parse_time(ocp['sectionTT']['runTimes']['@operationalReserve'])
        run_time_reserve = self.utl.parse_time(runTimes.get("operationalReserve"))
        #run_add_reserve = self.utl.parse_time(ocp['sectionTT']['runTimes']['@additionalReserve'])
        run_add_reserve = self.utl.parse_time(runTimes.get("additionalReserve"))
        total_run_time = (run_time_min.total_seconds() + run_time_reserve.total_seconds() + run_add_reserve.total_seconds()) / 60
        return total_run_time
    
    def getDwellTime(self, ocp):
        stopTimes = ocp.find('stopDescription').find("stopTimes")
        
        dwell_min = self.utl.parse_time(stopTimes.get("minimalTime"))
        dwell_reserve = self.utl.parse_time(stopTimes.get("operationalReserve"))
        dwell_add_reserve = self.utl.parse_time(stopTimes.get("additionalReserve"))
        dwell_shuntingTime = self.utl.parse_time(stopTimes.get("shuntingTime"))
        dwell_clearanceTime = self.utl.parse_time(stopTimes.get("clearanceTime"))
        
        total_dwell_time = (dwell_min.total_seconds() + dwell_reserve.total_seconds() + dwell_add_reserve.total_seconds() + dwell_shuntingTime.total_seconds() + dwell_clearanceTime.total_seconds())/60
        return total_dwell_time

    def calc_sim_delay(self):
        #return random.randint(0,3)
        return 0
    def dwell_process(self, duration):
        yield self.env.timeout(duration)

    def	running_process(self, duration, delay):
		#move process is a train operation between 2 stations
        yield self.env.timeout(duration+delay)
    
class SimulatedRailway(object):
    def __init__(self, _linePlan, ttDoc):
        self.linePlan = railml.ScopedLinePlan(_linePlan)
        self.utl = utilities.Utilities()
        #self.linePlan = _linePlan
        self.scheduledTimetable = ttDoc
        #before simulation begins both scheduled and calculated timetables are same
        self.calculatedTimetable = copy.deepcopy(self.scheduledTimetable)
        self.timetableStatsInfo = utilities.TimetableStatInformation()  
        self.env = simpy.Environment()
        self.psgrData = pd.DataFrame()
        self.odm = pd.DataFrame()
        self.StationsData = pd.DataFrame()
    @property
    def ScheduledTimetable(self):
        return self.scheduledTimetable
    @property 
    def CalculatedTimetable(self):
        return self.calculatedTimetable

    @CalculatedTimetable.setter
    def CalculatedTimetable(self, value):
        self.calculatedTimetable = value

    def gen_station_frame(self, train):
        stations = pd.DataFrame(columns = ['ocpRef', 'ocpPayload'])
        for item in train['OcpsTT']['ocpTT']:
            stations.loc[len(stations)] =  [item['@ocpRef'], 0]
        self.StationsData  = stations.set_index('ocpRef')
        ##################################### TT3#################################
    def runSimulation(self):
        # this method create simulatedTrain instance for each train in the line plan
        # each train run should return a modified copy of railML timetable for the train itinirary
        # the railway object tp cpmbine these version producing an entire timetable for the subject hour
        #self.odm = pd.DataFrame.from_csv(self.utl.get_data_path('l1sat.odm'))
        #self.psgr_data = pd.DataFrame.from_csv(self.utl.get_data_path('PA010.csv'))
        if True:
            train_set = set([])
            #self.gen_station_frame(train_dict)
            start =  self.linePlan.StartingTime
            previous_train = None
        
            for i in range(1,self.linePlan.TrainCount+1):
                _trainID = self.linePlan.StartingTrainID + ((i - 1)*2)
                #trainPart = 
                trainStartingTime = self.utl.convert_datetime_to_time(self.utl.convert_time_to_datetime(self.linePlan.StartingTime) 
                                                       + datetime.timedelta(minutes=self.linePlan.T * (i-1)))
                atrain = SimulatedTrain(self, self.env,
                                        copy.deepcopy(self.CalculatedTimetable.getroot().find('timeTable').find('trainParts').find('trainPart')),
                                        trainStartingTime, previous_train, _trainID)
                train_set.add(atrain)
                previous_train = atrain
                
            self.env.run(self.linePlan.getSimulationMaxTimeInt())#until=self.end_time)
            
            TT_TOTAL_WT = TT_TOTAL_PLF = TT_TOTAL_P =  0
            self.CalculatedTimetable.getroot().find('timeTable').find('trainParts').clear()
            for train in train_set:
                #print("train " + str(train.Code))
                #print("WT: " + str(train.WT_TOTAL), "PLF_RAW:" + str(train.PLF_RAW_TOTAL))
                TT_TOTAL_WT+=train.WT_TOTAL
                TT_TOTAL_PLF+=train.PLF_RAW_TOTAL
                TT_TOTAL_P+=train.P_TOTAL
                #combine each train modified trainPart object into one railML timetable
                self.CalculatedTimetable.getroot().find('timeTable').find('trainParts').append(train.train_TT)
            #TT_TOTAL_PLF = TT_TOTAL_PLF / (self.linePlan.TrainCount * atrain.capacity)
            #TT_TOTAL_WT = TT_TOTAL_WT / TT_TOTAL_P 
            #self.utl.write_data_file('calc_timetable.xml', self.CalculatedTimetable)
            self.CalculatedTimetable.write('SimTimetables\calc_timetable_' + str(self.linePlan.OperationalHour.hour) + '.xml')
            print(self.CalculatedTimetable)
            #self.env.exit()
    def ODM(self, ocp_from, ocp_to):
        return self.odm.loc[ocp_from][ocp_to]