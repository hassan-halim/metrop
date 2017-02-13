import os
import json
from utilities import Debugger
from railml import railMLTTParser
import multiprocessing.process as process
import multiprocessing.pool as pool
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array
import copy
import simpy
import simulator.FAPSimulator as simulator
class TimetableGenerator:
    def __init__(self):
        #create objects 
        self.TimetableTemplate = None
    def callf(self):
        return str(os.getpid()) 
    
    def GenerateHour(self,hour_lineplan, state_file="", isdebug = False):
        #return  self.TimetableTemplate
        # Edit timetable XML to remove parent nodes
        # Check how many trains in the plan (n) and copy the timetable n times
        # update the n timetable instances with the departure and arrival generated by the simulator
        # N.B. hour_lineplan is a dictionary object
        #_numtrains = hour_lineplan[LinePlanDataStruct.T_FIELD]
        #_trainops = list()
        railway = simulator.SimulatedRailway(hour_lineplan, self.TimetableTemplate)
        railway.runSimulation()
    
    def Generate(self,template_file, lineplan_file, state_file, isdebug = False):
        #for el in lpdata["hourly_hw"]:
        #    print (el["hour"])
        #load the json file into lpdata
        tt_parser = railMLTTParser()
        tt_parser.ReadTTRailMLFile(template_file)
        self.TimetableTemplate = tt_parser.RailMLTTFile

        with open(lineplan_file) as lineplanfileobj:
            lpdata = json.load(lineplanfileobj)
        if(isdebug):
            Debugger.PrintDebugMessage("Lineplan file loaded successfully, content:", lpdata)

        if(isdebug):
            Debugger.PrintDebugMessage("Timetable file loaded successfully, content:", tt_parser.RailMLTTFile)
        #test generate one hour
        #self.GenerateHour(lpdata["hourly_hw"][0])
        #return
        with pool.Pool(os.cpu_count()) as p:
            result = p.map(self.GenerateHour, lpdata["hourly_hw"])
            #tt_parser.DumpRailMLFileFromList("data\outrailfile", result)
            print ("Completed")
            print(os.getpid())
            