import json
import debugger
import railMLTTParser as TTParser
import multiprocessing.process as process
import multiprocessing.pool as pool
from multiprocessing import Process, Lock
from multiprocessing.sharedctypes import Value, Array

class TimetableGenerator:
    def __init__(self):
        #create objects 
        self.TimetableTemplate = None

    def GenerateHour(self,hour_lineplan, state_file="", isdebug = False):
        return  self.TimetableTemplate
    
    def Generate(self,template_file, lineplan_file, state_file, isdebug = False):   
        #for key, value in lpdata.items():
        #     if key == "hourly_hw" : 
        #         for el in lpdata[key]:
        #             print (el["hour"])

        #for el in lpdata["hourly_hw"]:
        #    print (el["hour"])
        #load the json file into lpdata
        tt_parser = TTParser.railMLTTParser(template_file)
        tt_parser.ReadTTRailMLFile()
        self.TimetableTemplate = tt_parser.RailMLTTString

        with open(lineplan_file) as lineplanfileobj:
            lpdata = json.load(lineplanfileobj)
        if(isdebug):
            debugger.PrintDebugMessage("Lineplan file loaded successfully, content:", lpdata)

        if(isdebug):
            debugger.PrintDebugMessage("Timetable file loaded successfully, content:", tt_parser.RailMLTTFile)
        
        with pool.Pool(5) as p:
            print('\n')
            print(p.map(self.GenerateHour, lpdata["hourly_hw"]))

