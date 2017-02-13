import xml.etree.ElementTree as ET
import datetime
from utilities import Utilities
import math
class LinePlanDataStruct:
    HOURE_FIELD = "hour"
    T_FIELD = "T"
    HOUR_START_TIME_FIELD = "starting_time"
    HOUR_START_TRAIN_ID_FIELD = "starting_trainID"
    MIN_HEADWAY_TIME_FIELD = "min_headway"
    MACRO_PLANNING_HORIZON_FIELD = "macro_planning_horizon"
    MICRO_PLANNING_HORIZON_FIELD =  "micro_planning_horizon"

class railMLTTParser:
    def __init__(self):
        self._file_name = None
        self.railMLTTFile = None
    
    def ReadTTRailMLFile(self, file_name):
        self._file_name = file_name
        self.railMLTTFile = ET.parse(self._file_name)
    def WriteRailMLFile(self, ofile_name ,railMLString):
        xml = ET.fromstring(railMLString)
        xml.Write(ofile_name);

    def WriteRailMLFileFromList(self, ofile_name ,railMLStringList):
        xml = ET.fromstringlist(railMLStringList)
        xml.Write(ofile_name);

    def DumpRailMLFileFromList(self, ofile_name, railMLStringList):
        _file = open(ofile_name, "wb")
        for ttstr in railMLStringList:
            _file.write(ttstr)
        _file.close()

    @property
    def RailMLTTFile(self):
        return self.railMLTTFile
    @property  
    def RailMLTTString(self):
        return ET.tostring(self.railMLTTFile, "utf-8")

class LinePlanFile:
    def __init__(self):
        #these parameter are changed later after loading the line plan file
        self.MicroPlanningHorizon = "1320"
        self.MacroPlanningHorizon = 60
        self.StartingHour = datetime.datetime(hour=5)
        self.EndingHour = datetime.datetime(hour=24)
        self.T = 10
        
    def loadLinePlan(self, lineplan_file):
        with open(lineplan_file) as lineplanfileobj:
            lpdata = json.load(lineplanfileobj)
 
class ScopedLinePlan:
    def __init__(self, _hourPlan):
        utl = Utilities()
        self.MicroPlanningHorizon = 60
        self.MacroPlanningHorizon = 1320
        self.OperationalHour = datetime.datetime.strptime(_hourPlan[LinePlanDataStruct.HOURE_FIELD],'%H:%M:%S').time()
        self.T = _hourPlan[LinePlanDataStruct.T_FIELD]
        #operation day start hour
        self.StartingTime = datetime.datetime.strptime(_hourPlan[LinePlanDataStruct.HOUR_START_TIME_FIELD],'%H:%M:%S').time()
        #get the strating simulation for this hour in integer format
        self.StartingSimulationTimeInt = utl.get_integer_time(self.StartingTime, self.OperationalHour)
        self.EndingTime = datetime.datetime.strptime('06:00:00','%H:%M:%S').time()
        self.TrainCount =  math.floor(self.MicroPlanningHorizon / self.T)
        self.StartingTrainID = _hourPlan[LinePlanDataStruct.HOUR_START_TRAIN_ID_FIELD]
        self.StationCount = 35
        self.ItiniraryTimeInMinutes = 60

    def getSimulationMaxTimeInt(self):
        # calculate the number of minutes needed from the first minute and untill the last train arrives its final station
        lastTrainStartingOp = self.MicroPlanningHorizon - self.T + self.StartingSimulationTimeInt
        return lastTrainStartingOp + self.ItiniraryTimeInMinutes
