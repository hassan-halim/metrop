import colorama
from colorama import Back, Fore, Style
import pandas as pd
import datetime
import re
import path

class Debugger:
    def PrintDebugMessage(header, content):
        colorama.init()
        print(Back.GREEN + Fore.WHITE+ header + Style.RESET_ALL)
        print(content)
    def PrintWarning(header, content):
        pass
    def PrintError(header, content):
        print(Back.GREEN + Fore.RED + header +"\n" + Style.RESET_ALL)
        print(content)


class OCPTTStatInformation(pd.DataFrame):
    def __init__(self, data = None, index = None, columns = None, dtype = None, copy = False):
        columns = ['TrainCode', 'OcpRef','OCPType', 'T0', 'T1', 'PsgrCount','PsgrWaitingTime', 'TrainPayload','OCPPayload', 'PLFRaw', 'K', 'R']
        return super().__init__(data, index, columns, dtype, copy)
    def add_info(self,train_code, ocp_ref,optype, t0, t1, psgr_count,psgr_wt, train_payload, ocp_payload, plf,K,R):
        next_index=len(self.index)
        self.loc[next_index] = [train_code,ocp_ref,optype, t0, t1, psgr_count,psgr_wt, train_payload, ocp_payload, plf, K,R]
    def __iter__(self):
        return super().__iter__()
'''Store list of all time table trips statistical figures '''
class TimetableStatInformation:
    def __init__(self):
        self.TripStatInfos = dict()
    
    def init_trip_stat_info(self, tripcode):
        self.TripStatInfos[tripcode] = OCPTTStatInformation()
    
    def record_trip_stat_info(self, tripcode, ocp_ref, optype, t0, t1, psgr_count, psgr_wt, train_payload=0, ocp_payload=0, plf=0, K=0, R=0):
        self.TripStatInfos[tripcode].add_info(tripcode, ocp_ref, optype, t0, t1, psgr_count, psgr_wt, train_payload, ocp_payload, plf,K,R)
class Utilities(object):
    TIME_FORMAT = '%H:%M:%S'
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    def get_integer_time(self, time, beginingTime):
        if type(time) is str:
            t1 = datetime.datetime.strptime(time,self.TIME_FORMAT).time()
        elif type(time) is datetime.datetime:
            t1 = datetime.time(hour= time.hour, minute = time.minute, second = time.second)
        elif type(time) is datetime.time:
            t1 = time

        if type(beginingTime) is str:
            t0 = datetime.datetime.strptime(beginingTime,self.TIME_FORMAT).time()
        elif type(beginingTime) is datetime.datetime:
            t0 = datetime.time(hour= beginingTime.hour, minute = beginingTime.minute, second = beginingTime.second)
        elif type(beginingTime) is datetime.time:
            t0 = beginingTime

        t = (datetime.datetime.combine(datetime.date.today(), t1)) - (datetime.datetime.combine(datetime.date.today(), t0))
        return int(t.seconds / 60)
    
    def convert_datetime_to_time(self, dt):
        return datetime.time(hour = dt.hour, minute = dt.minute, second = dt.second)

    def convert_time_to_datetime(self, dt):
        return datetime.datetime(year=2017,month=1, day=1, hour = dt.hour, minute = dt.minute, second = dt.second)
        
    def get_time(self, time_float, beginingTime, delta=True):
        if type(beginingTime) is str:
            start =  datetime.datetime.strptime(beginingTime,self.TIME_FORMAT).time()
            h = start.hour
            m = start.minute
            s = start.second
        elif type(beginingTime) is datetime.time:
            h = beginingTime.hour
            m = beginingTime.minute
            s = beginingTime.second
        time_int = int(time_float)
        s += math.floor((time_float - time_int) * 60)
        m += time_int % 60
        h += math.floor(time_int / 60)
        if delta:
            return datetime.timedelta(hours=h, minutes=m, seconds=s)
        else:
            return datetime.time(hour=h, minute=m,second=s)

        
    def write_data_file(self, file_name, date):
        file_path = self.get_data_path()
        fpath = path.join(file_path,file_name)
        file = open(fpath,'w')
        file.write(date)
        
    def get_data_path(self, file_name=''):
        script_path = path.dirname(__file__)
        if file_name =='':
            file_path = path.relpath('data/')
        else:
            file_path = path.relpath('data/'+ file_name)
        data_folder_path = path.join(script_path,file_path)
        return data_folder_path 

    def parse_time(self, time_str):
        regex = re.compile(r'((?P<hours>\d+?)hr)?((?P<minutes>\d+?)M)?((?P<seconds>\d+?)S)?')
        parts = regex.match(time_str)
        if not parts:
            return
        parts = parts.groupdict()
        time_params = {}
        for (name, param) in parts.items():
            if param:
                time_params[name] = int(param)
        return datetime.timedelta(**time_params)