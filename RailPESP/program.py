
import argparse
from generator import TimetableGenerator
import simulator.RealTimeSimulator as rsimulator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("option")
    parser.add_argument("--mode", default="gen", choices=["gen","sim"], help="Simulation run mode")
    parser.add_argument("--template",default="" ,help="Timetable railML template to generate timetable with")
    parser.add_argument("--lineplan", default="",help="line plan file")
    parser.add_argument("--debug", default=0,help="run in debug mode?")
    args = parser.parse_args()
    option = args.option
    mode = args.mode
    tt_template = args.template
    lineplan = args.lineplan
    debug = args.debug
    if(mode == "gen"):
        generator = TimetableGenerator()
        generator.Generate(tt_template,lineplan,"",debug)
    elif (mode=="sim"):
        simu = rsimulator.RealTimeSimulator()
        simu.TestRealTimeSimulation()

if __name__ == "__main__":
    main()
