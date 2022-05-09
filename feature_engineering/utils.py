import configparser

config = configparser.ConfigParser()
config.read('config.ini')
max_ticks = float(config["ELAPSED"]['max_ticks'])

def cal_sec(x):
    try:
        return min(x.total_seconds(), max_ticks)
    except:
        return -1