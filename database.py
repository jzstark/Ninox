# name of tables: asp; bsp; ssp_s10; pssp_s10_p2; ...
import csv
import os

default_dir = "~/Tmp/Ninox/data"

def init_db(dir):
    try:
        os.makedirs(dir)
    except Exception as e:
        return
