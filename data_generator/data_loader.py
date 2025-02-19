import os
import sys
import re
import glob
import tqdm
import numpy as np
import pandas as pd
import pickle as pkl

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from configs import DATA_DIR, hf_token

class DataCreator: 
    def __init__(self):
        ds_names = [""]
    
    def create(ds_name):
        pass
    

if __name__ == "__main__":
    pass
            



