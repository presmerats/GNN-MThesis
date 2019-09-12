
"""

Read all files inside tasks/
for each file:
    yaml or json to dict
    explode all combinations (sklearn or by )
    for each combination write yaml file

"""
import yaml
import os
import json
import logging
import sys
import shutil

parentdir = os.path.join(os.path.abspath('..'))
sys.path.insert(0,os.path.join(parentdir,'function_renaming'))
sys.path.insert(0,'..')
#sys.path.insert(0,'../..')
from function_renaming.TFM_function_renaming_baseline_models import *


# logger setup
logger = logging.getLogger('training_jobs')
logger.setLevel(logging.DEBUG)
log_file= 'trainings.log'
fh = logging.FileHandler(log_file)
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)


def get_last_id():
    """
        inspect all files inside a folder 
    """
    idx=0
    for r,d,f, in os.walk('trains/'):
        for file in f:
            i = file.find('.yml')
            if i>-1:
                id_str = file[5:i]
                if idx < int(id_str):
                    idx = int(id_str)

    return idx

idx = get_last_id()
for root, dirs, files in os.walk('tasks/',topdown=False):
    for f in files:
        # read task dict
        task = {}
        f = os.path.join(root,f)
        logger.debug(os.path.splitext(f))
        if os.path.splitext(f)[1] == '.yml' or os.path.splitext(f)[1] == '.yaml':
            task = yaml.load(open(f,'r'), Loader=yaml.FullLoader)
        else:
            task = json.load(open(f,'r'))

        # explode all combinations
        combos = unroll_all_possible_pipeline_model_combos(task,None, {})
        #pprint(combos)

        # write each of those as a yaml file in trains/
        for comb in combos:
            with open('trains/task_'+str(idx)+'.yml','w') as f2:
                yaml.dump(comb,f2)
                idx+=1

                try:
                    shutil.move(f,f.replace('tasks','done_tasks'))
                except:
                    pass





# filename1='trains/model01.yml'
# with open(filename1,'w') as f:
# 	model01 = {
# 		'name': 'model01'
# 	}
# 	yaml.dump(model01,f)
# 	print("writing ",filename1)


# filename1='trains/model02.yml'
# with open(filename1,'w') as f:
# 	model01 = {
# 		'name': 'model02'
# 	}
# 	yaml.dump(model01,f)
# 	print("writing ",filename1)