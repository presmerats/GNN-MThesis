
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

import sys
parentdir = os.path.join(os.path.abspath('..'))
print(os.path.join(parentdir,'function_renaming'))
sys.path.insert(0,os.path.join(parentdir,'function_renaming'))
sys.path.insert(0,'..')
#sys.path.insert(0,'../..')


from function_renaming.TFM_function_renaming_baseline_models import *


idx = 0
for root, dirs, files in os.walk('tasks/',topdown=False):
    for f in files:
        # read task dict
        task = {}
        f = os.path.join(root,f)
        print(os.path.splitext(f))
        if os.path.splitext(f)[1] == '.yml' or os.path.splitext(f)[1] == '.yaml':
            task = yaml.load(open(f,'r'), Loader=yaml.FullLoader)
        else:
            task = json.load(open(f,'r'))

        # explode all combinations
        combos = unroll_all_possible_pipeline_model_combos(task,None, {})

        # write each of those as a yaml file in trains/
        for comb in combos:
            with open('trains/task_'+str(idx)+'.yml','w') as f:
                yaml.dump(comb,f)
                idx+=1




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