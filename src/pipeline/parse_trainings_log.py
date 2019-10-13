import argparse
import yaml
from pprint import pprint
import logging
import os

from pathlib import Path
import sys
import inspect
import pkgutil
import traceback
import json
import pandas as pd



def process_training(training_lines):
    dataset_folder = ''
    training_time = -1.0
    results_file = ''
    task_name = ''
    next_is_dataset_folder = False
    #print("Processing training ")
    for line in training_lines:
        #print(line)
        if line.find('training time')>-1:
            i = line.find('training time')
            training_time = float(line[(i+15):-1].replace('\n',''))

        elif line.find('saving results to')>-1:
            i = line.find('saving results to')
            results_file = line[(i+17):]

        elif line.find('training trains/')>-1:
            i = line.find('task')
            task_name = line[i:]

        elif line.find('training with:')>-1:
            next_is_dataset_folder = True

        elif next_is_dataset_folder:
            dataset_folder = line[len('2019-09-15 11:17:20,489 - training_jobs - DEBUG - '):]
            next_is_dataset_folder = False

    # add info to the results file
    if results_file != '':
        try:
            with open(os.path.join('../function_renaming/',results_file.replace(' ','')),'r+') as f:
                resdict = json.load(f)
                for k,v in resdict.items():
                    for k2,v2 in v.items():
                        v2['dataset']=dataset_folder
                        v2['time']=training_time
                        v2['task']=task_name
                f.seek(0)
                json.dump(resdict,f)
                f.truncate()
                print(task_name, dataset_folder, results_file, training_time)

        except Exception as err:
            try:
                with open(os.path.join('../function_renaming/',results_file.replace('results','results/old').replace(' ','')),'r+') as f:
                    resdict = json.load(f)
                    for k,v in resdict.items():
                        for k2,v2 in v.items():
                            v2['dataset']=dataset_folder
                            v2['time']=training_time
                            v2['task']=task_name
                    f.seek(0)
                    json.dump(resdict,f)
                    f.truncate()
                    print(task_name, dataset_folder, results_file, training_time)
                    
            except Exception as err:
                pass

    return (task_name, dataset_folder, results_file, training_time)


def find_trainings_in_log():

    processed_trainings =[]
    training = []
    read_training = False 
    with open('trainings.log','r') as f:
        for line in f.readlines():
            line = line.replace('\n','')
            #print(line, line.find('training trains/'))

            if line.find('training trains')>-1:
                res = process_training(training)
                processed_trainings.append(res)
                # restart
                training = []
                read_training = True



            if read_training:
                #print(line)
                training.append(line)
                #print(training)
        


    pprint(processed_trainings)


def flatten_params(paramsdict):
    paramsstr=''
    for k,v in paramsdict.items():
        paramsstr+=k+"-" +str(v) + "_"

    #print(paramsstr)
    return str(paramsstr)

def times_summary():

    with open('training_times.csv','w') as fcsv:
        fcsv.write(
                                'date' + ", " + \
                                'modeltype' + ", " + \
                                'features' + ", " + \
                                'dataset' + ", " + \
                                'thetime' + ", " + \
                                'cv_score' + ", " + \
                                'macro_prec' + ", " + \
                                'macro_recall' + ", " + \
                                'macro_f1' + ", " + \
                                'params' + ", " + \
                                "\n")
        for root, dirs, files in os.walk('../function_renaming/results',topdown=False):
            for f in files:
                try:
                    resdict = json.load(open(os.path.join(root,f),'r'))

                    # read task, algorithm, dataset, time, macro-f1, macro precision, macro recall, cv-score
                    for k,v in resdict.items():
                        for k2,v2 in v.items():
                            try:
                                thetime = str(v2['time'])
                            except:
                                thetime = ''
                            cv_score = str(round(v2['cv_score'],3))
                            try:
                                dataset = v2['dataset']
                            except:
                                dataset = ''
                            try:
                                task = v2['task']
                            except:
                                task = ''
                            modeltype = k 
                            features = v2['features']
                            params = flatten_params(v2['params'])
                            #params += v2['epochs'] + '_' + v2['weight_decay']
                            macro_prec = str(round(v2['macro avg']['precision'],3))
                            macro_recall = str(round(v2['macro avg']['recall'],3))
                            macro_f1 = str(round(v2['macro avg']['f1-score'],3))
                            datetrain = k2

                            fcsv.write(
                                datetrain + ", " + \
                                modeltype + ", " + \
                                features + ", " + \
                                dataset + ", " + \
                                thetime + ", " + \
                                cv_score + ", " + \
                                macro_prec + ", " + \
                                macro_recall + ", " + \
                                macro_f1 + ", " + \
                                params + ", " + \
                                "\n")
                except Exception as err:
                    traceback.print_exc()
                    print("problem with "+f)


def report_latex(filename='results/V3_results.csv', label='FN_exp_v3',title='Function classification experiment results with dataset v3'):

    pd.set_option("display.max_colwidth", 30)
    report = pd.read_csv(filename)
    print(report.columns)

    report2 = pd.DataFrame(data={
        'Model': report[' modeltype'],
        #'parameters': report[' params'],
        'Features': report[' features'],
        'CV score': report[' cv_score'],
        'Precision macro average': report[' macro_prec'],
        'Recall macro average': report[' macro_recall'],
        'F1-macro': report[' macro_f1'],
        })



    # add
    #prefix="\\minipage\{1\\textwidth}\n\\\\\{\\footnotesize "
    #suffix=" }\\endminipage"
    # prefix="\\footnotesize "
    # suffix=" "

    # report2['parameters'] = report2.apply(lambda row: prefix + row['parameters'] + suffix, axis=1)

    display(report2)

    latex_str = report2.to_latex(index=False)

    latex_str = latex_str.replace('\\textbackslash ','\\')
    latex_str = latex_str.replace("llrrrr",'|llcccc|')
    latex_str = latex_str.replace("\\toprule",'\\hline')
    latex_str = latex_str.replace("\\midrule",'\\hline')
    latex_str = latex_str.replace("\\bottomrule",'\\hline')
    latex_str = "\\begin{table}[H]\n\\centering\n{\\footnotesize\n "+latex_str
    caption = "}\\label{"+label+"}\\caption{"+title+"}\n\\end{table}"
    latex_str = latex_str + caption


    print(latex_str)
    return latex_str





if __name__=='__main__':



    #find_trainings_in_log()    
    times_summary()