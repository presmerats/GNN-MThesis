import os
import glob

from doit import create_after

def task_create_folders():
    return{
        'targets': ['trains/','done/'],
        'actions': ['mkdir -pv trains','mkdir -pv done']

    }

def task_generate_job_batch():
    """Reads a config file with models(json file) and their possible parameters, then creates a txt file with instructions to train a model with each of the parameter and dataset combinations.
    Can be improved into reading many config yaml files """
    return {
        # force doit to always mark the task
        # as not up-to-date (unless target removed)
        'uptodate': [False],
        'file_dep': ['generate_job_batch.py'],
        'task_dep': ['create_folders'],
        #'targets': ['.running_jobs/list_of_jobs.txt'],
        'actions': ['python generate_job_batch.py'],
    }


@create_after(executed='generate_job_batch')
def task_generate_tasks():
    """Reads folder tasks/ with model training instructions in txt files, and generates a task for each txt file"""
    
    yield {
        'basename': 'generate_tasks',
        'name': None,
        # 'doc': 'docs for X',
        'watch': ['trains/'],
        'task_dep': ['create_folders'],
        }
    
    for root, dirs, files in os.walk('trains/',topdown=False):
        for f in files:
            #print(f)
            yield template_train_model(os.path.join(root,f))


def template_train_model(task_filename):
    """ reads txt task  trains/, reads dataset, instantiates model and performs cross validation training with final refit.
    Training results and model are saved to disk"""
    task_filename_only = os.path.basename(task_filename)
    return {
        'basename': 'generate_tasks',
        'task_dep': ['generate_job_batch'],
        'name': task_filename_only,
        #'file_dep': [task_filename], # does not work if mv
        'targets': ['tests/'+task_filename_only],
        'actions': [
            'python pipeline_train_model.py '+task_filename,
            #'rm '+task_filename
                ],
    }



def task_parse_results():
    """ Parses results from results folder and generates an html notebook result with a table presenting the results"""
    pass