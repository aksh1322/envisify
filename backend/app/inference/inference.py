from celery_app import makeinference as make_inference
from celery_app import get_list_failed_tasks, get_list_successful_tasks, delete_task 
from datetime import datetime

# Dictionary to store task information
task_info_dict = {}

# Function to initiate inference task
def makeinference(path, target_models, task_id=None, postback=None, email=None):
    start_time = datetime.now()
    task = None
    if task_id is not None:
        # Create a task with a specific ID
        task = make_inference.apply_async(args=[path, target_models, postback, email], task_id=task_id)
    else:
        # Create a task with a generated ID
        task = make_inference.delay(path, target_models, postback, email)
    # Store task information
    task_info_dict[task.id] = {'start_time': start_time, 'status': 'PENDING'} 
    return task

# Function to check the status of a specific job
def check_status(job_id):
    task_info = task_info_dict.get(job_id)
    if task_info:
        result = make_inference.AsyncResult(job_id)
        task_info['status'] = result.status
        task_info['recieved_time'] = result.date_done
    return task_info

# Function to get a list of active tasks
def get_active_task_list():
    active_tasks_list = []
    for task_id in task_info_dict.keys():
        task_info = task_info_dict[task_id]
        start_time = task_info.get('start_time')  
        result = make_inference.AsyncResult(task_id)
        task_status = result.status
        if task_status not in ('SUCCESS', 'FAILURE', 'REVOKED'):
            active_tasks_list.append({'task_id': task_id, 'status': task_status, 'start_time': start_time}) 
    return active_tasks_list        

# Function to get successful tasks
def get_successful_tasks():
   return get_list_successful_tasks()
    
# Function to get failed tasks    
def get_failed_tasks():
  return get_list_failed_tasks()   

# Function to delete a task by its ID
def delete_by_taskid(task_id):
    return delete_task(task_id)
