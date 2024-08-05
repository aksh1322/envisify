import os
import uuid
from fastapi import FastAPI, UploadFile, File, Form, Query, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html
from enum import Enum
from typing import Optional
import requests
from starlette.status import HTTP_401_UNAUTHORIZED
from app.upload_files.uploadCSV import upload_to_s3
from app.inference.inference import makeinference, check_status, get_active_task_list, get_successful_tasks, get_failed_tasks,delete_by_taskid

# Get S3 bucket name from environment variables
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', '')

# Define Enum classes for model names and task statuses
class ModelName(str, Enum):
    ethnicity = "ethnicity"
    age_model = "age"
    sentiment="sentiment"

class TaskStatus(str, Enum):
    pending = "pending"
    success = "success"
    failed = "failed"

# Initialize FastAPI app with metadata
app = FastAPI(
    title="Inference Application",
    description="This application offers APIs for running inferencing tasks using Celery and SageMaker. It also provides endpoints for monitoring the status of these tasks.",
    version="1.0.0",
    docs_url=None,
    openapi_url="/api/openapi.json"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up HTTP Basic Authentication
security = HTTPBasic()

# Get authentication credentials from environment variables
SWAGGER_USERNAME = os.getenv('SWAGGER_USERNAME')
SWAGGER_PASS = os.getenv('SWAGGER_PASS')
key = os.getenv('api_key')
group_tasks={}

# API endpoint to serve Swagger UI documentation
@app.get("/api/docs", include_in_schema=False)
async def get_documentation(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != SWAGGER_USERNAME or credentials.password != SWAGGER_PASS:
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    else:
        response = get_swagger_ui_html(openapi_url="/api/openapi.json", title="Inference Application")
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        response.set_cookie("authenticated", "true", max_age=0, expires=0)
        return response

# Middleware to clear authentication cookies
@app.middleware("http")
async def clear_auth_cookies(request, call_next):
    response = await call_next(request)
    if request.url.path == "/api/docs":
        response.delete_cookie("authenticated")
    return response

# Function to verify API key
def verify_credentials(api_key: str = Header(...)):
    if api_key != key:
        raise HTTPException(status_code=401, detail="Invalid API key")

# Health check endpoint
@app.get('/api/hc', summary="Health Check", description="Endpoint to perform a health check.")
def hc():
    return 'ok'

# Endpoint to upload CSV file and start inference task
@app.post('/api/upload', summary="Upload CSV", description="Endpoint to upload a CSV file for inferencing. The postback URL will be used for posting both success and failure results.")
async def upload_csv(
    model: Optional[str] = Form(None, description="Comma-separated names of the models to use for processing. If not provided, all models will be processed. Supported values are 'ethnicity','age' and 'sentiment'."),
    group_id:Optional[str]=Form(None,description="ID to group tasks with different task ids"),
    task_id: Optional[str] = Form(None, description="ID of the task."),
    file: UploadFile = File(..., description="CSV file to be uploaded."),
    email: Optional[str] = Form(None, description="Comma or semicolon separated email addresses for notification."),
    postback: Optional[str] = Form(None, description="result will be POSTED upon success or failure."),
    api_key: str = Header(..., description="API Key for authentication")
):
    verify_credentials(api_key)
    if task_id is not None:
        pending_task = check_status(task_id)
        if pending_task is not None:
            return{"success":False,"message": f"Duplicate job id:{task_id}"}
            
    if model is None or model == "":
        target_models = ['ethnicity', 'age','sentiment']
    else:
        target_models = [m.strip() for m in model.split(',')]
        
    if file.content_type != "text/csv":
        return{ "success":False,"message":"Only CSV files are allowed"}
    
    contents = await file.read()
    if not task_id:
        task_id = str(uuid.uuid4())
    
    try:
        path = upload_to_s3(contents, task_id, 'unprocessed') 
    except Exception as e:
        if postback:
            headers = {"api_key":key}
            data = {
                "task_id": task_id,
                "success": False,
                "error": str(e),
            }
            requests.post(postback, data=data,headers=headers) 
        raise
    
    try:
        if group_id:
            if group_id not in group_tasks:
                group_tasks[group_id] = []
            group_tasks[group_id].append(task_id)          
    
        if task_id:
            task = makeinference(path, target_models,task_id, postback, email)
        else:
            task = makeinference(path, target_models,postback=postback, email=email)
            
        return {"success": True, "message": f"CSV file submitted for processing with job id: {task.id}", "task_id": task.id}
    except Exception as e:
        if postback:
            headers = {"api_key":key}
            data = {
                "task_id": task.id,
                "success": False,
                "error": str(e),
            }
            requests.post(postback, data=data,headers=headers)
        raise HTTPException(status_code=500, detail=f"Failed to submit the CSV file for processing{e}")

# Endpoint to check the status of a specific job
@app.get('/api/check-status/{job_id}', summary="Check Job Status", description="Endpoint to check the processing status of a job.")
def check_job_status(job_id: str, api_key: str = Header(..., description="API Key for authentication")):
    verify_credentials(api_key)
    task_info = check_status(job_id)

    if task_info is None:
        return {"success":False, "message":'Task not found'}

    task_status = task_info.get('status')
    if task_status is None:
        return{"success":False, "message":'Task status not available'}

    response = {'job_id': job_id, 'status': task_status}

    if 'start_time' in task_info:
        response['start_time'] = task_info['start_time']

    if task_status == 'SUCCESS':
        response['result_url'] = f'https://{S3_BUCKET_NAME}.s3.amazonaws.com/csv/processed/{job_id}.csv'

    return response

# Endpoint to retrieve a list of tasks based on their status
@app.get('/api/task-list', summary="Task List", description="Endpoint to retrieve a list of tasks based on their status.")
def check_task_list(status: TaskStatus = Query(TaskStatus.pending, description="Status of the tasks to retrieve. Supported values are 'pending', 'success', and 'failed'."), api_key: str = Header(..., description="API Key for authentication")):
    verify_credentials(api_key)
    if status == TaskStatus.success:
        return get_successful_tasks()
    elif status == TaskStatus.failed:
        return get_failed_tasks()
    else:
        return get_active_task_list()
    
# Endpoint to delete a specific task by its ID
@app.delete('/api/task/{task_id}', summary="Delete task by task_id", description="Terminates the task using task_id.")    
async def delete_group_tasks(task_id: str, api_key: str = Header(..., description="API Key for authentication")):
    verify_credentials(api_key)
    return delete_by_taskid(task_id)

# Endpoint to delete all tasks associated with a group
@app.delete('/api/group/{group_id}', summary="Delete Tasks by Group", description="Deletes all tasks associated with a group.")
async def delete_tasks_by_group(group_id: str, api_key: str = Header(..., description="API Key for authentication")):
    verify_credentials(api_key)
    if group_id not in group_tasks:
        return{"success":False,"message":f"No tasks found for group ID: {group_id}"}
    
    tasks_in_group = group_tasks.pop(group_id)
    for task_id in tasks_in_group:
        result = delete_by_taskid(task_id)
        
    return {"success": True, "message": f"All tasks in group {group_id} have been deleted."}

# Endpoint to check the status of all tasks in a group
@app.get('/api/group_status/{group_id}', summary="Group Task Status", description="Checks Status of tasks in given group")
async def check_status_by_group(group_id: str, api_key: str = Header(..., description="API Key for authentication")):
    verify_credentials(api_key)
    if group_id not in group_tasks:
        return {"success": False, "message": f"No tasks found for group ID: {group_id}"}
    
    result_list = []
    tasks_in_group = group_tasks[group_id]
    
    for task_id in tasks_in_group:
        task_info = check_status(task_id)
        if task_info is not None:
            task_status = task_info.get('status')
            result_list.append({'task_id': task_id, 'status': task_status})
        else:
            result_list.append({'task_id': task_id, 'status': 'Task not found'})
    
    return {"success": True, "tasks": result_list}
