# celery_app.py

# Import necessary libraries and modules
from dotenv import load_dotenv
load_dotenv()
import os
from celery import Celery
from app.upload_files.uploadCSV import upload_to_s3
from app.inference.llm_inference import invoke_inference
from app.email.send_email import write_email
import json
import redis
from datetime import datetime, timezone
import pandas as pd
from io import StringIO, BytesIO
import logging
import requests
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential
import math

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Retry decorator for OpenAI API calls
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# Set up Celery and Redis configurations
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
redis_client = redis.StrictRedis.from_url(os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0'))
api_key = os.getenv('api_key')

# Function to create Celery app
def make_celery(app_name=__name__):
    return Celery(
        'tasks',
        broker=CELERY_BROKER_URL,
        result_backend=CELERY_RESULT_BACKEND,
        task_serializer='json',
        result_serializer='json',
        accept_content=['json']
    )

celery_app = make_celery()

# Function to download CSV from S3
def download_csv_from_s3(path):
    response = requests.get(path)
    response.raise_for_status()
    contents = response.content.decode('utf-8')
    return pd.read_csv(StringIO(contents))

# Function to get required columns based on target models
def get_required_columns(target_models):
    required_columns = ['content', 'extra_author_attributes.name', 'extra_author_attributes.description']
    if 'ethnicity' in target_models:
        required_columns.append('extra_source_attributes.world_data.country')
    return required_columns

# Function to clean OpenAI API response
def clean_response(response):
    return response.strip('```').strip()

# Function to create prompt for sentiment analysis
def create_prompt(sub_batch, start_index):
    prompt = "Analyze the sentiment of the following texts. For each text, provide a JSON object with 'index', 'emotions' (a comma-separated list of emotions), 'intensity' (1-5), and 'confidence' (0-1). Possible emotions include happy, excited, hope, empowered, love, gratitude, anger, sadness, fear, frustration, and neutral. Separate each JSON object with a newline.-\n\n"
    for idx, row in sub_batch.iterrows():
        prompt += f"Text {start_index + idx - sub_batch.index[0]}: {row['content']}\n"
    prompt += "\nRespond with exactly one JSON object per line, matching the number of input texts. Include the 'index', 'sentiment', 'intensity', and 'confidence' in each JSON object."
    return prompt

# Function to predict sentiment using OpenAI API
def predict_sentiment(batch):
    results = [None] * len(batch)
    batch_size = 25
    max_retries = 3

    for i in range(0, len(batch), batch_size):
        sub_batch = batch.iloc[i:i+batch_size]
        
        for attempt in range(max_retries):
            try:
                prompt = create_prompt(sub_batch, start_index=i)
                response = chat_completion_with_backoff(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a sentiment analysis expert. Respond only with valid JSON objects, one per line."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2,
                    max_tokens=2000
                )
                
                response_content = response['choices'][0]['message']['content'].strip()
                json_lines = response_content.split('\n')
                
                sub_results = parse_json_lines(json_lines)
                
                for result in sub_results:
                    if 'index' in result and 0 <= result['index'] < len(batch):
                        results[result['index']] = {
                            "sentiment": result.get("sentiment", "unknown"),
                            "intensity": result.get("intensity", 0),
                            "confidence": result.get("confidence", 0.0)
                        }
                
                if all(results[i:i+len(sub_batch)]):
                    break
                else:
                    print(f"Attempt {attempt + 1}: Received {len(sub_results)} valid results for {len(sub_batch)} inputs")
            except Exception as e:
                print(f"Error processing batch: {e}")

    # Fill in any missing results
    results = [result if result else {"sentiment": "unknown", "intensity": 0, "confidence": 0.0} for result in results]
    return results

# Function to parse JSON lines from OpenAI API response
def parse_json_lines(json_lines):
    results = []
    for line in json_lines:
        try:
            result = json.loads(line.strip())
            results.append(result)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {line}")
    return results

# Function to generate batches for processing
def generate_batches(df, required_columns, batch_size, model_name):
    row_count = len(df)
    num_batches = (row_count + batch_size - 1) // batch_size
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, row_count)
        batch = df.iloc[start_idx:end_idx]
        if model_name == 'sentiment':
            batch = batch[['content']]
        else:
            batch = batch[required_columns]
            batch['concatenated'] = batch.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
            batch = batch.drop(columns=required_columns)
        yield batch

# Function to handle inference errors
def handle_inference_error(task, error, postback, model_name):
    logging.error(f"Error during inference with model {model_name}: {error}")
    if postback:
        headers = {"api_key": api_key}
        data = {
            "task_id": task.request.id,
            "success": False,
            "error": str(error),
        }
        requests.post(postback, data=data, headers=headers)

# Function to save DataFrame to buffer
def save_df_to_buffer(df):
    processed_csv_buffer = BytesIO()
    df.to_csv(processed_csv_buffer, index=False)
    processed_csv_buffer.seek(0)
    return processed_csv_buffer

# Function to send postback request
def send_postback(task, postback, processed_csv_buffer, row_count, models_info):
    headers = {"api_key": api_key}
    files = {'file': (f'{task.request.id}_results.csv', processed_csv_buffer, 'text/csv')}
    data = {
        "task_id": task.request.id,
        "success": True,
        "records_processed": row_count,
        "completion_timestamp": datetime.now().isoformat(),
        "models": models_info
    }
    try:
        response = requests.post(postback, files=files, data=data, headers=headers)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send postback to {postback}: {e}")

# Function to send email notification
def send_email_notification(email, path):
    email_list = [e.strip() for e in email.replace(';', ',').split(',')]
    if email_list:
        primary_email = email_list[0]
        cc_emails = email_list[1:]
        subject = "Your Inference Results"
        write_email(primary_email, cc_emails, subject, path)

# Function to handle task errors
def handle_task_error(task, error, postback):
    logging.error(f"Error in makeinference task: {error}")
    if postback:
        headers = {"api_key": api_key}
        data = {
            "task_id": task.request.id,
            "success": False,
            "error": str(error),
        }
        requests.post(postback, data=data, headers=headers)

# Function to categorize reach
def categorize_reach(reach):
    categories = [
        (1_000_000, 'MEGA'),
        (500_000, 'MACRO'),
        (50_000, 'MID-TIER'),
        (10_000, 'MICRO'),
        (1_000, 'NANO')
    ]

    for threshold, category in categories:
        if reach >= threshold:
            return category
    return 'N/A'

# Main Celery task for making inferences
@celery_app.task(bind=True)
def makeinference(self, path, target_models, postback, email=None):
    try:
        # Download and process the CSV file
        df = download_csv_from_s3(path)
        required_columns = get_required_columns(target_models)
        row_count = len(df)
        batch_size = 1000
        num_batches = (row_count + batch_size - 1) // batch_size

        all_predictions_empty = True
        model_versions = {
            "ethnicity": "2.0",
            "age": "6.1",
            "sentiment": "1.0"
        }

        # Process each model
        for model_name in target_models:
            all_results = []
            
            print(f"Processing model: {model_name}")

            # Process batches
            for batch in generate_batches(df, required_columns, batch_size, model_name):
                try:
                    if model_name == "sentiment":
                        results = predict_sentiment(batch)
                    else:
                        results, content_type = invoke_inference(batch, model_name)
                    all_results.extend(results)
                except Exception as e:
                    handle_inference_error(self, e, postback, model_name)
                    raise

            # Process results
            if any(all_results):
                all_predictions_empty = False
            if model_name == 'sentiment':
                predictions = [result['sentiment'] for result in all_results]
                confidences = [result['confidence'] for result in all_results]
                intensity_of_sentiment = [result['intensity'] for result in all_results]
            else:
                predictions = [result[0] for result in all_results]
                confidences = [result[1] for result in all_results]

            # Validate predictions
            if len(predictions) != len(df):
                raise ValueError(f"Length of predictions ({len(predictions)}) does not match length of DataFrame ({len(df)})")

            # Add results to DataFrame
            df[f'result_{model_name}'] = predictions
            df[f'{model_name}_confidence'] = confidences
            df['Influencer_type'] = df['reach'].apply(categorize_reach)
            
            if model_name == 'sentiment':
                df[f'{model_name}_intensity'] = intensity_of_sentiment

        # Check if all predictions are empty
        if all_predictions_empty:
            raise ValueError("All predictions are empty")

        # Save results and send postback
        processed_csv_buffer = save_df_to_buffer(df)
        models_info = ",".join([f"{model}:{model_versions.get(model, 'unknown')}" for model in target_models])

        if postback:
            send_postback(self, postback, processed_csv_buffer, row_count, models_info)

        # Upload results to S3
        path = upload_to_s3(df.to_csv(index=False), self.request.id, result_type='processed')

        # Send email notification if email is provided
        if email:
            send_email_notification(email, path)

        return self.request.id

    except Exception as e:
        handle_task_error(self, e, postback)
        raise

# Function to get list of successful tasks
def get_list_successful_tasks():
    successful_tasks = []
    task_keys = redis_client.keys('celery-task-meta-*')
    current_time = datetime.now(timezone.utc)
    for task_key in task_keys:
        task_info_str = redis_client.get(task_key).decode('utf-8')
        task_info = json.loads(task_info_str)
        task_status = task_info['status']
        task_time_str = task_info['date_done']
        task_time = datetime.fromisoformat(task_time_str).astimezone(timezone.utc)
        time_difference = current_time - task_time
        total_hours = time_difference.total_seconds() / 3600
        task_id = task_info['result']
        if task_status == 'SUCCESS' and total_hours <= 24.0:
            successful_tasks.append({'task_id': task_id, 'status': task_status, 'completed_on': task_time})
    return successful_tasks

# Function to get list of failed tasks
def get_list_failed_tasks():
    failed_tasks = []
    task_keys = redis_client.keys('celery-task-meta-*')
    current_time = datetime.now(timezone.utc)
    for task_key in task_keys:
        task_info_str = redis_client.get(task_key).decode('utf-8')
        task_info = json.loads(task_info_str)
        task_status = task_info['status']
        task_time_str = task_info['date_done']
        task_time = datetime.fromisoformat(task_time_str).astimezone(timezone.utc)
        time_difference = current_time - task_time
        total_hours = time_difference.total_seconds() / 3600
        task_id = task_info['result']
        if task_status == 'FAILURE' and total_hours <= 24.00:
            failed_tasks.append({'task_id': task_id, 'status': task_status, 'completed_on': task_time})
    return failed_tasks

# Function to delete a task
def delete_task(task_id):
    celery_app.control.revoke(task_id, terminate=True)
    return {'success': True, 'message': f"task with task_id:{task_id} successfully deleted"}

# Function to delete tasks by group
def delete_task_by_group(taskid_list):
    results = []
    for task_id in taskid_list:
        result = delete_task(task_id)
        results.append(result)
    return results
