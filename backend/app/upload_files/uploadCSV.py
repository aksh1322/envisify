import boto3
import os
import pandas as pd
from io import StringIO,BytesIO

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME', 'sagemaker-us-east-1-518479883285')
s3 = boto3.resource("s3")
s3.meta.client.head_bucket(Bucket=S3_BUCKET_NAME)
s3_bucket = s3.Bucket(S3_BUCKET_NAME)


def combine_processed_csv(filename, target_models):
    try:
        files = {}
        processed_path = 'csv/processed'

        # Iterate through target models
        for model_name in target_models:
            folder_name = f'{processed_path}/{filename}/{model_name}'

            # Iterate through objects in the processed folder for each model
            for s3_object in s3_bucket.objects.filter(Prefix=folder_name):
                # Check if the object is a CSV or CSV.out file
                if s3_object.key.endswith('.csv') or s3_object.key.endswith('.csv.out'):
                    file_object = s3.Object(S3_BUCKET_NAME, s3_object.key)
                    file_content = file_object.get()['Body'].read().decode('utf-8')
                    lines = file_content.split('\n')
                    lines[0] = f'{model_name}_predictions'
                    file_content = '\n'.join(lines)

                    data = pd.read_csv(StringIO(file_content))

                    data.reset_index(drop=True, inplace=True)

                    data.index = range(len(data))

                    # Map DataFrame against index
                    files[model_name] = data
    except Exception as e:
        print(f"An error occurred while processing files from S3: {e}")
        return None

    try:
        # Read the unprocessed CSV file and map it against the index
        unprocessed_filename = f'csv/unprocessed/{filename}.csv'
        unprocessed_file_object = s3.Object(S3_BUCKET_NAME, unprocessed_filename)
        unprocessed_file_content = unprocessed_file_object.get()['Body'].read().decode('utf-8')
        unprocessed_data = pd.read_csv(StringIO(unprocessed_file_content))

        # Reset index for unprocessed data
        unprocessed_data.reset_index(drop=True, inplace=True)

        # Set index to 0 for unprocessed data
        unprocessed_data.index = range(len(unprocessed_data))

        # Include unprocessed data in the files dictionary
        files['unprocessed'] = unprocessed_data
    except Exception as e:
        print(f"An error occurred while processing the unprocessed CSV file: {e}")
        return None

    try:
        # Combine all DataFrames along the index axis
        merged_df = pd.concat(files.values(), axis=1)

        # Save the merged DataFrame to a CSV file
        output_filename = f'{filename}/{filename}-results'
        upload_to_s3(merged_df.to_csv(index=True), output_filename, result_type='processed')

        # Construct and return the S3 path
        return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/csv/processed/{output_filename}.csv"
    except Exception as e:
        print(f"An error occurred while saving the merged DataFrame to S3: {e}")
        return None



def upload_to_s3(content, filename, result_type='processed'):
    try:
        OBJECT_KEY = f'csv/{result_type}/{filename}.csv'
        if isinstance(content, str):
            content = BytesIO(content.encode('utf-8'))
        s3.Bucket(S3_BUCKET_NAME).Object(OBJECT_KEY).put(Body=content, ACL='public-read')
        return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{OBJECT_KEY}"
    except Exception as e:
        print(f"An error occurred while uploading to S3: {e}")
        raise  
