import os
import joblib
import logging
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from io import StringIO
import numpy as np
import json

# Main function to invoke inference
def invoke_inference(data, model_name):
    try:
        # Set up the model directory and load model artifacts
        # model_dir = os.path.join('/home/ubuntu/models', model_name)
        model_dir = "/home/signity/Ankit707/Pyhton Learning/envisify-upload-backend/finetuned_ageBERT_v6.1"
        model_artifacts = model_fn(model_dir, model_name)
        
        # Prepare input data based on model type
        if model_name == 'sentiment':
            texts = data['content'].tolist()
        else:
            texts = data['concatenated'].tolist() 
        
        # Make predictions
        predictions, confidences = predict_fn(texts, model_artifacts, model_name)
        
        # Format and return results
        results = output_fn(predictions, confidences, accept='list_of_tuples')
        return results
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        raise

# Function to load the model and its associated artifacts
def model_fn(model_dir, model_name):
    try:
        print("Loading model.", flush=True)
        model, tokenizer, label_encoder_or_id_to_label, device = lazy_load_model_and_tokenizer(model_dir, model_name)
        print("Done loading model.")
        return {'model': model, 'tokenizer': tokenizer, 'label_encoder_or_id_to_label': label_encoder_or_id_to_label, 'device': device}
    except Exception as e:
        logging.error(f"Error loading the model from {model_dir}: {e}")
        raise

# Function to lazily load the model, tokenizer, and label encoder
def lazy_load_model_and_tokenizer(model_path, model_name):
    try:
        print('lazy_load_model_and_tokenizer', flush=True)

        # Load label encoder or id2label mapping based on model type
        if model_name == 'sentiment':
            config_path = os.path.join(model_path, 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
                label_encoder_or_id_to_label = config.get('id2label', {})
                if not label_encoder_or_id_to_label:
                    raise ValueError(f"id2label not found in config.json at {config_path}")
        else:
            label_encoder_path = os.path.join(model_path, 'label_encoder.pkl')
            label_encoder_or_id_to_label = joblib.load(label_encoder_path)

        # Load model and tokenizer
        model = BertForSequenceClassification.from_pretrained(model_path)
        tokenizer = BertTokenizerFast.from_pretrained(model_path)

        # Set device to CPU
        device = torch.device("cpu")
        model.to(device)

        return model, tokenizer, label_encoder_or_id_to_label, device
    except Exception as e:
        logging.error(f"Error in lazy_load_model_and_tokenizer: {e}")
        raise

# Function to make predictions using the loaded model
def predict_fn(input_data, model_artifacts, model_name):
    try:
        print("Making predictions against the model.", flush=True)

        # Unpack model artifacts
        model = model_artifacts['model']
        tokenizer = model_artifacts['tokenizer']
        label_encoder_or_id_to_label = model_artifacts['label_encoder_or_id_to_label']
        device = model_artifacts['device']
        
        # Ensure input_data is a list
        texts = [input_data] if isinstance(input_data, str) else input_data
        
        # Make predictions
        predictions, confidences = predict_batch(tokenizer, texts, model, device)

        # Decode predictions based on model type
        if model_name == 'sentiment':
            decoded_labels = [label_encoder_or_id_to_label[str(pred)] for pred in predictions]
        else:
            decoded_labels = label_encoder_or_id_to_label.inverse_transform(predictions)
        
        # Replace 'American' with 'General' for the ethnicity model
        if model_name == 'ethnicity':
            decoded_labels = ['General' if label == 'American' else label for label in decoded_labels]

        print(f"decoded_labels {decoded_labels}", flush=True)
        return decoded_labels, confidences
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise

# Function to make predictions in batches
def predict_batch(tokenizer, texts, model, device):
    try:
        print('predict_batch', flush=True)

        batch_size = 20
        predictions = []
        confidences = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(num_batches):
            # Process texts in batches
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            batch_texts = texts[start_idx:end_idx]

            # Tokenize and encode batch
            encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)

            # Make predictions
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # Process outputs
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            confidence_scores, predicted_labels = torch.max(probabilities, 1)
            
            # Log batch predictions
            logging.info(f"Predictions for batch {i}: {predicted_labels.cpu().numpy()} with confidences: {confidence_scores.cpu().numpy()}")

            # Extend predictions and confidences
            predictions.extend(predicted_labels.cpu().numpy())
            confidences.extend(confidence_scores.cpu().numpy())

        return predictions, confidences
    except Exception as e:
        logging.error(f"Error in predict_batch: {e}")
        raise

# Function to format the prediction output
def output_fn(predictions, confidences, accept):
    print("prediction output.", flush=True)
    if accept == 'text/csv':
        output = StringIO()
        for pred, conf in zip(predictions, confidences):
            output.write(f"{pred},{conf}\n")
        return output.getvalue(), accept
    elif accept == 'application/json':
        return json.dumps({'predictions': predictions, 'confidences': confidences}), accept
    elif accept == 'list_of_tuples':
        return list(zip(predictions, confidences)), accept  # Return as list of tuples
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
