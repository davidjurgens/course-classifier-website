import os
import argparse
import pandas as pd
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, session, Response, jsonify
from werkzeug.utils import secure_filename
import tempfile
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm
import io
import time
import threading
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv'}

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_TOKEN')

# Model information
MODEL_INFO = {
    'two-digit': {
        'model_name': 'annamp/classifying-courses-at-scale-two-digit-roberta-base',
        'description': 'Two-digit CIP code classification (general category)'
    },
    'four-digit': {
        'model_name': 'annamp/classifying-courses-at-scale-four-digit-roberta-base',
        'description': 'Four-digit CIP code classification (intermediate specificity)'
    },
    'six-digit': {
        'model_name': 'annamp/classifying-courses-at-scale-six-digit-roberta-base',
        'description': 'Six-digit CIP code classification (most specific)'
    }
}

# Global variable to cache models and tokenizers
models = {}
tokenizers = {}

# Determine device (use MPS if available, otherwise CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using CUDA device")
else:
    device = torch.device("cpu")
    print(f"Using CPU device")

def repair_ccm(ccm, level):
    # Convert to string first to handle any numeric types
    ccm_str = str(ccm)

    # Handle case where there's no decimal point (integer codes)
    if "." not in ccm_str:
        two_digit = ccm_str.rjust(2, "0")
        if level == "two":
            return two_digit
        elif level == "four":
            return f"{two_digit}.00"
        elif level == "six":
            return f"{two_digit}.0000"

    parts = ccm_str.split(".")
    two_digit = parts[0].rjust(2, "0")

    if level == "two":
        return two_digit
    elif level == "four":
        try:
            end = parts[1].ljust(2, "0")
        except (IndexError, Exception):
            end = "00"
        return f"{two_digit}.{end}"
    elif level == "six":
        try:
            end = parts[1].ljust(4, "0")
        except (IndexError, Exception):
            end = "0000"
        return f"{two_digit}.{end}"

# Global progress tracking
progress_data = {}

# Load taxonomy mappings
ccm_taxonomy = {}
ccm_six_description = {}

def load_ccm_taxonomy():
    """Load CCM code taxonomy from Excel files"""
    global ccm_taxonomy
    global ccm_six_description

    try:
        # Load two-digit taxonomy
        two_digit_df = pd.read_excel('ccm_taxonomy_two.xlsx', dtype={'ccm_two': str})
        two_digit_df['ccm_two_str'] = two_digit_df['ccm_two'].astype(str).str.zfill(2)
        ccm_taxonomy['two-digit'] = dict(zip(two_digit_df['ccm_two_str'], two_digit_df['ccm_title']))

        # Load six-digit taxonomy - create mapping from repaired codes to titles
        six_digit_df = pd.read_excel('ccm_taxonomy_six.xlsx', dtype={'ccm_six': str})
        six_digit_mapping = {}
        six_digit_description = {}
        for idx, row in six_digit_df.iterrows():
            # Get the code from the file - it's a float like 1.0105
            code_in_file = row['ccm_six']

            # Convert to string and ensure proper format
            if isinstance(code_in_file, float):
                # Format the float to string with proper padding
                code_str = f"{code_in_file:.4f}"
            else:
                code_str = str(code_in_file)

            # Repair the code from the format in the file (e.g., 1.0105) to the standard format (01.0105)
            repaired_code = repair_ccm(code_str, 'six')
            title = row['ccm_title']
            description = row.get('ccm_description', '')  # Get description if it exists, otherwise empty string
            six_digit_mapping[repaired_code] = title
            six_digit_description[repaired_code] = description
        ccm_taxonomy['six-digit'] = six_digit_mapping

        # Store descriptions globally
        ccm_six_description = six_digit_description

        print(f"Loaded {len(ccm_taxonomy.get('two-digit', {}))} two-digit CCM codes")
        if ccm_six_description:
            # Count how many have descriptions
            description_count = sum(1 for desc in ccm_six_description.values() if desc and desc.strip())
            print(f"Loaded {len(ccm_six_description)} six-digit CCM codes, {description_count} with descriptions")
        #print(f"Loaded {len(ccm_taxonomy.get('six-digit', {}))} six-digit CCM codes")
        #print(f"Sample six-digit mappings: {list(six_digit_mapping.items())[:5]}")

        # Debug: Show some key mappings
        test_keys = ['01.0105', '05.0201', '11.0101', '15.1101']
        for key in test_keys:
            if key in six_digit_mapping:
                print(f"Taxonomy loaded: {key} -> {six_digit_mapping[key]}")
            else:
                print(f"Taxonomy missing: {key}")
    except Exception as e:
        print(f"Error loading CCM taxonomy: {e}")
        import traceback
        traceback.print_exc()
        ccm_taxonomy = {'two-digit': {}, 'six-digit': {}}

def get_ccm_name(code, level):
    """Get the full name for a CCM code"""
    if not code or code == 'N/A':
        return None

    # Convert to string and clean
    code_str = str(code).strip()

    # Level without the hyphen - extract first part (e.g., "two-digit" -> "two")
    level_clean = level.split('-')[0] if '-' in level else level

    # Try direct lookup first (in case the code is already in the right format)
    taxonomy = ccm_taxonomy.get(level, {})
    result = taxonomy.get(code_str, None)

    # If not found, try repairing the code
    if result is None:
        repaired_code = repair_ccm(code_str, level_clean)
        result = taxonomy.get(repaired_code, None)

        # If still not found and it looks like it might be missing a leading zero, try adding it
        if result is None and '.' in code_str:
            parts = code_str.split('.')
            if len(parts) == 2 and len(parts[0]) == 1:
                # Single digit before the decimal, add leading zero
                code_with_zero = f"0{code_str}"
                repaired_code = repair_ccm(code_with_zero, level_clean)
                result = taxonomy.get(repaired_code, None)

    # Debug logging for problematic lookups
    if result is None and taxonomy:
        sample_keys = list(taxonomy.keys())[:5]
        repaired = repair_ccm(code_str, level_clean)
        print(f"WARNING: Could not find name for '{code_str}' (repaired: '{repaired}') in {level} taxonomy (sample keys: {sample_keys})")

    return result

def get_ccm_description(code, level='six-digit'):
    """Get the description for a CCM code"""
    if not code or code == 'N/A':
        return None

    # Only six-digit codes have descriptions currently
    if level != 'six-digit':
        return None

    # Convert to string and clean
    code_str = str(code).strip()

    # Try direct lookup first
    if ccm_six_description and code_str in ccm_six_description:
        return ccm_six_description[code_str]

    # If not found, try repairing the code
    repaired_code = repair_ccm(code_str, 'six')
    if ccm_six_description and repaired_code in ccm_six_description:
        return ccm_six_description[repaired_code]

    return None

def load_progress_data():
    """Load progress data from file if it exists"""
    global progress_data
    progress_file = os.path.join(app.config['UPLOAD_FOLDER'], 'progress_data.json')
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
        except:
            progress_data = {}

def save_progress_data():
    """Save progress data to file"""
    progress_file = os.path.join(app.config['UPLOAD_FOLDER'], 'progress_data.json')
    try:
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f)
    except:
        pass

def convert_pandas_to_json_safe(obj):
    """Convert pandas/numpy data types to JSON-safe Python types"""
    if pd.isna(obj):
        return None
    elif isinstance(obj, (pd.Int64Dtype, pd.Float64Dtype)):
        return int(obj) if not pd.isna(obj) else None
    elif isinstance(obj, (int, float)) and not pd.isna(obj):
        return int(obj) if isinstance(obj, (int, float)) and obj == int(obj) else float(obj)
    elif hasattr(obj, 'item'):  # numpy scalars
        return obj.item()
    else:
        return str(obj)

def convert_for_json_dict(obj):
    """Convert values for JSON-safe dictionary keys"""
    if obj is None or pd.isna(obj):
        return None
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, (int, float)):
        return obj
    else:
        return str(obj)

# Load any existing progress data on startup
load_progress_data()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model(model_type):
    """Load model and tokenizer if not already loaded"""
    if model_type not in models:
        model_name = MODEL_INFO[model_type]['model_name']
        print(f"Loading model {model_name}...")

        # Pass token to from_pretrained method if available
        if HF_TOKEN:
            models[model_type] = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=HF_TOKEN
            )
            tokenizers[model_type] = AutoTokenizer.from_pretrained(
                model_name,
                token=HF_TOKEN
            )
        else:
            # If no token is provided, try to use the logged-in user's credentials
            print("No Hugging Face token found in environment variables.")
            print("Make sure you're logged in with `huggingface-cli login` or set HUGGINGFACE_TOKEN in .env file")
            models[model_type] = AutoModelForSequenceClassification.from_pretrained(model_name)
            tokenizers[model_type] = AutoTokenizer.from_pretrained(model_name)

        # Move model to device (MPS/GPU/CPU)
        models[model_type] = models[model_type].to(device)
        print(f"Moved model to {device}")

    return models[model_type], tokenizers[model_type]

def classify_text(model, tokenizer, text, return_probability=False):
    """Classify a single text using the provided model and tokenizer"""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Get predictions
    predictions = outputs.logits.argmax(dim=1).item()
    result = model.config.id2label[predictions]

    if return_probability:
        # Get probabilities using softmax
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        max_prob = probs[0][predictions].item()
        return result, max_prob
    else:
        return result

def process_csv(file_path, model_types, row_limit=None, progress_id=None, save_probabilities=False, random_sampling=False):
    """Process CSV file with course descriptions"""
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Apply row limit if specified
    if row_limit is not None:
        original_length = len(df)
        if random_sampling:
            # Randomly sample rows instead of taking the first N
            df = df.sample(n=min(row_limit, original_length), random_state=None).reset_index(drop=True)
            print(f"Randomly sampling {len(df)} rows from {original_length} total rows")
        else:
            df = df.head(row_limit)
            print(f"Processing {len(df)} rows out of {original_length} total rows")

    # Check if the required column exists, with course_title as fallback
    if 'course_description' not in df.columns and 'course_title' not in df.columns:
        raise ValueError("CSV file must contain either a 'course_description' or 'course_title' column")

    # Determine which column to use for classification
    text_column = 'course_description' if 'course_description' in df.columns else 'course_title'
    print(f"PROCESS CSV DEBUG - Available columns: {list(df.columns)}")
    print(f"PROCESS CSV DEBUG - Using text column: {text_column}")
    print(f"PROCESS CSV DEBUG - Sample text values: {df[text_column].head().tolist()}")

    # Add any missing optional columns with default values
    if 'school_year_enrolled' not in df.columns:
        df['school_year_enrolled'] = 'Unknown'
    if 'school_name' not in df.columns:
        df['school_name'] = 'Unknown'

    # Initialize progress tracking
    if progress_id:
        total_rows = len(df)
        total_models = len(model_types)
        total_items = total_rows * total_models
        progress_data[progress_id] = {
            'current': 0,
            'total': total_items,
            'current_model': '',
            'current_row': 0,
            'total_rows': total_rows,
            'start_time': time.time(),
            'status': 'processing'
        }
        print(f"Initialized progress_data for {progress_id}: {progress_data[progress_id]}")

    # Process each selected model type
    print(f"PROCESS CSV DEBUG - Processing models: {model_types}")
    for model_idx, model_type in enumerate(model_types):
        print(f"PROCESS CSV DEBUG - Starting {model_type} model (index {model_idx})")
        model, tokenizer = load_model(model_type)

        # Create output column name
        output_column = f"{model_type}_classification"
        print(f"PROCESS CSV DEBUG - Output column: {output_column}")

        # Update progress for model start
        if progress_id:
            progress_data[progress_id]['current_model'] = model_type
            progress_data[progress_id]['status'] = f'Processing {model_type} model'

        # Classify each course description/title
        classifications = []
        probabilities = []
        for row_idx, text in enumerate(df[text_column]):
            if pd.isna(text) or text == "":
                classifications.append("N/A")
                if save_probabilities:
                    probabilities.append(None)
            else:
                try:
                    if save_probabilities:
                        classification, probability = classify_text(model, tokenizer, text, return_probability=True)
                        classifications.append(classification)
                        probabilities.append(probability)
                    else:
                        classification = classify_text(model, tokenizer, text)
                        #print(f"CLASSIFY TEXT DEBUG - {text} -> {classification} (type: {type(classification)})")
                        classifications.append(classification)
                except Exception as e:
                    print(f"Error processing: {text[:30]}... - {str(e)}")
                    classifications.append("Error")
                    if save_probabilities:
                        probabilities.append(None)

            # Update progress
            if progress_id:
                current_item = model_idx * total_rows + row_idx + 1
                progress_data[progress_id]['current'] = current_item
                progress_data[progress_id]['current_row'] = row_idx + 1

                # Calculate ETA
                elapsed_time = time.time() - progress_data[progress_id]['start_time']
                if current_item > 0:
                    avg_time_per_item = elapsed_time / current_item
                    remaining_items = total_items - current_item
                    eta_seconds = remaining_items * avg_time_per_item
                    progress_data[progress_id]['eta'] = eta_seconds

                # Save progress data periodically
                if current_item % 10 == 0:  # Save every 10 items
                    save_progress_data()

        # Add classifications to dataframe
        df[output_column] = classifications

        # Add probability column if requested
        if save_probabilities:
            prob_column = f"{model_type}_probability"
            df[prob_column] = probabilities
            print(f"PROCESS CSV DEBUG - Added {prob_column} column")

        # Debug: Print classification results
        print(f"PROCESS CSV DEBUG - Added {output_column} column")
        print(f"PROCESS CSV DEBUG - Unique values in {output_column}: {df[output_column].nunique()}")
        print(f"PROCESS CSV DEBUG - Sample values: {df[output_column].value_counts().head()}")
        print(f"PROCESS CSV DEBUG - First 5 classifications: {df[output_column].head().tolist()}")
        print(f"PROCESS CSV DEBUG - Non-N/A values: {df[df[output_column] != 'N/A'][output_column].nunique()}")

    # Mark as completed
    if progress_id:
        progress_data[progress_id]['status'] = 'completed'
        progress_data[progress_id]['current'] = progress_data[progress_id]['total']
        save_progress_data()

    return df

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', model_info=MODEL_INFO)

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    # If user does not select file, browser also
    # submits an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    # Check selected models
    selected_models = request.form.getlist('models')
    print(f"UPLOAD DEBUG - Selected models: {selected_models}")
    print(f"UPLOAD DEBUG - Form data: {dict(request.form)}")
    if not selected_models:
        flash('Please select at least one model')
        return redirect(request.url)

    # Get row limit
    row_limit = request.form.get('rowLimit', '').strip()
    if row_limit:
        try:
            row_limit = int(row_limit)
            if row_limit < 1:
                flash('Row limit must be a positive number')
                return redirect(request.url)
        except ValueError:
            flash('Row limit must be a valid number')
            return redirect(request.url)
    else:
        row_limit = None

    # Get random sampling option
    random_sampling = 'randomSampling' in request.form

    # Get probability option
    save_probabilities = 'saveProbabilities' in request.form

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Generate unique progress ID
        progress_id = f"progress_{int(time.time())}_{filename}"
        print(f"Generated progress_id: {progress_id}")

        # Start background processing
        def process_in_background():
            try:
                # Process the file
                processed_df = process_csv(filepath, selected_models, row_limit, progress_id, save_probabilities, random_sampling)

                # Remove duplicates based on key identifying columns
                print(f"DUPLICATE REMOVAL - Original shape: {processed_df.shape}")

                # Define columns to use for duplicate detection
                # Priority: course_title + school_name, then course_description + school_name, then all columns
                duplicate_columns = []
                if 'course_title' in processed_df.columns and 'school_name' in processed_df.columns:
                    duplicate_columns = ['course_title', 'school_name']
                    print(f"DUPLICATE REMOVAL - Using course_title + school_name for deduplication")
                elif 'course_description' in processed_df.columns and 'school_name' in processed_df.columns:
                    duplicate_columns = ['course_description', 'school_name']
                    print(f"DUPLICATE REMOVAL - Using course_description + school_name for deduplication")
                else:
                    # Fallback: use all columns except classification columns
                    classification_cols = [col for col in processed_df.columns if col.endswith('_classification')]
                    duplicate_columns = [col for col in processed_df.columns if col not in classification_cols]
                    print(f"DUPLICATE REMOVAL - Using all non-classification columns for deduplication")

                # Remove duplicates, keeping the first occurrence
                original_count = len(processed_df)
                processed_df = processed_df.drop_duplicates(subset=duplicate_columns, keep='first')
                final_count = len(processed_df)
                duplicates_removed = original_count - final_count
                print(f"DUPLICATE REMOVAL - After deduplication shape: {processed_df.shape}")
                print(f"DUPLICATE REMOVAL - Removed {duplicates_removed} duplicate rows")

                # Save the results
                output_filename = f"processed_{filename}"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                processed_df.to_csv(output_path, index=False)

                # Mark progress as completed
                progress_data[progress_id]['status'] = 'completed'
                progress_data[progress_id]['output_file'] = output_path
                progress_data[progress_id]['output_filename'] = output_filename
                save_progress_data()

            except Exception as e:
                progress_data[progress_id]['status'] = 'error'
                progress_data[progress_id]['error'] = str(e)
                save_progress_data()

        # Start the background thread
        thread = threading.Thread(target=process_in_background)
        thread.start()

        # Store progress ID in session
        session['progress_id'] = progress_id

        # Initialize progress data immediately to avoid race condition
        progress_data[progress_id] = {
            'current': 0,
            'total': 0,  # Will be updated when processing starts
            'current_model': '',
            'current_row': 0,
            'total_rows': 0,
            'start_time': time.time(),
            'status': 'initializing'
        }

        # Redirect to progress page
        return redirect(url_for('progress'))
    else:
        flash('Invalid file type. Please upload a CSV file.')
        return redirect(url_for('index'))

@app.route('/progress')
def progress():
    if 'progress_id' not in session:
        flash('No processing session found')
        return redirect(url_for('index'))

    return render_template('progress.html', progress_id=session['progress_id'])

@app.route('/progress/stream')
def progress_stream():
    # Get progress_id from query parameter as fallback
    progress_id = request.args.get('progress_id') or session.get('progress_id')
    print(f"Progress stream requested for progress_id: {progress_id}")
    print(f"Available progress_data keys: {list(progress_data.keys())}")

    if not progress_id:
        return Response("No progress ID", status=400)

    def generate():
        import json
        while True:
            if progress_id in progress_data:
                data = progress_data[progress_id]
                yield f"data: {json.dumps(data)}\n\n"

                if data['status'] in ['completed', 'error']:
                    break
            else:
                # Debug: show available progress IDs
                available_ids = list(progress_data.keys())
                yield f"data: {json.dumps({'status': 'not_found', 'progress_id': progress_id, 'available_ids': available_ids})}\n\n"
                break

            time.sleep(0.5)  # Update every 500ms

    return Response(generate(), mimetype='text/event-stream')

@app.route('/result')
def result():
    if 'progress_id' not in session:
        flash('No processed file found')
        return redirect(url_for('index'))

    progress_id = session['progress_id']
    if progress_id not in progress_data or progress_data[progress_id]['status'] != 'completed':
        flash('File processing not complete')
        return redirect(url_for('index'))

    filename = progress_data[progress_id]['output_filename']
    return render_template('result.html', filename=filename)

@app.route('/download')
def download():
    if 'progress_id' not in session:
        flash('No processed file found')
        return redirect(url_for('index'))

    progress_id = session['progress_id']
    if progress_id not in progress_data or progress_data[progress_id]['status'] != 'completed':
        flash('File processing not complete')
        return redirect(url_for('index'))

    output_file = progress_data[progress_id]['output_file']
    output_filename = progress_data[progress_id]['output_filename']

    return send_file(
        output_file,
        as_attachment=True,
        download_name=output_filename,
        mimetype='text/csv'
    )

@app.route('/dashboard')
def dashboard():
    if 'progress_id' not in session:
        flash('No processed file found')
        return redirect(url_for('index'))

    progress_id = session['progress_id']
    if progress_id not in progress_data or progress_data[progress_id]['status'] != 'completed':
        flash('File processing not complete')
        return redirect(url_for('index'))

    # Read the processed data
    output_file = progress_data[progress_id]['output_file']
    df = pd.read_csv(output_file)

    # Debug: Print column information
    #print(f"Dashboard - Available columns: {list(df.columns)}")
    #print(f"Dashboard - Six-digit column exists: {'six-digit_classification' in df.columns}")
    #if 'six-digit_classification' in df.columns:
    #    print(f"Dashboard - Six-digit values: {df['six-digit_classification'].value_counts().head()}")
    #    print(f"Dashboard - Six-digit unique count: {df['six-digit_classification'].nunique()}")

    # Prepare dashboard data
    dashboard_data = prepare_dashboard_data(df)

    # Debug: Print dashboard data structure
    #print(f"Dashboard - Six-digit histogram data: {dashboard_data.get('six_digit_histogram', [])}")
    #print(f"Dashboard - Six-digit breakdown exists: {'six-digit_breakdown' in dashboard_data}")
    #print(f"Dashboard - All dashboard data keys: {list(dashboard_data.keys())}")
    if 'two-digit_breakdown' in dashboard_data:
        print(f"Dashboard - Two-digit breakdown data: {dashboard_data['two-digit_breakdown']}")
    #print(f"Dashboard - DataFrame shape: {df.shape}")
    #print(f"Dashboard - DataFrame columns: {list(df.columns)}")

    # Debug: Check ccm_names in dashboard data
    if 'ccm_names' in dashboard_data:
        print(f"Dashboard - ccm_names keys: {list(dashboard_data['ccm_names'].keys())}")
        if 'six-digit' in dashboard_data['ccm_names']:
            six_names = dashboard_data['ccm_names']['six-digit']
            print(f"Dashboard - six-digit CCM names count: {len(six_names)}")
            print(f"Dashboard - Sample six-digit CCM names: {list(six_names.items())[:5]}")
    else:
        print("Dashboard - ERROR: ccm_names not in dashboard_data!")

    return render_template('dashboard.html',
                         filename=progress_data[progress_id]['output_filename'],
                         data=dashboard_data)

@app.route('/api/courses/search')
def search_courses():
    if 'progress_id' not in session:
        return {'error': 'No processed file found'}, 404

    progress_id = session['progress_id']
    if progress_id not in progress_data or progress_data[progress_id]['status'] != 'completed':
        return {'error': 'File processing not complete'}, 404

    query = request.args.get('q', '').lower()
    if not query:
        return {'courses': []}

    # Read the processed data
    output_file = progress_data[progress_id]['output_file']
    df = pd.read_csv(output_file)

    # Search in course titles and descriptions
    search_columns = []
    if 'course_title' in df.columns:
        search_columns.append('course_title')
    if 'course_description' in df.columns:
        search_columns.append('course_description')

    if not search_columns:
        return {'courses': []}

    # Filter courses based on search query in course titles and descriptions
    mask = df[search_columns].apply(
        lambda x: x.astype(str).str.lower().str.contains(query, na=False)
    ).any(axis=1)

    # Also search in CCM code descriptions if six-digit classification exists
    if 'six-digit_classification' in df.columns:
        def matches_ccm_description(row):
            ccm_code = row.get('six-digit_classification', '')
            if pd.isna(ccm_code) or ccm_code == 'N/A':
                return False
            description = get_ccm_description(ccm_code, 'six-digit')
            if description:
                return query in str(description).lower()
            return False

        ccm_mask = df.apply(matches_ccm_description, axis=1)
        mask = mask | ccm_mask

    # Convert to list of dictionaries with proper data type conversion
    matching_df = df[mask]
    matching_courses = []
    for _, row in matching_df.iterrows():
        course_dict = {}
        for col, value in row.items():
            course_dict[col] = convert_pandas_to_json_safe(value)
        matching_courses.append(course_dict)

    return {'courses': matching_courses}

@app.route('/api/courses/<int:course_id>/related')
def get_related_courses(course_id):
    if 'progress_id' not in session:
        return {'error': 'No processed file found'}, 404

    progress_id = session['progress_id']
    if progress_id not in progress_data or progress_data[progress_id]['status'] != 'completed':
        return {'error': 'File processing not complete'}, 404

    # Read the processed data
    output_file = progress_data[progress_id]['output_file']
    df = pd.read_csv(output_file)

    # Find the course
    course = df[df['id'] == course_id]
    if course.empty:
        return {'error': 'Course not found'}, 404

    # Get the six-digit CIP code for this course
    six_digit_cip = convert_pandas_to_json_safe(course.iloc[0].get('six-digit_classification', ''))
    if not six_digit_cip or six_digit_cip == 'N/A':
        return {'related_courses': []}

    # Find all courses with the same six-digit CIP code
    related_df = df[df['six-digit_classification'] == six_digit_cip]

    # Convert to list of dictionaries with proper data type conversion
    related_courses = []
    for _, row in related_df.iterrows():
        course_dict = {}
        for col, value in row.items():
            course_dict[col] = convert_pandas_to_json_safe(value)
        related_courses.append(course_dict)

    return {'related_courses': related_courses}

@app.route('/api/ccm-names', methods=['GET'])
def get_ccm_names():
    """Get full names for CCM codes"""
    codes = request.args.getlist('codes')
    level = request.args.get('level', 'six-digit')

    names = {}
    for code in codes:
        name = get_ccm_name(code, level)
        if name:
            names[str(code)] = name

    return {'names': names}

@app.route('/api/classify-single', methods=['POST'])
def classify_single_course():
    """Classify a single course title/description"""
    data = request.get_json()
    course_text = data.get('course_text', '').strip()
    selected_models = data.get('models', ['six-digit'])  # Default to six-digit
    save_probability = data.get('save_probability', False)

    if not course_text:
        return {'error': 'Course text is required'}, 400

    if not selected_models:
        return {'error': 'At least one model must be selected'}, 400

    results = {}

    for model_type in selected_models:
        if model_type not in models:
            # Load model if not already loaded
            load_model(model_type)

        model = models[model_type]
        tokenizer = tokenizers[model_type]

        try:
            if save_probability:
                classification, probability = classify_text(model, tokenizer, course_text, return_probability=True)
                results[model_type] = {
                    'classification': classification,
                    'probability': probability
                }
            else:
                classification = classify_text(model, tokenizer, course_text)
                results[model_type] = {
                    'classification': classification
                }
        except Exception as e:
            results[model_type] = {
                'error': str(e)
            }

    return {'results': results}

@app.route('/api/courses/by-cip/<cip_code>')
def get_courses_by_cip(cip_code):
    """Get courses and schools for a specific CIP code"""
    print(f"API DEBUG - get_courses_by_cip called with cip_code: {cip_code} (type: {type(cip_code)})")

    if 'progress_id' not in session:
        print("API DEBUG - No progress_id in session")
        return {'error': 'No processed file found'}, 404

    progress_id = session['progress_id']
    print(f"API DEBUG - progress_id: {progress_id}")
    if progress_id not in progress_data or progress_data[progress_id]['status'] != 'completed':
        print(f"API DEBUG - progress_id not found or not completed. Available keys: {list(progress_data.keys())}")
        return {'error': 'File processing not complete'}, 404

    # Read the processed data
    output_file = progress_data[progress_id]['output_file']
    df = pd.read_csv(output_file)
    print(f"API DEBUG - DataFrame shape: {df.shape}, columns: {list(df.columns)}")

    # Convert cip_code to float for comparison
    try:
        cip_code_float = float(cip_code)
        print(f"API DEBUG - Converted cip_code to float: {cip_code_float}")
    except ValueError:
        print(f"API DEBUG - Could not convert cip_code to float: {cip_code}")
        return {'courses': [], 'schools': [], 'total_courses': 0, 'total_schools': 0}

    # Filter courses by CIP code
    if 'six-digit_classification' in df.columns:
        print(f"API DEBUG - Using six-digit_classification column")
        filtered_df = df[df['six-digit_classification'] == cip_code_float]
        print(f"API DEBUG - Filtered DataFrame shape: {filtered_df.shape}")
        print(f"API DEBUG - Unique six-digit values: {df['six-digit_classification'].unique()[:10]}")
    elif 'four-digit_classification' in df.columns:
        print(f"API DEBUG - Using four-digit_classification column")
        filtered_df = df[df['four-digit_classification'] == cip_code_float]
    elif 'two-digit_classification' in df.columns:
        print(f"API DEBUG - Using two-digit_classification column")
        filtered_df = df[df['two-digit_classification'] == cip_code_float]
    else:
        print("API DEBUG - No classification columns found")
        return {'courses': [], 'schools': [], 'total_courses': 0, 'total_schools': 0}

    # Convert to list of dictionaries with proper data type conversion
    courses = []
    for _, row in filtered_df.iterrows():
        course_dict = {}
        for col, value in row.items():
            course_dict[col] = convert_pandas_to_json_safe(value)
        courses.append(course_dict)

    # Get unique schools for this CIP code
    schools = filtered_df['school_name'].unique().tolist()
    schools = [convert_pandas_to_json_safe(school) for school in schools]

    result = {
        'courses': courses,
        'schools': schools,
        'total_courses': len(courses),
        'total_schools': len(schools)
    }

    print(f"API DEBUG - Returning result: {len(courses)} courses, {len(schools)} schools")
    print(f"API DEBUG - Sample course: {courses[0] if courses else 'No courses'}")
    print(f"API DEBUG - Schools: {schools}")

    return result

def prepare_dashboard_data(df):
    """Prepare data for dashboard visualization"""
    print(f"PREPARE DASHBOARD DEBUG - Input DataFrame shape: {df.shape}")
    print(f"PREPARE DASHBOARD DEBUG - Input DataFrame columns: {list(df.columns)}")

    data = {}

    # Get available classification columns
    classification_cols = [col for col in df.columns if col.endswith('_classification')]
    print(f"PREPARE DASHBOARD DEBUG - Classification columns found: {classification_cols}")

    # Prepare CIP code breakdowns by school
    for col in classification_cols:
        cip_level = col.replace('_classification', '')
        print(f"PREPARE DASHBOARD DEBUG - Processing {cip_level} level (column: {col})")

        # Count courses by school and CIP code
        school_cip_counts = df.groupby(['school_name', col]).size().reset_index(name='count')

        # Create charts data
        schools = df['school_name'].unique()
        cip_codes = df[col].unique()

        # Filter out NaN values from the unique CIP codes
        cip_codes = [c for c in cip_codes if pd.notna(c)]

        # Sort CIP codes by overall frequency (most common to least common)
        cip_frequency = df[col].value_counts()
        # Filter out NaN from sorted codes too
        sorted_cip_codes = [c for c in cip_frequency.index.tolist() if pd.notna(c)]
        #print(f"PREPARE DASHBOARD DEBUG - {cip_level} CIP codes sorted by frequency: {sorted_cip_codes[:10]}")
        #print(f"PREPARE DASHBOARD DEBUG - {cip_level} CIP codes types: {[type(c).__name__ for c in sorted_cip_codes[:10]]}")
        #print(f"PREPARE DASHBOARD DEBUG - {cip_level} Any None in sorted_cip_codes? {any(c is None for c in sorted_cip_codes)}")

        # Prepare data for charts
        chart_data = []
        for school in schools:
            school_data = {'school': convert_pandas_to_json_safe(school), 'cip_codes': {}, 'course_titles': {}}
            for cip in sorted_cip_codes:
                # Skip NaN values
                if pd.isna(cip):
                    continue

                count = school_cip_counts[
                    (school_cip_counts['school_name'] == school) &
                    (school_cip_counts[col] == cip)
                ]['count'].sum()
                # Convert to string and repair the code format to ensure consistent format
                cip_key = str(cip).strip()
                #print(f"PREPARE DASHBOARD DEBUG - Processing cip={cip} (type={type(cip).__name__}) -> cip_key={cip_key}")
                # Use repair_ccm to normalize the code format
                # Map cip_level (e.g., "six-digit") to repair_ccm level format (e.g., "six")
                level_clean = cip_level.split('-')[0] if '-' in cip_level else cip_level
                cip_normalized = repair_ccm(cip_key, level_clean)
                #print(f"PREPARE DASHBOARD DEBUG - Normalized to: {cip_normalized}")

                # Store using the normalized format
                school_data['cip_codes'][cip_normalized] = int(count)

                # Get course titles for this school and CIP code combination
                if cip_level in ['two-digit', 'four-digit', 'six-digit']:
                    school_cip_courses = df[
                        (df['school_name'] == school) &
                        (df[col] == cip)
                    ]
                    course_titles = []
                    if 'course_title' in df.columns:
                        course_titles = school_cip_courses['course_title'].dropna().unique().tolist()
                    elif 'course_description' in df.columns:
                        course_titles = school_cip_courses['course_description'].dropna().unique().tolist()

                    # Limit to first 10 course titles to prevent tooltip from being too large
                    course_titles = course_titles[:10]
                    # Store course titles using the normalized key
                    school_data['course_titles'][cip_normalized] = [convert_pandas_to_json_safe(title) for title in course_titles]

            chart_data.append(school_data)

        # Sort schools by total number of classifications (most to least)
        def get_total_classifications(school_data):
            return sum(school_data['cip_codes'].values())

        chart_data.sort(key=get_total_classifications, reverse=True)

        # For two-digit charts, limit to top 20 schools to prevent overcrowding
        if cip_level == 'two-digit':
            chart_data = chart_data[:20]


        # Extract sorted school names and log the sorting results
        sorted_schools = [school_data['school'] for school_data in chart_data]

        for school_data in chart_data:
            total = get_total_classifications(school_data)
            print(f"  {school_data['school']}: {total} total classifications")

        def fix_digits(s):
            if len(s) == 1:
                return f"0{s}"
            elif s.find('.') == 1:
                return f"0{s}"
            return s
        normalized_cip_codes = [fix_digits(str(c).strip()) for c in sorted_cip_codes]


        data[f'{cip_level}_breakdown'] = {
            'schools': sorted_schools,
            'cip_codes': normalized_cip_codes,
            'chart_data': chart_data
        }

        print(f"PREPARE DASHBOARD DEBUG - {cip_level}_breakdown created with {len(chart_data)} schools, {len(cip_codes)} CIP codes (sorted by total classifications)")

    # Overall statistics - convert pandas types to native Python types
    school_counts = df['school_name'].value_counts()
    schools_dict = {}
    for school, count in school_counts.items():
        schools_dict[convert_pandas_to_json_safe(school)] = convert_pandas_to_json_safe(count)

    data['stats'] = {
        'total_courses': convert_pandas_to_json_safe(len(df)),
        'total_schools': convert_pandas_to_json_safe(df['school_name'].nunique()),
        'schools': schools_dict
    }

    # Add hierarchical CIP filtering data
    data['hierarchical_cips'] = prepare_hierarchical_cip_data(df)

    # Add six-digit CIP histogram data
    data['six_digit_histogram'] = prepare_six_digit_histogram(df)

    # Add CCM code names mapping
    data['ccm_names'] = {}

    # Add two-digit names
    if 'two-digit_classification' in df.columns:
        cip_codes = df['two-digit_classification'].unique()
        names = {}
        for cip in cip_codes:
            name = get_ccm_name(cip, 'two-digit')
            if name:
                # Convert to string and normalize using repair_ccm
                cip_str = str(cip).strip()
                cip_normalized = repair_ccm(cip_str, 'two')
                # CRITICAL: Force both key and value to be strings
                key = str(cip_normalized)
                names[key] = str(name)
        data['ccm_names']['two-digit'] = names

    # Add six-digit names
    if 'six-digit_classification' in df.columns:
        cip_codes = df['six-digit_classification'].unique()
        names = {}
        for cip in cip_codes:
            name = get_ccm_name(cip, 'six-digit')
            if name:
                # Convert to string and normalize using repair_ccm
                cip_str = str(cip).strip()
                cip_normalized = repair_ccm(cip_str, 'six')
                # CRITICAL: Force both key and value to be strings
                key = str(cip_normalized)
                names[key] = str(name)
        data['ccm_names']['six-digit'] = names

    print(f"PREPARE DASHBOARD DEBUG - Final data keys: {list(data.keys())}")
    print(f"PREPARE DASHBOARD DEBUG - Six-digit histogram length: {len(data['six_digit_histogram'])}")

    # Debug: Check CCM names mapping
    if 'ccm_names' in data and 'six-digit' in data['ccm_names']:
        ccm_six_names = data['ccm_names']['six-digit']
        print(f"PREPARE DASHBOARD DEBUG - Six-digit CCM names count: {len(ccm_six_names)}")
        print(f"PREPARE DASHBOARD DEBUG - Sample CCM names: {list(ccm_six_names.items())[:5]}")
        print(f"PREPARE DASHBOARD DEBUG - Sample keys in mapping: {list(ccm_six_names.keys())[:10]}")

    return data

def prepare_hierarchical_cip_data(df):
    """Prepare hierarchical CIP data for faceted filtering"""
    hierarchical_data = {}

    # Two-digit to Four-digit mapping
    if 'two-digit_classification' in df.columns and 'four-digit_classification' in df.columns:
        two_to_four = df.groupby(['two-digit_classification', 'four-digit_classification']).size().reset_index(name='count')
        hierarchical_data['two_to_four'] = {}
        for _, row in two_to_four.iterrows():
            two_digit = convert_pandas_to_json_safe(row['two-digit_classification'])
            four_digit = convert_pandas_to_json_safe(row['four-digit_classification'])
            count = convert_pandas_to_json_safe(row['count'])

            if two_digit not in hierarchical_data['two_to_four']:
                hierarchical_data['two_to_four'][two_digit] = []
            hierarchical_data['two_to_four'][two_digit].append({
                'four_digit': four_digit,
                'count': count
            })

    # Four-digit to Six-digit mapping
    if 'four-digit_classification' in df.columns and 'six-digit_classification' in df.columns:
        four_to_six = df.groupby(['four-digit_classification', 'six-digit_classification']).size().reset_index(name='count')
        hierarchical_data['four_to_six'] = {}
        for _, row in four_to_six.iterrows():
            four_digit = convert_pandas_to_json_safe(row['four-digit_classification'])
            six_digit = convert_pandas_to_json_safe(row['six-digit_classification'])
            count = convert_pandas_to_json_safe(row['count'])

            if four_digit not in hierarchical_data['four_to_six']:
                hierarchical_data['four_to_six'][four_digit] = []
            hierarchical_data['four_to_six'][four_digit].append({
                'six_digit': six_digit,
                'count': count
            })

    return hierarchical_data

def prepare_six_digit_histogram(df):
    """Prepare histogram data for most common six-digit CIP codes"""
    #print(f"HISTOGRAM DEBUG - Checking for six-digit_classification column")
    #print(f"HISTOGRAM DEBUG - Available columns: {list(df.columns)}")

    if 'six-digit_classification' not in df.columns:
        print(f"HISTOGRAM DEBUG - six-digit_classification column not found, returning empty list")
        return []

    # Get counts of six-digit CIP codes
    six_digit_counts = df['six-digit_classification'].value_counts().head(20)  # Top 20
    #print(f"HISTOGRAM DEBUG - Found {len(six_digit_counts)} unique six-digit CIP codes")
    #print(f"HISTOGRAM DEBUG - Top 5 counts: {six_digit_counts.head()}")

    histogram_data = []
    for cip_code, count in six_digit_counts.items():

        # Normalize the CIP code to ensure consistent formatting
        cip_str = str(cip_code).strip()
        if cip_str.find('.') == 1:
            cip_str = f"0{cip_str}"
        histogram_data.append({
            'cip_code': convert_pandas_to_json_safe(cip_str),
            'count': convert_pandas_to_json_safe(count),
            'percentage': convert_pandas_to_json_safe(round((count / len(df)) * 100, 2))
        })

    print(f"HISTOGRAM DEBUG - Created histogram with {len(histogram_data)} entries")
    return histogram_data

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run the Flask application')
    parser.add_argument('-P', '--port', type=int, default=5000, help='Port to run the server on (default: 5000)')
    args = parser.parse_args()

    # Load CCM taxonomy on startup
    print("Loading CCM taxonomy...")
    load_ccm_taxonomy()

    # Load all models on startup
    print("Loading all models on startup...")
    for model_type in MODEL_INFO.keys():
        try:
            load_model(model_type)
            print(f"✓ Loaded {model_type} model: {MODEL_INFO[model_type]['model_name']}")
        except Exception as e:
            print(f"✗ Failed to load {model_type} model: {str(e)}")

    print("Model loading complete!")
    app.run(debug=True, host='127.0.0.1', port=args.port)