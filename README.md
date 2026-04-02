# Course Description Classifier

A Flask-based web application that allows users to upload CSV files containing course descriptions and classify them using Hugging Face models from the [Classifying Courses at Scale](https://huggingface.co/collections/annamp/classifying-courses-at-scale-66e470e6126f9577f89fd417) collection.

## Features

- Upload CSV files with course descriptions or titles
- Choose from three different classification models:
  - Two-digit CIP code classification (general category)
  - Four-digit CIP code classification (intermediate specificity)
  - Six-digit CIP code classification (most specific)
- Process files in the background
- Download results as CSV files
- **NEW**: Interactive dashboard with visualizations
- **NEW**: Course search functionality
- **NEW**: Related courses discovery by CIP code
- **NEW**: School-based analysis and charts
- **NEW**: Support for additional columns (school_name, school_year_enrolled)
- **NEW**: Row limit processing (process only a subset of rows for testing)

## Requirements

- Python 3.8+
- Flask
- Pandas
- PyTorch
- Transformers (Hugging Face)
- Hugging Face account (for API access)

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd course-descriptions
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up Hugging Face authentication (choose one method):

   **Option 1: Use the Hugging Face CLI (recommended)**
   ```
   pip install huggingface_hub
   huggingface-cli login
   ```
   Follow the prompts to log in with your Hugging Face account.

   **Option 2: Use a Hugging Face token**
   - Create a copy of `.env.example` and name it `.env`:
     ```
     cp .env.example .env
     ```
   - Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Edit the `.env` file and replace `your_huggingface_token_here` with your actual token

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://127.0.0.1:5000`

## Usage

1. Prepare a CSV file with either a `course_description` or `course_title` column containing the text to classify.
2. Upload the CSV file using the web interface.
3. Optionally specify how many rows to process (useful for testing with large files).
4. Select which classification models you want to apply (two-digit, four-digit, and/or six-digit).
5. Click "Upload and Classify" to process the file.
6. Once processing is complete, you can:
   - View the interactive dashboard with charts and analysis
   - Search for specific courses
   - Find related courses by CIP code
   - Download the result file

## CSV Format

Your input CSV file should have either a `course_description` or `course_title` column. Additional optional columns enhance the dashboard experience:

**Required columns:**
- `course_description` OR `course_title` - Text content to classify

**Optional columns:**
- `school_name` - Name of the school/institution
- `school_year_enrolled` - Academic year
- `course_code` - Course identifier
- `department` - Academic department
- `credits` - Credit hours

**Example with all columns:**
```
id,course_code,course_title,course_description,department,credits,school_name,school_year_enrolled
1,CS101,Introduction to Computer Science,"Introduction to fundamental concepts of computer science and programming.",Computer Science,3,University of Technology,2023-2024
2,MATH201,Calculus I,"Limits, derivatives, and integrals of algebraic and trigonometric functions.",Mathematics,4,State University,2023-2024
```

**Example with minimal columns:**
```
id,course_title
1,Introduction to Computer Science
2,Calculus I
```

## Output

The output CSV will contain all original columns plus additional columns with the classifications:

- `two-digit_classification` (if selected)
- `four-digit_classification` (if selected)
- `six-digit_classification` (if selected)

## Dashboard Features

After processing your file, you can access the interactive dashboard which provides:

### Visualizations
- **CIP Code Breakdown Charts**: Bar charts showing the distribution of courses by CIP code level (2, 4, or 6 digits) across different schools
- **School Comparison**: Visual comparison of course offerings between institutions
- **Statistics Overview**: Total courses, number of schools, and course distribution

### Course Search
- **Text Search**: Search through course titles and descriptions
- **Related Courses**: Find courses with the same 6-digit CIP code
- **Course Details**: View comprehensive information about each course including school, department, and classification

### Navigation
- Easy access to download processed results
- Return to upload page for processing additional files
- Responsive design that works on desktop and mobile devices

## Row Limit Processing

For large CSV files, you can specify how many rows to process:

- **Default behavior**: Process all rows in the file
- **Row limit**: Enter a number to process only the first N rows
- **File preview**: The interface shows how many rows are in your uploaded file
- **Use cases**:
  - Test processing with a small subset before running on the full dataset
  - Process large files in batches
  - Reduce processing time for initial testing

**Example**: If your file has 1000 rows but you only want to test with 50 rows, enter "50" in the row limit field.

## Model Information

This application uses models from the [Classifying Courses at Scale](https://huggingface.co/collections/annamp/classifying-courses-at-scale-66e470e6126f9577f89fd417) collection:

- [annamp/classifying-courses-at-scale-two-digit-roberta-base](https://huggingface.co/annamp/classifying-courses-at-scale-two-digit-roberta-base)
- [annamp/classifying-courses-at-scale-four-digit-roberta-base](https://huggingface.co/annamp/classifying-courses-at-scale-four-digit-roberta-base)
- [annamp/classifying-courses-at-scale-six-digit-roberta-base](https://huggingface.co/annamp/classifying-courses-at-scale-six-digit-roberta-base)

These models classify course descriptions according to the Classification of Instructional Programs (CIP) taxonomy.

## Troubleshooting

If you encounter the error `is not a local folder and is not a valid model identifier listed on 'https://huggingface.co/models'`, it means the application can't authenticate with Hugging Face. Make sure you've followed one of the authentication methods described above.