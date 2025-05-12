# Answer Evaluation System

## Overview
An advanced AI-powered system for evaluating student answers using natural language processing techniques.

## Features
- Single answer evaluation
- Batch answer evaluation
- CSV file processing
- Detailed scoring metrics
- Constructive feedback generation

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/answer-evaluation-system.git
cd answer-evaluation-system
```

2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download NLTK resources
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

## Running the Application

### Local Development
```bash
uvicorn app:app --reload
```

### Deployment
The application is configured for easy deployment on Render.com.

## API Endpoints

- `GET /`: Root endpoint with welcome message
- `POST /evaluate`: Evaluate a single answer
- `POST /evaluate-batch`: Evaluate multiple answers
- `POST /evaluate-csv`: Evaluate answers from a CSV file

## Example Request
```python
import requests

# Single answer evaluation
response = requests.post('http://localhost:8000/evaluate', json={
    'student_answer': 'Machine learning is...',
    'model_answer': 'Machine learning is a branch of AI...'
})
print(response.json())
```

## Scoring Metrics
- Overall Score
- Semantic Similarity
- Keyword Matching
- Grammar Evaluation
- Constructive Feedback

## Technologies
- FastAPI
- scikit-learn
- NLTK
- Pandas
- TensorFlow

## License
[Your License Here]
```

## Deployment Workflow
1. Push to GitHub
2. Connect to Render
3. Set build command: `pip install -r requirements.txt`
4. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## Contributing
Contributions are welcome! Please read the contributing guidelines.
