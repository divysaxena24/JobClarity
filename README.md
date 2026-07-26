# JobClarity

> AI-powered job description analysis and skill gap detection platform.

## 📋 Overview

**JobClarity** is an end-to-end machine learning pipeline that helps job seekers and recruiters gain clarity on job market requirements. The project analyzes job descriptions, identifies key skills, detects skill gaps, and provides actionable recommendations.

## 🏗️ Project Structure

```
JobClarity/
│
├── data/                   # Data storage
│   ├── raw/                # Raw, unprocessed data
│   └── processed/          # Cleaned and processed data
│
├── notebooks/              # Jupyter notebooks for EDA & prototyping
│
├── src/                    # Source code
│   ├── data/               # Data ingestion and collection modules
│   ├── validation/         # Data validation and schema enforcement
│   ├── preprocessing/      # Data cleaning and preprocessing pipelines
│   ├── features/           # Feature engineering and extraction
│   ├── models/             # Model training, evaluation, and inference
│   ├── api/                # FastAPI endpoints and API logic
│   └── utils/              # Shared utilities and helper functions
│
├── tests/                  # Unit and integration tests
│
├── artifacts/              # Experiment artifacts, figures, reports
│
├── models/                 # Saved trained model files
│
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
└── Dockerfile              # Docker container definition
```

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip / conda

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/JobClarity.git
cd JobClarity

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```bash
# Run data pipeline
python -m src.data.ingest

# Start the API server
uvicorn src.api.main:app --reload

# Run tests
pytest tests/
```

## 📦 Modules

| Module        | Description                                               |
|---------------|-----------------------------------------------------------|
| `data`        | Data collection, scraping, and loading from sources       |
| `validation`  | Schema validation, data quality checks, and error handling|
| `preprocessing` | Text cleaning, normalization, and transformation pipelines|
| `features`    | Feature extraction (TF-IDF, embeddings, skill extraction) |
| `models`      | ML models for classification, clustering, and recommendation |
| `api`         | FastAPI REST endpoints for inference and management        |
| `utils`       | Logging, config management, helper functions              |

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 🐳 Docker

```bash
# Build image
docker build -t jobclarity .

# Run container
docker run -p 8000:8000 jobclarity
```

## 📄 License

MIT
