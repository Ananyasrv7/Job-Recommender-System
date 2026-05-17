# Job Recommender & Skill Gap Analysis System

## Overview
This project presents an AI-powered Job Recommendation and Skill Gap Analysis System built using Sentence-BERT (SBERT) and the ESCO Ontology dataset. The system semantically matches user resumes and skills with relevant job roles while also identifying missing skills required for target careers. 
Unlike traditional keyword-based job recommendation systems, this project uses semantic embeddings and cosine similarity to understand contextual meaning between resumes, job descriptions, and skills.

## Key Features
- Semantic job recommendation using SBERT embeddings
- Skill gap analysis for target job roles
- Resume-to-job matching using cosine similarity
- ESCO ontology integration with 3000+ occupations and 13000+ skills
- Threshold optimization for accurate recommendations
- Evaluation metrics visualization:
  - Accuracy (0.998)
  - Precision (0.997)
  - Recall (0.999)
  - F1-score (0.998)
  - ROC Curve (AUC = 1.0000)
  - Confusion Matrix

## Technologies Used
- Python
- Sentence-BERT (SBERT)
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- NLP
- ESCO Dataset
- Cosine Similarity

## Dataset
The project uses the ESCO v1.1.1 dataset containing:
- 3000+ occupations
- 13000+ skills
- 120000+ occupation-skill relations

## System Workflow
1. Data preprocessing using ESCO dataset
2. Fine-tuning SBERT model on occupation-skill pairs
3. Generating semantic embeddings
4. Computing cosine similarity
5. Recommending relevant job roles
6. Identifying missing skills for career development

## Model Performance
At the optimized threshold of 0.45, the system achieved:

- Accuracy: 99.8%
- Precision: 99.7%
- Recall: 99.9%
- F1 Score: 99.8%

## Project Structure
- `prepare_data.py` → Dataset preprocessing
- `train_sbert.py` → SBERT model fine-tuning
- `evaluate.py` → Model evaluation and threshold optimization
- `match.py` → Job recommendation system
- `skill_gap.py` → Skill gap analysis
- `esco-sbert/` → Trained SBERT model
- `checkpoints/` → Model checkpoints

## How to Run

Clone the repository:

```bash
git clone https://github.com/Ananyasrv7/Job-Recommender-System.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the job recommendation system:

```bash
python match.py
```

Run the skill gap analysis:

```bash
python skill_gap.py
```

## Research Contribution
This project proposes a two-phase semantic recommendation framework:
1. Job Recommendation
2. Skill Gap Identification

The system extends existing CareerBERT-based approaches by introducing actionable skill-gap insights for users.

## Future Improvements
- Web application deployment
- Resume upload and parsing
- Real-time job API integration
- Multilingual support using ESCO taxonomy
- Enhanced career path recommendations

## Author
Ananya Srivastava & Shivam Mittal
Manipal University Jaipur

## Research Paper
Title:
"An Approach Towards Skill Gap Analyzing Job Recommendation Using Sentence-BERT and ESCO Ontology"
