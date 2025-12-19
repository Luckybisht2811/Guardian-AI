# 🛡️ Guardian-AI — AI-Powered Crime Pattern Analysis System

## 📌 Overview
Guardian-AI is an AI-powered web application designed to analyze real-world crime datasets, uncover hidden patterns, and predict crime trends.  
The system provides interactive visualizations and insights to assist law enforcement and researchers in proactive decision-making.

---

## 🚀 Key Features
- 🔍 Analyze and visualize crime patterns across regions  
- 🤖 Predict crime trends using machine learning models  
- 🗺️ Geospatial analysis of crime hotspots and resource distribution  
- 🖥️ Interactive Streamlit dashboard for real-time insights  
- ⚡ End-to-end data workflow from preprocessing to live deployment  
- 🐳 Dockerized application for easy deployment and scalability  

---

## 🛠️ Tech Stack
- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn, Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn, Plotly  
- **Web Framework / UI:** Streamlit  
- **Containerization:** Docker  
- **Version Control:** Git & GitHub  

---

## 🎯 Use Cases
- Crime pattern analysis for law enforcement  
- Predictive policing and resource allocation  
- Academic research in criminology and data science  
- Interactive dashboards for non-technical stakeholders  

---

## 🚀 Future Enhancements
- Integrate real-time crime data API  
- Add deep learning models for more accurate predictions  
- Implement user authentication and role-based access  
- Expand geospatial analytics with interactive maps  

---

## 👤 Author
**Lalit Singh Bisht**  
Software Engineer (Python) | Backend & Data Engineering | Applied ML  
GitHub: [https://github.com/Luckybisht2811](https://github.com/Luckybisht2811)

---

## 📂 Project Structure
```text
Guardian-AI/
│
├── .devcontainer/                   # Configuration files for VS Code Dev Container
├── component_datasets/              # Datasets for different components of the system
├── continuous_learning_and_feedback/ # Modules for model retraining and feedback integration
├── crime_pattern_analysis/          # Scripts and notebooks for crime pattern discovery
├── criminal_profiling/              # Modules for profiling criminals based on historical data
├── predictive_modeling/recidivism_prediction/ # ML models and scripts for recidivism prediction
├── resources_allocation/            # Scripts for optimizing police resource distribution
├── app/                             # Streamlit / web application front-end files
├── assets/                          # Static files like images, icons, logos
├── models/recidivism_model/         # Saved ML models for crime prediction
├── pipelines/                        # Data pipelines for preprocessing, training, and evaluation
├── devcontainer.json                 # VS Code dev container configuration file
├── Dockerfile                        # Docker configuration for containerized deployment
├── packages.txt                       # Additional package dependencies
├── requirements.txt                   # Python project dependencies
└── README.md                          # Project documentation


⚙️ Installation & Usage
1️⃣ Clone the repository
git clone https://github.com/Luckybisht2811/Guardian-AI.git
cd Guardian-AI

2️⃣ Using Python (local setup)
pip install -r requirements.txt
streamlit run app.py

3️⃣ Using Docker
# Build the Docker image
docker build -t guardian-ai .

# Run the Docker container
docker run -p 8501:8501 guardian-ai

# Open your browser at http://localhost:8501


