# AutoML with Streamlit
- [AutoML with Streamlit](#automl-with-streamlit)
- [🚧 Development Status](#-development-status)
  - [📍 Project Roadmap](#-project-roadmap)
  - [✅ Phase 1: Core Functionality (Completed)](#-phase-1-core-functionality-completed)
  - [🚧 Phase 2: Advanced Preprocessing \& Visualizations (In Progress)](#-phase-2-advanced-preprocessing--visualizations-in-progress)
  - [🔨 Phase 3: Model Training Engine](#-phase-3-model-training-engine)
  - [🧠 Phase 4: Automation \& Explainability](#-phase-4-automation--explainability)
  - [📦 Phase 5: Export \& Deployment](#-phase-5-export--deployment)
  - [🎨 Phase 6: UI/UX Enhancements](#-phase-6-uiux-enhancements)
- [📥 Installation and Running Instructions](#-installation-and-running-instructions)
  - [📋 Prerequisites](#-prerequisites)
    - [📦 Step 1: Install Required Packages](#-step-1-install-required-packages)
    - [🚀 Step 2: Run the Streamlit Application](#-step-2-run-the-streamlit-application)
  - [🛠️ Troubleshooting](#️-troubleshooting)
  - [🧪 Example Workflow](#-example-workflow)
  - [📚 Additional Resources](#-additional-resources)
- [🔧 Key Features:](#-key-features)
- [⚠️ Known Issues](#️-known-issues)
  - [1. 🚨 Unused Settings Are Saved](#1--unused-settings-are-saved)
  - [2. 🚨 Refactor Functions into a Class for Readability and Maintainability - *(🔧 Work In Progress)*](#2--refactor-functions-into-a-class-for-readability-and-maintainability----work-in-progress)

# 🚧 Development Status
The project is currently under active development. During this process, various bugs may arise and are being addressed regularly.
<p align="center">
  <img src="./data/ui_images/under_construction.png" alt="Robot Under Construction" width="30%">
</p>

This project is an AutoML platform designed to streamline machine learning workflows through an intuitive web interface. It is currently under active development, and new features are continuously being added.

## 📍 Project Roadmap

---

## ✅ Phase 1: Core Functionality (Completed)
- [x] CSV file upload support  
- [x] Load/save predefined configuration  
- [x] ML task selection (Classification, Regression)  
- [x] Train-test split options  
- [x] Basic preprocessing: drop columns,duplicate removal, fill missing values, encode categoricals  

---

## 🚧 Phase 2: Advanced Preprocessing & Visualizations (In Progress)
- [x] Outlier detection and handling 
  - [x] Visualize outliers 
  - [x] Implement detection methods 
  - [x] Provide options: remove, flag, or adjust outliers
    - [x] Transform the data 
    - [x] Impute the outliers 
    - [x] Flag outliers as a separate feature
    - [x] Remove the outliers
- [ ] Handle multicollinearity
- [ ] Imbalanced class handling 
- [ ] Feature scaling and transformation  
- [ ] Skew handling and normalization  
- [ ] Date/datetime conversions and time-based feature engineering  
- [ ] Resampling support for time series  
- [ ] Interactive EDA and correlation plots  

---

## 🔨 Phase 3: Model Training Engine
- [ ] Implement Classification, Regression, Clustering, Dimensionality Reduction  
- [ ] Time series forecasting module  
- [ ] Train all models with comparison view  
- [ ] Hyperparameter tuning (GridSearchCV, Optuna)  
- [ ] Model performance visualizations (ROC, Confusion Matrix, etc.)  

---

## 🧠 Phase 4: Automation & Explainability
- [ ] AutoML (TPOT/AutoSklearn/FLAML integration)  
- [ ] Model explainability (SHAP, LIME)  
- [ ] Auto feature selection based on importance  
- [ ] Custom metric support and scoring configurability  

---

## 📦 Phase 5: Export & Deployment
- [ ] Download trained model & preprocessing pipeline  
- [ ] Export ready-to-use inference script  
- [ ] Export complete notebook for reproducibility  
- [ ] Generate FastAPI app for model inference  
- [ ] Docker deployment setup  

---

## 🎨 Phase 6: UI/UX Enhancements
- [ ] Sidebar with step navigation  
- [ ] Save/load project sessions  
- [ ] Improved loading indicators and error feedback  
- [ ] Custom transformation builder in UI  

# 📥 Installation and Running Instructions

This project is developed using **Python 3.12.3** on a **Ubuntu 24.04.2 LTS** environment and **Python 3.11.9** on a **Windows 11**. Follow the steps below to install dependencies and run the **AutoML Streamlit application**.

## 📋 Prerequisites
**Python 3.8 or higher**: Ensure Python is installed. Download Python.
**pip**: Python package manager (comes installed with Python).
**Virtual Environment** (optional but recommended): To avoid dependency conflicts, create a virtual environment:
```bash
python -m venv automl_venv
source automl_venv/bin/activate
```

### 📦 Step 1: Install Required Packages

All dependencies are listed in the `requirements.txt` file.
1. Open a terminal in the project directory.
2. Run the following command:

```bash
pip install -r requirements.txt
```
This will install essential packages like `streamlit==1.20.0`, `pandas`, `scikit-learn`, `torch`, and more.

**Note:**
If you encounter any issues, make sure pip is up-to-date:
```bash
pip install --upgrade pip
```

### 🚀 Step 2: Run the Streamlit Application
You have two options to run the app:
**✅ Option 1: Using Visual Studio Code with Streamlit Runner Extension**
1. Install the **Streamlit Runner** extension:
  - Open VSCode → Go to Extensions (Ctrl+Shift+X).
  - Search for **Streamlit Runner** from **joshrmosier** and install it.
2. Open main.py in VSCode.
3. Right-click on main.py and select **Run with Streamlit**.
The app will automatically open in your browser.
**✅ Option 2: Using the Terminal**
1. In the terminal, navigate to the project directory.
2. Run:
```bash
streamlit run main.py
```
This will launch the app in your default web browser. 

## 🛠️ Troubleshooting
**Module Not Found:** Ensure all packages are installed correctly. If needed, activate your virtual environment and rerun:
```bash
pip install -r requirements.txt
```
**Python Version:** Confirm your Python version:
```bash
python --version
```

Make sure it is **3.8+**.

## 🧪 Example Workflow

```bash
# 1. Clone the repository
git clone <repository-url>
cd <repository-name>

# 2. Set up and activate a virtual environment
python -m venv automl_venv
source automl_venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run main.py
```

## 📚 Additional Resources
- [Streamlit Documentation](https://docs.streamlit.io/)
- Open an **Issue** if you encounter any problems.

# 🔧 Key Features:
1. Upload Your Data: Load your dataset in .csv format. – ✅
2. Load Predefined Settings: Optionally load a saved configuration file to speed up the process. – ✅
3. Choose ML Task: Select the type of machine learning task you want to perform (e.g., classification, regression). – ✅
4. Data Splitting Options: Define the train-test split ratio and set a random seed for reproducibility. – ✅
5. Data Preprocessing:
   - Drop unnecessary columns – ✅
   - Fill in missing values – ✅
   - Convert categorical columns to numerical – ✅
   - Remove outliers via "Visualize Column" – ✅
   - Handle multicollinearity – 🔧
   - Address class imbalance – 🔧
   - Apply feature scaling – 🔧
   - Handle skewed data – 🔧
   - Create new features from existing ones – 🔧
   - Convert date columns to datetime format (if not already) – 🔧
   - Handle seasonality – 🔧
   - Resample data – 🔧
   - Perform Exploratory Data Analysis (EDA) – 🔧
6. Train Models: Choose a specific model or train all available models to compare performance. – 🚧
   - Supported model types: 
     - Regression - 🚧
     - Time Series Forecasting - 🚧
     - Classification - 🚧
     - Clustering - 🚧
     - Dimensionality Reduction - 🚧
7. Evaluate Model: Review the model’s performance on the test set using key metrics. – 🚧
8. Download Model & Inference Script: Export the trained model and a ready-to-use inference script. – 🚧

# ⚠️ Known Issues

## 1. 🚨 Unused Settings Are Saved

- All parameter values  are saved to the configuration file, **even if the associated method was not selected or applied**.
- This leads to unnecessary clutter in the configuration file, making it harder to maintain or understand later.

## 2. 🚨 Refactor Functions into a Class for Readability and Maintainability - *(🔧 Work In Progress)*
- Currently, all outlier detection, visualization, and handling logic is implemented as scattered standalone functions.
- This structure leads to:
  - Reduced readability and increased cognitive load when navigating the code
  - Duplication of logic and inconsistent parameter management
  - Difficulty in maintaining and extending the system
- All logic should be encapsulated into related class to promote clarity, state reuse, and modular design.