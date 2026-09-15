🏥 AI Medical Diagnostic Portal (CodeAlpha Task 4)
An interactive Machine Learning web application built using Python, Scikit-Learn, and Streamlit to predict disease risk (Benign vs. Malignant) based on clinical biomarker metrics. Developed as part of the CodeAlpha Machine Learning Virtual Internship.

🚀 Live Demo & Screenshots
Live App Link: Click here to view live web app (Aap yahan apna Streamlit Cloud URL dal sakte hain deployment ke baad)
🛠️ Project Structure
diseaseapp.py: The frontend web application interface built with Streamlit.
Disease Prediction from Medical.py: The machine learning training pipeline script.
disease_model.pkl: The trained Random Forest classification model.
scaler.pkl: StandardScaler object used for data normalization.
feature_names.pkl: List of clinical features utilized by the model.
requirements.txt: Python package dependencies for cloud deployment.
⚙️ How to Run Locally
Clone the repository:
git clone [https://github.com/uk792486-sudo/Disease-Prediction-from-Medical_CodeAlpha](https://github.com/uk792486-sudo/Disease-Prediction-from-Medical_CodeAlpha)
cd Prediction-from-Medical_CodeAlpha
**Bash
  python "Disease Prediction from Medical.py"
  **bash 
  python "Disease Prediction from Medical.py"
  **bash
       streamlit run diseaseapp.py 
