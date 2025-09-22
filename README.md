# Task 1 - Data Cleaning & Preprocessing  

## Objective  
The goal of this task is to clean and preprocess the Titanic dataset to prepare it for machine learning. Steps include:  
- Handling missing values  
- Encoding categorical features  
- Standardizing numerical features  
- Detecting and removing outliers  
- Visualizing data before and after preprocessing  

## Tools & Libraries  
- Python  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

## Dataset  
Titanic Dataset from Kaggle:  
👉 https://www.kaggle.com/datasets/yasserh/titanic-dataset  

## Steps Performed  
1. Imported and explored the dataset.  
2. Handled missing values (Age, Embarked, dropped Cabin).  
3. Encoded categorical variables (Sex, Embarked).  
4. Standardized numerical features (Age, Fare).  
5. Visualized and removed outliers using the IQR method.  
6. Created comparison plots (before vs after cleaning).  

## Visualizations  
- Age & Fare Before Cleaning -> images/before_cleaning.png  
- Fare Outliers (Standardized, Before Removal) -> images/fare_outliers.png  
- Age & Fare After Cleaning -> images/after_cleaning.png  
- Survival by Sex (After Cleaning) -> images/survival_by_sex.png  

## Key Learnings  
- Different ways of handling missing data (median, mode, dropping columns).  
- Importance of encoding categorical features.  
- Difference between normalization and standardization.  
- Detecting outliers with boxplots and removing them with IQR.  
- Preprocessing ensures higher data quality and helps improve model accuracy.  

## Repository Structure  
📦 Task1_Data_Cleaning  
 ┣ 📜 task1_data_cleaning.py  
 ┣ 📜 titanic.csv  
 ┣ 📜 README.txt  
 ┗ 📂 images  
    ┣ 📜 before_cleaning.png  
    ┣ 📜 fare_outliers.png  
    ┣ 📜 after_cleaning.png  
    ┗ 📜 survival_by_sex.png  

## How to Run  
1. GitHub : https://github.com/Mohammedshair/Task1_Data-cleaning-Preprocessing.git
   cd Task1_Data_Cleaning  

2. Install dependencies:  
   pip install pandas numpy matplotlib seaborn scikit-learn  

3. Run the script:  
   python task1_data_cleaning.py  

4. Check the images/ folder for saved plots.  
