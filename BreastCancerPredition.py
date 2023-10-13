# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load and preprocess the Kaggle dataset
data = pd.read_csv('Breast_cancer_data.csv')  # Update the dataset filename

# Replace 'diagnosis' with the actual column name that contains the diagnosis information
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Create and train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a function to predict diagnoses for new patient data
def predict_diagnosis(new_patient_data):
    diagnosis = model.predict([new_patient_data])
    if diagnosis[0] == 'B':
        return "Benign"
    else:
        return "Malignant"

# Lists to store patient data and corresponding diagnoses
patient_data = []
patient_diagnoses = []

# Command-line interface to take patient data and make predictions
while True:
    print("Enter patient data:")
    mean_radius = float(input("Mean Radius: "))
    mean_texture = float(input("Mean Texture: "))
    mean_perimeter = float(input("Mean Perimeter: "))
    mean_area = float(input("Mean Area: "))
    mean_smoothness = float(input("Mean Smoothness: "))

    input_data = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness]

    result = predict_diagnosis(input_data)

    # Append patient data and diagnosis to lists for visualization
    patient_data.append(input_data)
    patient_diagnoses.append(result)

    print(f"Predicted Diagnosis: {result}")

    # Ask the user if they want to continue
    continue_input = input("Do you want to make another prediction? (yes/no): ").strip().lower()
    if continue_input != 'yes':
        break

# Separate data points for benign and malignant diagnoses
benign_data = [patient_data[i] for i in range(len(patient_data)) if patient_diagnoses[i] == "Benign"]
malignant_data = [patient_data[i] for i in range(len(patient_data)) if patient_diagnoses[i] == "Malignant"]

# Check if there are data points for each diagnosis before creating a scatter plot
if benign_data:
    benign_data = list(zip(*benign_data))
    plt.scatter(benign_data[0], benign_data[1], c='green', label='Benign', marker='o')

if malignant_data:
    malignant_data = list(zip(*malignant_data))
    plt.scatter(malignant_data[0], malignant_data[1], c='red', label='Malignant', marker='x')

plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.title('Patient Data Scatter Plot')
plt.legend()
plt.show()

# Count the occurrences of each diagnosis
diagnosis_counts = data['diagnosis'].value_counts()

# Create a bar chart to visualize the diagnosis distribution with rotated x-axis labels
plt.figure(figsize=(6, 6))
diagnosis_counts.plot(kind='bar', color=['green', 'red'])
plt.title('Diagnosis Distribution')
plt.xlabel('Diagnosis')
plt.ylabel('Count')
plt.xticks(rotation=90)  # Rotate x-axis labels by 90 degrees
plt.show()
