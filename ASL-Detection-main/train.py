import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the enhanced dataset
data_dict = pickle.load(open('./data_asl.pickle', 'rb'))

# Extract data and labels
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Encode labels as integers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    data, encoded_labels, test_size=0.2, shuffle=True, stratify=encoded_labels, random_state=42
)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Predict on the test set
y_predict = model.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_predict)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print("\nClassification Report:\n")
print(classification_report(y_test, y_predict, target_names=label_encoder.classes_))

# Save the trained model and label encoder
with open('model_asl_rf.p', 'wb') as model_file:
    pickle.dump({'model': model, 'label_encoder': label_encoder}, model_file)

print("Model and label encoder saved as 'model_asl_rf.p'")
