import autokeras as ak
from tensorflow.keras.utils import plot_model
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from utils import *

# Download datasets

training_sets, testing_sets = load_datasets()

# Load training dataset
print('Loading training dataset...', end='')
train_x = np.load(os.path.join("./model_cache/train_data", training_sets[0]))
train_y = np.load(os.path.join("./model_cache/train_data", training_sets[1]))
print('done.')

# Load testing dataset
print('Loading testing dataset...', end='')
test_x = np.load(os.path.join('./model_cache/test_data', testing_sets[0]))
test_y = np.load(os.path.join('./model_cache/test_data', testing_sets[1]))
print('done.')

# Instantiate the image classifier
classifier = ak.ImageClassifier(max_trials=1)

classifier.fit(train_x, train_y, epochs=1)

y_pred = classifier.predict(test_x)

f1 = f1_score(test_y, y_pred)
accuracy = accuracy_score(test_y, y_pred)

print(f'F1: {f1:.3f}    Accuracy: {accuracy:.3f}')

# Get the best model and show its architecture
model = classifier.export_model()
model.save_model('./model_cache/best_model', save_format='tf')
plot_model(model)
