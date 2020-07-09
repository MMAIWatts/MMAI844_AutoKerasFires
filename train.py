import autokeras as ak
from utils import *
from sklearn.metrics import f1_score, accuracy_score
from tensorflow.keras.datasets import mnist

# Download datasets

training_sets, testing_sets = load_datasets()

# Load training dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Customize the model
input_node = ak.ImageInput()
output_node = ak.Normalization()(input_node)
output_node1 = ak.ConvBlock()(output_node)
# output_node2 = ak.ConvBlock()(input_node)
# output_node3 = ak.ConvBlock()(input_node)
# output_node = ak.Merge()([output_node1, output_node2, output_node3])
output_node = ak.ClassificationHead()(output_node1)

classifier = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=1, overwrite=True)

# Instantiate the image classifier
# classifier = ak.ImageClassifier(max_trials=5, num_classes=10)

classifier.fit(X_train, y_train, epochs=2)

y_pred = classifier.predict(X_test)

f1 = f1_score(y_test, y_pred, average='micro')
accuracy = accuracy_score(y_test, y_pred)

print(f'F1: {f1:.3f}    Accuracy: {accuracy:.3f}')

# Get the best model and show its architecture
model = classifier.export_model()
model.save('./model_cache/best_model.h5')
