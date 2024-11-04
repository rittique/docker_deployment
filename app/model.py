import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a random forest classifier
model = RandomForestClassifier()
model.fit(X=X, y=y)

# save the model
joblib.dump(model, 'model.joblib')