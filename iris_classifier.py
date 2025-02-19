from sklearn.datasets import load_iris
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

#Step-1 load iris data
iris = load_iris()
x= iris.data
y= iris.target
##Enable autologging
mlflow.sklearn.autolog()
#Step 2 Split data
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)
with mlflow.start_run():
    model = LogisticRegression(max_iter = 200)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    print(f"Accuracy: {accuracy}")
    mlflow.log_metric("accuracy",accuracy)
    print("Classification Report:")
    print(report)
    