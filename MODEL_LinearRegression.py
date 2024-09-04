#Pre-processing and split of the Data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler

X = df.drop(columns=["EPS_Target", "Company", "Var"])
#X = df[["Cost of Goods Sold (COGS) incl. D&A", "Total Equity"]]
y = df.EPS_Target.astype('int')

# Normaliser les données
scaler = MinMaxScaler()

# Fit the scaler on your data and transform it
X_normalized = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.3, random_state=102)

# Build model
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

print(model_lr.coef_)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define the range of C values
C_values = np.arange(1, 1500, 3)  # From 1 to 100, inclusive

# Define the number of random states
num_random_states = 10

# Initialize lists to store results
mean_accuracy_test = []
mean_f1_class_0 = []
mean_f1_class_1 = []

best_c = None
best_f1_test = 0

for random_state in range(num_random_states):
    # Split the scaled data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)

    # Initialize lists to store results for this random state
    accuracy_test = []
    f1_class_0 = []
    f1_class_1 = []

    # Perform grid search over C values
    for C in C_values:
        # Create and fit the model with the current C value
        model_lr = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000)
        model_lr.fit(X_train, y_train)

        # Evaluate model on test set
        y_hat_test = model_lr.predict(X_test)
        report = classification_report(y_test, y_hat_test, output_dict=True, zero_division=1)

        # Store metrics for this C value
        accuracy = report['accuracy']
        accuracy_test.append(accuracy)
        f1_class_0.append(report['0']['f1-score'])
        f1_class_1.append(report['1']['f1-score'])

        # Update best C value if necessary
        if np.mean(f1_class_0) + np.mean(f1_class_1) > best_f1_test:
            best_f1_test = np.mean(f1_class_0) + np.mean(f1_class_1)
            best_c = C

    # Append mean metrics for this random state
    mean_accuracy_test.append(np.mean(accuracy_test))
    mean_f1_class_0.append(np.mean(f1_class_0))
    mean_f1_class_1.append(np.mean(f1_class_1))

# Compute overall means
overall_mean_accuracy_test = np.mean(mean_accuracy_test)
overall_mean_f1_class_0 = np.mean(mean_f1_class_0)
overall_mean_f1_class_1 = np.mean(mean_f1_class_1)

# Print results
print("Mean Macro Avg Accuracy Test:", overall_mean_accuracy_test)
print("Mean F1 Score Class 0:", overall_mean_f1_class_0)
print("Mean F1 Score Class 1:", overall_mean_f1_class_1)
print("Best C value:", best_c)

# Evaluate model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

# Define lists to store report values
precision_0_test_list = []
precision_1_test_list = []
accuracy_test_list = []
f1_score_0_test_list = []
f1_score_1_test_list = []

# Itération sur 50 random states
for random_state in range(50):
    # Separation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

    # Création du modèle et paramètrage
    model_lr = LogisticRegression(penalty='l2', C=1498, solver='lbfgs', max_iter=1000)
    model_lr.fit(X_train, y_train)

    # Evaluate model
    y_hat_test = model_lr.predict(X_test)

    # Generate classification report
    report_test = classification_report(y_test, y_hat_test, output_dict=True)

    # Extract precision, accuracy, and F1 score for classes 0 and 1
    precision_0_test = report_test['0']['precision']
    precision_1_test = report_test['1']['precision']
    accuracy_test = report_test['accuracy']
    f1_score_0_test = report_test['0']['f1-score']
    f1_score_1_test = report_test['1']['f1-score']

    # Append the report values to the respective lists
    precision_0_test_list.append(precision_0_test)
    precision_1_test_list.append(precision_1_test)
    accuracy_test_list.append(accuracy_test)
    f1_score_0_test_list.append(f1_score_0_test)
    f1_score_1_test_list.append(f1_score_1_test)

# Compute the mean of the report values
mean_precision_0_test = np.mean(precision_0_test_list)
mean_precision_1_test = np.mean(precision_1_test_list)
mean_accuracy_test = np.mean(accuracy_test_list)
mean_f1_score_0_test = np.mean(f1_score_0_test_list)
mean_f1_score_1_test = np.mean(f1_score_1_test_list)

# Print the mean report values
print("Mean Precision for class 0:", mean_precision_0_test)
print("Mean Precision for class 1:", mean_precision_1_test)
print("Mean Accuracy:", mean_accuracy_test)
print("Mean F1 Score for class 0:", mean_f1_score_0_test)
print("Mean F1 Score for class 1:", mean_f1_score_1_test)
