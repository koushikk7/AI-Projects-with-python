import random
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from datasets import load_dataset

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 0
    set_seed(seed)

    # Load and preprocess dataset
    X, y = load_dataset("titanic")

    # Handling missing values and categorical encoding
    numeric_features = ['age', 'fare']
    categorical_features = ['sex', 'embarked', 'class']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Split data into train and test partitions with 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Define the models
    model1 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', LogisticRegression(random_state=seed))])
    model2 = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', RandomForestClassifier(random_state=seed))])

    # Evaluate model using cross-validation
    scores1 = cross_val_score(model1, X_train, y_train, cv=4, scoring='accuracy')
    scores2 = cross_val_score(model2, X_train, y_train, cv=4, scoring='accuracy')

    print("Logistic Regression CV Scores:", scores1)
    print("Random Forest CV Scores:", scores2)

    # Check which model performed better on average
    mean_score_lr = np.mean(scores1)
    mean_score_rf = np.mean(scores2)
    final_model = model1 if mean_score_lr > mean_score_rf else model2
    final_model.fit(X_train, y_train)
    predictions = final_model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print("Test Set Accuracy:", accuracy)
    print("Final selected model based on CV scores:", "Logistic Regression" if final_model.steps[-1][1] == LogisticRegression() else "Random Forest")

    # Grid search for Random Forest parameters
    param_grid = {
        'classifier__n_estimators': [10, 50, 100, 200],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(model2, param_grid, cv=4, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    print("Best parameters for Random Forest:", grid_search.best_params_)
    print("Best cross-validation score for Random Forest: {:.2f}".format(grid_search.best_score_))
