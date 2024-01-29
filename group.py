import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import streamlit as st

# Load the dataset from a CSV file
df = pd.read_csv('career_dataset (2).csv')

# Check and clean column names
df.columns = df.columns.str.strip()

# Split features (X) and target variable (y)
X = df.drop('Group', axis=1)
y = df['Group']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf = RandomForestClassifier()

# Define the hyperparameters grid
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],   # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],   # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]      # Minimum number of samples required at each leaf node
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Perform GridSearchCV to find the best hyperparameters
grid_search.fit(X_train, y_train)

# Save the best model
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

# Define the Streamlit app
def main():
    st.title('Career Interest Group Prediction')

    # Create a sidebar for selecting features
    feature_selections = {}
    for feature in X.columns:
        if feature != 'Personal Statement':
            if 'Personal Statement' not in feature:
                if X[feature].dtype == 'object':
                    feature_selections[feature] = st.sidebar.selectbox(f"Select {feature}", ['Low', 'Medium', 'High'])
                else:
                    feature_selections[feature] = st.sidebar.slider(f"Select {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

    # Collect personal statement separately
    personal_statement = st.sidebar.text_area('Personal Statement', 'This is a sample personal statement.')

    # Append personal statement to feature selections
    feature_selections['Personal Statement'] = personal_statement

    # Convert the selected features into a DataFrame
    selected_features_df = pd.DataFrame([feature_selections])

    # Preprocess the selected features (one-hot encoding)
    selected_encoded_df = pd.get_dummies(selected_features_df, drop_first=True)

    # Ensure that the selected features have the same columns as the training data
    missing_cols = set(X.columns) - set(selected_encoded_df.columns)
    for col in missing_cols:
        selected_encoded_df[col] = 0

    # Reorder the columns to match the order during training
    selected_encoded_df = selected_encoded_df[X.columns]

    # Make predictions using the trained model
    prediction = best_model.predict(selected_encoded_df)

    # Display the predicted career interest group
    st.subheader('Predicted Career Interest Group:')
    st.write(prediction[0])

    # Display images based on prediction
    if prediction[0] == 'Technology':
        st.image('/path/to/technology_image.jpg', caption='Technology')
    elif prediction[0] == 'Business':
        st.image('/path/to/business_image.jpg', caption='Business')
    elif prediction[0] == 'Others':
        st.image('/path/to/others_image.jpg', caption='Others')

if __name__ == "__main__":
    main()
