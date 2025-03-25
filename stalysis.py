import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px


def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns=['Unnamed: 0'])
    X = data['statement'].dropna()
    y = data['status'][X.index]
    return X, y

def train_model(X, y, model_type="Naive Bayes"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using Bi-grams for better context
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Bi-grams added here
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Model selection based on user choice
    if model_type == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_type == "SVM":
        model = SVC(kernel='linear')
    else:  # Default model: Naive Bayes
        model = MultinomialNB(alpha=1.0)
    
    model.fit(X_train_tfidf, y_train)
    return model, tfidf, X_test, y_test

def classify_new_statement(model, tfidf, statement):
    statement_tfidf = tfidf.transform([statement])
    prediction = model.predict(statement_tfidf)
    confidence = model.predict_proba(statement_tfidf).max()  # Max probability for confidence score
    return prediction[0], confidence

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Sentiment Analysis Web App"),
    html.Div([
        html.Label("Upload your dataset (CSV):"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload CSV'),
            multiple=False
        ),
        html.Div(id='output-data-upload'),
    ]),
    html.Div([
        html.Label("Select Model:"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Naive Bayes', 'value': 'Naive Bayes'},
                {'label': 'Logistic Regression', 'value': 'Logistic Regression'},
                {'label': 'SVM (Support Vector Machine)', 'value': 'SVM'}
            ],
            value='Naive Bayes',
        ),
    ]),
    html.Div([
        html.Label("Enter a statement for sentiment analysis:"),
        dcc.Textarea(
            id='user-input',
            style={'width': '100%', 'height': 100},
            placeholder="Enter statement here..."
        ),
        html.Button('Analyze', id='analyze-button'),
        html.Div(id='prediction-output'),
    ]),
    html.Div(id='model-performance'),
    html.Div(id='dataset-insights'),
])


@app.callback(
    Output('output-data-upload', 'children'),
    Output('prediction-output', 'children'),
    Output('model-performance', 'children'),
    Output('dataset-insights', 'children'),
    Input('upload-data', 'contents'),
    Input('user-input', 'value'),
    Input('analyze-button', 'n_clicks'),
    Input('model-dropdown', 'value'),
)
def update_output(contents, user_input, n_clicks, model_type):
    if contents is None:
        return html.Div(["Please upload a CSV file to proceed."]), None, None, None

    # Read the uploaded file content
    import base64
    import io

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    uploaded_file = io.StringIO(decoded.decode('utf-8'))

    X, y = load_data(uploaded_file)
    model, tfidf, X_test, y_test = train_model(X, y, model_type)

    # Analyze the user's input
    if n_clicks is not None and user_input is not None and user_input.strip():
        predicted_class, confidence = classify_new_statement(model, tfidf, user_input)
        prediction_output = f"Predicted Sentiment: `{predicted_class}`\nConfidence: {confidence:.2f}"
    else:
        prediction_output = "Please enter a valid statement."

    # Show model performance
    y_pred = model.predict(tfidf.transform(X_test))
    accuracy = accuracy_score(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    model_performance = html.Div([
        html.H4(f"Model Accuracy: {accuracy * 100:.2f}%"),
        html.Pre(classification_rep),
    ])

    # Dataset insights (Sample and Sentiment Distribution)
    sentiment_counts = y.value_counts()
    fig = px.bar(
        sentiment_counts,
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Count'},
        title="Sentiment Distribution"
    )

    dataset_insights = html.Div([
        html.H4("Dataset Sample:"),
        html.Table([
            html.Tr([html.Th("Statement"), html.Th("Label")])] +
            [html.Tr([html.Td(X[i]), html.Td(y[i])]) for i in range(5)]
        ),
        html.H4("Sentiment Distribution:"),
        dcc.Graph(figure=fig)
    ])

    return html.Div(f"Dataset loaded successfully!"), prediction_output, model_performance, dataset_insights


if __name__ == '__main__':
    app.run_server(debug=True)

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import accuracy_score, classification_report
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.express as px

# # Load and Train model
# def load_data(uploaded_file):
#     data = pd.read_csv(uploaded_file)
#     data = data.drop(columns=['Unnamed: 0'])
#     X = data['statement'].dropna()
#     y = data['status'][X.index]
#     return X, y

# def train_model(X, y):
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Using Bi-grams for better context
#     tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))  # Bi-grams added here
#     X_train_tfidf = tfidf.fit_transform(X_train)
#     X_test_tfidf = tfidf.transform(X_test)
    
#     # Model: Naive Bayes with smoothing and class weights
#     model = MultinomialNB(alpha=1.0)  # You can adjust alpha for better smoothing
#     model.fit(X_train_tfidf, y_train)
#     return model, tfidf, X_test, y_test

# def classify_new_statement(model, tfidf, statement):
#     statement_tfidf = tfidf.transform([statement])
#     prediction = model.predict(statement_tfidf)
#     confidence = model.predict_proba(statement_tfidf).max()  # Max probability for confidence score
#     return prediction[0], confidence

# # Initialize the Dash app
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Sentiment Analysis Web App"),
#     html.Div([
#         html.Label("Upload your dataset (CSV):"),
#         dcc.Upload(
#             id='upload-data',
#             children=html.Button('Upload CSV'),
#             multiple=False
#         ),
#         html.Div(id='output-data-upload'),
#     ]),
#     html.Div([
#         html.Label("Enter a statement for sentiment analysis:"),
#         dcc.Textarea(
#             id='user-input',
#             style={'width': '100%', 'height': 100},
#             placeholder="Enter statement here..."
#         ),
#         html.Button('Analyze', id='analyze-button'),
#         html.Div(id='prediction-output'),
#     ]),
#     html.Div(id='model-performance'),
#     html.Div(id='dataset-insights'),
# ])


# @app.callback(
#     Output('output-data-upload', 'children'),
#     Output('prediction-output', 'children'),
#     Output('model-performance', 'children'),
#     Output('dataset-insights', 'children'),
#     Input('upload-data', 'contents'),
#     Input('user-input', 'value'),
#     Input('analyze-button', 'n_clicks'),
# )
# def update_output(contents, user_input, n_clicks):
#     if contents is None:
#         return html.Div(["Please upload a CSV file to proceed."]), None, None, None

#     # Read the uploaded file content
#     import base64
#     import io

#     content_type, content_string = contents.split(',')
#     decoded = base64.b64decode(content_string)
#     uploaded_file = io.StringIO(decoded.decode('utf-8'))

#     X, y = load_data(uploaded_file)
#     model, tfidf, X_test, y_test = train_model(X, y)

#     # Analyze the user's input
#     if n_clicks is not None and user_input is not None and user_input.strip():
#         predicted_class, confidence = classify_new_statement(model, tfidf, user_input)
#         prediction_output = f"Predicted Sentiment: `{predicted_class}`\nConfidence: {confidence:.2f}"
#     else:
#         prediction_output = "Please enter a valid statement."

#     # Show model performance
#     y_pred = model.predict(tfidf.transform(X_test))
#     accuracy = accuracy_score(y_test, y_pred)
#     classification_rep = classification_report(y_test, y_pred)

#     model_performance = html.Div([
#         html.H4(f"Model Accuracy: {accuracy * 100:.2f}%"),
#         html.Pre(classification_rep),
#     ])

#     # Dataset insights (Sample and Sentiment Distribution)
#     sentiment_counts = y.value_counts()
#     fig = px.bar(
#         sentiment_counts,
#         x=sentiment_counts.index,
#         y=sentiment_counts.values,
#         labels={'x': 'Sentiment', 'y': 'Count'},
#         title="Sentiment Distribution"
#     )

#     dataset_insights = html.Div([
#         html.H4("Dataset Sample:"),
#         html.Table([
#             html.Tr([html.Th("Statement"), html.Th("Label")])] +
#             [html.Tr([html.Td(X[i]), html.Td(y[i])]) for i in range(5)]
#         ),
#         html.H4("Sentiment Distribution:"),
#         dcc.Graph(figure=fig)
#     ])

#     return html.Div(f"Dataset loaded successfully!"), prediction_output, model_performance, dataset_insights


# if __name__ == '__main__':
#     app.run_server(debug=True)
