from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open('fake_news_model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news_text']
        
        # 1. Vectorize the input text
        vectorized_input = vectorizer.transform([news])
        
        # 2. Make the prediction (Assuming 0 = Real, 1 = Fake based on your Colab screenshot)
        prediction = model.predict(vectorized_input)
        
        if prediction[0] == 0:
            result = "The news is Real ✅"
        else:
            result = "The news is Fake ❌"
            
        return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)