from flask import Flask, jsonify, request
from flask_cors import CORS

from SentimentRNN import SentimentRNN
from model_function import tokenize_input, preprocess_string, padding_
from torch.utils.data import TensorDataset, DataLoader
import torch

app = Flask(__name__)
CORS(app)
device = torch.device('cpu')

model = SentimentRNN(2, 1001, 256, 64, 1, drop_prob=0.5)
model.load_state_dict(torch.load('sentiment_model.pth',map_location=device))
model.eval()

@app.route('/')
def index():
    return 'succes'

def predict(sentiment_model, data, hidden, device):
    for i in data:
        h = tuple([each.data for each in hidden])
        output, h = sentiment_model(i[0].to(device),h)
        return torch.round(output.squeeze()).detach().numpy()

@app.route('/api/check-sentiment-data', methods=['POST'])
def check_sentiment_data():
    list_input = [request.json['data']]
    data_padding = padding_(tokenize_input(list_input), 500)
    data_tensor = TensorDataset(torch.from_numpy(data_padding))
    batch_size = 50
    data_loader= DataLoader(data_tensor, batch_size=batch_size)
    
    
    result = predict(model, data_loader, model.init_hidden(len(list_input)), device)
    return jsonify({'predict': str(result)}), 200
    
if __name__ == '__main__':
    app.run(debug=True)