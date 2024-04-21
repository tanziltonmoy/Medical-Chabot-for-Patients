import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import torch
import os
import pickle
from config import *
from utils import *
from models import *
from train import *
from inference import *

# Load the vocabulary and model
voc_filename = os.path.join('model', 'voc.pkl')
with open(voc_filename, 'rb') as voc_file:
    voc = pickle.load(voc_file)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    return checkpoint

checkpoint_path = os.path.join('model', 'best_model_val_per.tar')
checkpoint = load_checkpoint(checkpoint_path)

hidden_size = 512
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
attn_model = 'dot'

embedding = torch.nn.Embedding(voc.num_words, hidden_size)
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

encoder.load_state_dict(checkpoint['en'])
decoder.load_state_dict(checkpoint['de'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
decoder = decoder.to(device)
encoder.eval()
decoder.eval()

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Home", href="#")),
            dbc.NavItem(dbc.NavLink("About", href="#"))
        ],
        brand="Medical Chatbot for Patients",
        className="text-center my-4",
        brand_href="#",
        color="primary",
        dark=True,
    ),
html.Div([
    html.H1("ChatBot Response Generator", className="text-center my-4"),
    dbc.Row([
        dbc.Col(dcc.Textarea(id='input-text', style={'height': '100px'}, placeholder="Type your medical query here..."), width=50),
    ]),
    dbc.Row([
        dbc.Col(dbc.Button("Submit", id='submit-button', color="success", className="my-2 mx-auto", n_clicks=0), width={"size": 6, "offset": 3}),
    ]),
    dbc.Row([
        dbc.Col(html.Div(id='response-text', className="my-2 p-3 bg-light border text-center mx-auto"), width=12)
    ]),
    dbc.Row([
        dbc.Col(html.Footer("© 2024 Medical Chatbot Project, All Rights Reserved", className="text-center my-4 mx-auto"), width=12)
    ]),
], className="mx-auto", style={"width": "80%", "max-width": "800px"})
], fluid=True)

@app.callback(
    Output('response-text', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('input-text', 'value')]
)
def update_output(n_clicks, input_text):
    if n_clicks > 0 and input_text is not None:
        try:
            response = evaluate(encoder, decoder, input_text, voc, PAD_token, SOS_token, UNK_token, EOS_token, device, max_length=50)
            return dbc.Alert(response, color="info", className="p-2")
        except Exception as e:
            return dbc.Alert(f"An error occurred: {str(e)}", color="danger", className="p-2")
    return "Enter a question and press submit to get a response."

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)
