from dash import Dash, html, dcc, Input, Output
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__, external_stylesheets=external_stylesheets)
_data_cache = {}

def serve_layout():
    return html.Div(id='main',children=[
        dbc.Container(
            children=[
                dbc.Row(
                    children=[
                            dbc.Col(
                            children=[
                            html.H5('Number of rows'),
                            dcc.Input('20',type='number',id='nrows'),
                        ]
                        ),
                dbc.Col(
                        children=[
                            html.H5('Number of columns'),
                            dcc.Input('2',type='number',id='ncols'),
                        ]
                    ),
                dbc.Col(
                        children=
                        [
                            html.H5('Percentage of nans'),
                            dcc.Input('5',type='number',id='percentage'),
                        ]
                    ),
                dbc.Col(
                        children=
                        [   
                            html.H5(''),
                            dbc.Button("Generate data", id='generate',color="primary", className="me-1",n_clicks=0),
                        ]
                    ),
            ]
        ),
        dbc.Container(
            children=[

            ]
        ),
        dbc.Container(children=[
            
                dbc.Row(
                    children=[
                        dbc.Col(id='heatmap'),
                        dbc.Col(id='heatmap_filled'),
                    ]
                    
                ),
                dbc.Row(
                    children=[
                        dbc.Col(children=[html.H5('Tolerance'),dcc.Input('1e-5',id='tol')]),
                        dbc.Col(children=[html.H5('Iteration'),dcc.Input('100',id='iter')]),
                        dbc.Col(children=[html.H5('Select the method'),dcc.RadioItems(['ISTA', 'FISTA','ADMM'], 'ADMM',id='method')]),
                        dbc.Col(children=[dbc.Button("Fill matrix", id='fill',color="primary", className="me-1",n_clicks=0),]),
                        
                    ]
                ),
                
                ]
            ),
           
            ],className='container pt-3 text-center'
        )
    ])
app.layout=serve_layout
@app.callback(
    Output('heatmap','children'),
    Input('generate','n_clicks'),
    State('nrows','value'),
    State('ncols','value'),
    State('percentage','value'),
)
def generate_data(n_clicks,nrows,ncols,percentage):
    ncol=int(ncols)
    nrow=int(nrows)
    data=np.random.randn(nrow,ncol)
    
    number_nans=int(ncol*nrow*int(percentage)/100)
    
    np.put(data,np.random.choice(data.size,number_nans,replace=False),np.nan)
    _data_cache[1]=data
    fig = go.Figure(data=go.Heatmap(
                    z=data,
                    x=[f'{i+1}' for i in range(ncol)],
                    y=[f'{i+1}' for i in range(nrow)],
                    text=data,
                    texttemplate="%{text:.2f}",
                    showscale=False,
                   
                   ))
    
    return [dcc.Graph(figure=fig)]
@app.callback(
    Output('heatmap_filled','children'),
    Input('fill','n_clicks'),
    State('tol','value'),
    State('iter','value'),
    State('method','value'),
)
def fill_matrix(n_clicks,tol,iter,method):
    data=_data_cache.get(1)
    fig = go.Figure(data=go.Heatmap(
                    z=data,
                    text=data,
                    texttemplate="%{text:.2f}",
                    showscale=False,
                   
                   ))
    
    return [dcc.Graph(figure=fig)]

if __name__ == '__main__':
    app.run_server(debug=True)
