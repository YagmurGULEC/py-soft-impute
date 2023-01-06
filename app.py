from dash import Dash, html
from dash  import  html
from dash.dependencies import Input, Output,State
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
import soft_impute
from dash import dcc
from plotly.subplots import make_subplots
from dash.exceptions import PreventUpdate

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(__name__,external_stylesheets=[dbc.themes.BOOTSTRAP,dbc.icons.BOOTSTRAP],meta_tags=[{'name': 'viewport',
                     'content': 'width=device-width, initial-scale=1.0'}])
_data_cache = {}
navbar = dbc.NavbarSimple(
    children=[
        
        html.A(
            dbc.NavItem(dbc.NavLink(html.H2(html.I(className="bi bi-linkedin me-2 m-3")))),
            href="https://www.linkedin.com/in/ya%C4%9Fmur-g%C3%BCle%C3%A7-a52111204/"
        ),
        html.A(
            dbc.NavItem(dbc.NavLink(html.H2(html.I(className="bi bi-github me-2 m-3")))),
            href="https://github.com/YagmurGULEC/MoviesWithDashPlotly"
        ),
    ],
    brand="Imputing missing data with iterative soft-thresholding",
    brand_href="#",
    color="dark",
    dark=True,
)
def serve_layout():
    return html.Div(id='main',children=[
        navbar,
        dbc.Container(
            children=[ 
                dbc.Row(
                    children=[
                            dbc.Col(
                            children=[
                            html.H5('Number of rows'),
                            dcc.Input('10',type='number',id='nrows'),
                        ]
                        ),
                dbc.Col(
                        children=[
                            html.H5('Number of columns'),
                            dcc.Input('10',type='number',id='ncols'),
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
                            dbc.Button("Generate data", id='generate',color="primary", className="me-2 my-3",n_clicks=0),
                        ]
                    ),
            ],className='p-3'),
            ],className='container text-center  bg-light'
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
                        dbc.Col(children=[html.H5('Tolerance'),dcc.Input('1e-10',id='tol')]),
                        dbc.Col(children=[html.H5('Iteration'),dcc.Input('100',id='iter')]),
                        dbc.Col(children=[html.H5('Beta'),dcc.Input('0.01',id='beta')]),
                        dbc.Col(children=[html.H5('Select the method'),dcc.RadioItems(['ISTA', 'FISTA','ADMM'], 'ADMM',id='method')]),
                        dbc.Col(children=[dbc.Button("Fill matrix", id='fill',color="primary", className="me-1",n_clicks=0),]),
                        
                    ]
                ),
                dbc.Row(
                    children=[dbc.Button("Plot residual", id='residual_button',color="primary", className="me-1 custom",n_clicks=0)],
                className="row mx-auto"
                )
                
                ],className="container justify-content-center"
            ),
        dbc.Container(id='residual'
           
        ),
        dcc.Store(id='session', storage_type='session'),
            ]
        )

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
    data=soft_impute.generate_data(nrows,ncols,percentage)
    
    _data_cache[1]=data
    fig = go.Figure(data=go.Heatmap(
                    z=data,
                    x=[f'{i+1}' for i in range(ncol)],
                    y=[f'{i+1}' for i in range(nrow)],
                    text=data,
                    texttemplate="%{text:.0f}",
                    colorscale='Greys',
                    showscale=False,),
                    layout=go.Layout(
                        title="Original",
                                
                    ),
                    )
    
    return [dcc.Graph(figure=fig)]
@app.callback(
    Output('heatmap_filled','children'),
    Input('fill','n_clicks'),
    State('tol','value'),
    State('iter','value'),
    State('method','value'),
    State('beta','value'),
)
def fill_matrix(n_clicks,tol,iter,method,beta):
    if n_clicks==0:
        return []
    else:
        data=_data_cache.get(1)
        clf=None
        if method=='ADMM':
            clf=soft_impute.ADMM(beta=float(beta),maxit=int(iter),thresh=float(tol))
        elif (method=='FISTA'):
            clf=soft_impute.FISTA(beta=float(beta),maxit=int(iter),thresh=float(tol))
        elif (method=='ISTA'):
            clf=soft_impute.Impute(beta=float(beta),maxit=int(iter),thresh=float(tol))
        clf.fit(data)
        Ximputed=clf.transform() 
        _data_cache[2]=clf.get_residual()
        fig = go.Figure(data=go.Heatmap(
                    z=Ximputed,
                    x=[f'{i+1}' for i in range(data.shape[1])],
                    y=[f'{i+1}' for i in range(data.shape[0])],
                    text=Ximputed,
                    texttemplate="%{text:.0f}",
                    colorscale='Greys',
                    showscale=False,),
                    layout=go.Layout(title="Imputed with "+method+ "  beta="+beta,)
                    )
        return [dcc.Graph(figure=fig)]

@app.callback(
    Output('residual','children'),
    Input('residual_button','n_clicks'),
    )
def generate_plots(n_clicks):
    if n_clicks==0:
        return []
    else:
        df=_data_cache.get(2)
        fig = make_subplots(rows=1, cols=2,subplot_titles=['Convergence Rate', 'Cost function'])
        
        if df is None:
            fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),row=1, col=1)
            fig.add_trace(go.Scatter(x=[20, 30, 40], y=[50, 60, 70]),row=1, col=2)
        else:
            fig.add_trace(go.Scatter(x=df.index, y=df['convergence']),row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['cost']),row=1, col=2)
        fig['layout']['xaxis']['title']='Iteration'
        fig['layout']['xaxis2']['title']='Iteration'
        fig['layout']['yaxis']['title']='Convergence Rate'
        fig['layout']['yaxis2']['title']='Cost function'
        return [dcc.Graph(figure=fig)]

        
if __name__ == '__main__':
    app.run_server(debug=True)
