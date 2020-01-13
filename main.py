'''
About the data:


-coeficiente de balanceamento para semanas com feriados



'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR


'''
Constants:
'''
dataframe = pd.read_csv('Alasca_nordeste.csv', sep=';')
macroeconomia = pd.read_csv('macroeconomia.csv', sep=';')


'''
---------------------------------------------
----------- Data processing functios --------
---------------------------------------------
'''

def format_sale_string(value):
    '''
    Convert sales number string to floating point
    '''
    value = value.replace('.', '')
    value = value.replace(',', '.')
    return float(value)


def format_sales_column(data):
    '''
    Expects a dataframe column
    '''
    sold = np.array(data)
    formated_sold = []

    for value in sold:
        formated_sold.append(format_sale_string(value))

    return np.array(formated_sold)



def group_sales_by_year(year):
    '''
    - year should be an integer
    - sales_type = sm, total, indiretos or route
    - Select specific rows from dataframe: https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values
    '''

    df = dataframe.loc[(dataframe['Year'] == year)]
    return df



def group_year_sales_by_month(year, sales_type):
    year = int(year)
    '''
        - year should be given as integer
        - Returns a numpy array with the sales of a given type of a given yeay grouped
        by month
    '''

    df = group_sales_by_year(year)
    max_month = int(np.array(df['Month_gregoriano'])[np.array(df['Month_gregoriano']).size - 1])

    m_sales = []
    #TODO: diminuir a complexidade desse algoritmo
    for i in range(1, max_month + 1):
        df_m = df.loc[(dataframe['Month_gregoriano'] == i)]
        sales = format_sales_column(df_m[sales_type])
        m_sale = 0
        for j in range(0, sales.size-1):
            m_sale = m_sale + sales[j]
        m_sales.append(m_sale)

    return m_sales



def plot_sales(df = dataframe, year=None, sm = True, total = True, route = True, indiretos = True):
    '''
    -Choose wich channels should be ploted. Default plots all channels
    -Choose year (should be an integer). Default show all years
    '''
    if year != None:
        df = group_sales_by_year(year)
    if total:
        total_sales = format_sales_column(df['Total'])
    if sm:
        sm_sales = format_sales_column(df['SM'])
    if route:
        route_sales = format_sales_column(df['ROUTE'])
    if indiretos:
        indiretos_sales = format_sales_column(df['INDIRETOS'])

    plt.plot(total_sales, 'r')
    plt.plot(sm_sales , 'b' )
    plt.plot(indiretos_sales , 'g')
    plt.plot(route_sales, 'y')

    plt.show()



'''
---------------------------------------------
----------- Data analysis functions ----------
---------------------------------------------
'''

def train_rbf_estimator(traning_year):
    '''
    Treina o modelo no ano de 2017 e tenta prever as vendas de 2018 da forma mais simples possivel
    '''

    df = group_sales_by_year(traning_year)

    weeks = np.linspace(0, np.array(df["Total"]).size,np.array(df["Total"]).size)
    weeks = np.reshape(weeks,(len(weeks), 1))
    sales =  format_sales_column(df['Total'])

    #TODO how to adjust the radial basis function parameters ?
    svr_rbf = SVR(kernel= 'rbf', C= 1e11, gamma= 0.01)
    svr_rbf.fit(weeks, sales)

    plt.plot(weeks, sales, color= 'black', label= 'Data')
    plt.plot(weeks, svr_rbf.predict(weeks), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel

    plt.show()

train_rbf_estimator(2015)

def plot_sales_and_macro(year):
    '''
        - Year should be given as a String
    '''
    df = macroeconomia[macroeconomia['Data'].str.contains(year)]

    #TODO: encapsular as funcoes e tirar as variaveis que eu coloquei direto na mao

    fig, ax1 = plt.subplots()
    months  = np.linspace(1, 12, 12) #one year divded

    #gotta format the data
    dataset = []
    for i in np.array(df['PMC']):
        dataset.append(float(i.replace(',', '.')))


    ax1.set_xlabel('Months')
    ax1.set_ylabel('PMC', color='red')
    ax1.plot(months, dataset, color='red')

    ax1.tick_params(axis='y', labelcolor='red')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    #get sales for a given year
    sales = group_year_sales_by_month(year, "Total")

    ax2.set_ylabel('sales', color='blue')  # we already handled the x-label with ax1
    ax2.plot(months, sales, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

'''
Tests
'''


#train_rbf_estimator(2017)

#df = group_sales_by_year(2017)
#sales =  format_sales_column(df['Total'])
#plt.plot(sales)
#plt.show()

