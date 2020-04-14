import pandas as pd #import pandas for Python data structure
import numpy as np # Import Numpy for Numerical manipulation
import seaborn as sns # Seaborn Module for Plotting
import matplotlib.pyplot as plt # 
import folium # Module to plot maps
import branca
import googlemaps
import plotly # Interactive Visualization
import plotly.graph_objects as go # import graph object from Plotly module
import plotly.express as px
import plotly.io as pio
# offline mode
from plotly.offline import init_notebook_mode, iplot # Import iplot and Offline from Plotly
import warnings
warnings.filterwarnings("ignore")
init_notebook_mode(connected = True) # Set Offline plotting to True
pio.templates.default = "simple_white"

### Set the Visualization parameters
# Initializing Visualization set
sns.set(style = "whitegrid
        palette = "pastel",
        color_codes = True)
sns.mpl.rc("figure", figsize = (15,10))

### 1. Import product_a.csv dataset into python pandas data frame df_product_a
df_product_a = pd.read_csv("C:/Users/k64068013/Downloads/product_a.csv",
                           names = ["Index","Date","Price","Total_Vol","Plu1",
                                    "Plu2","Plu3","Bags_t","Bags_s","Bags_l",
                                    "Bags_lx","Type","Year","Location"],
                           skiprows = 1, parse_dates=True)
 
 ##### Glimpse of data
 df_product_a.head()
 df_product_a.shape
 
 ### 2. Convert Date field to a suitable datetime data type
 ##### Let's begin with checking the current data type of Date field
 df_product_a['Date'].dtypes
 
 #### Change to Appropriate Date Time
 df_product_a['Date'] = pd.to_datetime(df_product_a['Date'], format= '%Y-%m-%d')
 
 ### 3. Values of the year column do not match with the values of the date_w column. Correct the values of the year column.
 df_product_a.loc[:,("Date","Year")].head()
 ##### Extract the year from Date column and pass it to the year column
 df_product_a.Year = df_product_a.Date.dt.year 
 
 ### 4. Create df_stats with the following details from df_product_a 
 #### Columns: filed_name, minimum, maximum, mean, standard deviation, 
 #### variance, mode, median, 10th, 20th .. 90th percentiles, 1st, 2nd and 3rd quartiles, 
 #### interquartile distance, skewness and kurtosis.
 
 df_stats = pd.DataFrame(df_product_a.describe().T)
 df_stats["Variance"] = df_product_a.var()
 df_stats["Median"] = df_product_a.median()
 df_stats["Skewness"] = df_product_a.skew()
 df_stats["Kurtosis"] = df_product_a.kurtosis() 
 df_quant = pd.DataFrame(df_product_a.quantile([0.1,0.2,0.9]).T)
 df_quant.columns = ['10th', "20th", "90th"]
 df_stats = pd.concat([df_stats,df_quant], axis = 1)
 df_st = df_stats.reset_index()
 df_st = df_st.rename(columns = {"index" : "Field_name"})
 df_st
 
 ### 8. Create a Pearson correlation matrix (it is a square matrix) between all the possible fields. 
 ### What are the conclusions you make?
 corrMatrix = df_product_a.corr(method="pearson")
 print("The Pearson Correlation Matrix for Df_Product_a is: \n",corrMatrix)
 sns.heatmap(corrMatrix, annot = True)
 
 ### 9. Create a Spearman’s Rank correlation matrix (it is a square matrix) 
 ### between all the possible fields.
 spCorrMatrix = df_product_a.corr(method="spearman")
 print("The Spearman Correlation for the given dataset is : \n", spCorrMatrix)
 sns.heatmap(spCorrMatrix,annot = True)
 
 ### 10. Create a seaborne pairplot for df_product_a. What are the conclusions you can make using the analysis sofar
 sns.mpl.rc("figure", figsize = (20,20))
 sns.pairplot(df_product_a, diag_kind="kde", corner=True, 
              kind="reg", palette="husl")
 
 ### 11. Using Plotly, draw weekly and monthly time-series graphs of the numeric fields
 df_copy = df_product_a.copy(deep=True)
 df_month = df_copy.groupby(['Date']).mean().reset_index()
 
 #### Define functions to create traces and layout
 # Create a function to create traces
 def traces(x, y, mode, name):
     trace = go.Scatter(
         x=x,
         y=y,
         mode=mode,
         name=name)
     return trace

 # Define a function to customize the layout


 def layouts(title1, title2, title3):
     layout = go.Layout(
         title=title1,
         xaxis=dict(
             title=title2,
             showgrid=False),
         yaxis=dict(
             title=title3),
         hovermode="closest")
     # return Layout
     return layout
     
#### Weekly and Monthly time series data for Price Variable
# Create a trace for Price
trace1 = traces(x=df_month.Date, y=df_month.Price,
                mode="lines+markers", name="Price")
# add trace to data
data = [trace1]
# Customize the layout
layout = layouts(title1="Weekly and Monthly Time series data for Price", 
                 title2="Date", 
                 title3="Price (in $US)")
# Create the figure
figure = dict(data=data, layout=layout)
# Plot the figure
iplot(figure)

#### Weekly and Monthly Time series data for Total Volume
# Create a trace for Total Volume
trace1 = traces(x=df_month.Date, y=df_month.Total_Vol,
                mode="lines+markers", name="Total Volume")
# add trace to data
data = [trace1]
# Customize the layout
layout = layouts(title1="Weekly and Monthly Time series data for Total Volume",
                 title2="Date",
                 title3="Total Volume")

figure = dict(data=data, layout=layout)
iplot(figure)

#### Weekly and Monthly Time Series for PLU's
# Create a trace for Plu1
trace1 = traces(x=df_month.Date, y=df_month.Plu1,
                mode="lines+markers", name="Plu1")
# Create a trace for Plu2
trace2 = traces(x=df_month.Date, y=df_month.Plu2,
                mode="lines", name="Plu2")
# Create a trace for Plu3
trace3 = traces(x=df_month.Date, y=df_month.Plu3,
                mode="lines", name="Plu3")
# add trace to data
data = [trace1, trace2, trace3]
# Customize the layout
layout = layouts(title1="Weekly and Monthly Time series data for Plu Variable",
                 title2="Date",
                 title3="PLU Variables")
# Create the plots
figure = dict(data=data, layout=layout)
# Display the plots
iplot(figure)

#### Weekly and monthly time series for Plu3 Variable
# Create a trace for Plu3
trace3 = traces(x=df_month.Date, y=df_month.Plu3,
                mode="lines+markers", name="Plu3")

# Add the trace to Plotly data
data = [trace3]

# Customize the layout
layout = layouts(title1="Weekly and Monthly Time series data for Plu3 Variable",
                 title2="Date",
                 title3="Plu3")

# Create the plot
fig = dict(data=data, layout=layout)
# Display the plot
iplot(fig)

#### Weekly and Monthly Time Series for Bags
# Create a trace for Bags_l
trace1 = traces(x=df_month.Date, y=df_month.Bags_l,
                mode="lines", name='Bags_l')
# Create a trace for Bags_lx
trace2 = traces(x=df_month.Date, y=df_month.Bags_lx,
                mode="lines", name="Bags_lx")
# Create a trace for Bags_s
trace3 = traces(x=df_month.Date, y=df_month.Bags_s,
                mode="lines", name="Bags_s")
# Create a trace for Bags_t
trace4 = traces(x=df_month.Date, y=df_month.Bags_t,
                mode="lines", name="Bags_t")
# Add traces to data
data = [trace1, trace2, trace3, trace4]

# Customize the layout
layout = layouts(title1="Weekly and Monthly Time series for Bags",
                 title2="Date",
                 title3="Bags Variable")
# Create the figure
figure = dict(data=data, layout=layout)
# Display the plots
iplot(figure)

### 12. Draw year based location and type bar charts using Plotly.
#### Location by Year
df_1 = df_product_a.groupby(['Location','Year']).mean().reset_index()
df_2 = df_1[df_1.Location != 'TotalUS']
px.bar(df_2, x = "Year", y = "Plu1", color = "Location", barmode = 'group')

#### Product Type by Year
df_3 = df_product_a.groupby(['Type','Year']).mean().reset_index()
px.bar(df_3, x = "Year", y = "Price", color = "Type", barmode="group")

### 13. Compare and contrast the prices of each type,
### each location and {location and type} combination. 
### Visualise the results using suitable plots.

df_count = df_product_a.groupby(['Type','Location']).mean().reset_index()
df_count

#### Price by Type
fig = go.Figure(data = [go.Bar(x = df_count.Type, y = df_count.Price)])
fig.update_layout(title_text = 'Price by Type')
fig.show()

#### Price by Location
df_count = df_product_a.groupby('Location').mean().reset_index()
fig = px.line(df_count, x = "Location", y = "Price")
fig.show()

#### Average price by Location and Type
df_count = df_product_a.groupby(['Location','Type']).mean().reset_index()
fig = px.line(df_count, x = "Location", y = "Price", color = "Type")
fig.show()

### 14. Visualise data on a folium map. 
### The locations should have markers with a colour range based on the mean values of bags_t. 
### Tooltips should show the total values of bags_t. and total values of bag_t for each type. 
### When markers are clicked, the average values of all numeric fields should be shown.

df_loc = pd.DataFrame(df_product_a.Location.unique(), columns = ["City"])
Key = "Your Gmaps Key" 
gmaps = googlemaps.Client(key=Key)

##### Define a Function to get the co-ordinated for address/City in our Dataset
def get_coordinates(address):
    geocode_result = gmaps.geocode(str(address))
    if len(geocode_result) > 0:
        return list(geocode_result[0]['geometry']['location'].values())
    else:
        return [np.NaN, np.NaN]
 
coordinates = df_loc['City'].apply(lambda x:
                                        pd.Series(get_coordinates(x), index = ['LATITUDE', 'LONGITUDE']))
df_loc = pd.concat([df_loc[:], coordinates[:]], axis="columns")
df_loc = df_loc.dropna()

df_loc = df_loc.rename(columns={'City':'Location'})
df_loc.head()

##### Merge the co-ordinates with the original dataset to be used in the Folium
df_product_a = df_product_a.merge(df_loc, on="Location")
df_product_a.head()

# Group the data by Location as we need the data by Location
df_group = df_product_a.groupby('Location').mean().reset_index()
# separate the data by Type as we need Bags_t information for Each type
df_ta = df_product_a[df_product_a["Type"] == "A"]  # for Type A
df_tc = df_product_a[df_product_a["Type"] == "C"]  # For Type C
df_t = pd.DataFrame(df_product_a.groupby('Location')[
                    'Bags_t'].sum()).reset_index()  # Total of Bags_t
df_ta = pd.DataFrame(df_ta.groupby('Location')["Bags_t"].sum()).reset_index()
df_ta = df_ta.rename(columns={"Bags_t": "Type_A"})
df_tc = pd.DataFrame(df_tc.groupby('Location')["Bags_t"].sum()).reset_index()
df_tc = df_tc.rename(columns={"Bags_t": "Type_C"})
result = df_ta.merge(df_tc, on="Location")
df_map = df_group.merge(result, on="Location")
df_t = df_t.rename(columns={"Bags_t": "Bags_tt"})
df_map = df_map.merge(df_t, on='Location')
pd.options.display.float_format = '{:.4f}'.format
df_map.describe()
mean_bt = df_map['Bags_t'].mean()


def html_row(row):
    i = row
    
    Price = round(df_map['Price'].iloc[i],2)
    Total_vol = round(df_map['Total_Vol'].iloc[i],2)
    Plu1 = round(df_map['Plu1'].iloc[i],2)
    Plu2 = round(df_map['Plu2'].iloc[i],2)
    Plu3 = round(df_map['Plu3'].iloc[i],2)
    Bags_t = round(df_map['Bags_t'].iloc[i],2)
    Bags_s = round(df_map['Bags_s'].iloc[i],2)
    Bags_l = round(df_map['Bags_l'].iloc[i],2)
    Bags_lx = round(df_map['Bags_lx'].iloc[i],2)
    location = df_map['Location'].iloc[i]
    
    left_col_colour = "#2A799C"
    right_col_colour = "#C5DCE7"
    
    html = """<!DOCTYPE html>
<html>
<head>
<h4 style="margin-bottom:0"; width="300px">{}</h4>""".format(location) + """

</head>
    <table style="height: 126px; width: 300px;">
<tbody>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Price</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Price) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Total_vol</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Total_vol) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Plu1</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Plu1) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Plu2</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Plu2) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Plu3</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Plu3) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Bags_t</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Bags_t) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Bags_s</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Bags_s) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Bags_l</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Bags_l) + """
</tr>
<tr>
<td style="background-color: """+ left_col_colour +""";"><span style="color: #ffffff;">Bags_lx</span></td>
<td style="width: 200px;background-color: """+ right_col_colour +""";">{}</td>""".format(Bags_lx) + """
</tr>
</tbody>
</table>
</html>
"""
    return html
    
# basic map
map1 = folium.Map(
    location=[42.652579, -73.756232],
    tiles="cartodbpositron",
    zoom_start=3)

for i in range(0, len(df_map)):
    html = html_row(i)
    iframe = branca.element.IFrame(html=html, width=400, height=300)
    popup = folium.Popup(iframe, parse_html=True)
    bags_t = int(df_map['Bags_t'].iloc[i])

    # Function to change the marker color
    # according to the mean value of Bags_t
    if bags_t in range(1000000, 5000000):
        col = "blue"
    elif bags_t in range(10000, 100000):
        col = "green"
    elif bags_t in range(100000, 1000000):
        col = "red"
    else:
        col = "purple"

    tooltip = "Total Value for:  {} <br> Bags_T: {}<br> Type A: {}<br> Type C: {}<br> Click for more".format(
        df_map['Location'][i], round(df_map["Bags_tt"][i], 2), 
        round(df_map["Type_A"][i], 2), round(df_map["Type_C"][i], 2))
    folium.Marker([df_map['LATITUDE'].iloc[i], df_map['LONGITUDE'].iloc[i]],
                  popup=popup,
                  tooltip=tooltip,
                  icon=folium.Icon(color=col, icon='info-sign')).add_to(map1)

 

   map1.save("Product_data.html")
