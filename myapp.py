import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

st.markdown("# Project 3")

st.write("""

Crop Yield Prediction. Create a machine learning model to forecast crop yields based on factors such as weather patterns, soil characteristics, and agricultural practices.

""")

st.markdown("## Analysis of Crop Yield Data")
st.write("Compare the total yield of **wheat**, **barley**, and **maize** for New Zealand as a whole across different years.")

st.set_option('deprecation.showPyplotGlobalUse', False)

crop_df = pd.read_csv(r"C:\Users\dress\assignment three group 297201\crop_df.csv")

plt.figure(figsize=(10, 6))

crop_type = crop_df[['Year', 'Wheat Tonnes', 'Barley Tonnes', 'Maize grain Tonnes']]

# Reshape the data into long format
crop_type_long = pd.melt(crop_type, id_vars=['Year'], var_name='Crop', value_name='Yield')

# Plot
sns.barplot(x="Year", y="Yield", hue="Crop", data=crop_type_long)
plt.title('Total crop yield for 5 years in New Zealand')
plt.xlabel('Year')
plt.ylabel('Crop yield total')
plt.xticks(rotation=45)

# Show plot using st.pyplot()
st.pyplot()

st.write("""According to the bar plot, the wheat yield exceeds the yield of barley and maize over the five-year period. The highest wheat yield was observed in 2020. However, after 2020, the wheat yield gradually declined. In contrast, the maize grain yield remained relatively stable throughout the considered period. The barley yield showed a tendency to fluctuate, increasing and decreasing from year to year.

Visualize how the total crop yield changes over 5 years.""")

total_NZ_data = crop_df[crop_df['Region'] == 'Total New Zealand']



# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 6))

# Plot data for Wheat Tonnes
axs[0].plot(total_NZ_data['Year'], total_NZ_data['Wheat Tonnes'])
axs[0].set_ylabel('Wheat Tonnes')
axs[0].tick_params(axis='x', labelsize=8)

# Plot data for Barley Tonnes
axs[1].plot(total_NZ_data['Year'], total_NZ_data['Barley Tonnes'])
axs[1].set_ylabel('Barley Tonnes')
axs[1].tick_params(axis='x', labelsize=8)

# Plot data for Maize grain Tonnes
axs[2].plot(total_NZ_data['Year'], total_NZ_data['Maize grain Tonnes'])
axs[2].set_ylabel('Maize grain Tonnes')
axs[2].tick_params(axis='x', labelsize=8)

# Adjust layout and display plot
plt.tight_layout()
st.pyplot(fig)

st.write("""According to the plot of total crop yield in New Zealand, which combines data from both the North and South Islands, the wheat yield significantly decreases after 2020. The barley yield was declining until 2021, after which it started to show a slow increase. The maize yield peaked in 2021, and its trend was not consistently smooth throughout the considered period.

Generate a plot to show how the crop yield changes over 5 years for the South Island and North Island of New Zealand.""")

# Load your data
crop_yield_total = pd.read_csv(r"C:\Users\dress\assignment three group 297201\crop_yield_total.csv")



# Set figure size
plt.figure(figsize=(8, 12))

# Plot for Wheat
plt.subplot(3, 1, 1)
plot_yield_wheat = sns.barplot(x="Year", y="Wheat Tonnes", hue="Region", data=crop_yield_total)
plt.title('Total wheat yield for 5 years on South and North Island of New Zealand')
plt.ylabel("Wheat yield total")
plt.xticks(rotation=45)

# Plot for Barley
plt.subplot(3, 1, 2)
plot_yield_barley = sns.barplot(x="Year", y="Barley Tonnes", hue="Region", data=crop_yield_total)
plt.title('Total barley yield for 5 years on South and North Island of New Zealand')
plt.ylabel("Barley yield total")
plt.xticks(rotation=45)

# Plot for Maize grain
plt.subplot(3, 1, 3)
plot_yield_maize = sns.barplot(x="Year", y="Maize grain Tonnes", hue="Region", data=crop_yield_total)
plt.title('Total maize grain yield for 5 years on South and North Island of New Zealand')
plt.ylabel("Maize grain yield total")
plt.xticks(rotation=45)

# Show plots
st.pyplot()

st.write("""According to the plots of total crop yield for the past 5 years, it can be concluded that the South Island mainly excels in barley and wheat yield crops, while the North Island focuses on maize crops. Interestingly, the highest wheat yield was recorded on the South Island of New Zealand in 2020, during the Covid pandemic. The highest barley yield was achieved in 2019 on the South Island, and the North Island showed the highest maize yield in 2021. In general, there are no significant rises or drops in crop yield during the considered 5 years.

Visualize which regions are focusing on the cultivation of specific crops.
""")

# Load your data
crop_yield_melted = pd.read_csv(r"C:\Users\dress\assignment three group 297201\crop_yield_melted.csv")

# Filter data for each year and create bar plots
for year in range(2019, 2024):
    st.subheader(f"Crop Yield by Region in {year}")
    crop_yield_year = crop_yield_melted[crop_yield_melted['Year'] == year]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=crop_yield_year, x='Region', y='Tonnes', hue='Crop', palette='muted')
    plt.title(f"Crop Yield by Region in {year}", fontweight='bold')
    plt.xlabel('Region', fontweight='bold')
    plt.ylabel('Tonnes', fontweight='bold')
    plt.xticks(rotation=90)
    st.pyplot()

st.write('According to the five plots, the region of Canterbury consistently demonstrates the highest yield of wheat and barley. Meanwhile, Hawkes Bay consistently exhibits high maize yields. The Auckland region shows no significant maize yield throughout all years, and in 2021, it also demonstrates no significant barley yield. Only Hawke\'s Bay, located on the North Island of New Zealand, regularly shows notable results for wheat, barley, and maize crops.')

st.markdown("## Analysis of agricultural practice")
st.write("""Agricultural practices refer to the various techniques, methods, and processes that farmers use to cultivate crops and manage their farms. Consider the used fertilizer in the project.

Visualize how the amount of fertilizer changes through the years for different regions.""")



fertiliser_df = pd.read_csv(r"C:\Users\dress\assignment three group 297201\fertiliser_df.csv")

# Plot changes in fertilizer tonnes applied for each region
plt.figure(figsize=(12, 6))
sns.lineplot(data=fertiliser_df, x='Year', y='Fertilizer Tonnes Applied', hue='Region', marker='o')
plt.xlabel('Year')
plt.ylabel('Fertilizer Tonnes Applied')
plt.title('Changes in Fertilizer Tonnes Applied for each Region')
plt.xticks(rotation=45)
st.pyplot()


st.write("In general, for all regions, it is visible that there is a tendency for a decrease in fertilizer application, for some rgions this decrease is not significant, for other regions is more sufficient.  The amount of applied fertilizers has dramatically dropped for the Waikato region by 2023 compared to the year 2005.")

st.markdown("## Classifications model")
st.write("""The target variable (crop yield) is a numerical variable that cannot be predicted using a classification model. To address this limitation, an additional column with categorical values will be added to the dataset, providing ranges for the crop yield.
Initally generate the hist to visualize the distribution of crop yield.""")

# Load the DataFrame
crop_wether_soils_fertiliser_w_df = pd.read_csv(r"C:\Users\dress\assignment three group 297201\crop_wether_soils_fertiliser_w_df.csv")

# Filter out rows with zero wheat tonnes
crop_wether_soils_fertiliser_w_df = crop_wether_soils_fertiliser_w_df[crop_wether_soils_fertiliser_w_df['Wheat Tonnes'] != 0]

# Plot the histogram using Seaborn and Matplotlib
plt.figure(figsize=(8, 6))
sns.histplot(data=crop_wether_soils_fertiliser_w_df['Wheat Tonnes'], bins=20, kde=False, color='grey', edgecolor='black')
plt.title('Histogram of Wheat Yield', alpha=0.5)
plt.xlabel('Wheat Tonnes')
plt.ylabel('Frequency')

# Display the plot in Streamlit
st.pyplot(plt)

st.write("""For wheat yield following categories could be applied low (less 100000), moderate(from 100000 to 350000) and high (higher then 350000). Add the category to the dataset 'crop_wether_soils_fertiliser_w_df'.""")

category_wheat = pd.DataFrame({
    'Category': ['low', 'moderate', 'high'],
    'Range min': [0, 100001, 350001],
    'Range max': [100000, 350000, np.inf]
})

# Display the DataFrame using Streamlit
st.write(category_wheat)

st.write("""Apply the sam processe for barley yield. """)

# Load the DataFrame
crop_wether_soils_fertiliser_b_df = pd.read_csv(r"C:\Users\dress\assignment three group 297201\crop_wether_soils_fertiliser_b_df.csv")

# Filter out rows with zero wheat tonnes
crop_wether_soils_fertiliser_b_df = crop_wether_soils_fertiliser_b_df[crop_wether_soils_fertiliser_b_df['Barley Tonnes'] != 0]

# Plot the histogram using Seaborn and Matplotlib
plt.figure(figsize=(8, 6))
sns.histplot(data=crop_wether_soils_fertiliser_b_df['Barley Tonnes'], bins=20, kde=False, color='grey', edgecolor='black')
plt.title('Histogram of Barley Yield', alpha=0.5)
plt.xlabel('Barley Tonnes')
plt.ylabel('Frequency')
# Display the plot in Streamlit
st.pyplot(plt)

st.write("""For wheat yield following categories could be applied low (less 100000), moderate(from 100000 to 350000) and high (higher then 350000). Add the category to the dataset 'crop_wether_soils_fertiliser_w_df'.""")

category_barley = pd.DataFrame({
    'Category': ['low', 'moderate', 'high'],
    'Range min': [0, 100001, 250001],
    'Range max': [100000, 250000, np.inf]
})

# Display the DataFrame using Streamlit
st.write(category_barley)

st.write("""Apply the sam processe for maize grain yield. """)

# Load the DataFrame
crop_wether_soils_fertiliser_m_df = pd.read_csv(r"C:\Users\dress\assignment three group 297201\crop_wether_soils_fertiliser_m_df.csv")

# Filter out rows with zero maize grain tonnes
crop_wether_soils_fertiliser_m_df = crop_wether_soils_fertiliser_m_df[crop_wether_soils_fertiliser_m_df['Maize grain Tonnes'] != 0]

# Plot the histogram using Seaborn and Matplotlib
plt.figure(figsize=(8, 6))
sns.histplot(data=crop_wether_soils_fertiliser_m_df['Maize grain Tonnes'], bins=20, kde=False, color='grey', edgecolor='black')
plt.title('Histogram of Maize Grain Yield', fontsize=16)
plt.xlabel('Maize Grain Tonnes', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.grid(True)

# Display the plot in Streamlit
st.pyplot()

st.write("""The maize yield has diffrent categories. For maize yioeld following categories could be applied low (less 10000), moderate(from 10000 to 30000) and high (higher then 30000). """)

category_maize = pd.DataFrame({
    'Category': ['low', 'moderate', 'high'],
    'Range min': [0, 10001, 30001],
    'Range max': [10000, 30000, np.inf]
})

st.write(category_maize)

st.write("""The following classification models were applied to predict the category of crop yield for wheat, barley, and maize grain:

    Random forest classifier
    Decision tree classifier
    Gaussian Naive Bayes classifier

Results presemted in the table below
""")

clasifier_result = pd.DataFrame({
    'Method': ['Gaussian Naive Bayes classifier', 'Decision tree classifier', 'Random forest classifier'],
    'Wheat yield': [0.83, 0.50, 0.67],
    'Barley yield': [0.86, 0.93, 0.86],
    'Maize grain yield': [0.79, 0.86, 0.86]
})

st.write(clasifier_result)

st.write("""Analysis reveals that for wheat yield prediction the Gaussian Naive Bayes classifier has the highest accuracy, for the prediction of barley yield the Decision tree classifier has the best result and for maize grain prediction the  Random forest classifier and Decision tree classifier get the best prediction.
""")

html_content = """
# Project 3

Crop Yield Prediction. Create a machine learning model to forecast crop yields based on factors such as weather patterns, soil characteristics, and agricultural practices.

"""

# Offer download link for HTML file
st.markdown("[Download HTML file](myapp · Streamlit.htm)", unsafe_allow_html=True)
