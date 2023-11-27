import pandas as pd
import os

dataset = pd.read_csv("amazon_products_dataset.csv")

# Get the data which only have full description
description = dataset[dataset['product_long_description'].notna()] 

# only get the columns that we will be needing for NLP tasks. ie ( 'product_name', product_brand', product_long_description')
description = description[['product_name', 'product_brand', 'product_long_description']]

# since some of the multiple brands have same productname, we will create a new column to uniquely identify each plug name
description['combined_column'] = description['product_name'] + ' ' + description['product_brand']

current_directory = os.getcwd()

#Creating the dataset folder
data_folder = os.path.join(current_directory, 'basic_nlp', 'data')
os.makedirs(data_folder, exist_ok=True)

for index, row in description[['combined_column', 'product_long_description']].iterrows():
    folder_name = row['combined_column']
    description = row['product_long_description']
    folder_path = os.path.join(data_folder, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    data_file_path = os.path.join(folder_path, 'data.txt')
    with open(data_file_path, 'w' , encoding='utf-8') as data_file:
        data_file.write(description)







