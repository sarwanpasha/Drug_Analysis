from rdkit import Chem
from collections import Counter
import pandas as pd
import numpy as np
import itertools
from itertools import product

import timeit

import math

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import numpy as np


print("Pakages Loaded!!")


# Define the input and output file paths
input_file = "/olga-data1/Sarwan/SMILES_String_Regression/Dataset/df_drugbank_sm.csv"
# output_file = "output.csv"

# Read the input CSV file as a pandas dataframe
df = pd.read_csv(input_file)


print("Data Loaded!!")


# Define the k-mer length
#k = 9


smile_strings_1 = np.array(df['smiles'])
smile_strings_label_1 = np.array(df['solubility ALOGPS'])

smile_strings = []
smile_strings_label = []

for i in range(len(smile_strings_1)):
    aa = smile_strings_1[i]
    if str(aa)!="nan":
        asd = float(str(smile_strings_label_1[i]).replace(" g/l",""))
        if np.isnan(asd)==False:
            smile_strings.append(aa)
            smile_strings_label.append(asd)
            
len(smile_strings)


# In[37]:


idx = pd.Index(smile_strings_label) # creates an index which allows counting the entries easily
print('Here are all of the viral species in the dataset: \n', len(idx),"entries in total")
aq = idx.value_counts()
print(aq)


# In[38]:


# Convert SMILES to RDKit molecule
mol = Chem.MolFromSmiles(smile_strings[0])
#mol


# # Generating k-mers

# In[39]:


# seq_data
# attr_new
def build_kmers(sequence, ksize):
    # https://homolog.us/blogs/bioinfo/2017/10/25/intro-minimizer/
#     seq="ATGCGATATCGTAGGCGTCGATGGAGAGCTAGATCGATCGATCTAAATCCCGATCGATTCCGAGCGCGATCAAAGCGCGATAGGCTAGCTAAAGCTAGCA"
#     sequence = seq[:]

#     asd = str(sequence)
#     asd = np.array2string(sequence)
    
    string_parsing = []
    for ind_test in range(len(sequence)):
        string_parsing.append(str(sequence[ind_test]))
    
    asd = str(string_parsing)
    aa_lst_1 = asd.replace(",","")
    aa_lst_2 = aa_lst_1.replace("[","")
    aa_lst_3 = aa_lst_2.replace("\"","")
    aa_lst_4 = aa_lst_3.replace("]","")
    aa_lst_5 = aa_lst_4.replace("'","")
    aa_lst_6 = aa_lst_5.replace(" ","")
    aa_lst_6

#     print(aa_lst_6)
    seq = aa_lst_6[:]
#     rev=seq[::-1]
    
    Kmer=ksize
#     M=m_size
    L=len(seq)

#     minimizers = []
    k_mers_final = []
    for i in range(0, L-Kmer+1):

            sub_f=seq[i:i+Kmer]
            k_mers_final.append(sub_f)
#             print(sub_f,min)

#     print("unique minimizers = ",len(np.unique(minimizers)))
#     print("unique kmers = ",len(np.unique(k_mers_final)))

    return k_mers_final


# In[42]:



seq_data = smile_strings[:]
start = timeit.default_timer()


k_size_val = 9
spaced_kmer_length = 4 # 6

kmers_freq_vec = []
for protein_kmers in range(len(seq_data)):
    temp = seq_data[protein_kmers]
    k_mers_vals = build_kmers(temp,k_size_val) 
    kmers_freq_vec.append(k_mers_vals)
    
smiles_chars = '#%)(+-.0123456789=@ABCDEFGHIKLMNOPRSTVWXYZ[\\]abcdefgilmnoprstuy/$'

unique_seq_kmers_final_list = [''.join(c) for c in product(smiles_chars, repeat=spaced_kmer_length)]  

frequency_vector = []
#cnt_check2 = 0
for ii in range(len(kmers_freq_vec)):
    seq_tmp = kmers_freq_vec[ii]
    listofzeros = [0] * len(unique_seq_kmers_final_list)
    for j in range(len(seq_tmp)):
        aqs = seq_tmp[j]
        ind_tmp = unique_seq_kmers_final_list.index(aqs[0:spaced_kmer_length])
        listofzeros[ind_tmp] = listofzeros[ind_tmp] + 1
    frequency_vector.append(listofzeros)
    
stop = timeit.default_timer()
print("Spaced kmers Time : ", stop - start) 


# In[43]:


np.save("/olga-data1/Sarwan/SMILES_String_Regression/Dataset/Spaced_kmer_Drug_Bank_6951_seq_k_9_g_4.npy",frequency_vector)


# In[45]:


smile_strings_label = np.load("/olga-data1/Sarwan/SMILES_String_Regression/Dataset/Attribute_solubility_ALOGPS_7162.npy")


print("Data Saved!!")

# # Regression Models



# Assume that frequency_vector contains the k-mer spectrum embedding for SMILES strings, and smile_strings_label contains the solubility label.
X = np.array(frequency_vector)
y = np.array(smile_strings_label)

# Split the data into training and test sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train Test Split Done!!!")

# Initialize the models.
linear_reg = LinearRegression()
ridge_reg = Ridge(alpha=1)
lasso_reg = Lasso(alpha=0.1)
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the models.
linear_reg.fit(X_train, y_train)
print("Linear Regression Done!!")
ridge_reg.fit(X_train, y_train)
print("Ridge Regression Done!!")
lasso_reg.fit(X_train, y_train)
print("Lasso Regression Done!!")
rf_reg.fit(X_train, y_train)
print("RF Regression Done!!")
gb_reg.fit(X_train, y_train)
print("GB Regression Done!!")

# Make predictions on the test set.
y_pred_linear_reg = linear_reg.predict(X_test)
y_pred_ridge_reg = ridge_reg.predict(X_test)
y_pred_lasso_reg = lasso_reg.predict(X_test)
y_pred_rf_reg = rf_reg.predict(X_test)
y_pred_gb_reg = gb_reg.predict(X_test)




# In[ ]:


# # Compute the evaluation metrics.
# print("Mean Absolute Error (MAE)")
# print("Linear Regression:", mean_absolute_error(y_test, y_pred_linear_reg))
# print("Ridge Regression:", mean_absolute_error(y_test, y_pred_ridge_reg))
# print("Lasso Regression:", mean_absolute_error(y_test, y_pred_lasso_reg))
# print("Random Forest Regression:", mean_absolute_error(y_test, y_pred_rf_reg))
# print("Gradient Boosting Regression:", mean_absolute_error(y_test, y_pred_gb_reg))

# print("Mean Squared Error (MSE)")
# print("Linear Regression:", mean_squared_error(y_test, y_pred_linear_reg))
# print("Ridge Regression:", mean_squared_error(y_test, y_pred_ridge_reg))
# print("Lasso Regression:", mean_squared_error(y_test, y_pred_lasso_reg))
# print("Random Forest Regression:", mean_squared_error(y_test, y_pred_rf_reg))
# print("Gradient Boosting Regression:", mean_squared_error(y_test, y_pred_gb_reg))

# print("Root Mean Squared Error (RMSE)")
# print("Linear Regression:", np.sqrt


# In[ ]:


# Compute evaluation metrics for each model.
linear_reg_mae = mean_absolute_error(y_test, y_pred_linear_reg)
ridge_reg_mae = mean_absolute_error(y_test, y_pred_ridge_reg)
lasso_reg_mae = mean_absolute_error(y_test, y_pred_lasso_reg)
rf_reg_mae = mean_absolute_error(y_test, y_pred_rf_reg)
gb_reg_mae = mean_absolute_error(y_test, y_pred_gb_reg)

linear_reg_mse = mean_squared_error(y_test, y_pred_linear_reg)
ridge_reg_mse = mean_squared_error(y_test, y_pred_ridge_reg)
lasso_reg_mse = mean_squared_error(y_test, y_pred_lasso_reg)
rf_reg_mse = mean_squared_error(y_test, y_pred_rf_reg)
gb_reg_mse = mean_squared_error(y_test, y_pred_gb_reg)

linear_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_linear_reg))
ridge_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge_reg))
lasso_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso_reg))
rf_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_reg))
gb_reg_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb_reg))

linear_reg_r2 = r2_score(y_test, y_pred_linear_reg)
ridge_reg_r2 = r2_score(y_test, y_pred_ridge_reg)
lasso_reg_r2 = r2_score(y_test, y_pred_lasso_reg)
rf_reg_r2 = r2_score(y_test, y_pred_rf_reg)
gb_reg_r2 = r2_score(y_test, y_pred_gb_reg)

linear_reg_evs = explained_variance_score(y_test, y_pred_linear_reg)
ridge_reg_evs = explained_variance_score(y_test, y_pred_ridge_reg)
lasso_reg_evs = explained_variance_score(y_test, y_pred_lasso_reg)
rf_reg_evs = explained_variance_score(y_test, y_pred_rf_reg)
gb_reg_evs = explained_variance_score(y_test, y_pred_gb_reg)

# Print the evaluation metrics for each model.
print("Linear Regression:")
print("MAE:", linear_reg_mae)
print("MSE:", linear_reg_mse)
print("RMSE:", linear_reg_rmse)
print("R^2:", linear_reg_r2)
print("EVS:", linear_reg_evs)

print("Ridge Regression:")
print("MAE:", ridge_reg_mae)
print("MSE:", ridge_reg_mse)
print("RMSE:", ridge_reg_rmse)
print("R^2:", ridge_reg_r2)
print("EVS:", ridge_reg_evs)

print("Lasso Regression:")
print("MAE:", lasso_reg_mae)
print("MSE:", lasso_reg_mse)
print("RMSE:", lasso_reg_rmse)
print("R^2:", lasso_reg_r2)
print("EVS:", lasso_reg_evs)

print("Random Forest Regression:")
print("MAE:", rf_reg_mae)
print("MSE:", rf_reg_mse)
print("RMSE:", rf_reg_rmse)
print("R^2:", rf_reg_r2)
print("EVS:", rf_reg_evs)

print("Gradient Boosting Regression:")
print("MAE:", gb_reg_mae)
print("MSE:", gb_reg_mse)
print("RMSE:", gb_reg_rmse)
print("R^2:", gb_reg_r2)
print("EVS:", gb_reg_evs)

