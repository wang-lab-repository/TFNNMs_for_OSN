import numpy as np
import keras
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
import rdkit
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs
from scipy import stats
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
import shap

# Calculate the performance corresponding to the samples in the dataset
def train_new_data(data, path, min, max):
    model = keras.models.load_model(path, compile = False)
    per = data['Permeability']
    sel = data['Rejection/Selectivity']
    new_data = (data - min) / max
    y_predicted = model.predict(new_data)
    data_per = per*y_predicted[:, 0]
    data_sel = sel*y_predicted[:, 1]
    val_data = pd.concat([data_per, data_sel], axis = 1)
    #val_data.to_excel("***.xlsx", index_label = False)
    index = []
    for i in range(len(val_data)):
        index.append(i)
    val_data.index = index
    return val_data

# Constructing the dataset for optimization
def make_data(data, size_list, load_list, amine_list, chloride_list):
    datas = []
    for i in size_list:
        for j in load_list:
            for k in amine_list:
                for t in chloride_list:
                    for u in range(data.shape[0]):
                        nul = data.iloc[u]
                        nul[0] = i
                        nul[1] = j
                        nul[11] = k
                        nul[12] = t
                        datas.append(list(nul))
    return pd.DataFrame(datas, columns=data.columns)

# Plotting shap summary graphs
def plot_SSP(model, col, x_train, x_test):
    shap.initjs()
    backgroud = x_train.iloc[np.random.choice(x_train.shape[0], 400, replace = False)]
    explainer = shap.DeepExplainer(model, backgroud.values)
    shap_values = explainer.shap_values(x_test.values[0: 100])
    shap.summary_plot(shap_values[1], x_test.iloc[0: 100], max_display = 10, show = False) # , plot_type = "bar"
    # shap.force_plot(explainer.expected_value[0], shap_values[0], x_test, show = False)
    # plt.savefig('RS.tif', dpi=600, bbox_inches='tight', pad_inches=0.1)
    shap.dependence_plot(col, shap_values[0], x_test.iloc[0:100], show=False)
    # shap.dependence_plot("Chloride concentration", shap_values[0], x_test)
    # shap.dependence_plot("Amine concentration", shap_values[1], x_test)
    # shap.dependence_plot("NP size", shap_values[1], x_test,show=False)


# Perform recursive feature selection
def select_feature(x, y, x_unseen):
    RFC_ = RandomForestRegressor()
    selector = RFE(estimator=RFC_, n_features_to_select=61, step=1)
    selector = selector.fit(x, y)
    index = selector.support_
    columns = x.columns
    selected_columns = []
    for i, _ in enumerate(index):
        if index[i] == True:
            for j, col in enumerate(columns):
                if i == j:
                    selected_columns.append(col)
    return x[selected_columns], x_unseen[selected_columns]

# MACCS encoding
def get_maccs(mol):
    mol = Chem.MolFromSmiles(mol)
    fp1 = MACCSkeys.GenMACCSKeys(mol)
    return fp1.ToBitString()

# ECFP encoding
def get_ecfp(mol):
    mol = Chem.MolFromSmiles(mol)
    #fp1_morgan = AllChem.GetMorganFingerprint(mol,2)
    #print (fp1_morgan.GetLength())
    fp1_morgan_hashed = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits = 1024)
    return fp1_morgan_hashed.ToBitString()
# Feature filling using machine learning models
def feature_impute(df_all, target, use_f):
    df_mv = df_all[[use_f, target]].copy()
    df_mv = df_mv.dropna()
    x = np.array(df_mv[use_f]).reshape(-1, 1)
    y = np.array(df_mv[target]).reshape(-1, 1)

    x_train, x_test , y_train , y_test = train_test_split(x, y, test_size=0.2, random_state = 42)
    print("RF: ")
    rf = RandomForestRegressor(random_state = 42)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    scores = cross_val_score(rf, x_train, y_train , cv = 3, scoring='r2')
    score = np.mean(scores)
    print("Train R2: %.3f (+/- %.3f) \n" %(score, scores.std()))
    print("Test R2: %.3f\n" %(r2_score(y_test, pred)))

    # As RF is best:

    name = target + "_imputed"
    p = rf.predict(np.array(df_all[use_f]).reshape(-1,1))
    df_all[name] = p

    return df_all, rf

def feature_impute_exiting(df_all, target, use_f, model):
    # As RF is best:
    name = target + "_imputed"
    p = model.predict(np.array(df_all[use_f]).reshape(-1, 1))
    df_all[name] = p
    return df_all

def rf_fill(df_all, target):
    df_all[target] = df_all[target].fillna(df_all[target + "_imputed"])
    # drop unrequired featues
    df = df_all.drop([target + "_imputed"], axis = 1, inplace=True)