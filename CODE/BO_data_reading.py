import pandas as pd

###########################################################################
# generative the data
df_labeled = pd.read_csv('descriptors_labeled_019.csv')
df_unlabeled = pd.read_csv('descriptors_unlabeled_019.csv')

### preprocessing additional data
df_DFT = pd.read_csv('ActiveLearning_G_K_preprocessed_019.csv')
df_DFT = df_DFT.dropna(axis = 'rows')
DFT_descriptors = df_unlabeled.loc[df_unlabeled.name.isin([df_DFT['ID'][0]])]
for iter in range(1,len(df_DFT),1):
    DFT_descriptors = DFT_descriptors.append(df_unlabeled.loc[df_unlabeled.name.isin([df_DFT['ID'][iter]])],ignore_index=True)

temptemp = pd.concat([df_unlabeled, DFT_descriptors])
df_unlabeled = temptemp.drop_duplicates(keep = False)

DFT_descriptors = DFT_descriptors.assign(ID = df_DFT['ID'], g_dft = df_DFT['Gavg'], k_dft = df_DFT['Bavg'])
df_additional = DFT_descriptors

descriptors_labeled =[
'density','spg','volume','SiOSi_average','SiOSi_gmean','SiOSi_hmean','SiOSi_max','SiOSi_mean','SiOSi_min','SiOSi_skew','SiOSi_std','SiOSi_var','SiO_average','SiO_gmean','SiO_hmean','SiO_max','SiO_mean','SiO_min','SiO_skew','SiO_std','SiO_var','ASA','AV','NASA','NAV','VolFrac','largest_free_sphere','largest_included_sphere','largest_included_sphere_free','max_dim','min_dim','mode_dim','g_dft','k_dft']
descriptors_unlabeled = [
'density','spg','volume','SiOSi_average','SiOSi_gmean','SiOSi_hmean','SiOSi_max','SiOSi_mean','SiOSi_min','SiOSi_skew','SiOSi_std','SiOSi_var','SiO_average','SiO_gmean','SiO_hmean','SiO_max','SiO_mean','SiO_min','SiO_skew','SiO_std','SiO_var','ASA','AV','NASA','NAV','VolFrac','largest_free_sphere','largest_included_sphere','largest_included_sphere_free','max_dim','min_dim','mode_dim','g_gbr','k_gbr']

descriptors_for_all_labeled = list(set(descriptors_labeled))
descriptors_for_all_unlabeled = list(set(['name'] + descriptors_unlabeled))

descriptors_labeled = df_labeled[descriptors_for_all_labeled]
descriptors_unlabeled = df_unlabeled[descriptors_for_all_unlabeled]

# adding additional labels
descriptors_labeled = descriptors_labeled.append(df_additional[descriptors_for_all_labeled], ignore_index=True)

## making DB files
descriptors_labeled.to_csv('descriptors_labeled_020.csv', index=False)
descriptors_unlabeled.to_csv('descriptors_unlabeled_020.csv', index=False)