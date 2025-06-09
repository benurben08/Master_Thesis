import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Settings import Directory, Plotting_Parameters

directory = Directory()
plot_params = Plotting_Parameters()

def find_similar_profiles(x_profile, model, fixed_features=None):
    if fixed_features is None:
        fixed_features = model.features.copy()
        fixed_indices = [i for i, item in enumerate(model.features) if item in fixed_features]
        index_to_remove = [model.features.index('M_contribution'),
                           model.features.index('V_contribution')]
        fixed_indices = [i for i in fixed_indices if i not in index_to_remove]
        fixed_indices = np.array(fixed_indices)
    else:
        fixed_features = fixed_features
        fixed_indices = [i for i, item in enumerate(model.features) if item in fixed_features]

    profiles_similar = []
    x_profile_features = x_profile[model.features].to_numpy().astype(np.float32).flatten()[fixed_indices]
    for idx in range(len(model.x_data)):
        if np.all(model.x_data[idx][fixed_indices] == x_profile_features):
            profiles_similar.append(idx)
            
    print(f'{len(profiles_similar)} profiles found with the same fixed features')

    return profiles_similar

def plot_MV_interaction(x_profile,model,df,n=10,fixed_features=None):
    if fixed_features is None:
        profiles_similar = find_similar_profiles(x_profile, model)
    else:
        profiles_similar = find_similar_profiles(x_profile, model, fixed_features=fixed_features)

    M_Rd_truth = df["M_Rd"].to_numpy()[profiles_similar]
    V_Rd_truth = df["V_Rd"].to_numpy()[profiles_similar]

    M_Rd_pred, V_Rd_pred, targets_predicted = model.get_MV_interaction_prediction(x_profile,n_predictions=n)
    
    fig, ax = plt.subplots(figsize=(5,5))
    plt.scatter(M_Rd_pred, V_Rd_pred, 
                s=50, 
                color=plot_params.morecolors[1], 
                edgecolors=plot_params.morecolors[0], 
                linewidth=1,                        
                label="Predicted")
    
    plt.plot(M_Rd_pred, V_Rd_pred,
             color=plot_params.morecolors[1], 
             linewidth=1.0, 
             linestyle='--', 
             alpha=0.5)

    plt.scatter(M_Rd_truth, V_Rd_truth,
                s=50, 
                color=plot_params.morecolors[3], 
                edgecolors=plot_params.morecolors[2],  
                linewidth=1,
                label="Ground Truth")
    
    if "M_Rd" in x_profile.keys() and "V_Rd" in x_profile.keys():
        plt.scatter(x_profile["M_Rd"], x_profile["V_Rd"],
                    s=50, 
                    color=plot_params.morecolors[5], 
                    edgecolors=plot_params.morecolors[4],  
                    linewidth=1,
                    label="Input Profile")

    plt.xlabel("MRd [kNm]")
    plt.ylabel("VRd [kN]")
    plt.grid(True, which='major', color='k', linestyle='--', linewidth=0.5, alpha=0.5)

    legend_labels = ['Predicted', 'Ground Truth']
    legend_handles = [
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=plot_params.morecolors[1], markeredgecolor=plot_params.morecolors[0], markersize=7),
        plt.Line2D([], [], marker='o', color='w', markerfacecolor=plot_params.morecolors[3], markeredgecolor=plot_params.morecolors[2], markersize=7),
    ]

    if "M_Rd" in x_profile.keys() and "V_Rd" in x_profile.keys():
        legend_labels.append("Input Profile")
        legend_handles.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor=plot_params.morecolors[5], markeredgecolor=plot_params.morecolors[4], markersize=7))

    plt.xlim([0,None])
    plt.ylim([0,None])

    plt.legend(legend_handles, legend_labels)

    plt.show()