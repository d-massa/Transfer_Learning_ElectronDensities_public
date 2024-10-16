import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define the data specific to 'E_F'
data_efermi = {
        'Formula': {
            'Image': {
                'FT': [0.77, 0.63, 0.60],
            },
            'Text': {
                'FT': [0.91, 0.71, 0.86],
            },
            'Image+Text': {
                'FT': [0.99, 0.78, 0.88],
            },
        },
        'Keywords': {
            'Image': {
                'FT': [0.82, 0.72, 0.63],
            },
            'Text': {
                'FT': [0.92, 0.84, 0.86],
            },
            'Image+Text': {
                'FT': [0.97, 0.84, 0.87],
            },
        },
        'All': {
            'Image': {
                'FT': [0.80, 0.64, 0.56],
            },
            'Text': {
                'FT': [0.94, 0.77, 0.87],
            },
            'Image+Text': {
                'FT': [0.97, 0.80, 0.84],
            },
        }
    }
data_EF= {
    
       'Formula': {
            'Image': {
                'FT': [0.71, 0.62, 0.72],
            },
            'Text': {
                'FT': [0.88, 0.65, 0.64],
            },
            'Image+Text': {
                'FT': [0.95, 0.59, 0.68],
            },
        },
        'Keywords': {
            'Image': {
                'FT': [0.61, 0.55, 0.49],
            },
            'Text': {
                'FT': [0.94, 0.75, 0.84],
            },
            'Image+Text': {
                'FT': [0.94, 0.68, 0.78],
            },
        },
        'All': {
            'Image': {
                'FT': [0.61, 0.53, 0.65],
            },
            'Text': {
                'FT': [0.92, 0.71, 0.60],
            },
            'Image+Text': {
                'FT': [0.95, 0.70, 0.73],
            },
        }
}

data_kvrh={
        'Formula': {
            'Image': {
                'FT': [0.77, 0.84, 0.84],
            },
            'Text': {
                'FT': [0.90, 0.81, 0.82],
            },
            'Image+Text': {
                'FT': [0.97, 0.87, 0.87],
            },
        },
        'Keywords': {
            'Image': {
                'FT': [0.8, 0.85, 0.84],
            },
            'Text': {
                'FT': [0.94, 0.76, 0.83],
            },
            'Image+Text': {
                'FT': [0.95, 0.81, 0.81],
            },
        },
        'All': {
            'Image': {
                'FT': [0.79, 0.83, 0.79],
            },
            'Text': {
                'FT': [0.80, 0.75, 0.75],
            },
            'Image+Text': {
                'FT': [0.92, 0.75, 0.71],
            },
        }
    }

# Prepare data for heatmap
rows = []
values = []

for case, modalities in data_efermi.items():
    for modality, ft_data in modalities.items():
        for ft, r2_values in ft_data.items():
            rows.append(f'{case} - {modality} - {ft}')
            values.append(r2_values)

values = np.array(values)

# Plot the heatmap
plt.figure(figsize=(11, 8))
sns.set(font_scale=2.6)  # Increase the overall font size

# Create heatmap with smaller cell spacing
ax = sns.heatmap(values, annot=True, cmap="Reds", cbar_kws={'label': '$R^2$'}, linewidths=1.0, 
                 fmt=".2f", vmin=0, vmax=1, annot_kws={"size": 24}, square=False)

# Set axis labels
ax.set_yticklabels(rows, rotation=0, fontsize=24)  # Increase font size for y-tick labels
ax.set_xticklabels(['Train', 'Val', 'Test'], rotation=0, fontsize=24)  # Increase font size for x-tick labels
ax.figure.axes[-1].yaxis.label.set_size(30)  # Increase font size for colorbar label

# Draw rectangles around the different 'text amount classes'
rectangle_params1 = {
    'facecolor': 'none',
    'edgecolor': 'black',
    'linewidth': 6,
}
rectangle_params2 = {
    'facecolor': 'none',
    'edgecolor': 'black',
    'linewidth': 6,
}
rectangle_params3 = {
    'facecolor': 'none',
    'edgecolor': 'black',
    'linewidth': 6,
}


# Rectangle for 'Formula' (first 6x3 block)
plt.gca().add_patch(plt.Rectangle((0, 0), 3, 6, **rectangle_params1))

# Rectangle for 'Keywords' (next 6x3 block)
plt.gca().add_patch(plt.Rectangle((0, 3), 3, 6, **rectangle_params2))

# Rectangle for 'All' (final 6x3 block)
plt.gca().add_patch(plt.Rectangle((0, 6), 3, 6, **rectangle_params3))


# Titles and labels
plt.title('b) $E_{fermi}$',loc='left',pad=20)
plt.tight_layout()
plt.savefig('Results_data_efermi_highlighted_XS.png')
#plt.show()
