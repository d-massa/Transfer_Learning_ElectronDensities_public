import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data organization:
# Structure: [Property][Wording Case][Modality][No FT/FT][Train/Test/Hold]

# Define the data as a nested dictionary:
data = {
    'E_fermi': {
        'Formula': {
            'Image': {
                'No FT': [0.62, 0.58, 0.47],
                'FT': [0.77, 0.63, 0.60],
            },
            'Text': {
                'No FT': [0.78, 0.59, 0.65],
                'FT': [0.91, 0.71, 0.86],
            },
            'Image+Text': {
                'No FT': [0.97, 0.69, 0.82],
                'FT': [0.99, 0.78, 0.88],
            },
        },
        'Keywords': {
            'Image': {
                'No FT': [0.73, 0.67, 0.63],
                'FT': [0.82, 0.72, 0.63],
            },
            'Text': {
                'No FT': [0.85, 0.49, 0.62],
                'FT': [0.92, 0.84, 0.86],
            },
            'Image+Text': {
                'No FT': [0.97, 0.83, 0.86],
                'FT': [0.97, 0.84, 0.87],
            },
        },
        'All': {
            'Image': {
                'No FT': [0.68, 0.62, 0.49],
                'FT': [0.80, 0.64, 0.56],
            },
            'Text': {
                'No FT': [0.77, 0.54, 0.60],
                'FT': [0.94, 0.77, 0.87],
            },
            'Image+Text': {
                'No FT': [0.98, 0.76, 0.80],
                'FT': [0.97, 0.80, 0.84],
            },
        }
    },
    'E_F': {
        'Formula': {
            'Image': {
                'No FT': [0.62, 0.59, 0.65],
                'FT': [0.71, 0.62, 0.72],
            },
            'Text': {
                'No FT': [0.90, 0.38, 0.61],
                'FT': [0.88, 0.65, 0.64],
            },
            'Image+Text': {
                'No FT': [0.94, 0.56, 0.60],
                'FT': [0.95, 0.59, 0.68],
            },
        },
        'Keywords': {
            'Image': {
                'No FT': [0.31, 0.24, 0.06],
                'FT': [0.61, 0.55, 0.49],
            },
            'Text': {
                'No FT': [0.89, 0.45, 0.68],
                'FT': [0.94, 0.75, 0.84],
            },
            'Image+Text': {
                'No FT': [0.94, 0.55, 0.63],
                'FT': [0.94, 0.68, 0.78],
            },
        },
        'All': {
            'Image': {
                'No FT': [0.50, 0.49, 0.58],
                'FT': [0.61, 0.53, 0.65],
            },
            'Text': {
                'No FT': [0.76, -0.04, 0.18],
                'FT': [0.92, 0.71, 0.60],
            },
            'Image+Text': {
                'No FT': [0.93, 0.58, 0.57],
                'FT': [0.95, 0.70, 0.73],
            },
        }
    },
    'K_vrh': {
        'Formula': {
            'Image': {
                'No FT': [0.69, 0.81, 0.77],
                'FT': [0.77, 0.84, 0.84],
            },
            'Text': {
                'No FT': [0.83, 0.74, 0.77],
                'FT': [0.90, 0.81, 0.82],
            },
            'Image+Text': {
                'No FT': [0.94, 0.84, 0.86],
                'FT': [0.97, 0.87, 0.87],
            },
        },
        'Keywords': {
            'Image': {
                'No FT': [0.72, 0.81, 0.82],
                'FT': [0.8, 0.85, 0.84],
            },
            'Text': {
                'No FT': [0.81, 0.68, 0.75],
                'FT': [0.94, 0.76, 0.83],
            },
            'Image+Text': {
                'No FT': [0.94, 0.8, 0.83],
                'FT': [0.95, 0.81, 0.81],
            },
        },
        'All': {
            'Image': {
                'No FT': [0.66, 0.79, 0.72],
                'FT': [0.79, 0.83, 0.79],
            },
            'Text': {
                'No FT': [0.71, 0.59, 0.62],
                'FT': [0.80, 0.75, 0.75],
            },
            'Image+Text': {
                'No FT': [0.89, 0.78, 0.71],
                'FT': [0.92, 0.75, 0.71],
            },
        }
    }
}

# Prepare data for heatmap
rows = []
values = []

for prop, cases in data.items():
    for case, modalities in cases.items():
        for modality, ft_data in modalities.items():
            for ft, r2_values in ft_data.items():
                rows.append(f'{prop} - {case} - {modality} - {ft}')
                values.append(r2_values)

values = np.array(values)

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.set(font_scale=0.8)

ax = sns.heatmap(values, annot=True, cmap="Reds", cbar_kws={'label': '$R^2$'}, linewidths=.5, fmt=".2f", vmin=0, vmax=1)

# Set axis labels
ax.set_yticklabels(rows, rotation=0)
ax.set_xticklabels(['Train', 'Test', 'Hold'], rotation=0)

# Titles and labels
#plt.title('$R^2$ Values Across Different Regression Experiments', fontsize=14)
#plt.xlabel('Dataset', fontsize=12)
plt.ylabel('Experiment Configuration', fontsize=12)

plt.tight_layout()
plt.savefig('Results_MultimodalSolution.png')
#plt.show()

