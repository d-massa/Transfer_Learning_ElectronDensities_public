import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'FCC_properties_w_dens.csv'
df = pd.read_csv(file_path)

# Select interesting scalar properties and the number of elements
properties = ['formation_energy_per_atom', 'efermi', 'k_vrh']
nelements = df['nelements']

plt.rcParams.update({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 18, 'xtick.labelsize': 14, 'ytick.labelsize': 14})

# Set up the figure
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

# Define color palette
palette = sns.color_palette("deep", as_cmap=False)

# Create a stacked histogram for each property
for i, prop in enumerate(properties):
    sns.histplot(df, x=prop, hue='nelements', multiple='stack', palette=palette, ax=axes[i])
    axes[i].set_title(f'{prop}')

# Adjust the layout
plt.tight_layout()

# Show plot
plt.savefig('Distrib.png')

# Set up the figure for violin plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Create a violin plot for each property
for i, prop in enumerate(properties):
    sns.violinplot(x='nelements', y=prop, data=df, palette="rocket", ax=axes[i])
    #axes[i].set_title(f'{prop}')
    axes[i].set_xlabel('nelements')
    axes[i].set_ylabel(prop)

# Adjust the layout
plt.tight_layout()

# Show plot
plt.savefig('Violin_Distrib.png')


# Load the dataset
file_path = './Newdata_normalized_zscore_interval/filtered_FCC_properties.csv'
df = pd.read_csv(file_path)

# Set up the figure for violin plots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Create a violin plot for each property
props_names=['$E_F$','$E_{fermi}$','$K_{vrh}$']
for i, prop in enumerate(properties):
    sns.violinplot(x='nelements', y=prop, data=df, palette="rocket", ax=axes[i])
    
    #axes[i].set_title(f'{prop}')
    axes[i].set_xlabel('$N_{elem}$')
    axes[i].set_ylabel(props_names[i])

# Adjust the layout
plt.tight_layout()

# Show plot
plt.savefig('Violin_Distrib_afterclean.png')


#------------------------
# Set up the figure for 3 subplots

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

plt.rcParams.update({'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 22, 'xtick.labelsize': 25, 'ytick.labelsize': 25})

palette = sns.color_palette("rocket", as_cmap=False)

# Loop over the properties and create violin plots
for i, prop in enumerate(properties):
    # Violin plot
    sns.violinplot(x='nelements', y=prop, data=df, palette=palette, ax=axes[i])

    # Set labels for each subplot
    axes[i].set_xlabel('$N_{elem}$')
    axes[i].set_ylabel(props_names[i])

    # Calculate and display the number of samples for each 'nelements' group
    sample_counts = df['nelements'].value_counts().sort_index()

    # Add text at the top of the plot for each number of elements with matching colors
    for j, (nelem, count) in enumerate(sample_counts.items()):
        # Use the same color from the palette
        color = palette[j % len(palette)]
        axes[i].annotate(f'n={count}', xy=(nelem-1, axes[i].get_ylim()[1]), 
                         xytext=(0, 10), textcoords='offset points', 
                         ha='center', fontsize=18, color=color)

# Adjust layout and show plot
plt.tight_layout()
plt.savefig('Violin_Distrib_afterclean2.png')