# Adjusting the layout for the new figure with three subplots

# Creating a figure with a grid layout: 2 rows, 2 columns
# Adjusted width_ratios so that the left column (for KDE of PC2) is wider
fig = plt.figure(figsize=(12, 6), dpi=400)
gs = fig.add_gridspec(2, 2, height_ratios=[1, 4], width_ratios=[1, 4], hspace=0.05, wspace=0.05)

# Scatter plot moved to the right
ax_scatter = fig.add_subplot(gs[1, 1])
sns.scatterplot(x='PC1', y='PC2', hue='cell_type', data=pca_data, alpha=0.98, ax=ax_scatter, marker='o')
ax_scatter.set_xlabel('PC1')
ax_scatter.set_ylabel('PC2')
ax_scatter.legend(title='Cell Type', bbox_to_anchor=(1, 0.5), loc='center left')

# KDE plot for PC1 above the scatter plot (no change needed here)
ax_kde_pc1 = fig.add_subplot(gs[0, 1], sharex=ax_scatter)
for cell_type in pca_data['cell_type'].unique():
    sns.kdeplot(pca_data[pca_data['cell_type'] == cell_type]['PC1'], ax=ax_kde_pc1)
ax_kde_pc1.set_ylabel('Density')
ax_kde_pc1.set_xlabel('')  # Hide x-axis labels

# KDE plot for PC2 moved to the left of the scatter plot
ax_kde_pc2 = fig.add_subplot(gs[1, 0], sharey=ax_scatter)
for cell_type in pca_data['cell_type'].unique():
    sns.kdeplot(pca_data[pca_data['cell_type'] == cell_type]['PC2'], ax=ax_kde_pc2, vertical=True)
ax_kde_pc2.set_xlabel('Density')
ax_kde_pc2.set_ylabel('')  # Hide y-axis labels

# Invert the x-axis to flip the plot
ax_kde_pc2.invert_xaxis()

# Hide x and y labels of scatter plot to avoid duplication
ax_scatter.set_xlabel('')
ax_scatter.set_ylabel('')

# Setting an overall title for the figure
fig.suptitle('PCA Analysis: Scatter and KDE Plots', fontsize=16)

# Show the plots
plt.show()