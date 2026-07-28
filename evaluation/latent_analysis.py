import umap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def create_latent_transformers(latent_transformers,z,random_state=0):

    latent_transformers.append(umap.UMAP(n_components=2, random_state=random_state).fit(z))


    return latent_transformers


def z_plot(z_transform,z_cf_transform,ef,ef_cf_target,k,mean_dist_umap,plt_name):
    vmin, vmax = 0, 1

    cmap_red = cm.get_cmap("Reds").copy()
    cmap_blue = cm.get_cmap("Blues").copy()

    # Improve visibility near white ends
    cmap_red.set_bad(color="lightgray")
    cmap_blue.set_bad(color="lightgray")
    cmap_red.set_under("darkred")
    cmap_blue.set_under("darkblue")
    cmap_red.set_over("black")
    cmap_blue.set_over("black")

    # --- Create scientific-style plot ---
    plt.figure(figsize=(7, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Scatter for z (red spectrum)
    sc1 = plt.scatter(
        z_transform[:, 0], z_transform[:, 1],
        c=ef, cmap=cmap_red,
        vmin=vmin, vmax=vmax,
        s=40, alpha=0.8, edgecolors='none', label="z"
    )

    # Scatter for z_cf (blue spectrum)
    sc2 = plt.scatter(
        z_cf_transform[:, 0], z_cf_transform[:, 1],
        c=ef_cf_target, cmap=cmap_blue,
        vmin=vmin, vmax=vmax,
        s=40, alpha=0.8, edgecolors='none', label="z_cf"
    )

    # --- Colorbars ---
    # Use inset_axes for better spacing control


    ax = plt.gca()

    # cbar1 (left)
    cax1 = inset_axes(ax, width="3%", height="35%", loc="upper left",
                      bbox_to_anchor=(1.05, 0.55, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar1 = plt.colorbar(sc1, cax=cax1)
    cbar1.set_label("y", fontsize=10)

    # cbar2 (right)
    cax2 = inset_axes(ax, width="3%", height="35%", loc="lower left",
                      bbox_to_anchor=(1.05, 0.1, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar2 = plt.colorbar(sc2, cax=cax2)
    cbar2.set_label("y_cf", fontsize=10)

    # --- Labels & aesthetics ---
    plt.title(f"Latent Space Comparison\n(mean {k}-NN distance = {mean_dist_umap:.4f})", fontsize=12)
    plt.xlabel("UMAP dim 1", fontsize=11)
    plt.ylabel("UMAP dim 2", fontsize=11)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()

    # --- Save high-quality figure ---
    plt.savefig(f"{plt_name}_latent_comparison.png", dpi=600, bbox_inches="tight")
    plt.close()

def z_diff_plot(z_change_transform,cluster_labels,cluster_method,n_clusters,ef_change,plt_name):
    '''
    # Step 4: create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Left subplot: Clusters ---
    sns.scatterplot(
        ax=axes[0],
        x=z_change_transform[:, 0],
        y=z_change_transform[:, 1],
        hue=cluster_labels,
        palette="tab10",
        s=40,
        alpha=0.8,
        legend="full"
    )
    axes[0].set_title(f"z - z_cf ({cluster_method}, {n_clusters} clusters)", fontsize=11)
    axes[0].set_xlabel("UMAP dim 1")
    axes[0].set_ylabel("UMAP dim 2")
    axes[0].legend(title="Cluster", fontsize=8, loc='best')

    # --- Right subplot: ef_change (blue colormap) ---
    norm = Normalize(vmin=0, vmax=1)
    cmap_blue = cm.get_cmap("Blues")

    sc = axes[1].scatter(
        z_change_transform[:, 0],
        z_change_transform[:, 1],
        c=ef_change,
        cmap=cmap_blue,
        norm=norm,
        s=40,
        alpha=0.8,
        edgecolors='none'
    )
    axes[1].set_title("z - z_cf colored by y-y_cf", fontsize=11)
    axes[1].set_xlabel("UMAP dim 1")
    axes[1].set_ylabel("UMAP dim 2")

    # Add colorbar for ef_change
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("y-y_cf", fontsize=9) #(blue spectrum)

    # --- Layout & save ---
    plt.tight_layout()
    plt.savefig(plt_name + "_z_change_subplots.png", dpi=600, bbox_inches="tight")
    #plt.show()
    plt.close()
    '''
    # Step 4: create figure with two subplots
    #fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    '''
    # --- Left subplot: Clusters ---
    sns.scatterplot(
        ax=axes[0],
        x=z_change_transform[:, 0],
        y=z_change_transform[:, 1],
        hue=cluster_labels,
        palette="tab10",
        s=40,
        alpha=0.8,
        legend="full"
    )
    axes[0].set_title(f"z - z_cf ({cluster_method}, {n_clusters} clusters)", fontsize=11)
    axes[0].set_xlabel("UMAP dim 1")
    axes[0].set_ylabel("UMAP dim 2")
    axes[0].legend(title="Cluster", fontsize=8, loc='best')
    '''

    ax=axes

    # --- Right subplot: ef_change (blue colormap, no normalization) ---
    vmin, vmax = -1, 1  # show full raw range
    cmap_blue = cm.get_cmap("Blues").copy()

    # improve visibility near white
    cmap_blue.set_under("darkblue")
    cmap_blue.set_over("black")
    cmap_blue.set_bad(color="lightgray")

    sc = ax.scatter(
        z_change_transform[:, 0],
        z_change_transform[:, 1],
        c=ef_change,
        cmap=cmap_blue,
        vmin=vmin,
        vmax=vmax,
        s=40,
        alpha=0.8,
        edgecolors='none'
    )
    ax.set_title("z - z_cf colored by y - y_cf", fontsize=11)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")

    cax = inset_axes(ax, width="3%", height="60%", loc="center left",
                     bbox_to_anchor=(1.05, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("y - y_cf (blue spectrum)", fontsize=9)

    # --- Layout & save ---
    plt.tight_layout()
    plt.savefig(plt_name + "_z_change_subplots.png", dpi=600, bbox_inches="tight")
    plt.close()

def analyze_and_plot_latent_space(latent_transformer,z, z_cf,ef,ef_cf_target, plt_name, random_state=0, k=1, use_fit_transform=False):
    """
    latent_transformers: list [umap_model, z_transform]
    z_cf: counterfactual samples, shape (n_samples, n_features)
    plt_name: path to save figure
    k: number of neighbors for mean distance
    """
    #if use_fit_transform:
    batch_size,replicates,z_dim =z.shape
    z=z.reshape((batch_size*replicates,z_dim))
    z_cf=z_cf.reshape((batch_size*replicates,z_dim))

    latent_transformer=umap.UMAP(n_components=2, random_state=random_state).fit(z)

    z_transform = latent_transformer.transform(z)  # shape (batch_size, 2)
    z_cf_transform = latent_transformer.transform(z_cf)  # shape (batch_size, 2)

    # --- Compute mean k-nearest neighbor distance ---
    nbrs = NearestNeighbors(n_neighbors=k).fit(z_transform)
    distances, _ = nbrs.kneighbors(z_cf_transform)
    distances=distances.reshape(batch_size,replicates)
    distances=np.mean(distances,axis=1)
    mean_dist_umap = np.mean(distances)
    # --- Compute mean k-nearest neighbor distance ---
    nbrs = NearestNeighbors(n_neighbors=k).fit(z)
    distances, _ = nbrs.kneighbors(z_cf)
    distances = distances.reshape(batch_size,replicates)
    distances=np.mean(distances,axis=1)
    mean_dist_raw = np.mean(distances)

    z_transform, z_cf_transform=np.mean(np.reshape(z_transform,(batch_size,replicates,2)),axis=1),np.mean(np.reshape(z_cf_transform,(batch_size,replicates,2)),axis=1)

    # --- Create Seaborn scatterplot ---
    # two colors
    '''
    data = np.concatenate(
        [z_transform, z_cf_transform], axis=0
    )
    labels = ["z"] * len(z_transform) + ["z_cf"] * len(z_cf_transform)
    df = {
        "x": data[:, 0],
        "y": data[:, 1],
        "label": labels,
    }

    plt.figure(figsize=(7, 6))
    sns.scatterplot(x="x", y="y", hue="label", data=df, alpha=0.7, s=40, palette=["tab:blue", "tab:orange"])
    plt.title(f"Latent Space Comparison (mean {k}-NN distance = {mean_dist_umap:.4f})")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Embedding Type")
    plt.tight_layout()
    plt.savefig(plt_name+'_z.png', dpi=300)
    #plt.show()
    plt.close()
    '''
    '''
    # --- Normalize color values between 0 and 1 ---
    norm_red = Normalize(vmin=0, vmax=1)
    norm_blue = Normalize(vmin=0, vmax=1)

    cmap_red = cm.get_cmap("Reds")
    cmap_blue = cm.get_cmap("Blues")

    # --- Create scientific-style plot ---
    plt.figure(figsize=(7, 6))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Scatter for z (red spectrum)
    sc1 = plt.scatter(
        z_transform[:, 0], z_transform[:, 1],
        c=ef, cmap=cmap_red, norm=norm_red,
        s=40, alpha=0.8, edgecolors='none', label="z"
    )

    # Scatter for z_cf (blue spectrum)
    sc2 = plt.scatter(
        z_cf_transform[:, 0], z_cf_transform[:, 1],
        c=ef_cf_target, cmap=cmap_blue, norm=norm_blue,
        s=40, alpha=0.8, edgecolors='none', label="z_cf"
    )

    # --- Colorbars ---
    cbar1 = plt.colorbar(sc1, fraction=0.046, pad=0.04)
    cbar1.set_label("y", fontsize=10)#(red spectrum)
    cbar2 = plt.colorbar(sc2, fraction=0.046, pad=0.08)
    cbar2.set_label("y_cf", fontsize=10)# (blue spectrum)

    # --- Labels & aesthetics ---
    plt.title(f"Latent Space Comparison\n(mean {k}-NN distance = {mean_dist_umap:.4f})", fontsize=12)
    plt.xlabel("UMAP dim 1", fontsize=11)
    plt.ylabel("UMAP dim 2", fontsize=11)
    plt.legend(frameon=True, fontsize=10, loc="best")
    plt.tight_layout()

    # --- Save high-quality figure ---
    plt.savefig(f"{plt_name}_latent_comparison.png", dpi=600, bbox_inches="tight")
    #plt.show()
    plt.close()
    '''
    z_plot(z_transform, z_cf_transform, ef, ef_cf_target, k, mean_dist_umap, plt_name)

    ############################################################################

    z_change=z-z_cf

    z_change_transform=umap.UMAP(n_components=2, random_state=random_state).fit_transform(z_change)
    # todo do a clustering on  z_change_transform (estimate number of cluster and do clustering) + make a seaborn scatterplot with z_change_transform colorcoded with the corresponding clusters

    # Step 2: clustering
    try:
        import hdbscan
        print('hdbscan')
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        cluster_labels = clusterer.fit_predict(z_change_transform)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        cluster_method = "HDBSCAN"
    except ImportError:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        print('KMeans')
        # fallback: estimate cluster number for KMeans
        best_k, best_score = 2, -1
        for k in range(2, min(10, len(z_change_transform))):
            labels = KMeans(n_clusters=k, random_state=random_state).fit_predict(z_change_transform)
            score = silhouette_score(z_change_transform, labels)
            if score > best_score:
                best_score, best_k = score, k
        kmeans = KMeans(n_clusters=best_k, random_state=random_state)
        cluster_labels = kmeans.fit_predict(z_change_transform)
        n_clusters = best_k
        cluster_method = f"KMeans (k={best_k})"

    # Step 3: plot results
    ef_change=ef-ef_cf_target
    '''
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=z_change_transform[:, 0],
        y=z_change_transform[:, 1],
        hue=cluster_labels,
        palette="tab10",
        s=40,
        alpha=0.8,
        legend="full"
    )
    plt.title(f"z-z_cf ({cluster_method}, {n_clusters} clusters)")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.tight_layout()
    #plt.show()
    plt.savefig(plt_name+'z_change.png', dpi=300)
    plt.close()
    '''
    #z_change_transform=np.mean(np.reshape(z_change_transform, (batch_size, replicates, 2)), axis=1)
    #cluster_labels=np.reshape(z_change_transform, (batch_size, replicates))
    #labels = np.apply_along_axis(lambda r: np.bincount(r).argmax(), 1, cluster_labels)
    #print(labels.shape)
    #z_diff_plot(z_change_transform, labels, cluster_method, n_clusters, ef_change, plt_name)




    return mean_dist_umap,mean_dist_raw,n_clusters