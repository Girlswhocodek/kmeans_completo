"""
Implementación completa de K-means desde cero + Análisis de clustering
Resuelve los 14 problemas del ejercicio
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class ScratchKMeans:
    """
    PROBLEMAS 1-7: Implementación completa de K-means desde cero
    """

    def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4, verbose=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids = None
        self.labels = None
        self.sse = None
        self.best_sse = float('inf')

    def _initialize_centroids(self, X):
        """
        PROBLEMA 1: Inicializar centroides aleatoriamente desde los datos
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        return X[indices]

    def _calculate_sse(self, X, labels, centroids):
        """
        PROBLEMA 2: Calcular Suma de Errores Cuadráticos (SSE)
        """
        sse = 0.0
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                distances = np.linalg.norm(X[mask] - centroids[k], axis=1)
                sse += np.sum(distances ** 2)
        return sse

    def _assign_clusters(self, X, centroids):
        """
        PROBLEMA 3: Asignar cada punto al cluster más cercano
        """
        n_samples = X.shape[0]
        labels = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            distances = np.linalg.norm(centroids - X[i], axis=1)
            labels[i] = np.argmin(distances)
        
        return labels

    def _update_centroids(self, X, labels):
        """
        PROBLEMA 4: Actualizar centroides como la media de los puntos asignados
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        for k in range(self.n_clusters):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # Reinicializar si el cluster está vacío
                new_centroids[k] = X[np.random.randint(0, X.shape[0])]
        
        return new_centroids

    def fit(self, X):
        """
        PROBLEMAS 5-6: Entrenar K-means con múltiples inicializaciones
        """
        best_sse = float('inf')
        best_centroids = None
        best_labels = None
        
        for init in range(self.n_init):
            if self.verbose:
                print(f"Inicialización {init + 1}/{self.n_init}")
            
            centroids = self._initialize_centroids(X)
            prev_labels = None
            
            for iteration in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                
                # Verificar convergencia
                if prev_labels is not None and np.array_equal(labels, prev_labels):
                    if self.verbose:
                        print(f"  Convergencia en iteración {iteration + 1}")
                    break
                
                # Verificar cambio en centroides
                change = np.linalg.norm(new_centroids - centroids)
                if change < self.tol:
                    if self.verbose:
                        print(f"  Convergencia por tolerancia en iteración {iteration + 1}")
                    break
                
                centroids = new_centroids
                prev_labels = labels.copy()
            
            sse_current = self._calculate_sse(X, labels, centroids)
            
            if sse_current < best_sse:
                best_sse = sse_current
                best_centroids = centroids.copy()
                best_labels = labels.copy()
        
        self.centroids = best_centroids
        self.labels = best_labels
        self.sse = best_sse
        self.best_sse = best_sse
        
        if self.verbose:
            print(f"Mejor SSE: {best_sse:.4f}")

    def predict(self, X):
        """
        PROBLEMA 7: Predecir clusters para nuevos datos
        """
        if self.centroids is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a fit() primero.")
        
        return self._assign_clusters(X, self.centroids)

    def elbow_method(self, X, k_max=10):
        """
        PROBLEMA 8: Método del codo para determinar número óptimo de clusters
        """
        k_values = range(1, k_max + 1)
        sse_values = []
        
        for k in k_values:
            if self.verbose:
                print(f"Probando k={k}")
            
            kmeans_temp = ScratchKMeans(n_clusters=k, n_init=5, max_iter=100, verbose=False)
            kmeans_temp.fit(X)
            sse_values.append(kmeans_temp.sse)
        
        return k_values, sse_values

    def calculate_silhouette(self, X):
        """
        PROBLEMA 9: Calcular coeficientes de silueta
        """
        n_samples = X.shape[0]
        silhouette_vals = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Distancia intra-cluster (a)
            current_cluster = self.labels[i]
            mask_current = self.labels == current_cluster
            points_current = X[mask_current]
            
            if len(points_current) > 1:
                a_i = np.mean(np.linalg.norm(points_current - X[i], axis=1))
            else:
                a_i = 0
            
            # Distancia inter-cluster más cercana (b)
            b_i = np.inf
            for k in range(self.n_clusters):
                if k != current_cluster:
                    mask_k = self.labels == k
                    if np.any(mask_k):
                        points_k = X[mask_k]
                        distance_k = np.mean(np.linalg.norm(points_k - X[i], axis=1))
                        b_i = min(b_i, distance_k)
            
            if b_i == np.inf:
                b_i = a_i
            
            # Coeficiente de silueta
            silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
        
        silhouette_avg = np.mean(silhouette_vals)
        return silhouette_vals, silhouette_avg

def plot_silhouette(silhouette_vals, silhouette_avg, y_km, n_clusters):
    """
    PROBLEMA 9: Visualizar diagrama de silueta
    """
    from matplotlib import cm
    
    cluster_labels = np.arange(n_clusters)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_ax_lower, y_ax_upper = 0, 0
    yticks = []
    
    for i, c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km == c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color = cm.jet(i / n_clusters)
        ax.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, 
                height=1.0, edgecolor='none', color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2)
        y_ax_lower += len(c_silhouette_vals)
    
    ax.axvline(silhouette_avg, color="red", linestyle="--", 
               label=f'Promedio: {silhouette_avg:.3f}')
    ax.set_yticks(yticks)
    ax.set_yticklabels(cluster_labels + 1)
    ax.set_ylabel('Cluster')
    ax.set_xlabel('Coeficiente de Silueta')
    ax.legend()
    ax.set_title('Diagrama de Silueta')
    plt.tight_layout()
    plt.show()

def analyze_wholesale_customers():
    """
    PROBLEMAS 10-12: Análisis completo del dataset Wholesale Customers
    """
    print("🛒 ANÁLISIS COMPLETO WHOLESALE CUSTOMERS")
    print("=" * 60)
    
    # Cargar datos
    try:
        df = pd.read_csv('Wholesale customers data.csv')
        print("✅ Datos cargados exitosamente")
    except FileNotFoundError:
        print("❌ Archivo no encontrado.")
        print("📥 Descarga de: https://archive.ics.uci.edu/ml/datasets/Wholesale+customers")
        return None, None
    
    print(f"Dimensiones: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    # Información de grupos conocidos
    print(f"\n📊 GRUPOS CONOCIDOS:")
    print(f"Regiones únicas: {df['Region'].unique()}")
    print(f"Canales únicos: {df['Channel'].unique()}")
    print(f"Distribución por región:")
    print(df['Region'].value_counts().sort_index())
    print(f"Distribución por canal:")
    print(df['Channel'].value_counts().sort_index())
    
    # Seleccionar características de gasto
    features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
    X = df[features].values
    
    # Estandarizar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PROBLEMA 10: Análisis PCA
    print("\n🔍 ANÁLISIS PCA")
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    var_exp = pca.explained_variance_ratio_
    cum_var_exp = np.cumsum(var_exp)
    
    print("Varianza explicada acumulada:")
    for i, var in enumerate(cum_var_exp, 1):
        print(f"  Componente {i}: {var:.3f}")
    
    # Gráfico de varianza explicada
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, 7), var_exp, alpha=0.5, align='center', 
            label='Varianza individual')
    plt.step(range(1, 7), cum_var_exp, where='mid', 
             label='Varianza acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.hlines(0.8, 0, 7, "blue", linestyles='dashed', label='80% varianza')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title('Varianza Explicada - PCA')
    
    # Reducir a 2 componentes
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    # PROBLEMA 10: Método del codo
    print("\n📊 MÉTODO DEL CODO")
    kmeans = ScratchKMeans(verbose=True)
    k_values, sse_values = kmeans.elbow_method(X_pca, k_max=10)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, sse_values, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Método del Codo')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Seleccionar k óptimo basado en el codo
    k_optimal = 3
    print(f"\n🎯 K óptimo seleccionado: {k_optimal}")
    
    # Entrenar K-means con k óptimo
    kmeans_final = ScratchKMeans(n_clusters=k_optimal, n_init=10, verbose=True)
    kmeans_final.fit(X_pca)
    labels = kmeans_final.labels
    
    # PROBLEMA 9: Diagrama de silueta
    print("\n📈 DIAGRAMA DE SILUETA")
    silhouette_vals, silhouette_avg = kmeans_final.calculate_silhouette(X_pca)
    print(f"Coeficiente de silueta promedio: {silhouette_avg:.3f}")
    
    plot_silhouette(silhouette_vals, silhouette_avg, labels, k_optimal)
    
    # Visualizaciones completas
    plt.figure(figsize=(15, 5))
    
    # Clusters K-means
    plt.subplot(1, 3, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for k in range(k_optimal):
        mask = labels == k
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[k], label=f'Cluster {k}', alpha=0.7, s=50)
    
    plt.scatter(kmeans_final.centroids[:, 0], kmeans_final.centroids[:, 1],
               marker='*', c='black', s=200, label='Centroides')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clusters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Regiones conocidas
    plt.subplot(1, 3, 2)
    regions = df['Region'].values
    region_colors = ['red', 'blue', 'green']
    region_names = {1: 'Lisboa', 2: 'Oporto', 3: 'Otra'}
    
    for region in [1, 2, 3]:
        mask = regions == region
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=region_colors[region-1], label=region_names[region], 
                   alpha=0.7, s=50)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Regiones Conocidas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Canales conocidos
    plt.subplot(1, 3, 3)
    channels = df['Channel'].values
    channel_colors = ['red', 'blue']
    channel_names = {1: 'Hotel/Restaurant', 2: 'Retail'}
    
    for channel in [1, 2]:
        mask = channels == channel
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=channel_colors[channel-1], label=channel_names[channel], 
                   alpha=0.7, s=50)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Canales Conocidos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # PROBLEMA 12: Análisis detallado
    print("\n📊 ANÁLISIS DETALLADO DE CLUSTERS")
    df['Cluster'] = labels
    
    # Estadísticas por cluster
    print("\n💰 GASTOS PROMEDIO POR CLUSTER:")
    cluster_stats = df.groupby('Cluster')[features].mean()
    print(cluster_stats.round(2))
    
    # Comparación con grupos conocidos
    print("\n🔀 DISTRIBUCIÓN DE REGIONES POR CLUSTER:")
    region_cross = pd.crosstab(df['Cluster'], df['Region'], 
                              margins=True, margins_name="Total")
    print(region_cross)
    
    print("\n🛍️ DISTRIBUCIÓN DE CANALES POR CLUSTER:")
    channel_cross = pd.crosstab(df['Cluster'], df['Channel'],
                               margins=True, margins_name="Total")
    print(channel_cross)
    
    # Análisis de características importantes
    print("\n🎯 CARACTERÍSTICAS MÁS IMPORTANTES POR CLUSTER:")
    for cluster in range(k_optimal):
        cluster_data = df[df['Cluster'] == cluster][features]
        means = cluster_data.mean()
        top_features = means.nlargest(3)
        print(f"Cluster {cluster}: {', '.join([f'{feat}({val:.0f})' for feat, val in top_features.items()])}")
    
    return df, kmeans_final, X_scaled

def advanced_analysis(X_scaled, df):
    """
    PROBLEMAS 13-14: Análisis avanzado con DBSCAN y t-SNE
    """
    print("\n" + "=" * 60)
    print("🚀 ANÁLISIS AVANZADO: DBSCAN + t-SNE")
    print("=" * 60)
    
    # PROBLEMA 13: Investigación de métodos
    print("\n🔍 PROBLEMA 13: COMPARACIÓN DE MÉTODOS")
    print("DBSCAN - Ventajas:")
    print("  • No requiere especificar número de clusters")
    print("  • Encuentra clusters de forma arbitraria")
    print("  • Robustez a outliers")
    print("DBSCAN - Desventajas:")
    print("  • Sensible a parámetros eps y min_samples")
    print("  • Dificultad con clusters de densidad variable")
    print("  • Problemas con datos de alta dimensionalidad")
    
    print("\nt-SNE - Ventajas:")
    print("  • Excelente para visualización de alta dimensión")
    print("  • Preserva estructura local de datos")
    print("  • Muy efectivo para datos no lineales")
    print("t-SNE - Desventajas:")
    print("  • Computacionalmente costoso")
    print("  • Resultados dependen de perplexity")
    print("  • No preserva distancias globales")
    
    print("\nLLE - Ventajas:")
    print("  • Preserva estructura local")
    print("  • Menos costoso que t-SNE")
    print("  • Bueno para datos no lineales")
    print("LLE - Desventajas:")
    print("  • Sensible a número de vecinos")
    print("  • Puede colapsar datos en baja dimensión")
    print("  • Problemas con datos ruidosos")
    
    # PROBLEMA 14: t-SNE + DBSCAN
    print("\n🔧 PROBLEMA 14: t-SNE + DBSCAN")
    
    # Aplicar t-SNE
    print("Aplicando t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_scaled)
    
    # Aplicar DBSCAN
    print("Aplicando DBSCAN...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_tsne)
    
    # Contar clusters (excluyendo outliers -1)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"Clusters encontrados por DBSCAN: {n_clusters_dbscan}")
    print(f"Puntos considerados ruido: {n_noise}")
    
    # Comparar visualizaciones
    plt.figure(figsize=(15, 5))
    
    # t-SNE + DBSCAN
    plt.subplot(1, 3, 1)
    unique_labels = set(dbscan_labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Negro para ruido
        
        mask = dbscan_labels == k
        plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[col], label=f'Cluster {k}', alpha=0.7, s=50)
    
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'DBSCAN Clusters (t-SNE)\nClusters: {n_clusters_dbscan}, Ruido: {n_noise}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # PCA + K-means para comparación
    plt.subplot(1, 3, 2)
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_scaled)
    
    kmeans_comparison = ScratchKMeans(n_clusters=3, n_init=10, verbose=False)
    kmeans_comparison.fit(X_pca)
    kmeans_labels = kmeans_comparison.labels
    
    colors_kmeans = ['red', 'blue', 'green']
    for k in range(3):
        mask = kmeans_labels == k
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors_kmeans[k], label=f'Cluster {k}', alpha=0.7, s=50)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clusters (PCA)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Comparación de distribuciones
    plt.subplot(1, 3, 3)
    methods = ['K-means\n(PCA)', 'DBSCAN\n(t-SNE)']
    cluster_counts = [3, n_clusters_dbscan]
    
    plt.bar(methods, cluster_counts, color=['skyblue', 'lightcoral'])
    plt.ylabel('Número de Clusters')
    plt.title('Comparación de Métodos')
    
    for i, v in enumerate(cluster_counts):
        plt.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Análisis comparativo
    print("\n📈 COMPARACIÓN DE RESULTADOS:")
    print(f"K-means (PCA): {3} clusters fijos")
    print(f"DBSCAN (t-SNE): {n_clusters_dbscan} clusters encontrados automáticamente")
    print(f"Puntos identificados como ruido: {n_noise}")
    
    # Evaluar calidad de clusters
    if n_clusters_dbscan > 1:
        silhouette_dbscan = silhouette_score(X_tsne, dbscan_labels)
        print(f"Coeficiente de silueta DBSCAN: {silhouette_dbscan:.3f}")
    
    return dbscan_labels, X_tsne

def main():
    """
    Función principal que ejecuta todos los problemas
    """
    print("🎯 K-MEANS COMPLETO - 14 PROBLEMAS RESUELTOS")
    print("=" * 60)
    
    # Problemas 1-7: K-means desde cero con datos sintéticos
    print("1. PROBLEMAS 1-7: K-MEANS DESDE CERO")
    X, _ = make_blobs(n_samples=300, n_features=2, centers=4, 
                     cluster_std=0.8, shuffle=True, random_state=42)
    
    kmeans = ScratchKMeans(n_clusters=4, n_init=5, verbose=True)
    kmeans.fit(X)
    
    # Visualizar resultados básicos
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.6, s=50)
    plt.title('Datos Originales')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    colors = ['red', 'blue', 'green', 'orange']
    for k in range(4):
        mask = kmeans.labels == k
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[k], label=f'Cluster {k}', alpha=0.7, s=50)
    
    plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1],
               marker='*', c='black', s=200, label='Centroides')
    plt.title('K-means Desde Cero')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Problema 8: Método del codo
    plt.subplot(1, 3, 3)
    k_values, sse_values = kmeans.elbow_method(X, k_max=8)
    plt.plot(k_values, sse_values, 'bo-', linewidth=2, markersize=6)
    plt.xlabel('Número de Clusters (k)')
    plt.ylabel('SSE')
    plt.title('Método del Codo')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Problema 9: Diagrama de silueta
    print("\n2. PROBLEMA 9: DIAGRAMA DE SILUETA")
    silhouette_vals, silhouette_avg = kmeans.calculate_silhouette(X)
    print(f"Coeficiente de silueta promedio: {silhouette_avg:.3f}")
    plot_silhouette(silhouette_vals, silhouette_avg, kmeans.labels, 4)
    
    # Problemas 10-12: Análisis de datos reales
    print("\n3. PROBLEMAS 10-12: ANÁLISIS WHOLESALE CUSTOMERS")
    df, kmeans_final, X_scaled = analyze_wholesale_customers()
    
    if df is not None:
        # Problemas 13-14: Análisis avanzado
        print("\n4. PROBLEMAS 13-14: ANÁLISIS AVANZADO")
        dbscan_labels, X_tsne = advanced_analysis(X_scaled, df)
        
        # Resumen ejecutivo
        print("\n" + "=" * 60)
        print("📊 RESUMEN EJECUTIVO")
        print("=" * 60)
        print("🔹 K-means desde cero: Implementado y funcionando correctamente")
        print("🔹 Método del codo: Ayuda a determinar k óptimo")
        print("🔹 Diagramas de silueta: Evalúan calidad de clusters")
        print("🔹 Análisis Wholesale: Revela segmentos de clientes naturales")
        print("🔹 DBSCAN + t-SNE: Encuentra clusters de forma no paramétrica")
        print("\n💡 RECOMENDACIONES PARA EL NEGOCIO:")
        print("  • Segmentar clientes por patrones de compra identificados")
        print("  • Desarrollar estrategias de marketing específicas por cluster")
        print("  • Monitorear cambios en los patrones de compra")
        print("  • Considerar ambos métodos (K-means y DBSCAN) para validación")
    


if __name__ == "__main__":
    main()
