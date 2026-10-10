#Funciones (Desarrolladas para la clase de Ciencia y Analítica de Datos)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
import seaborn as sns


class AnalizadorDeDatos:

  def __init__(self, df):
    self.df = df

  
  def graficar_histograma(self, columna, eje, height_stat="count", num_bins=70):
    """
    Grafica un histograma y una curva de densidad kernel para una columna numérica.
    """
    # Eliminar valores nulos para el histograma y KDE
    serie = self.df[columna].dropna() 
    
    sns.histplot(x=serie, ax=eje, stat=height_stat, bins=num_bins, label='Histograma')
    sns.kdeplot(x=serie, ax=eje, color='red', linewidth=2, label='KDE')
    x_vals = np.linspace(serie.min(), serie.max(), num_bins)
    normal_ref = norm.pdf(x_vals, loc=serie.mean(), scale=serie.std())

    # Si el histograma está en conteos, escalar las curvas
    if height_stat == 'count':
      n = len(serie)
      
      # Evitar división por cero
      rango = serie.max() - serie.min()
      rango = rango if rango > 0 else 1 
      
      bin_width = rango / num_bins
      normal_ref = normal_ref * n * bin_width

      # Reescalar altura de la curva KDE ya dibujada
      if eje.lines:  # Verificar si hay líneas dibujadas
        line = eje.lines[0]  # línea KDE dibujada
        y_kde = line.get_ydata()
        y_kde_scaled = y_kde * n * bin_width
        line.set_ydata(y_kde_scaled)

    eje.plot(x_vals, normal_ref, linewidth=2, color="orange", label='Normal ref')
    eje.set_title(f"Distribución de {columna}({height_stat})")
    eje.legend()

  def graficar_histogramas(
      self,
      columnas: list[str],
      rowsqty: int,colsqty: int,
      size_x: int,size_y: int,
      height_stat: str = "count",
      num_bins: int = 70
  ) -> None:
      """
      Grafica histogramas, curvas KDE y una distribución
      normal de referencia para múltiples columnas.
      """

      if not columnas:
          raise ValueError("Debe proporcionar al menos una columna.")

      if num_bins < 1:
          raise ValueError("num_bins debe ser mayor que 0.")

      # Configurar gráfico
      fig, ejes = plt.subplots(rowsqty,colsqty,figsize=(size_x, size_y))

      # Convertir ejes en lista plana
      if rowsqty * colsqty == 1:
          ejes = [ejes]
      else:
          ejes = np.array(ejes).ravel()

      # Graficar cada variable
      for columna, eje in zip(columnas, ejes):

          serie = self.df[columna].dropna()
          sns.histplot(x=serie,ax=eje,stat=height_stat,bins=num_bins,label="Histograma")
          sns.kdeplot(x=serie,ax=eje,color="red",linewidth=2,label="KDE")
          x_vals = np.linspace(serie.min(),serie.max(),num_bins)
          normal_ref = norm.pdf(x_vals,loc=serie.mean(),scale=serie.std())

          # Ajustar escalas cuando el histograma es de conteos
          if height_stat == "count":

              n = len(serie)

              rango = serie.max() - serie.min()
              rango = rango if rango > 0 else 1

              bin_width = rango / num_bins

              normal_ref = (normal_ref * n * bin_width)

              # Reescalar KDE
              if eje.lines:
                  line = eje.lines[0]
                  y_kde = line.get_ydata()
                  y_kde_scaled = (y_kde * n * bin_width)
                  line.set_ydata(y_kde_scaled)

          eje.plot(x_vals,normal_ref,linewidth=2,color="orange",label="Normal ref")
          eje.set_title(f"Distribución de {columna}")
          eje.legend(loc="upper right", fontsize=9)

       
  
  def imprimir_tabla_de_frecuencia(self, columna) -> pd.DataFrame:
    """
    Imprime una tabla de frecuencia y frecuencia relativa para una columna categórica.
    """
        
    # Conteo por categoría
    freq_abs = self.df[columna].value_counts()
    # Frecuencia por categoría (%)
    freq_rel = self.df[columna].value_counts(normalize=True) * 100
    # Se combinan las frecuencias
    frecuencia_df = pd.DataFrame({
        'Frecuencia': freq_abs,
        'Frecuencia Relativa (%)': freq_rel
    })
    # Se aplica formato para que sea porcentaje con dos decimales
    frecuencia_df['Frecuencia Relativa (%)'] = frecuencia_df['Frecuencia Relativa (%)'].map("{:.2f}%".format)
    # Salida
    
    return frecuencia_df


  
  def graficar_frecuencias_en_categorias(
    self,
    rowsqty: int,
    colsqty: int,
    size_x: int,
    size_y: int,
    columns: list[str] | None = None) -> None:
    """
    Grafica las frecuencias de las categorías en un subplot.
    """
    if(columns is None):
      columns = self.df.select_dtypes(
        include=['object', 'category']        
        ).columns.tolist()
      
    #Configurar gráfico
    fig, ejes = plt.subplots(rowsqty, colsqty, figsize=(size_x, size_y))
    if((rowsqty*colsqty == 1)):
      ejes = [ejes]
    else:
      ejes = ejes.ravel()

    for columna, eje in zip(self.df[columns], ejes):
      if(self.df[columna].nunique() > 10):
        #Calcular conteo y porcentaje
        conteo = self.df[columna].value_counts()
        porcentaje = (conteo / conteo.sum()) * 100  # ← cálculo de % sobre total
        top_10 = conteo.head(10)
        top_10_pct = porcentaje.head(10)

        #Graficar
        eje.barh(top_10.index, top_10,)

        # ETIQUETAS

        # Mostrar etiquetas de porcentaje
        for i, (cat, val) in enumerate(top_10.items()):
          pct = top_10_pct[cat]
          eje.text(val + (val * 0.01), i, f"{pct:.1f}%", va='center', fontsize=10)

        #Ejes y etiquetas
        # Ajustar el límite del eje X (reservar espacio para texto)
        eje.set_xlim(0, top_10.max() * 1.15)
        eje.set_ylabel(f"top 10 {columna}")
        eje.set_xlabel('count', fontsize=12)
        eje.set_title(f"Frecuencias para {columna}, Alta cardinalidad", fontsize=14)
        eje.invert_yaxis()
      else:
        # Obtener categorías ordenadas
        order = self.df[columna].value_counts().index
        sns.countplot(y=self.df[columna], ax=eje, order=order)


        # Añadir etiquetas manualmente (porcentaje sobre barras)
        total = len(self.df[columna])
        max_val = self.df[columna].value_counts().max()

        for p in eje.patches:
          val = p.get_width()
          pct = (val / total) * 100
          eje.text(val + (total * 0.01), p.get_y() + p.get_height()/2,
                  f"{pct:.1f}%", va='center', fontsize=10)

        # ETIQUETAS
        eje.set_xlim(0, max_val * 1.15)
        eje.set_ylabel(f"{columna}", fontsize=12)
        eje.set_xlabel('count', fontsize=12)
        eje.set_title(f"Frecuencias para {columna}", fontsize=14)
    plt.show()


  
  def eliminar_duplicados(self):
    """Elimina los registros duplicados del DataFrame, reinicia el índice y devuelve el DataFrame actualizado.
    """  
    #Verificar duplicados
    print(f"Duplicados encontrados: {self.df.duplicated().sum()}")
    #Eliminar  duplicados
    print(f"Eliminando duplicados...")
    self.df.drop_duplicates(inplace=True)
    #Reiniciar índice
    print(f"Reiniciando índice...")
    self.df.reset_index(drop=True, inplace=True)
    #Mostrar 0 duplicados
    print(f"Duplicados finales: {self.df.duplicated().sum()}")
    return self.df