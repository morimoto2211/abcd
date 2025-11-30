import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

class Entrada:
    def __init__(self, n: int, m: int, k: int, pos: np.array):
        assert n > 0, "El número de dimensiones debe ser mayor a 0"
        assert m > 0, "El número de pasos debe ser mayor a 0"
        assert k > 0, "El número de caminatas debe ser mayor a 0"
        self.n = n
        self.m = m
        self.k = k
        self.pos = pos

class Caminatas_aleatorias(Entrada):
    def __init__(self, n: int, m: int, k: int, pos: np.array, df: pd.DataFrame):
        super().__init__(n, m, k, pos)
        self.df = df

    @staticmethod
    def distancia_euclideana(pos, n):
        sum_val = 0
        for dimension in range(n):
            sum_val += pos[dimension] ** 2
        return np.sqrt(sum_val)

    def caminatas(self):
        n = self.n
        m = self.m
        k = self.k
        pos = self.pos
        df = self.df

        df = pd.DataFrame(columns=['Caminata', 'Posicion Final', 'Distancia', 'Tiempo'])

        for caminata in range(k):
            pos = np.array([0] * n)
            tiempo_i = time.time()
            for paso in range(m):
                for dimension in range(n):
                    if np.random.randint(0, 2) == 1:
                        pos[dimension] += 1
                    else:
                        pos[dimension] -= 1

            tiempo_f = time.time()
            tiempo_ejecucion = tiempo_f - tiempo_i

            df.loc[caminata, 'Caminata'] = caminata
            df.loc[caminata, 'Posicion Final'] = str(pos)
            df.loc[caminata, 'Distancia'] = Caminatas_aleatorias.distancia_euclideana(pos, n)
            df.loc[caminata, 'Tiempo'] = tiempo_ejecucion

        self.df = df
        return df

def main():
    st.set_page_config(page_title = "Movimiento Browniano", 
                       layout = "wide", 
                       initial_sidebar_state = "expanded")
    
    st.title("Movimiento Browniano")

    st.sidebar.header("Parámetros")
    
    dimensiones = st.sidebar.number_input(
        label = "Número de dimensiones",
        min_value = 1,
        max_value = 1000000,
        value = 2
    )
    
    pasos = st.sidebar.number_input(
        label  = "Número de pasos",
        min_value = 1,
        max_value = 1000001,
        value = 1
    )
    
    caminatas = st.sidebar.number_input(
        label ="Número de caminatas",
        min_value = 1,
        max_value = 1000002,
        value = 1
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        ejecutar_simulacion = st.button("Ejecutar Simulación")
    with col2:
        descargar_csv = st.button("Descargar CSV")

    if ejecutar_simulacion:
        st.info(f"Simulación {dimensiones}D, {pasos} pasos y {caminatas} caminatas")
        
        entrada = Entrada(dimensiones, pasos, caminatas, np.array([0] * dimensiones))
        algoritmo = Caminatas_aleatorias(entrada.n, entrada.m, entrada.k, entrada.pos, pd.DataFrame())
        
        resultado_df = algoritmo.caminatas()
        
        st.subheader("Resultados")
        st.dataframe(resultado_df)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Caminatas", len(resultado_df))
        with col2:
            st.metric("Distancia Promedio", f"{resultado_df['Distancia'].mean():.3f}")
        with col3:
            st.metric("Distancia Máxima", f"{resultado_df['Distancia'].max():.3f}")
        with col4:
            st.metric("Tiempo Total (s)", f"{resultado_df['Tiempo'].sum():.3f}")

        st.subheader("Gráficas")
        
        if dimensiones == 1:
            st.markdown("#### Posiciones Finales")
            x = []

            for pos_str in resultado_df['Posicion Final']:
                parts = pos_str.strip('[]').split()
                if len(parts) >= 1:
                    x.append(float(parts[0]))
            
            fig, ax = plt.subplots(figsize = (10, 4))
            ax.scatter(range(len(x)), x)
            ax.set_xlabel('Caminata')
            ax.set_ylabel('Posición Final')
            ax.set_title('Posiciones Finales')
            ax.grid(True)
            st.pyplot(fig)

        elif dimensiones == 2:
            st.markdown("#### Posiciones Finales")
            x = []
            y = []
            
            for pos_str in resultado_df['Posicion Final']:
                parts = pos_str.strip('[]').split()
                if len(parts) >= 2:
                    x.append(float(parts[0]))
                    y.append(float(parts[1]))
            
            fig, ax = plt.subplots(figsize = (10, 8))
            ax.scatter(x, y)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title('Posiciones Finales')
            ax.grid(True)
            st.pyplot(fig)
        
        elif dimensiones == 3:
            st.markdown("#### Posiciones Finales")
            x = []
            y = []
            z = []
            
            for pos_str in resultado_df['Posicion Final']:
                parts = pos_str.strip('[]').split()
                if len(parts) >= 3:
                        x.append(float(parts[0]))
                        y.append(float(parts[1]))
                        z.append(float(parts[2]))
            
            fig = plt.figure(figsize = (10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x, y, z)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('Posiciones Finales')
            st.pyplot(fig)
        
        else:
            st.info(f"Visualización no disponible para >3 dimensiones")

        st.markdown("#### Distribución de distancias")
        fig, ax = plt.subplots(figsize = (10, 6))
        ax.hist(resultado_df['Distancia'], edgecolor = 'black')
        ax.set_xlabel('Distancia Euclidiana')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de distancias')
        ax.grid(True)
        st.pyplot(fig)

        st.session_state.resultado_df = resultado_df

    if descargar_csv and hasattr(st.session_state, 'resultado_df'):
        csv_data = st.session_state.resultado_df.to_csv(index = False)
        st.download_button(
            label = "Descargar CSV",
            data = csv_data,
            file_name = "movimiento_browniano.csv"
        )
    elif descargar_csv:
        st.warning("Ejecutar primero")

if __name__ == "__main__":
    main()