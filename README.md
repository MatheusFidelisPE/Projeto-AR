# Estudo de Parâmetros com Ambientes Diversos  

## Descrição  

Este projeto investiga o impacto de diferentes valores de parâmetros fundamentais em algoritmos de aprendizagem por reforço, incluindo **epsilon, número de passos, gamma, alfa e taxa de aprendizagem**.  

O objetivo é entender como pequenas variações nesses parâmetros podem afetar o desempenho do agente ao longo dos episódios de treinamento e teste.  

## Metodologia  

1. **Treinamento**  
   - O agente será treinado por **N** episódios, ajustando os parâmetros mencionados.  

2. **Execução e Avaliação**  
   - Após o treinamento, o agente executará **M** episódios.  
   - A recompensa obtida ao longo desses **M** episódios será registrada.  
   - Para minimizar efeitos estocásticos, repetiremos essa execução **X** vezes, calculando a **média** das recompensas e o **desvio padrão**.  


