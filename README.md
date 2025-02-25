## Introdução
O aprendizado por reforço é uma abordagem poderosa para treinar agentes a tomarem decisões em ambientes desconhecidos. Este artigo explora o N STEP SARSA. Utilizando os ambientes *FrozenLake-v1*, *Race Track* do OpenAI Gym, analisamos como diferentes hiperparâmetros fundamentais afetam o desempenho do agente ao longo de N episódios de treinamento.

### Conceitos Fundamentais - O que são Hiperparâmetros?

O algoritmo apresentado pertence à família *Temporal-Difference Learning*, que atualiza a função de valor Q(s, a) com base em interações individuais do agente com o ambiente. A fórmula básica para atualização da tabela Q é:

$Q(s,a) \ &larr; Q(s,a) + \alpha \times (r + \gamma V(s') - Q(s,a))$

Onde:
- \( $\alpha$ \) é a taxa de aprendizado.
- \( $\gamma$ \) é o fator de desconto do futuro.
- $( r )$ é a recompensa obtida ao executar a ação $( a )$ no estado $( s )$.
- $V(s')$ é a estimativa do valor do próximo estado $( s' )$.

### Por que o Estudo de Hiperparâmetros é Essencial no Aprendizado por Reforço?
Quando falamos em algoritmos de aprendizado por reforço, como o n-step SARSA, uma das etapas mais importantes — e muitas vezes negligenciada — é o estudo de hiperparâmetros. Mas por que eles são tão cruciais para o sucesso do seu modelo? Vamos explorar um pouco mais abaixo:

1. *Desempenho do Algoritmo*:
Hiperparâmetros mal ajustados podem levar a resultados desastrosos. Por exemplo, uma taxa de aprendizado muito alta pode fazer com que o algoritmo oscile e nunca convirja para uma solução ótima. Por outro lado, uma taxa muito baixa pode tornar o aprendizado extremamente lento. O mesmo vale para o fator de desconto: se for muito baixo, o agente pode ignorar recompensas futuras importantes; se for muito alto, pode ficar preso em recompensas de curto prazo.

2. *Adaptação ao Problema*:
Cada problema de aprendizado por reforço é único. Um conjunto de hiperparâmetros que funciona bem em um ambiente pode não ser eficaz em outro. Por exemplo, em um ambiente com recompensas esparsas, uma taxa de exploração mais alta pode ser necessária para garantir que o agente descubra ações úteis. O estudo de hiperparâmetros permite ajustar o algoritmo para se adaptar ao problema específico que você está resolvendo.

3. *Equilíbrio entre Exploração e Exploitação*:
Um dos desafios do aprendizado por reforço é equilibrar a exploração (tentar novas ações) e a exploração (usar o conhecimento atual). A taxa de exploração (ε) é um hiperparâmetro crítico para isso. Se for muito alta, o agente pode nunca aprender a política ótima; se for muito baixa, ele pode ficar preso em ações subótimas.

4. *Eficiência e Velocidade de Convergência*:
Hiperparâmetros bem ajustados podem acelerar significativamente o tempo de treinamento. Por exemplo, escolher um número de passos (n) adequado no n-step SARSA pode melhorar a eficiência das atualizações da função de valor, reduzindo o tempo necessário para o algoritmo convergir.

## Metodologia

### Treinamento

O agente será treinado por um número N de episódios, durante os quais serão ajustados diversos parâmetros fundamentais, incluindo:
- **Epsilon ( $\epsilon$ )**: Define a probabilidade de o agente explorar novas ações em vez de explorar o conhecimento adquirido.
- **Número de passos**: Influencia o tempo de aprendizado do agente e sua capacidade de adaptação ao ambiente.
- **Gamma ( $\gamma$ )**: Representa o fator de desconto, que define a importância das recompensas futuras.
- **Alfa ( $\alpha$ )**: Taxa de aprendizado, que controla o impacto das novas experiências na atualização da Q-table.
- **Taxa de aprendizado**: Regula a velocidade com que o agente aprende a partir das experiências.

O objetivo principal desta investigação é compreender como pequenas variações nesses parâmetros afetam o desempenho do agente ao longo do treinamento, buscando otimizar sua performance.

### Execução e Avaliação

Após o treinamento, o agente executará M episódios para testar sua capacidade de tomada de decisão. Durante essa fase, serão registradas as recompensas obtidas para cada episódio.

Para minimizar efeitos estocásticos e garantir resultados mais confiáveis, repetiremos essa execução X vezes, calculando a média das recompensas e o desvio padrão. Isso permitirá avaliar a estabilidade e a eficácia do aprendizado do agente.

## Q-Learning

O **Q-Learning** é um algoritmo *off-policy*, pois assume que o agente sempre escolherá a melhor ação possível no futuro, independentemente da política seguida durante o treinamento. A atualização da Q-table usa:

$V(s') = \max_{a'} Q(s', a')$

Isso significa que, ao atualizar Q(s, a), ele considera a melhor ação disponível no estado seguinte.

O código implementa esse algoritmo e treina o agente no ambiente FrozenLake, executando 10.000 episódios e armazenando as recompensas obtidas. O método *epsilon-greedy* é utilizado para equilibrar exploração e explotação.

## SARSA

O **SARSA** é um algoritmo *on-policy*, o que significa que ele atualiza a Q-table considerando a próxima ação realmente escolhida pelo agente, ao invés da melhor ação teórica. Sua fórmula de atualização é:

$V(s') = Q(s', a')$

Isso faz com que SARSA tenha um comportamento mais conservador que o Q-Learning, pois a estimativa de futuro é baseada na política de treinamento.

### Resultados do SARSA

Executamos o SARSA no mesmo ambiente e comparamos os resultados com o Q-Learning. O agente tende a aprender de maneira mais segura, evitando caminhos arriscados. Os gráficos revelam a diferença no padrão de aprendizado e a influência dos parâmetros ajustados no desempenho final.

Sarsa personalizado para treinar e testar
Experimento 1: RaceTrack

a) Alterando o LR

![Image](https://github.com/user-attachments/assets/f4f74c9e-530a-4221-962d-9bd14d157800)

b) Alterando o EPSILON

![Image](https://github.com/user-attachments/assets/9527c224-e594-46c6-bdb3-a59c7ef11440)




Experimento 2: Fronzen Lake

a) Alterando o Alfa

![Image](https://github.com/user-attachments/assets/fcea98d9-84b6-4f01-91af-04b0b900d21d)


b) Alterando o EPSILON

![Image](https://github.com/user-attachments/assets/2b313868-5069-4611-9cc6-987244a5126c)



Experimento 3: Cliff Walking

a) Alterando o Alfa

b) Alterando o EPSILON


Experimento 4: Taxi

![Image](https://github.com/user-attachments/assets/912acb9b-f941-449a-820f-95922d619a36)


## Conclusão

Comparando os três algoritmos, podemos destacar:
- **Q-Learning**: Melhor desempenho a longo prazo, mas pode ser mais instável no aprendizado inicial.
- **SARSA**: Mais conservador, resultando em trajetórias seguras, mas possivelmente menos eficientes.
- **Expected-SARSA**: Equilíbrio entre estabilidade e desempenho, ideal para aplicações que exigem previsibilidade.

Além disso, a análise dos parâmetros fundamentais evidencia como pequenas variações podem impactar o desempenho do agente, reforçando a importância de um ajuste criterioso desses hiperparâmetros para otimizar o aprendizado.
