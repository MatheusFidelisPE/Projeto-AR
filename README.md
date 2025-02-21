## Introdução

O aprendizado por reforço é uma abordagem poderosa para treinar agentes a tomarem decisões em ambientes desconhecidos. Este artigo explora três algoritmos clássicos dessa abordagem: Q-Learning, SARSA e Expected-SARSA. Utilizando o ambiente *FrozenLake-v1* do OpenAI Gym, analisamos como diferentes parâmetros fundamentais afetam o desempenho do agente ao longo dos episódios de treinamento e teste.

## Conceitos Fundamentais

Os três algoritmos apresentados pertencem à família *Temporal-Difference Learning*, que atualiza a função de valor Q(s, a) com base em interações individuais do agente com o ambiente. A fórmula básica para atualização da tabela Q é:

$Q(s,a) \ &larr; Q(s,a) + \alpha \times (r + \gamma V(s') - Q(s,a))$

Onde:
- \( $\alpha$ \) é a taxa de aprendizado.
- \( $\gamma$ \) é o fator de desconto do futuro.
- $( r )$ é a recompensa obtida ao executar a ação $( a )$ no estado $( s )$.
- $V(s')$ é a estimativa do valor do próximo estado $( s' )$.

Cada algoritmo difere na maneira como estima \( V(s') \), afetando seu comportamento de aprendizado.

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

### Resultados do Q-Learning

Após o treinamento, avaliamos a Q-table e observamos uma melhora na tomada de decisão do agente. A média das recompensas nos últimos 20 episódios indica se o aprendizado foi eficaz. Os gráficos gerados mostram a evolução do desempenho do agente ao longo dos episódios, considerando as variações nos parâmetros analisados.

![image](https://github.com/user-attachments/assets/4207fd32-5260-4345-91bc-c8f9d237ef1e)


## SARSA

O **SARSA** é um algoritmo *on-policy*, o que significa que ele atualiza a Q-table considerando a próxima ação realmente escolhida pelo agente, ao invés da melhor ação teórica. Sua fórmula de atualização é:

$V(s') = Q(s', a')$

Isso faz com que SARSA tenha um comportamento mais conservador que o Q-Learning, pois a estimativa de futuro é baseada na política de treinamento.

### Resultados do SARSA

Executamos o SARSA no mesmo ambiente e comparamos os resultados com o Q-Learning. O agente tende a aprender de maneira mais segura, evitando caminhos arriscados. Os gráficos revelam a diferença no padrão de aprendizado e a influência dos parâmetros ajustados no desempenho final.

Sarsa personalizado para treinar e testar
Experimento 1: RaceTrack
a) Alterando o LR
![image](https://drive.google.com/file/d/1BV2i4yX62prVsticQLH6b1A6J6rQ2xbn/view?usp=drive_link)

b) Alterando o EPSILON

Experimento 2: Fronzen Lake
a) Alterando o Alfa

b) Alterando o EPSILON


Experimento 3: Cliff Walking
a) Alterando o Alfa

b) Alterando o EPSILON


Experimento 4: Taxi


## Expected-SARSA

O **Expected-SARSA** combina aspectos dos dois algoritmos anteriores. Em vez de usar a melhor ação (como no Q-Learning) ou a ação escolhida (como no SARSA), ele calcula um valor esperado com base na distribuição de probabilidades da política atual:

$V(s') = \sum_{a'} \pi(a'|s') Q(s',a')$

Isso resulta em uma estratégia de aprendizado mais estável, pois leva em conta todas as ações possíveis, ponderadas por suas probabilidades.

### Resultados do Expected-SARSA

A execução do Expected-SARSA no mesmo ambiente mostra um desempenho intermediário entre os dois algoritmos anteriores, proporcionando uma curva de aprendizado mais suave e menos sujeita a flutuações abruptas. A análise detalhada revela como os diferentes valores dos parâmetros afetam a estabilidade e a eficiência do aprendizado.

## Conclusão

Comparando os três algoritmos, podemos destacar:
- **Q-Learning**: Melhor desempenho a longo prazo, mas pode ser mais instável no aprendizado inicial.
- **SARSA**: Mais conservador, resultando em trajetórias seguras, mas possivelmente menos eficientes.
- **Expected-SARSA**: Equilíbrio entre estabilidade e desempenho, ideal para aplicações que exigem previsibilidade.

Além disso, a análise dos parâmetros fundamentais evidencia como pequenas variações podem impactar o desempenho do agente, reforçando a importância de um ajuste criterioso desses hiperparâmetros para otimizar o aprendizado.
