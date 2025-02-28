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
- **Taxa de aprendizado**: Regula a velocidade com que o agente aprende a partir das experiências.

O objetivo principal desta investigação é compreender como pequenas variações nesses parâmetros afetam o desempenho do agente ao longo do treinamento, buscando otimizar sua performance.

#### Definição dos Hiperparâmetros
Foram selecionados três conjuntos de valores para os hiperparâmetros:

1.Número de Passos (nstep): [1, 2, 4, 8, 16].
* O valor de nstep define quantos passos à frente o algoritmo considera para atualizar a função de valor. Valores menores tendem a ser mais enviesados, enquanto valores maiores podem introduzir mais variância.

2. Taxa de Aprendizado (lr): [0, 0.2, 0.4, 0.6, 0.8, 1].
* A taxa de aprendizado controla o tamanho das atualizações na função de valor. Valores muito altos podem causar instabilidade, enquanto valores muito baixos podem tornar o aprendizado lento.

3. Taxa de Exploração (epsilon): [0, 0.2, 0.4, 0.6, 0.8, 1].
* A taxa de exploração determina a probabilidade de o agente explorar novas ações em vez de seguir a política atual. Valores mais altos incentivam mais exploração, enquanto valores mais baixos evitam exploração.

#### Estrutura do Experimento
O experimento foi dividido em etapas claras para garantir uma análise robusta e confiável:

1. Combinação de Hiperparâmetros:
* Para cada valor de nstep, foram testados todos os valores de lr. Por exemplo, para nstep = 1, o algoritmo foi treinado com lr = 0, depois lr = 0.2, e assim por diante, até lr = 1.

2. Treinamento por Episódios:
* Cada combinação de nstep e lr foi treinada por um número fixo de episódios. Ao final de cada treinamento, a média das últimas 100 execuções foi calculada para avaliar o desempenho do modelo. Essa métrica foi escolhida para capturar a estabilidade do aprendizado ao longo do tempo.

3. Redução da Aleatoriedade:
* Para garantir que os resultados não fossem influenciados por flutuações aleatórias, cada combinação de nstep e lr foi executada 10 vezes. A média dessas execuções foi usada como métrica final, proporcionando uma avaliação mais confiável do desempenho do algoritmo.

| nstep | lr  |  
|-------|-----| 
| 1     | 0   | 
| 1     | 0.2   | 
| 1     | 0.4   | 
| 1     | 0.6   | 
| 1     | 0.8 | 
| ...   | ... | 
| 16    | 0  | 
| 16    | 0.2   | 
| 16    | 0.4   | 
| 16    | 0.6   | 
| 16    | 0.8   | 

Tabela 1: Tabela de experimentos

### Execução e Avaliação

Após o treinamento, o agente executará M episódios para testar sua capacidade de tomada de decisão. Durante essa fase, serão registradas as recompensas obtidas para cada episódio.

Para minimizar efeitos estocásticos e garantir resultados mais confiáveis, repetiremos essa execução X vezes, calculando a média das recompensas e o desvio padrão. Isso permitirá avaliar a estabilidade e a eficácia do aprendizado do agente.


## SARSA N STEPS

O **SARSA** é um algoritmo *on-policy*, o que significa que ele atualiza a Q-table considerando a próxima ação realmente escolhida pelo agente, ao invés da melhor ação teórica. Sua fórmula de atualização é:

$V(s') = Q(s', a')$

Isso faz com que SARSA tenha um comportamento mais conservador que o Q-Learning, pois a estimativa de futuro é baseada na política de treinamento.

### Resultados do SARSA

Executamos o SARSA no mesmo ambiente e comparamos os resultados com o Q-Learning. O agente tende a aprender de maneira mais segura, evitando caminhos arriscados. Os gráficos revelam a diferença no padrão de aprendizado e a influência dos parâmetros ajustados no desempenho final.

Sarsa personalizado para treinar e testar

# Experimentos

### Experimento 1: Alterando o LR com o ambiente do *RaceTrack*

![Image](https://github.com/user-attachments/assets/f4f74c9e-530a-4221-962d-9bd14d157800)
## 1. Taxas de aprendizado mais altas melhoram o desempenho
   - **Aumento do retorno médio**  
     Conforme a taxa de aprendizado (*learning rate* ou *lr*) aumenta, o retorno médio também cresce. Isso significa que o agente consegue aprender de forma mais rápida e eficiente.  
   - **Relação com Alfa**  
     O parâmetro Alfa, que está diretamente relacionado à taxa de aprendizado, influencia diretamente a velocidade de aprendizado. Quanto maior o valor de Alfa, mais rápido o agente converge para uma política ótima.  
   - **Impacto no treinamento**  
     Taxas de aprendizado mais altas permitem que o agente ajuste seus parâmetros de forma mais ágil, o que é especialmente útil em fases iniciais do treinamento.  

---

## 2. Valores mais altos de nstep resultam em melhor desempenho
   - **Definição de nstep**  
     O parâmetro *nstep* refere-se ao número de passos utilizados para atualizar a política de aprendizado. Valores mais altos significam que o agente considera um horizonte mais amplo de informações antes de realizar uma atualização.  
   - **Estabilidade e eficiência**  
     Utilizar um valor maior de *nstep* favorece a estabilidade do aprendizado, pois reduz a variância nas atualizações da política. Isso resulta em um processo de treinamento mais consistente e confiável.  
   - **Melhoria no desempenho**  
     Com um *nstep* maior, o agente consegue capturar relações de longo prazo entre ações e recompensas, o que leva a políticas mais robustas e eficientes.  

---

## 3. Diminuição da diferença entre curvas em lr alto
   - **Estabilização das curvas**  
     Para taxas de aprendizado (*lr*) muito altas, as curvas de desempenho tendem a se aproximar e estabilizar. Isso indica que o impacto do aumento da taxa de aprendizado diminui após um certo ponto.  
   - **Limite prático**  
     O efeito positivo de aumentar a taxa de aprendizado parece ter um limite prático. Após esse ponto, incrementos adicionais em *lr* não resultam em melhorias significativas no desempenho.  
   - **Possíveis razões**  
     Esse comportamento pode ser explicado por fatores como saturação na capacidade de aprendizado do modelo ou instabilidades causadas por taxas de aprendizado excessivamente altas. 
---

### Experimento 2: Alterando o EPSILON com o ambiente do *RaceTrack*

![Image](https://github.com/user-attachments/assets/9527c224-e594-46c6-bdb3-a59c7ef11440)

## 1. Valores mais altos de Épsilon prejudicam o desempenho
   - **Redução do retorno médio**  
     Conforme o valor de Épsilon aumenta, o retorno médio torna-se mais negativo. Isso ocorre porque um Épsilon maior prioriza a exploração, fazendo com que o agente escolha ações aleatórias com maior frequência.  
   - **Impacto na exploração**  
     A exploração excessiva impede que o agente utilize efetivamente o conhecimento já adquirido, reduzindo sua capacidade de maximizar recompensas ao longo do tempo.  
   - **Consequências**  
     O agente acaba desperdiçando oportunidades de aproveitar ações ótimas já aprendidas, resultando em um desempenho inferior.  

---

## 2. N-step maior leva a melhor desempenho
   - **Eficiência no aprendizado**  
     Valores mais altos de *nstep* estão associados a um aprendizado mais eficiente, pois permitem que o agente considere um horizonte mais amplo de informações antes de atualizar sua política.  
   - **Melhoria na recompensa acumulada**  
     Com um *nstep* maior, o agente consegue capturar relações de longo prazo entre ações e recompensas, resultando em uma recompensa acumulada mais alta.  
   - **Estabilidade do treinamento**  
     O uso de um *nstep* maior também contribui para a estabilidade do processo de aprendizado, reduzindo a variância nas atualizações da política.  

---

## 3. Para valores baixos de Épsilon, as curvas estão mais separadas
   - **Maior impacto do nstep**  
     Quando o valor de Épsilon é baixo, o impacto do parâmetro *nstep* torna-se mais evidente. Isso ocorre porque a exploração reduzida permite que as diferenças na eficiência do aprendizado sejam mais visíveis.  
   - **Separação das curvas**  
     As curvas de desempenho para diferentes valores de *nstep* ficam mais distantes, indicando que a escolha de um *nstep* adequado é crucial quando a exploração é limitada.  

---

### Experimento 3: Alterando o Alfa com o ambiente do *Fronzen Lake*

![Image](https://github.com/user-attachments/assets/fcea98d9-84b6-4f01-91af-04b0b900d21d)

## 1. Valores intermediários de Alfa (entre 0.4 e 0.6) são ideais
   - **Pico do retorno médio**  
     O retorno médio atinge seu valor máximo quando Alfa está na faixa de 0.4 a 0.6. Isso indica que essa taxa de aprendizado é a mais eficaz para o agente melhorar sua política de forma consistente.  
   - **Equilíbrio entre aprendizado e estabilidade**  
     Nessa faixa, o agente consegue atualizar suas ações de maneira eficiente, sem sofrer grandes oscilações ou instabilidades no processo de aprendizado.  
   - **Resultados otimizados**  
     A escolha de um Alfa intermediário permite um equilíbrio ideal entre explorar novas ações e aproveitar o conhecimento já adquirido.  

---

## 2. Alfa muito baixo (próximo de 0) prejudica o aprendizado
   - **Aprendizado lento**  
     Quando Alfa está próximo de 0, o agente demora mais para atualizar os valores das ações, resultando em um processo de aprendizado extremamente lento.  
   - **Falta de adaptação**  
     Com uma taxa de aprendizado muito baixa, o agente não consegue se adaptar rapidamente às mudanças no ambiente, o que prejudica sua capacidade de melhorar a política ao longo do tempo.  
   - **Impacto negativo**  
     O desempenho do agente fica comprometido, pois ele não consegue aproveitar as recompensas obtidas para ajustar suas ações de forma eficiente.  

---

## 3. Alfa muito alto (próximo de 1) pode causar instabilidade
   - **Comportamento instável**  
     Quando Alfa se aproxima de 1, o agente pode se tornar instável, reagindo de forma exagerada às novas recompensas e ignorando o conhecimento acumulado anteriormente.  
   - **Oscilações no aprendizado**  
     A alta taxa de aprendizado faz com que o agente mude sua política de maneira abrupta, resultando em grandes variações no desempenho e dificultando a convergência para uma solução ótima.  
   - **Desperdício de aprendizado passado**  
     O agente não consegue aproveitar efetivamente as informações já aprendidas, o que reduz sua eficiência e confiabilidade.  

---

## 4. Nstep menor tem vantagem no Frozen Lake
   - **Desempenho superior**  
     No ambiente Frozen Lake, um valor menor de *nstep* apresentou um desempenho melhor em comparação com outros ambientes mais complexos.  
   - **Características do ambiente**  
     O Frozen Lake é um ambiente simples e determinístico, onde um *nstep* menor permite que o agente atualize sua política de forma mais rápida e direta, sem a necessidade de considerar um horizonte amplo de informações.  
   - **Diferença em relação a ambientes complexos**  
     Em ambientes mais complexos, um *nstep* maior geralmente é mais eficiente, mas no Frozen Lake, a simplicidade do ambiente favorece uma abordagem mais imediatista.  

---

### Experimento 4: Alterando o EPSILON com o ambiente do *Fronzen Lake*

![Image](https://github.com/user-attachments/assets/2b313868-5069-4611-9cc6-987244a5126c)

## 1. Compensação Exploração-Explotação
   - **Para nstep menores (1 e 2)**  
     Uma exploração moderada (em torno de *epsilon = 0.2*) é vantajosa. Isso permite que o agente explore o ambiente de forma eficiente, descobrindo caminhos melhores sem comprometer o aproveitamento do conhecimento já adquirido.  
   - **Para nstep maiores (4 e 16)**  
     A exploração excessiva se torna prejudicial. O agente se beneficia mais de uma fase inicial de exploração, seguida por uma fase de explotação, onde ele utiliza o conhecimento já aprendido para maximizar as recompensas.  

---

## 2. Impacto do nstep
   - **Melhor desempenho geral**  
     Os valores de *nstep = 1* e *nstep = 16* apresentam os melhores resultados, mas com comportamentos distintos em relação ao *epsilon*. Enquanto *nstep = 1* se beneficia de uma exploração moderada, *nstep = 16* exige um equilíbrio mais cuidadoso entre exploração e explotação.  
   - **Pior desempenho**  
     O valor *nstep = 8* demonstra o desempenho mais fraco, sugerindo que esse valor pode não ser adequado para o ambiente em questão.  
   - **Escolha do nstep ideal**  
     O valor ideal de *nstep* depende do equilíbrio entre a capacidade do agente de aprender com experiências recentes e a necessidade de considerar recompensas futuras em um horizonte mais amplo.  

---

## 3. Convergência
   - **Comportamento das curvas de recompensa**  
     As curvas de recompensa mostram que o algoritmo SARSA converge para diferentes níveis de desempenho, dependendo dos hiperparâmetros configurados, como *nstep* e *epsilon*.  
   - **Importância da escolha de hiperparâmetros**  
     A seleção adequada de *nstep* e *epsilon* é fundamental para garantir um bom desempenho do agente. Uma combinação bem ajustada desses parâmetros permite que o algoritmo atinja um equilíbrio ideal entre exploração e explotação, maximizando as recompensas ao longo do tempo.  

---

### Experimento 5: Alterando o Alfa com o ambiente do *Taxi*

![Image](https://github.com/user-attachments/assets/b4605edd-351e-48f0-bda0-4e94324e4f6d)

## 1. Taxa de Aprendizado Alta é Fundamental
   - **Importância de uma taxa de aprendizado elevada**  
     No ambiente Taxi, uma taxa de aprendizado (*alfa*) alta é crucial para que o agente aprenda de maneira eficiente e rápida.  
   - **Exigência do ambiente**  
     O ambiente Taxi demanda um aprendizado ágil e adaptativo, onde o agente precisa ajustar suas ações rapidamente com base nas recompensas recebidas.  
   - **Resultado prático**  
     Taxas de aprendizado mais altas permitem que o agente atualize sua política de forma dinâmica, garantindo um desempenho melhor em um curto espaço de tempo.  

---

## 2. N-Step Menores São Mais Eficientes
   - **Vantagem de nstep menores**  
     Valores menores de *nstep* (como 1 e 2) proporcionam um aprendizado mais rápido e um desempenho ligeiramente superior no ambiente Taxi.  
   - **Benefício de experiências recentes**  
     Isso indica que o agente se beneficia mais ao aprender com experiências imediatas, em vez de considerar um horizonte amplo de informações.  
   - **Eficiência no ambiente Taxi**  
     A natureza do ambiente Taxi favorece uma abordagem mais direta, onde o foco em ações recentes é mais eficaz do que planejamentos de longo prazo.  

---

## 3. Convergência Rápida
   - **Convergência para um platô de desempenho**  
     O algoritmo SARSA converge rapidamente para um nível estável de desempenho a partir de *alfa = 0.6*.  
   - **Simplicidade do ambiente**  
     Esse comportamento sugere que o ambiente Taxi é relativamente simples, permitindo que o algoritmo encontre uma política ótima em um tempo razoável.  
   - **Eficácia do algoritmo**  
     A rápida convergência demonstra que o SARSA é adequado para ambientes como o Taxi, onde decisões rápidas e adaptativas são essenciais para o sucesso.  

---

### Experimento 6: Alterando o EPSILON com o ambiente do *Taxi*

![Image](https://github.com/user-attachments/assets/218ec2a2-a759-4b9d-bc86-7e0bfba8a3b3)

## 1. Exploração Mínima é Ideal
   - **Melhor desempenho com baixo epsilon**  
     No ambiente Taxi, o desempenho ótimo é alcançado com uma exploração mínima, ou seja, valores baixos de *epsilon*.  
   - **Ambiente determinístico**  
     Isso sugere que o ambiente Taxi é relativamente previsível e determinístico, onde o agente obtém melhores resultados ao seguir a política já aprendida, em vez de explorar ações aleatórias.  
   - **Benefício da explotação**  
     Com um *epsilon* baixo, o agente prioriza a explotação do conhecimento adquirido, resultando em decisões mais eficientes e consistentes.  

---

## 2. N-Step Menores São Mais Estáveis
   - **Estabilidade com nstep menores**  
     Valores menores de *nstep* (como 1, 2 e 4) demonstram um comportamento mais estável em relação à variação do *epsilon*.  
   - **Aprendizado com experiências recentes**  
     Isso indica que, no ambiente Taxi, o agente se beneficia mais ao aprender com experiências imediatas, em vez de considerar um horizonte amplo de informações.  
   - **Eficiência e consistência**  
     A estabilidade observada com *nstep* menores reforça a ideia de que o ambiente Taxi favorece uma abordagem mais direta e focada em recompensas de curto prazo.  

---

## 3. N-Step 16 tem comportamento diferente
   - **Comportamento distinto do nstep 16**  
     O valor *nstep = 16* apresenta um comportamento diferente em comparação com os valores menores. Isso pode indicar que, para passos mais longos, o agente ainda está em fase de aprendizado no início da variação do *epsilon*.  
   - **Aprendizado em horizonte amplo**  
     Com um *nstep* maior, o agente considera um horizonte mais amplo de informações, o que pode levar a um aprendizado mais lento e menos estável no início.  
   - **Impacto no desempenho**  
     Esse comportamento sugere que, no ambiente Taxi, valores maiores de *nstep* podem não ser tão eficientes quanto valores menores, especialmente em fases iniciais de exploração. 


# Conclusão

Os experimentos realizados com os algoritmos **Q-Learning**, **SARSA** e **Expected-SARSA** em diferentes ambientes (**RaceTrack**, **Frozen Lake** e **Taxi**) evidenciaram a relevância da configuração adequada dos hiperparâmetros, como **taxa de aprendizado (Alfa)**, **fator de exploração (Épsilon)** e **nstep**, para o desempenho dos algoritmos. Abaixo, são apresentados os principais insights e conclusões:

#### 1. **Taxa de Aprendizado (Alfa)**
   - **Efeito Geral**: Taxas de aprendizado mais elevadas geralmente melhoram o desempenho, principalmente em ambientes que demandam decisões ágeis e adaptativas, como o **Taxi**. Contudo, valores extremamente altos (próximos de 1) podem gerar instabilidade, enquanto valores muito baixos (próximos de 0) resultam em um aprendizado lento.
   - **Faixa Ideal**: No ambiente **Frozen Lake**, valores intermediários de Alfa (entre 0.4 e 0.6) mostraram-se mais eficazes, equilibrando aprendizado e estabilidade. Já no **Taxi**, taxas mais altas (a partir de 0.6) foram essenciais para uma convergência rápida.

#### 2. **Fator de Exploração (Épsilon)**
   - **Exploração vs. Explotação**: Valores elevados de Épsilon prejudicam o desempenho, especialmente em ambientes determinísticos como o **Taxi**, onde a exploração mínima (Épsilon baixo) é ideal. Por outro lado, em ambientes mais complexos, como o **RaceTrack**, uma exploração moderada (em torno de 0.2) pode ser vantajosa, principalmente para valores menores de nstep.
   - **Impacto no Aprendizado**: A exploração excessiva (Épsilon alto) resulta em um retorno médio mais negativo, pois o agente prioriza ações aleatórias em vez de aproveitar o conhecimento já adquirido.

#### 3. **Nstep**
   - **Ambientes Simples vs. Complexos**: Em ambientes simples e determinísticos, como o **Frozen Lake** e o **Taxi**, valores menores de nstep (1 e 2) mostraram-se mais eficientes, permitindo um aprendizado rápido e estável. Já em ambientes mais complexos, como o **RaceTrack**, valores maiores de nstep (16) foram mais eficazes, pois permitem ao agente considerar um horizonte mais amplo de informações.
   - **Estabilidade e Desempenho**: Valores maiores de nstep tendem a reduzir a variância nas atualizações da política, resultando em um aprendizado mais estável. No entanto, em alguns casos, como no **Taxi**, nstep maiores (16) podem levar a um aprendizado mais lento e menos estável no início.

#### 4. **Comparação entre Algoritmos**
   - **SARSA**: Demonstrou ser eficiente em ambientes como o **Taxi**, onde a convergência rápida é crucial. O algoritmo se beneficia de taxas de aprendizado altas e valores menores de nstep, especialmente em ambientes determinísticos.
   - **Q-Learning e Expected-SARSA**: Embora não tenham sido explicitamente comparados nos experimentos, espera-se que o **Expected-SARSA** apresente um desempenho mais estável em relação ao **Q-Learning**, principalmente em ambientes estocásticos, devido à sua capacidade de reduzir a variância nas atualizações da política.

#### 5. **Convergência e Estabilidade**
   - **Convergência Rápida**: Em ambientes simples, como o **Taxi** e o **Frozen Lake**, os algoritmos convergem rapidamente para um platô de desempenho, especialmente com taxas de aprendizado adequadas e valores menores de nstep.
   - **Estabilidade**: A estabilidade do aprendizado é influenciada pela combinação de Alfa, Épsilon e nstep. Valores intermediários de Alfa e Épsilon, juntamente com nstep adequado ao ambiente, são fundamentais para garantir um aprendizado estável e eficiente.

### Considerações Finais
Os resultados destacam a importância de ajustar os hiperparâmetros de acordo com as características do ambiente. Em ambientes simples e determinísticos, como o **Frozen Lake** e o **Taxi**, valores menores de nstep e taxas de aprendizado mais altas são ideais. Já em ambientes mais complexos, como o **RaceTrack**, valores maiores de nstep e uma exploração moderada são mais eficazes. Além disso, a escolha do algoritmo (SARSA, Q-Learning ou Expected-SARSA) deve considerar a natureza do ambiente, com o **Expected-SARSA** sendo preferível em cenários estocásticos devido à sua maior estabilidade.
Em resumo, a otimização dos hiperparâmetros é crucial para maximizar o desempenho dos algoritmos de aprendizado por reforço, e a escolha adequada depende diretamente das características do ambiente e dos objetivos do treinamento.
