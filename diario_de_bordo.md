

# Trabalho de Ciencia de Dados + Inteligencia Artificial

**2026-09-12**

O meu objetivo é pegar algum dataset de relacionado a venda de produtos ou algo do tipo.
Até agora o que achei interessante é um sobre o risco de pendencias financeiras de pessoas o [Buy Now Pay Later (BNPL) Default Risk](./studing/BNPL_Financial_Default_Risk_Dataset.csv) - [link kaggle](https://www.kaggle.com/datasets/itzzomkar/buy-now-pay-later-bnpl-default-risk/data) e o outro de PDV de vendas de uma padaria [The Bread Basket](./studing/bread%20basket.csv) - [link kaggle](https://www.kaggle.com/datasets/mittalvasu95/the-bread-basket).

Estou em conversa com o gemini para buscar um dataset interessante para ter algum conhecimento familiaridade posterior [link chat gemini](https://share.gemini.google/mrGdy6mdUZ0e).

Tive uma ideia, estava buscando informações sobre vendas para utilizar no futuro e então lembrei que tenho dados no sistema que trabalho de vendas porem ainda não cheguei a refatorar esse ponto. Sei que contem algumas incosistencias mas pela diversão vou tentar trazer informações das vendas da expointer 2026 e analisar a viabilidade deles para uma predição com base no horario e dia e conjunto de itens a serem vendidos.

**2026-09-14**

Realizei a busca do dataset via [SQL](./get_dataset.sql) e iniciei revisão do conteúdo de Ciencia de Dados fazendo testando alguns comandos


**2026-09-20*

Não realizei anotações mas durante a semana tirei 1h para analisar e apresnder um pouco sobre as bibliotecas pandas, sklearn, numpy, seaborn, etc.
Hoje me deparo como um dia antes do trabalho e preciso otimizar o tempo para conseguir atigir um bom resultado com auxilio da IA.
Fiz um monte de testes nesses ultimos dias e percebi que tenho muitas informações para tratar nesse dataset mas bora lá.


Reorganização estrutural e modular do notebook [`eda_v2.ipynb`](./eda_v2.ipynb) em 7 seções lógicas:
1. Setup do Ambiente e Configurações Globais (com constantes em padrão PEP 8: `RANDOM_STATE = 42`, `TIMEZONE`, etc.);
2. Carga, Tipagem Primária e Auditoria Financeira Bruta (preservando `nome_produto_bruto` para auditoria e validando a invariante de descontos);
3. Diagnóstico Exploratório de Inconsistências (tabela de dispersão de preços e quantidades, destacando produtos de embalagens múltiplas como caixas e pacotes);
4. Agrupamento Semântico e Consolidação de Produtos (isolamento do dicionário de mapeamento em célula dedicada);
5. Validação Estatística Pós-Agrupamento (avaliação do comportamento do desvio padrão pós-replace);
6. Engenharia de Atributos Contextuais e de Cesta (features temporais `turno`, `dia_horario`, repetições da cesta `qnt_repeticoes` e exportação para `base.csv`);
7. Sandbox Experimental de Agrupamento Não Supervisionado (prototipagem do K-Means e PCA desacoplados do pré-processamento).

*Decisão arquitetural adicional:* Descarte da coluna `pedido_vendedor` (e `item_pedido_user_id`) logo na Seção 2 do pipeline. Motivo: nas feiras, os atendentes se alternam no mesmo terminal com login único/compartilhado, tornando o dado de autoria do vendedor corrompido, inconsistente e irrelevante para a clusterização de perfis de cestas no checkout. Com isso, eliminou-se também o `LabelBinarizer` na Seção 6 e a `base.csv` passou a ter 16 colunas limpas.

*Adição da Seção 4 — Validação de Integridade (Data Quality):*
Implementação de auditoria transacional comparando a soma dos itens de cada pedido ($\sum \text{quantidade} \times \text{preço\_unitário}$) com o subtotal do cabeçalho (`pedido_subtotal`), via `np.isclose(..., atol=0.01)`.
- **Resultado:** 2.318 pedidos válidos (94,34%) vs. 139 pedidos divergentes (5,66%).
- **Exibição dos Dados Divergentes:** Apresentação da tabela de pedidos divergentes com deltas, tabela transacional com os 185 itens afetados e ranking de SKUs mais frequentes (*Rapadura Assada Pacote x6*, *Alfajor Preto Pacote x4* e *Rapadura Melado Pacote x3*), comprovando que o alto desvio decorre do apontamento do valor da embalagem fechada no preço unitário.

*Implementação do Mecanismo de Proporção com Fallback Condicional:*
Construção do motor de contingência não-destrutivo para sanar as inconsistências de embalagem:
- Criação das colunas adicionais `fator_k`, `quantidade_ajustada`, `preco_unitario_ajustado` e `valor_item_ajustado`.
- Dupla conferência contra o `pedido_subtotal` com tolerância de R$ 0,01.
- Classificação e consolidação: `ORIGINAL_VALIDO` (2.318 pedidos / 94,34% mantendo dados brutos de origem), `AJUSTADO_PROPORCAO` (139 pedidos / 5,66% sanados pela proporção) e `QUARENTENA_RESIDUAL` (0 casos).
- Taxa de conformidade consolidada atingiu 100,00% com maior delta igual a 0.000000. As grandezas `quantidade_final` e `preco_unitario_final` agora alimentam o K-Means com total coerência métrica.

**2026-09-21**

*Refatoração Modular da Seção 4.6 (Mecanismo de Validação e Contingência):*
A Seção 4.6 do notebook [`eda_v2.ipynb`](./eda_v2.ipynb) foi refatorada e desacoplada em 5 subetapas conceituais bem definidas, eliminando a densidade do bloco único anterior e estabelecendo uma convenção clara de escopos de dados:
1. **4.6.1 Extração do Fator de Embalagem ($k$) e Hipótese Proporcional:** Trabalho isolado no DataFrame temporário `df_ajustes_temp` (preservando o `df` principal inalterado).
2. **4.6.2 Auditoria de Dupla Checagem (Double-Check):** Agregação no nível de pedido em `df_auditoria_fallback_audit` com prova real matemática e classificação determinística dos status de validação (`ORIGINAL_VALIDO`, `AJUSTADO_PROPORCAO`, `QUARENTENA_RESIDUAL`).
3. **4.6.3 Consolidação no Dataset Principal:** Integração das grandezas auditadas e metadados de linhagem diretamente no DataFrame corporativo oficial (`df`).
4. **4.6.4 Isolamento da Quarentena Residual:** Governança preventiva isolando transações anômalas em `df_quarentena_audit` (0 registros / 100% de conformidade contábil).
5. **4.6.5 Demonstração de Prova Real (Antes vs. Depois):** Comparação tabular transparente de pedido ajustado demonstrando delta contábil zero.
6. **4.6.6 Auditoria Global de Integridade Contábil (Validação Pré-Seção 5):** Prova real exaustiva sobre 100% dos 2.457 pedidos e 3.113 itens, comprovando matematicamente que $\sum (Q_{\text{ajustada}} \times P_{\text{ajustado}}) \equiv \text{pedido\_subtotal}$ nos 139 pedidos corrigidos e $\sum (Q_{\text{final}} \times P_{\text{final}}) \equiv \text{pedido\_subtotal}$ no universo total (100,00% de conformidade, $\Delta \text{ máx} = 0.000000$, assertividade fail-fast e eliminação de colunas transitórias).

*Aprimoramento do Cálculo Proporcional (Eliminação de Magic Numbers):*
- Substituição da constante empírica `preco >= 10.0` por um modelo estatístico robusto baseado na proximidade à escala central típica do SKU: $|\frac{P}{k} - \tilde{P}_{\text{SKU}}| < |P - \tilde{P}_{\text{SKU}}|$.
- **Preservação de flutuações unitárias:** Variações naturais no preço de venda unitário (ex.: doces entre R$ 5,00 e R$ 7,00) são respeitadas e mantidas.
- **Harmonização de Fardos Fechados:** 11 pedidos onde clientes compraram fardos unitários (ex.: 1 pacote de alfajor x4 por R$ 20) foram harmonizados para unidades físicas de consumo ($Q_{\text{final}} = Q \times k$ e $P_{\text{final}} = P / k$), preservando 100% da integridade do subtotal e reduzindo o desvio padrão dos preços de todos os produtos para `0.0000`.
- Inclusão do metadado `tipo_escala` em `base.csv` para rastreabilidade de linhagem.

*Correção Precoce de Nomenclatura no Cadastro do PDV (Rapadura Grão Moído Pacote x3):*
- Identificou-se que o produto `Rapadura de Melado Grão Moído (Pacote x1)` foi cadastrado com erro de digitação no PDV, correspondendo na realidade a um fardo com 3 unidades vendido por R$ 20,00.
- A correção foi posicionada precocemente na Seção 2 (Carga e Sanitização Inicial) na coluna `nome_produto`, preservando `nome_produto_bruto` para auditoria.
- Com isso, a Seção 4 extraiu automaticamente $k = 3$, convertendo as 16 transações para 3 unidades de consumo a R$ 6,67 (estabilizando o desvio padrão em 0.0000 com conservação contábil perfeita de R$ 20,00), e a Seção 5 consolidou o rótulo `RAPADURA_DE_MELADO_GRÃO_MOÍDO_(PACOTE_X3)` no dicionário `mapa_produtos`.

*Saneamento de Registros Espúrios com Quantidade Zerada (Outliers de Digitação no PDV):*
- **Detecção e Auditoria Granular:** Identificaram-se 2 registros espúrios com `quantidade_item == 0` no início do pipeline (Seção 2.1 de `eda_v2.ipynb`):
  1. `ebbed809-e6f4-4b4a-9850-60eaa218e98b` (Expo Afubra): `Alfajor Preto (Caixa x16)` com quantidade 0 (o pedido continha também `Alfajor Preto (Pacote x4)` com quantidade 4 a R$ 5,00, totalizando os R$ 20,00 do subtotal).
  2. `4680d5c5-3262-4e60-829e-5a9f976edaff` (EXPOBENTO): `Rapadura de Melado` com quantidade 0 (o pedido continha também `Rapadura de Melado (Pacote x3)` com quantidade 3 a R$ 6,67, totalizando os R$ 20,00 do subtotal).
- **Prova Real de Não-Impacto:** Como $Q = 0 \implies Q \times P = \text{R\$\,}0,00$, a remoção não afeta o balanço financeiro dos pedidos, que continuam fechando exatamente os R$ 20,00 de subtotal.
- **Preservação Amostral:** Nenhum pedido foi descartado (2.457 pedidos únicos mantidos 100%).
- **Efeito Sanitizador:** Ajuste correto da diversidade da cesta (`qnt_repeticoes` passou de 2 para 1 em ambos os pedidos) e `base.csv` exportado com 3.113 linhas estritamente íntegras.

*Implementação da Seção 8: Agrupamento Não Supervisionado, Diagnóstico Multidimensional e Recomendação (K-Means & PCA):*
A Seção 8 do notebook [`eda_v2.ipynb`](./eda_v2.ipynb) foi completamente refatorada e expandida em 6 subetapas modulares, alinhadas às diretrizes do Prof. Rhauani Fazul e do [Harness de Agentes](./AGENTS.md):
1. **8.1 Preparação, Codificação e Padronização:** Isolamento de instâncias independentes de `LabelEncoder` (`le_metodo`, `le_prod`, `le_turno`) e padronização contínua rigorosa via `StandardScaler` sobre 8 features (`dia_horario`, `prod_code`, `metodo_code`, `turno_code`, `preco_unitario_final`, `quantidade_final`, `valor_final_pedido`, `qnt_repeticoes_num`).
2. **8.2 Otimização Experimental do Hiperparâmetro $K$:**
   - Varredura de $K \in [2, 8]$ calculando simultaneamente Inércia (WCSS / Método do Cotovelo), Coeficiente de Silhueta, Calinski-Harabasz e Davies-Bouldin.
   - **Convergência Matemática:** O índice Calinski-Harabasz atingiu seu ápice absoluto exatamente em $K=5$ (score de 788.31, superando os 732.19 de $K=4$ e 759.68 de $K=6$), coincidindo com o ponto de inflexão e estabilização da inércia.
   - Exportação do painel comparativo em alta resolução: [`reports/figures/01_otimizacao_k_cotovelo_silhueta.png`](./reports/figures/01_otimizacao_k_cotovelo_silhueta.png).
3. **8.3 Ajuste do Modelo Definitivo ($K=5$) e Semântica de Negócio:**
   - Treinamento determinístico `KMeans(n_clusters=5, random_state=42, n_init=10)`.
   - Identificação e batismo dos 5 perfis de clientes do PDV:
     - *C0 (15,8%):* Cuca Matinal (Abertura / Café da manhã da feira às 10h);
     - *C1 (33,4%):* Doces Tradicionais Vespertinos (Rapaduras de Melado a R$ 7,00);
     - *C2 (30,2%):* Padaria Familiar (Cucas Alemãs inteiras de R$ 22,00 para viagem);
     - *C3 (20,1%):* Lanches Rápidos (Alfajores individuais da tarde);
     - *C4 (0,4%):* Atacado e Grandes Encomendas corporativas (Ticket médio > R$ 2.800,00).
   - Exportação do heatmap de centróides (Z-score): [`reports/figures/04_heatmap_centroides_clusters.png`](./reports/figures/04_heatmap_centroides_clusters.png).
4. **8.4 Visualização Multidimensional 2D com PCA e Biplot de Cargas:**
   - Projeção ortogonal retendo 40,5% da variância ($\text{PC}_1$: 23,2% para volume da cesta; $\text{PC}_2$: 17,3% para momento temporal).
   - Centróides projetados com destaque e inclusão dos vetores de carga (*loadings*) do Biplot, revelando a direção das forças de cada feature no plano.
   - Exportação: [`reports/figures/02_pca_2d_clusters_biplot.png`](./reports/figures/02_pca_2d_clusters_biplot.png).
5. **8.5 Projeção Tridimensional PCA ($\text{PC}_1 \times \text{PC}_2 \times \text{PC}_3$ via Axes3D):**
   - Incorporação da 3ª componente elevando a variância explicada acumulada para 54,2%, isolando o eixo de Preço Unitário Final.
   - Exportação: [`reports/figures/03_pca_3d_clusters.png`](./reports/figures/03_pca_3d_clusters.png).
6. **8.6 Motor de Inferência e Simulação de Recomendação no Checkout:**
   - Pipeline de simulação em tempo real para um novo cliente no PDV (compra de 1 alfajor às 16h).
   - Classificação em $C_3$, projeção estelar no mapa PCA e acionamento determinístico de cross-selling ("Leve mais 3 alfajores com desconto progressivo no pacote x4!").
   - Exportação: [`reports/figures/05_simulacao_checkout_recomendacao.png`](./reports/figures/05_simulacao_checkout_recomendacao.png).

*Auditoria Global de Integridade Contábil Consolidada (Pré-Seção 5):*
- Inclusão formal da subseção **4.6.6** em [`eda_v2.ipynb`](./eda_v2.ipynb), posicionada estrategicamente como portão de qualidade (*quality gate*) imediatamente antes do agrupamento semântico da Seção 5.
- **Validação Dupla:**
  1. $\sum (\text{quantidade\_ajustada} \times \text{preco\_unitario\_ajustado}) == \text{pedido\_subtotal}$ nos 139 pedidos com status `AJUSTADO_PROPORCAO` (100,00% de conformidade, 0 erros);
  2. $\sum (\text{quantidade\_final} \times \text{preco\_unitario\_final}) == \text{pedido\_subtotal}$ na totalidade dos 2.457 pedidos únicos do dataset (100,00% de conformidade, $\Delta \text{ máx} = 0.000000$).
- Implementação de asserções duras (`assert`) de *fail-fast* e descarte automático de colunas transitórias de checagem, mantendo a integridade absoluta das 16 colunas exportadas em `base.csv`.

*Exportação do Modelo e CLI Simples de Checkout (Seção 8.7):*
- **Exportação Direta no Notebook (`modelo_checkout.joblib`):**
  - Inclusão da Seção 8.7 em [`eda_v2.ipynb`](./eda_v2.ipynb) salvando de forma concisa e direta o dicionário com o modelo `kmeans` ($K=5$), o escalonador `scaler` (`StandardScaler`), os dicionários de classes dos `LabelEncoder`s, a lista de features, os rótulos de negócio (`cluster_labels`), as regras de cross-selling (`regras_recomendacao`) e o dicionário semântico (`mapa_produtos`).
  - Arquivo consolidado e leve gerado na raiz do projeto (`modelo_checkout.joblib`).
- **Interface Simples de Linha de Comando ([`cli.py`](./cli.py)):**
  - Implementação de um script direto, sequencial e autoexplicativo na raiz do repositório (~80 linhas), sem classes complexas ou microsserviços pesados.
  - Carrega o arquivo `modelo_checkout.joblib`, aplica a mesma lógica de higienização e codificação segura, e aciona o K-Means para exibir o cluster e a recomendação no terminal.
  - **Eliminação de Valores Default e Validação Completa de Entradas:**
    - Removidos todos os valores padrão (`default`) dos argumentos do CLI.
    - Exigência estrita de fornecimento explícito dos 6 parâmetros transacionais: `--produto`, `--hora` (0 a 23), `--preco` (> 0), `--quantidade` (> 0), `--metodo` (`Dinheiro`, `Cartão`, `Pix`) e `--itens` (cesta $\ge 1$).
    - Caso qualquer parâmetro seja omitido, a execução é interrompida listando os campos ausentes e exibindo um exemplo formatado de comando válido.
  - Testado e validado com sucesso:
    - Alfajor às 16h $\rightarrow$ `[3] C3: Lanche Rápido (Alfajor)` com sugestão do pacote x4;
    - Cuca Alemã às 10h $\rightarrow$ `[0] C0: Cuca Matinal (Abertura)` com sugestão para café da manhã;
    - Bloqueio com erro e sugestão em caso de typos (ex.: `alfajo` $\rightarrow$ sugestão de `ALFAJOR`);
    - Bloqueio com erro informativo ao omitir parâmetros (ex.: `python cli.py`).
- **Roteiro Automatizado de Apresentação e Demonstração Prática ([`demo.sh`](./demo.sh)):**
  - Criação de script Bash executável com formatação ANSI colorida e narrativa de negócio estruturada em 8 cenários.
  - Cobre exaustivamente a ativação dos 5 clusters identificados ($C_0$ a $C_4$), a consulta de catálogo e a demonstração ao vivo dos guardrails (bloqueio de omissão, detecção de typos via `difflib` e rejeição de formas de pagamento não homologadas).
  - Suporta modo interativo (com pausas entre casos para explicação oral) e modo contínuo (`./demo.sh --auto`). Validação executada com 100% de sucesso.

*Reestruturação Metodológica: Análise Preliminar de Cargas Fatoriais e Semântica Antes dos Plots 2D e 3D (Seções 8.4 e 8.5 de `eda_v2.ipynb`):*
- **Fluxo Conceitual Padronizado:** Adoção rigorosa do padrão *Fundamentação Estatística (Tabela de Cargas) $\rightarrow$ Gráfico Diagnóstico de Barras $\rightarrow$ Síntese Semântica de Negócio $\rightarrow$ Auditoria Empírica $\rightarrow$ Projeção Visual (Dispersão)* antes de qualquer scatter plot multidimensional.
- **Projeção 2D (Seção 8.4):**
  - Cargas e contribuições de $PC_1$ e $PC_2$ apresentadas na Tabela 8.4.1 e Gráfico de Barras Divergentes [`reports/figures/02b_pca_loadings_barplots.png`](./reports/figures/02b_pca_loadings_barplots.png).
  - **Eliminação de Redundâncias (8.4.2):** Substituição da repetição de números da tabela por uma **Síntese Operacional de Negócio** orientada a decisões do balcão (Eixo X = porte da compra: avulso vs atacado; Eixo Y = rotina da feira: café da manhã vs rush da tarde/noite).
  - Validação no PDV via Casos Extremos (8.4.3) antes do scatter plot com biplot de cargas (8.4.4).
- **Projeção 3D (Seção 8.5) com Simetria Eixo por Eixo ($X, Y, Z$):**
  - Incorporação formal dos 3 eixos simultaneamente elevando a variância explicada acumulada para 54,2%.
  - Tabela comparativa (8.5.1) e **Painel Diagnóstico Triplo de Cargas Fatoriais** em alta resolução: [`reports/figures/03b_pca_loadings_3d_triplo.png`](./reports/figures/03b_pca_loadings_3d_triplo.png), confrontando $PC_1$, $PC_2$ e $PC_3$ lado a lado.
  - **Análise Tridimensional Eixo por Eixo (8.5.2):** Mapeamento do hipercubo sob 3 coordenadas independentes ($X$ = Volume da Cesta; $Y$ = Momento Temporal; $Z$ = Sensibilidade de Preço Unitário do SKU).
  - **Auditoria Empírica de Casos Extremos no Eixo Z (8.5.3):** Prova cabal no balcão de que o topo de $Z$ isola Cucas Nobres (R$ 25,00) e a base agrupa Doces Populares de desembolso unitário baixo (Alfajores e Rapaduras a R$ 5,00 a R$ 6,00).
  - Dispersão tridimensional com centróides e rótulos semânticos nos eixos (`Axes3D` em 8.5.4).