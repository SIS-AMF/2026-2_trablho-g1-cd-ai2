

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