

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