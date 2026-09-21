

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