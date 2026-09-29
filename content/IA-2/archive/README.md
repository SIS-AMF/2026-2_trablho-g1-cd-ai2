# Acervo Histórico — Trabalho 1: Inteligência Artificial II (2026/02)

Este diretório (`content/IA-2/archive/`) constitui o **arquivo histórico oficial** do **Trabalho 1 de Inteligência Artificial II (2026/02)**, desenvolvido no curso de **Sistemas de Informação da Faculdade Antonio Meneghetti (AMF)** sob docência do **Prof. Rhauani Fazul**.

---

## 1. Por que este diretório Archive existe?

Durante a primeira etapa do semestre, o repositório foi utilizado para conceber, validar e documentar a esteira completa de Machine Learning voltada ao Ponto de Venda (PDV) da empresa **KAlimentos** (clusterização K-Means de transações de feiras gastronômicas e recomendação de produtos em tempo de checkout para incremento de ticket médio).

Com a conclusão e entrega do Trabalho de IA-2, a estrutura do projeto precisou evoluir para acomodar o próximo desafio acadêmico: o **Trabalho 1 de Ciência dos Dados (Prof. Marcelo Bortoluzzi Diaz)**.

Para **não perder o contexto, a rastreabilidade experimental e a reprodutibilidade** do projeto KAlimentos, todos os artefatos técnicos, modelos treinados, scripts, relatórios e especificações de agentes foram centralizados e renomeados semanticamente neste diretório de arquivo histórico.

---

## 2. Catálogo Detalhado de Arquivos Arquivados

Abaixo está o mapeamento completo dos artefatos arquivados, seu nome original na raiz e a respectiva função técnica:

| Arquivo no Archive | Nome Original na Raiz | Categoria | Papel Técnico e Descrição |
| :--- | :--- | :--- | :--- |
| [`README.md`](./README.md) | *(Novo)* | Documentação | Este documento de catálogo e memorial descritivo do acervo histórico de IA-2. |
| [`README_ia2_original.md`](./README_ia2_original.md) | `README.md` | Documentação | Documentação arquitetural original completa do projeto KAlimentos (contexto de negócio, pipeline, fórmulas e diagramas). |
| [`diario_de_bordo_ia2.md`](./diario_de_bordo_ia2.md) | `diario_de_bordo.md` | Histórico / Log | Registro cronológico detalhado de todos os experimentos, hipóteses, decisões de engenharia de atributos e ajustes de modelo. |
| [`harness_operacional_ia2.md`](./harness_operacional_ia2.md) | `AGENTS.md` | Harness de IA | Especificação completa do Harness de Agentes Autônomos (personas, guardrails, invariantes financeiras e checklist operacional de IA-2). |
| [`clusterizacao_checkout_kmeans.ipynb`](./clusterizacao_checkout_kmeans.ipynb) | `eda_v2.ipynb` | Notebook Jupyter | Notebook com a esteira completa: auditoria estatística, sanitização, PCA 2D/3D com cargas fatoriais, varredura de $K$ (cotovelo/silhueta) e clusterização K-Means. |
| [`cli_modelo_recomendacao_checkout.py`](./cli_modelo_recomendacao_checkout.py) | `cli.py` | Aplicação Python | Interface interativa de linha de comando que simula o checkout do PDV, inferindo o cluster da compra e sugerindo cross-selling em tempo real. |
| [`demo_checkout_pdv.sh`](./demo_checkout_pdv.sh) | `demo.sh` | Shell Script | Script de demonstração automatizada com casos de teste pré-definidos para a CLI de recomendação. |
| [`modelo_checkout_kmeans.joblib`](./modelo_checkout_kmeans.joblib) | `modelo_checkout.joblib` | Artefato de ML | Pipeline de inferência serializado contendo o modelo `KMeans(n_clusters=5)`, os escaladores (`StandardScaler`) e as matrizes de recomendação. |
| [`base_kalimentos_processada.csv`](./base_kalimentos_processada.csv) | `base.csv` | Dataset | Dataset transacional resultante do pré-processamento e da engenharia de atributos temporais e contextuais (sem nulos). |
| [`get_dataset_kalimentos.sql`](./get_dataset_kalimentos.sql) | `get_dataset.sql` | Consulta SQL | Consulta SQL original utilizada para extração e consolidação dos dados do banco relacional de feiras da KAlimentos. |
| [`datasets/`](./datasets/) | `studing/` | Datasets Brutos | Acervo de dados originais de estudo: `KalimentosFeirasVendas.csv`, `bread_basket.csv`, `BNPL_Financial_Default_Risk_Dataset.csv` e `VendasFeiras.csv`. |
| [`reports/`](./reports/) | `reports/` | Entregáveis | Diretório contendo os entregáveis formais da disciplina (descrito a seguir). |

---

## 3. Estrutura do Diretório de Relatórios (`reports/`)

Dentro de [`reports/`](./reports/), encontram-se os entregáveis oficiais da disciplina:

- **[`reports/relatorio_final.pdf`](./reports/relatorio_final.pdf):** Documento final formal em PDF (13 páginas) diagramado com rigor acadêmico, cobrindo as 9 seções exigidas (Dataset, Problema, Preparação, Estratégia, Modelagem, Avaliação, Resultados, Demonstração e Harness de Agentes).
- **[`reports/relatorio_final.md`](./reports/relatorio_final.md):** Código-fonte em Markdown com toda a redação técnica do relatório acadêmico.
- **[`reports/relatorio_final.html`](./reports/relatorio_final.html):** Versão HTML formatada para visualização direta em navegadores.
- **[`reports/figures/`](./reports/figures/):** Figuras de diagnóstico e avaliação em alta resolução:
  - `01_otimizacao_k_cotovelo_silhueta.png`: Curva do método do cotovelo e coeficiente de silhueta para $k \in [2, 10]$.
  - `02_pca_2d_clusters_biplot.png`: Projeção bidimensional com vetores de carregamento do PCA.
  - `02b_pca_loadings_barplots.png`: Decomposição das cargas fatoriais nos componentes PC1 e PC2.
  - `03_pca_3d_clusters.png`: Dispersão tridimensional dos agrupamentos (PC1, PC2 e PC3).
  - `03b_pca_loadings_3d_triplo.png`: Contribuição das features nas três primeiras dimensões latentes.
  - `04_heatmap_centroides_clusters.png`: Heatmap interpretativo dos centróides dos 5 clusters identificados.
  - `05_simulacao_checkout_recomendacao.png`: Gráfico de dispersão da simulação de inferência em tempo real.

---

## 4. Como Executar e Consultar os Artefatos Arquivados

Caso seja necessário reproduzir os resultados de IA-2:

### Execução da CLI de Recomendação:
A partir da raiz do projeto, utilizando o ambiente virtual Python (`.venv`):
```bash
./.venv/bin/python3 content/IA-2/archive/cli_modelo_recomendacao_checkout.py --help
```

### Execução da Demonstração Automatizada:
```bash
bash content/IA-2/archive/demo_checkout_pdv.sh
```

### Visualização do Relatório:
O relatório oficial compilado pode ser visualizado via PDF:
```bash
xdg-open content/IA-2/archive/reports/relatorio_final.pdf
```
