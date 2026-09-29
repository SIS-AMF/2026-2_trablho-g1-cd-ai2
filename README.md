# Hub de Projetos — Ciência de Dados & Inteligência Artificial II

Repositório central de desenvolvimento prático, pipelines analíticos e relatórios acadêmicos das disciplinas de **Ciência dos Dados (G0544)** e **Inteligência Artificial II** do curso de **Sistemas de Informação** da **Faculdade Antonio Meneghetti (AMF)** — Semestre 2026/02.

---

## 1. Visão Geral do Repositório

O projeto integra esteiras de dados, experimentação estatística e modelagem de Machine Learning alinhadas às diretrizes curriculares da AMF:

| Disciplina | Docente | Código | Status Atual | Foco Prático |
| :--- | :--- | :--- | :--- | :--- |
| **Ciência dos Dados** | Prof. Marcelo Bortoluzzi Diaz | G0544 | 🚀 **Em Desenvolvimento** | Estatística, auditoria de outliers, EDA multivariada, engenharia de recursos e modelos preditivos. |
| **Inteligência Artificial II** | Prof. Rhauani Fazul | — | ✅ **Concluído & Arquivado** | Clusterização K-Means, redução de dimensionalidade PCA e recomendação de produtos no PDV KAlimentos. |

---

## 2. Acervo Histórico do Trabalho 1 — Inteligência Artificial II

O primeiro grande trabalho prático do semestre (sistema de suporte à decisão e recomendação de cross-selling no checkout do PDV para a **KAlimentos**) foi integralmente consolidado e catalogado em:

👉 **[`content/IA-2/archive/`](./content/IA-2/archive/)**

Nesse acervo histórico encontram-se preservados:
- **Notebook Completo:** [`clusterizacao_checkout_kmeans.ipynb`](./content/IA-2/archive/clusterizacao_checkout_kmeans.ipynb) (higienização, PCA biplot/3D, método do cotovelo, silhueta e perfis dos 5 clusters).
- **CLI do Checkout:** [`cli_modelo_recomendacao_checkout.py`](./content/IA-2/archive/cli_modelo_recomendacao_checkout.py) e roteiro [`demo_checkout_pdv.sh`](./content/IA-2/archive/demo_checkout_pdv.sh).
- **Modelo Serializado:** [`modelo_checkout_kmeans.joblib`](./content/IA-2/archive/modelo_checkout_kmeans.joblib).
- **Relatório Oficial:** [`relatorio_final.pdf`](./content/IA-2/archive/reports/relatorio_final.pdf) (13 páginas acadêmicas), Markdown e visualizador HTML com as 7 figuras do pipeline.
- **Harness Operacional Legado:** [`harness_operacional_ia2.md`](./content/IA-2/archive/harness_operacional_ia2.md).
- **Catálogo Detalhado:** Consulte o [`README.md` do archive](./content/IA-2/archive/README.md) para detalhes de cada arquivo.

---

## 3. Estrutura do Diretório Raiz

```text
.
├── .agents/                        # Harness operacional e personas de agentes para Ciência de Dados
│   └── .gitkeep
├── content/                        # Materiais pedagógicos oficiais da AMF
│   ├── CD/                         # Slides, notebooks de aula e Plano de Ensino de Ciência dos Dados
│   │   ├── Plano_de_Ensino_Ciencia_dos_dados_02_2026.docx
│   │   ├── Titanic.ipynb
│   │   ├── Pinguins_exemplos.ipynb
│   │   ├── carros.ipynb
│   │   ├── Pos__Identificação_outlier.ipynb
│   │   └── ...
│   ├── IA-1/                       # Acervo de IA I (ML Clássico, Redes Neurais, NLP, GenAI)
│   └── IA-2/                       # Acervo de IA II e pasta archive/ com o Trabalho 1
│       ├── archive/                # Acervo histórico do Trabalho 1 (KAlimentos)
│       └── ...
├── studing/                        # Datasets brutos de estudo e benchmarks
│   ├── BNPL_Financial_Default_Risk_Dataset.csv
│   ├── bread_basket.csv
│   ├── KalimentosFeirasVendas.csv
│   └── VendasFeiras.csv
├── requirements.txt                # Dependências Python homologadas (pandas, polars, scikit-learn, etc.)
└── README.md                       # Apresentação institucional e mapa do repositório
```

---

## 4. Configuração do Ambiente de Desenvolvimento

Para executar os notebooks e scripts de qualquer disciplina deste repositório:

1. **Ativar o Ambiente Virtual:**
   ```bash
   source .venv/bin/activate
   ```
2. **Instalação / Verificação de Dependências:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Execução de Jupyter Notebooks:**
   ```bash
   jupyter notebook
   ```
