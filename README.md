# Lord of the Cine

Sistema híbrido de recomendação de filmes desenvolvido em Python com Machine Learning. O projeto combina preferências do usuário, similaridade de conteúdo, avaliação média e popularidade para gerar recomendações personalizadas e explicáveis.

## Sobre o projeto

O Lord of the Cine permite que uma pessoa avalie alguns filmes e receba recomendações de títulos ainda não vistos. O sistema constrói um perfil a partir dessas avaliações e calcula um score híbrido para cada filme do catálogo.

O score combina três componentes:

- **Perfil do usuário:** similaridade entre o perfil construído e o conteúdo dos filmes.
- **Avaliação média:** qualidade geral do filme segundo as avaliações do dataset.
- **Popularidade:** quantidade de avaliações recebidas pelo filme, com transformação logarítmica.

Na interface Streamlit, os pesos desses componentes podem ser ajustados por controles deslizantes. As recomendações também apresentam uma explicação baseada em gêneros, tags, similaridade textual e avaliação média.

## Tecnologias

- Python 3.10 ou superior
- pandas e NumPy
- scikit-learn
- SciPy
- Joblib
- PyArrow
- Matplotlib
- Streamlit
- Jupyter Notebook

## Dataset

O projeto utiliza o **MovieLens Latest Small**, disponibilizado pelo GroupLens. O conjunto contém dados de filmes, avaliações, tags e identificadores externos.

Fonte oficial:

- [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/latest/)
- [Download direto do ml-latest-small.zip](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)
- [Documentação do dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html)

Após extrair o arquivo, a estrutura esperada é:

```text
data/raw/ml-latest-small/
├── links.csv
├── movies.csv
├── ratings.csv
└── tags.csv
```

## Estrutura do projeto

```text
LordOfTheCine/
├── data/
│   ├── raw/                     # Dados originais do MovieLens
│   ├── processed/               # Dados tratados em CSV e Parquet
│   └── cache/
├── models/
│   ├── tfidf_matrix.joblib      # Matriz TF-IDF dos filmes
│   └── tfidf_vectorizer.joblib  # Vetorizador TF-IDF treinado
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_content_based.ipynb
│   └── 04_evaluation.ipynb
├── reports/
│   ├── figures/                 # Gráficos da avaliação
│   ├── evaluation_analysis.md
│   └── evaluation_results.csv
├── src/
│   ├── app/                     # Aplicação Streamlit
│   ├── data/                    # Pré-processamento
│   ├── evaluation/              # Métricas, avaliação e gráficos
│   ├── explainability/          # Explicações das recomendações
│   ├── features/                # Extração das features TF-IDF
│   └── models/                  # Recomendadores
├── requirements.txt
└── README.md
```

## Instalação

Clone o repositório e entre na pasta do projeto:

```bash
git clone <URL_DO_REPOSITORIO>
cd LordOfTheCine
```

Crie um ambiente virtual:

```bash
python -m venv .venv
```

Ative o ambiente no Linux ou macOS:

```bash
source .venv/bin/activate
```

No Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Execução rápida

Se os arquivos abaixo já estiverem presentes, a aplicação pode ser iniciada diretamente:

```text
data/processed/movies_processed.parquet
models/tfidf_matrix.joblib
```

Execute:

```bash
python -m streamlit run src/app/streamlit_app.py
```

O Streamlit exibirá no terminal o endereço local, normalmente:

```text
http://localhost:8501
```

Na aplicação:

1. Pesquise um filme na barra lateral.
2. Escolha uma nota entre 0,5 e 5,0.
3. Adicione mais filmes para formar seu perfil.
4. Ajuste os pesos de perfil, nota média e popularidade.
5. Consulte as recomendações e suas explicações.

Evite formar um perfil somente com notas `3.0`, pois essa é a nota neutra usada pelo modelo.

## Pipeline completo

Use estas etapas quando os dados processados ou os arquivos `.joblib` ainda não existirem.

### 1. Preparar os dados brutos

Baixe e extraia o `ml-latest-small.zip` dentro de `data/raw/`, mantendo o caminho:

```text
data/raw/ml-latest-small/
```

### 2. Pré-processar os dados

```bash
python src/data/preprocessing.py
```

Essa etapa:

- normaliza gêneros e tags;
- extrai o ano dos títulos;
- calcula média, quantidade e desvio-padrão das avaliações;
- cria o texto de conteúdo usado pelo TF-IDF;
- salva os resultados em CSV e Parquet.

Arquivos gerados:

```text
data/processed/movies_processed.csv
data/processed/movies_processed.parquet
data/processed/ratings_processed.csv
data/processed/ratings_processed.parquet
```

### 3. Gerar as features TF-IDF

```bash
python src/features/text_features.py
```

Arquivos gerados:

```text
models/tfidf_vectorizer.joblib
models/tfidf_matrix.joblib
```

### 4. Iniciar a aplicação

```bash
python -m streamlit run src/app/streamlit_app.py
```

## Arquivos CSV, Parquet e Joblib

Os arquivos CSV e Parquet em `data/processed/` representam os dados tratados:

- **CSV:** formato textual, fácil de abrir em editores e planilhas.
- **Parquet:** formato binário, mais compacto e eficiente para leitura com pandas, além de preservar os tipos das colunas.

O projeto usa os arquivos Parquet durante a execução. Os CSVs são mantidos para inspeção e compatibilidade com outras ferramentas.

Os arquivos Joblib em `models/` armazenam objetos gerados pelo treinamento:

- `tfidf_vectorizer.joblib`: vocabulário e configuração do vetorizador.
- `tfidf_matrix.joblib`: representação vetorial do conteúdo dos filmes.

Esses arquivos evitam recalcular as features toda vez que a aplicação é iniciada. Eles podem ser recriados executando novamente `src/features/text_features.py`.

## Recomendadores

O projeto possui três componentes principais:

- **ContentRecommender:** encontra filmes semelhantes a partir do conteúdo textual.
- **UserProfileRecommender:** cria um vetor de preferências com base nas avaliações informadas.
- **HybridRecommender:** combina perfil, avaliação média e popularidade em um score final.

Também é possível executar os testes demonstrativos dos módulos:

```bash
python src/models/content_recommender.py
python src/models/user_profile.py
python src/models/hybrid_recommender.py
python src/explainability/explanations.py
```

## Avaliação quantitativa


- **Precision@10:** proporção de filmes relevantes entre as dez primeiras recomendações.
- **Recall@10:** proporção dos filmes relevantes do teste recuperada nas dez primeiras posições.
- **NDCG@10:** qualidade da ordenação, valorizando itens relevantes nas primeiras posições.
- **Coverage:** proporção do catálogo recomendada ao menos uma vez.

O protocolo:

- seleciona usuários com pelo menos 20 avaliações;
- usa separação de 80% para treino e 20% para teste;
- considera relevante uma avaliação maior ou igual a 4,0;
- usa seed fixa para permitir reprodutibilidade;
- compara o baseline de popularidade com cinco configurações híbridas.

Para executar a avaliação:

```bash
python src/evaluation/evaluate.py
```

O resultado é salvo em:

```text
reports/evaluation_results.csv
```

Para gerar os gráficos:

```bash
python src/evaluation/plot_results.py
```

Os gráficos são salvos em `reports/figures/`.

### Resultados

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|---|---:|---:|---:|---:|
| Popularity Baseline | 0.1290 | 0.1063 | 0.1706 | 0.0055 |
| Hybrid 70/20/10 | 0.0212 | 0.0304 | 0.0359 | **0.0844** |
| Hybrid 60/20/20 | 0.0459 | 0.0554 | 0.0673 | 0.0608 |
| Hybrid 50/30/20 | 0.0616 | 0.0711 | 0.0876 | 0.0488 |
| Hybrid 40/30/30 | 0.1144 | 0.1097 | 0.1513 | 0.0284 |
| Hybrid 30/30/40 | **0.1457** | **0.1278** | **0.1942** | 0.0154 |

O **Hybrid 30/30/40** apresentou os melhores valores de Precision@10, Recall@10 e NDCG@10. O **Hybrid 70/20/10** apresentou a maior cobertura.

Os resultados evidenciam um trade-off: aumentar o peso da popularidade melhora a precisão, enquanto aumentar o peso do perfil do usuário amplia a cobertura do catálogo.

Consulte também:

- [`reports/evaluation_results.csv`](reports/evaluation_results.csv)
- [`reports/evaluation_analysis.md`](reports/evaluation_analysis.md)
- [`notebooks/04_evaluation.ipynb`](notebooks/04_evaluation.ipynb)

## Notebooks

Para trabalhar com os notebooks, instale opcionalmente o JupyterLab:

```bash
python -m pip install jupyterlab
```

Depois, inicie o ambiente:

```bash
python -m jupyter lab
```

Eles documentam a análise exploratória, o pré-processamento, o modelo baseado em conteúdo e a avaliação quantitativa.

## Reprodução completa

Depois de instalar as dependências e preparar o MovieLens, todo o fluxo principal pode ser reproduzido nesta ordem:

```bash
python src/data/preprocessing.py
python src/features/text_features.py
python src/evaluation/evaluate.py
python src/evaluation/plot_results.py
python -m streamlit run src/app/streamlit_app.py
```

## Licença

O código do projeto é distribuído sob a licença MIT. Consulte o arquivo [`LICENSE`](LICENSE).

O dataset MovieLens possui termos próprios de uso definidos pelo GroupLens. Consulte a [documentação oficial do dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html).
