# Analisador de Sentimentos em Português

## Descrição
Este projeto tem como objetivo criar um modelo de *Machine Learning* para analisar sentimentos em textos e classificá-los como positivos ou negativos. Para isso, utilizamos uma base de dados do IMDb (Internet Movie Database), traduzida para o português, contendo aproximadamente 50 mil resenhas de filmes.

## Base de Dados
A base de dados utilizada neste projeto está disponível no Kaggle, sob o nome **"IMDB PT-BR"**. Trata-se de uma tradução automática de uma base original do IMDb, permitindo a análise de sentimentos em português.

### Download da Base de Dados
1. Acesse o [Kaggle](https://www.kaggle.com/).
2. Busque por **"IMDB PT-BR"**.
3. Escolha a opção correspondente e faça o download dos dados.
4. Salve o arquivo `imdb-reviews-pt-br.csv` na pasta `dados/` do projeto.

## Ferramentas Utilizadas
- **Python** (Linguagem principal do projeto)
- **Pandas** (Para manipulação e análise dos dados)
- **Google Colab** (Ambiente de execução dos códigos, mas pode utilizar qual preferir)
- **Scikit-Learn** (Para criação do modelo de *Machine Learning*)

## Estrutura do Projeto
```
/
|-- dados/
|   |-- imdb-reviews-pt-br.csv  # Arquivo com as resenhas traduzidas
|
|-- notebooks/
|   |-- analise_dados.ipynb     # Análise exploratória dos dados
|   |-- modelo_treinamento.ipynb # Treinamento do modelo de ML
|
|-- README.md                   # Documentação do projeto
```

## Como Começar
1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install pandas scikit-learn numpy
   ```
3. Execute o *notebook* `notebooks/analise_dados.ipynb` para visualizar os dados.
4. Execute `notebooks/modelo_treinamento.ipynb` para treinar o modelo.

## Resultados Esperados
Após o treinamento do modelo, ele será capaz de classificar novas resenhas como positivas ou negativas, permitindo análises automáticas de opinião.


