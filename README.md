# Analisador de Sentimentos em Português usando DistilBERT

Este projeto implementa um analisador de sentimentos em português utilizando o modelo **DistilBERT**, otimizado para eficiência computacional e execução em GPUs no Kaggle ou Google Colab.

## Descrição

O modelo analisa resenhas de filmes em português do dataset [**IMDB PT-BR**](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr) e as classifica como **positivas ou negativas**. Para isso, utilizamos o **DistilBERT**, uma versão otimizada do BERT, fine-tuned para esta tarefa específica.

## Principais Características

- Utiliza **DistilBERT**, versão otimizada do BERT
- Implementa **early stopping** para evitar overfitting
- Usa **DataLoader** para eficiência na alimentação de dados
- Aplica **técnicas avançadas de pré-processamento de texto** com **spaCy**, incluindo:
  - **Lematização**
  - **Remoção de stopwords, pontuação, espaços, URLs e e-mails**
- Utiliza **GradScaler e autocast** para melhor eficiência em GPUs

## Dependências

- PyTorch  
- Transformers (Hugging Face)  
- spaCy  
- scikit-learn  
- pandas  
- numpy  

## Estrutura do Projeto

1. **Importação e pré-processamento dos dados** (remoção de stopwords, lematização, tokenização)  
2. **Tokenização e preparação dos datasets** para treino e teste  
3. **Definição e treinamento do modelo** DistilBERT  
4. **Avaliação do modelo** com métricas de precisão  
5. **Salvamento do modelo treinado**  

## Como Usar

1. Faça upload do notebook para o Kaggle ou Google Colab  
2. Certifique-se de ter o dataset IMDB PT-BR disponível  
3. Execute as células do notebook na ordem  
4. O modelo treinado será salvo no diretório de trabalho  

## Resultados

O modelo atinge uma **acurácia competitiva** na classificação de sentimentos em português, demonstrando a eficiência do DistilBERT para essa tarefa.

## Lições Aprendidas

Durante o desenvolvimento, algumas melhorias foram implementadas para otimizar desempenho e qualidade do modelo:

- **Mudança para DistilBERT**: Essa versão mais leve do BERT reduz o custo computacional sem comprometer a performance.  
- **Pré-processamento avançado com spaCy**: A remoção de stopwords e a lematização melhoraram a representação textual.  
- **Uso de GradScaler e autocast**: Melhorou a eficiência computacional no treinamento em GPUs.  
- **Refinamento do dataset**: Reduzimos as colunas para apenas os dados essenciais (`text_pt`, `sentiment`), otimizando a entrada para o modelo.  

Essas melhorias tornaram o modelo mais rápido, eficiente e preciso para a classificação de sentimentos.
