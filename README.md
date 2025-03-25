# Analisador de Sentimentos em Português usando BERT

Este projeto implementa um analisador de sentimentos em português utilizando o modelo BERT pré-treinado, otimizado para execução em TPU no Kaggle.

## Descrição

O modelo analisa resenhas de filmes em português do dataset IMDB PT-BR e as classifica como positivas ou negativas. Utiliza-se o BERT base em português da NeuralMind, fine-tuned para esta tarefa específica.

## Principais Características

- Utiliza BERT pré-treinado em português
- Implementa early stopping
- Usa DataLoader distribuído para eficiência
- Aplica pré-processamento de texto com spaCy

## Dependências

- PyTorch
- Transformers (Hugging Face)
- spaCy
- scikit-learn
- pandas
- numpy

## Estrutura do Projeto

1. Importação e pré-processamento dos dados
2. Tokenização e preparação dos datasets
3. Definição e treinamento do modelo
4. Avaliação do modelo
5. Salvamento do modelo treinado

## Como Usar

1. Faça upload do notebook para o Kaggle
2. Certifique-se de ter o dataset IMDB PT-BR disponível
3. Execute as células em ordem
4. O modelo treinado será salvo no diretório de trabalho do Kaggle

## Resultados

O modelo atinge uma acurácia competitiva na classificação de sentimentos em português, demonstrando a eficácia do BERT para esta tarefa.
