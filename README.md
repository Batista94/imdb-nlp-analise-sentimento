# Analisador de Sentimentos em Português usando DistilBERT

Este projeto implementa um analisador de sentimentos em português utilizando o modelo **DistilBERT**, otimizado para eficiência computacional e execução em GPUs no Kaggle ou Google Colab.

## Descrição

O modelo analisa resenhas de filmes em português do dataset [**IMDB PT-BR**](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr) e as classifica como **positivas ou negativas**. Para isso, utilizamos o **DistilBERT**, uma versão otimizada do BERT, fine-tuned para essa tarefa específica.

## Principais Características

- Utiliza **DistilBERT**, versão otimizada do BERT  
- Implementa **early stopping** para evitar overfitting  
- Usa **DataLoader** para eficiência na alimentação de dados  
- Aplica **técnicas avançadas de pré-processamento de texto** com **spaCy**, incluindo:  
  - **Lematização**  
  - **Remoção de stopwords, pontuação, espaços, URLs e e-mails**  
- Utiliza **GradScaler e autocast** para melhor eficiência em GPUs  

## Dependências

- `PyTorch`  
- `Transformers` (Hugging Face)  
- `spaCy`  
- `scikit-learn`  
- `pandas`  
- `numpy`  

## Estrutura do Projeto

1. **Importação e pré-processamento dos dados** (remoção de stopwords, lematização, tokenização)  
2. **Tokenização e preparação dos datasets** para treino e teste  
3. **Definição e treinamento do modelo** DistilBERT  
4. **Avaliação do modelo** com métricas de precisão  
5. **Salvamento do modelo treinado**  

## Modelo Pré-treinado

Este repositório inclui os arquivos do modelo DistilBERT já treinado para análise de sentimentos em português. Os arquivos disponíveis são:

- `config.json`: Configuração do modelo  
- `model.safetensors`: Pesos do modelo treinado  
- `special_tokens_map.json`: Mapeamento de tokens especiais  
- `tokenizer_config.json`: Configuração do tokenizador  
- `vocab.txt`: Vocabulário utilizado pelo modelo  

### Como Carregar o Modelo Pré-treinado

Para utilizar o modelo já treinado sem precisar retreiná-lo, você pode carregá-lo diretamente com a biblioteca Transformers:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Carregar o tokenizador e o modelo do diretório local
tokenizer = AutoTokenizer.from_pretrained("./imdb-ptbrmodelo_bert_treinado")
model = AutoModelForSequenceClassification.from_pretrained("./imdb-ptbrmodelo_bert_treinado")

# Exemplo de uso para classificação
texto = "Este filme é maravilhoso, adorei cada minuto!"
inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
sentiment = "positivo" if prediction == 1 else "negativo"
print(f"Sentimento: {sentiment}")
