# Analisador de Sentimentos em Português com DistilBERT

Este projeto implementa um analisador de sentimentos em português utilizando o modelo **DistilBERT**, otimizado para eficiência computacional e execução em GPUs, sendo compatível com Kaggle e Google Colab.

## Descrição

O modelo analisa resenhas de filmes em português do dataset [**IMDB PT-BR**](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr) e as classifica como **positivas ou negativas**. Para isso, utilizamos o **DistilBERT**, uma versão compacta e eficiente do BERT, treinada especificamente para essa tarefa.

## Características Principais

- Baseado no **DistilBERT**, versão otimizada do BERT;
- Implementa **early stopping** para evitar overfitting;
- Usa **DataLoader** para uma alimentação eficiente dos dados;
- Aplica **técnicas avançadas de pré-processamento** com **spaCy**, incluindo:
  - **Lematização**;
  - **Remoção de stopwords, pontuação, espaços, URLs e e-mails**;
- Utiliza **GradScaler e autocast** para melhor desempenho em GPUs.

## Dependências

- PyTorch  
- Transformers (Hugging Face)  
- spaCy  
- scikit-learn  
- pandas  
- numpy  

## Estrutura do Projeto

1. **Importação e pré-processamento dos dados** (remoção de stopwords, lematização, tokenização);
2. **Tokenização e separação dos datasets** para treino e teste;
3. **Definição e treinamento do modelo** DistilBERT;
4. **Avaliação do modelo** com métricas de desempenho;
5. **Salvamento do modelo treinado** para reutilização.

## Modelo Treinado

O repositório inclui duas versões do modelo treinado:

- **Pasta `imdb-ptbrmodelo_bert_treinado`**:
  - `config.json`: Configuração do modelo;
  - `model.safetensors`: Pesos do modelo treinado;
  - `special_tokens_map.json`: Mapeamento de tokens especiais;
  - `tokenizer_config.json`: Configuração do tokenizador;
  - `vocab.txt`: Vocabulário utilizado pelo modelo.
- **Arquivo `melhor_modelo.pt`**: Modelo salvo no formato PyTorch (265.63 MB).

### Como Carregar o Modelo Treinado

#### Opção 1: Usando os arquivos Hugging Face

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Carregar o tokenizador e o modelo do diretório local
tokenizer = AutoTokenizer.from_pretrained("./imdb-ptbrmodelo_bert_treinado")
model = AutoModelForSequenceClassification.from_pretrained("./imdb-ptbrmodelo_bert_treinado")

# Exemplo de classificação
texto = "Este filme é maravilhoso, adorei cada minuto!"
inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = model(**inputs)
prediction = outputs.logits.argmax(-1).item()
sentiment = "positivo" if prediction == 1 else "negativo"
print(f"Sentimento: {sentiment}")
```

#### Opção 2: Usando o arquivo PyTorch

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Carregar o modelo
modelo = torch.load('melhor_modelo.pt')
modelo.eval()

# Carregar o tokenizador
tokenizer = AutoTokenizer.from_pretrained("./imdb-ptbrmodelo_bert_treinado")

# Exemplo de classificação
texto = "Este filme é maravilhoso, adorei cada minuto!"
inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = modelo(**inputs)
prediction = outputs.logits.argmax(-1).item()
sentiment = "positivo" if prediction == 1 else "negativo"
print(f"Sentimento: {sentiment}")
```

### Benefícios do Modelo Treinado

- **Redução de tempo**: Evita um treinamento longo e custoso;
- **Economia de recursos**: Dispensa necessidade de GPUs para treino;
- **Resultados imediatos**: Pode ser utilizado diretamente para inferência.

## Como Utilizar

### Opção 1: Usando o modelo treinado (recomendado)
1. Clone este repositório;
2. Carregue o modelo conforme demonstrado na seção "Como Carregar o Modelo Treinado";
3. Utilize o modelo para classificar novos textos.

### Opção 2: Treinando o modelo do zero
1. Faça upload do notebook no Kaggle ou Google Colab;
2. Certifique-se de ter o dataset IMDB PT-BR disponível;
3. Execute todas as células do notebook;
4. O modelo treinado será salvo no diretório de trabalho.

## Resultados

O modelo alcançou uma **acurácia competitiva** na classificação de sentimentos em português, comprovando a eficiência do DistilBERT para essa tarefa.

## Aprendizados Durante o Desenvolvimento

Ao longo do projeto, algumas melhorias foram implementadas para aumentar a qualidade e eficiência do modelo:

- **Uso do DistilBERT**: Reduziu o custo computacional sem prejudicar a performance;
- **Pré-processamento avançado com spaCy**: Melhorou a qualidade dos dados textuais;
- **Uso de GradScaler e autocast**: Otimizou a execução em GPUs;
- **Ajuste do dataset**: Apenas colunas essenciais (`text_pt`, `sentiment`) foram mantidas para melhor eficiência.

Essas melhorias tornaram o modelo mais rápido, eficiente e preciso na classificação de sentimentos.

