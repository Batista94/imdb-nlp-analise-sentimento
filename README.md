# Analisador de Sentimentos em Português com DistilBERT  

Este projeto implementa um analisador de sentimentos em português utilizando o modelo DistilBERT, otimizado para eficiência computacional e execução em GPUs. Ele é compatível com Google Colab e Kaggle, permitindo fácil experimentação e reuso.  

## Visão Geral  

O modelo classifica resenhas de filmes em português do dataset [IMDB PT-BR](https://www.kaggle.com/datasets/luisfredgs/imdb-ptbr) como positivas ou negativas. Para isso, utilizamos o DistilBERT, uma versão compacta e eficiente do BERT, treinada especificamente para essa tarefa.  

## Principais Características  

- Baseado no DistilBERT, uma versão otimizada do BERT  
- Implementa early stopping para evitar overfitting  
- Usa DataLoader para alimentação eficiente dos dados  
- Aplica técnicas avançadas de pré-processamento com spaCy, incluindo:  
  - Lematização  
  - Remoção de stopwords, pontuação, espaços, URLs e e-mails  
- Utiliza GradScaler e autocast para melhor desempenho em GPUs  

## Dependências  

Certifique-se de instalar todas as bibliotecas necessárias antes de rodar o código:  

```bash
pip install torch transformers spacy scikit-learn pandas numpy
```  

## Estrutura do Projeto  

1. Importação e pré-processamento dos dados (remoção de stopwords, lematização, tokenização)  
2. Tokenização e separação dos datasets para treino e teste  
3. Definição e treinamento do modelo DistilBERT  
4. Avaliação do modelo com métricas de desempenho  
5. Salvamento do modelo treinado para reutilização  

## Modelo Treinado  

Devido ao tamanho dos arquivos, o modelo treinado não está armazenado diretamente neste repositório. No entanto, ele pode ser baixado no Kaggle:  

[Download do Modelo Treinado](https://www.kaggle.com/code/wesleibatista/imbd-nlp-ptbr/output)  

O download inclui:  

- `config.json` – Configuração do modelo  
- `model.safetensors` – Pesos do modelo treinado  
- `special_tokens_map.json` – Mapeamento de tokens especiais  
- `tokenizer_config.json` – Configuração do tokenizador  
- `vocab.txt` – Vocabulário utilizado pelo modelo  

### Como Carregar o Modelo Treinado  

Após baixar os arquivos, extraia-os no diretório do projeto e utilize o código abaixo:  

#### Opção 1: Usando os arquivos Hugging Face  

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Carregar o modelo e o tokenizador do diretório local
tokenizer = AutoTokenizer.from_pretrained("./modelo_treinado")
model = AutoModelForSequenceClassification.from_pretrained("./modelo_treinado")

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

# Carregar o modelo salvo em PyTorch
modelo = torch.load('melhor_modelo.pt')
modelo.eval()

# Carregar o tokenizador
tokenizer = AutoTokenizer.from_pretrained("./modelo_treinado")

# Exemplo de classificação
texto = "Este filme é maravilhoso, adorei cada minuto!"
inputs = tokenizer(texto, return_tensors="pt", padding=True, truncation=True, max_length=512)
outputs = modelo(**inputs)
prediction = outputs.logits.argmax(-1).item()
sentiment = "positivo" if prediction == 1 else "negativo"
print(f"Sentimento: {sentiment}")
```  

## Como Utilizar  

### Opção 1: Utilizando o Modelo Treinado (Recomendado)  

1. Clone este repositório:  
```bash
git clone https://github.com/seuusuario/seurepositorio.git
cd seurepositorio
```  
2. Baixe o modelo do Kaggle e extraia os arquivos no diretório do projeto  
3. Siga as instruções da seção "Como Carregar o Modelo Treinado"  

### Opção 2: Treinando o Modelo do Zero  

1. Acesse o [notebook no Kaggle](https://www.kaggle.com/code/wesleibatista/imbd-nlp-ptbr)  
2. Certifique-se de ter o dataset IMDB PT-BR disponível  
3. Execute todas as células do notebook  
4. O modelo treinado será salvo no diretório de trabalho  

## Resultados  

O modelo atingiu uma acurácia competitiva na classificação de sentimentos em português, comprovando a eficiência do DistilBERT para essa tarefa.  

Principais melhorias aplicadas:  

- Uso do DistilBERT: Redução do custo computacional sem comprometer a performance  
- Pré-processamento avançado com spaCy: Melhor qualidade dos dados textuais  
- Uso de GradScaler e autocast: Execução otimizada em GPUs  
- Ajuste do dataset: Apenas colunas essenciais (`text_pt`, `sentiment`) foram mantidas  

Essas otimizações tornaram o modelo mais rápido, eficiente e preciso na classificação de sentimentos.  

## Aprendizados Durante o Desenvolvimento  

Durante o desenvolvimento deste projeto, foram exploradas diversas técnicas que aprimoraram a análise de sentimentos em português:  

- Otimização com DistilBERT: Permitiu uma execução mais eficiente em comparação com o BERT tradicional  
- Fine-tuning em Português: Melhorou a precisão do modelo ao lidar com nuances da língua portuguesa  
- Processamento de Texto Avançado: Técnicas como lematização e remoção de stopwords aumentaram a qualidade dos dados  

Esse projeto representa um avanço significativo na classificação de sentimentos em português, tornando-o um ótimo recurso para análises de opinião e aplicações em NLP.  

## Próximos Passos  

- Treinar o modelo com um dataset maior para melhorar a generalização  
- Explorar técnicas de data augmentation para ampliar a base de treino  
- Implementar uma API para permitir inferências em tempo real  

## Contribuições  

Contribuições são bem-vindas. Se você deseja melhorar o projeto, siga os passos abaixo:  

1. Faça um fork deste repositório  
2. Crie uma branch para sua feature (`git checkout -b minha-feature`)  
3. Implemente as mudanças e faça commit (`git commit -m "Minha contribuição"`)  
4. Envie um pull request e aguarde a revisão  

Se este projeto foi útil para você, não se esqueça de dar uma estrela no repositório.  
