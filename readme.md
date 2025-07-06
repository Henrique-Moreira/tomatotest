# Análise Comparativa de Arquiteturas de Deep Learning para Segmentação Semântica de Tomates

## 📋 Descrição do Projeto

Este repositório contém a implementação e análise comparativa de três arquiteturas de redes neurais convolucionais (CNNs) para segmentação semântica de frutos de tomate em imagens de campo.

## 🎯 Objetivo

O objetivo principal é determinar qual arquitetura de deep learning é mais eficaz para a tarefa de segmentação semântica de tomates em condições reais de campo, contribuindo para o avanço da agricultura de precisão através da fenotipagem automatizada.

## 📊 Dataset Utilizado

### 1. Escolha do Dataset
- **Dataset**: `tomatotest` - Dataset público disponibilizado por He et al. (2024)
- **Fonte**: Coletado no Mountain Horticultural Crops Research and Extension Center, NC, EUA
- **Disponível em**: [https://huggingface.co/datasets/XingjianLi/tomatotest](https://huggingface.co/datasets/XingjianLi/tomatotest)
- **Características**:
  - 21.367 imagens capturadas em ambiente real de campo
  - Resolução original: 2448 × 2048 pixels
  - Capturadas com robô terrestre autônomo (HuskyBot) equipado com sistema de câmera estéreo
  - Condições desafiadoras: iluminação variável, oclusão por folhas, fundos complexos

### 2. Justificativa da Escolha
O dataset foi escolhido por:
- **Complexidade adequada**: Imagens reais de campo com condições desafiadoras
- **Relevância para pesquisa**: Aplicação direta em agricultura de precisão
- **Qualidade dos dados**: Anotações precisas e dataset bem estruturado
- **Escala apropriada**: Quantidade significativa de dados para treinamento robusto

## 🔍 Análise Exploratória de Dados

### Pré-processamento Realizado
1. **Conversão de Formato**: Extração de arquivos HDF5 (.h5) para PNG
2. **Geração de Máscaras**: Criação de máscaras binárias de segmentação
3. **Redimensionamento**: Imagens redimensionadas para 256×256 pixels
4. **Augmentação de Dados**: Aplicação de espelhamento horizontal

### Análise Estatística
- **Proporção de Pixels de Tomate**: Média de 0,57% por imagem
- **Desbalanceamento de Classes**: Identificado forte desbalanceamento (muitos pixels de fundo vs. poucos pixels de tomate)
- **Distribuição dos Dados**: 80% para treinamento, 20% para validação

### Visualizações Implementadas
- Histograma da proporção de pixels de tomate por máscara
- Gráficos de evolução das métricas de validação por época
- Curvas de perda durante o treinamento
- Exemplos qualitativos de predições do modelo

## 🤖 Técnicas de Aprendizado de Máquina

### Técnica Aplicada: Segmentação Semântica (Deep Learning)

#### Justificativa da Escolha
A **segmentação semântica** foi escolhida porque:
- **Precisão na localização**: Necessária para delimitar exatamente os contornos dos tomates
- **Aplicação prática**: Essencial para contagem precisa de frutos e estimativa de produção
- **Vantagem sobre outras técnicas**: 
  - Classificação de imagens: Muito simples (um rótulo por imagem)
  - Detecção de objetos: Apenas caixas delimitadoras, sem contornos precisos
  - Segmentação semântica: Classificação pixel a pixel, permitindo medições precisas

### Algoritmos Implementados

#### 1. U-Net
- **Arquitetura**: Encoder-decoder com skip connections
- **Vantagens**: Preserva detalhes espaciais de alta resolução
- **Aplicação**: Ideal para localização precisa de objetos pequenos
- **Resultado**: **Melhor performance** - IoU: 0.8265, Dice/F1: 0.9033

#### 2. DeepLabV3
- **Arquitetura**: Convoluções atróficas com módulo ASPP
- **Vantagens**: Análise multi-escala sem perda de resolução
- **Aplicação**: Robusto para objetos de tamanhos variados
- **Resultado**: IoU: 0.5793, Dice/F1: 0.7309

#### 3. PSPNet (Pyramid Scene Parsing Network)
- **Arquitetura**: Módulo de pyramid pooling para contexto global
- **Vantagens**: Compreensão da estrutura geral da cena
- **Aplicação**: Ideal para parsing de cenas complexas
- **Resultado**: IoU: 0.5528, Dice/F1: 0.7120

### Hiperparâmetros Utilizados

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Épocas | 100 | Convergência adequada observada |
| Otimizador | Adam | Eficiente e amplamente usado |
| Função de Perda | Dice Loss + BCE | Combate desbalanceamento de classes |
| Resolução | 256×256 | Compromisso entre detalhes e viabilidade computacional |
| Augmentação | Espelhamento horizontal | Duplica dataset e melhora generalização |

### Métricas de Avaliação
- **IoU (Intersection over Union)**: Métrica mais rigorosa para segmentação
- **Dice Coefficient (F1-Score)**: Medida de sobreposição, mais flexível que IoU
- **Precisão**: Acurácia das predições positivas
- **Recall**: Completude das predições positivas

## 📈 Resultados Principais

### Performance Quantitativa
| Modelo | IoU | Dice/F1 | Precisão | Recall |
|--------|-----|---------|----------|--------|
| **U-Net** | **0.8265** | **0.9033** | **0.9487** | **0.8630** |
| DeepLabV3 | 0.5793 | 0.7309 | 0.7516 | 0.7140 |
| PSPNet | 0.5528 | 0.7120 | 0.7351 | 0.6903 |

### Principais Descobertas
1. **Superioridade da U-Net**: 42-50% melhor que as outras arquiteturas
2. **Importância dos Skip Connections**: Preservação de detalhes espaciais é crucial
3. **Eficácia da Abordagem Supervisionada**: Aprendizado direto supera métodos mais complexos para esta tarefa específica

## 🗂️ Estrutura do Repositório

```
├── README.md                              # Este arquivo
├── code/                                  # Código fonte
│   ├── experimento1/                      # Experimentos U-Net
│   ├── experimento2/                      # Experimentos U-Net (variações)
│   ├── experimento3/                      # Experimentos U-Net (variações)
│   ├── experimento4/                      # Experimentos U-Net (variações)
│   ├── experimento5/                      # Experimentos DeepLabV3
│   ├── experimento6/                      # Experimentos PSPNet
│   ├── data_augmentation_tomates.ipynb    # Análise e augmentação de dados
│   └── redimensionar imagens para 256.py # Pré-processamento
├── tomatotest/                            # Dataset e scripts
│   ├── data/                              # Dados brutos
│   ├── processed_data/                    # Dados processados
│   └── processed_data_256/                # Dados redimensionados
└── ARTIGO BASE - High‐Throughput Robotic... # Artigo de referência
```

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install torch torchvision opencv-python matplotlib numpy pandas
```

### Execução dos Experimentos
1. **Pré-processamento**: Execute `redimensionar imagens para 256.py`
2. **Análise Exploratória**: Abra `data_augmentation_tomates.ipynb`
3. **Treinamento U-Net**: Execute notebooks em `experimento1/` a `experimento4/`
4. **Treinamento DeepLabV3**: Execute notebook em `experimento5/`
5. **Treinamento PSPNet**: Execute notebook em `experimento6/`

## 📊 Análise dos Resultados

### Por que a U-Net foi Superior?
1. **Alinhamento Arquitetural**: Skip connections ideais para localização precisa
2. **Preservação de Detalhes**: Mantém informações espaciais de alta resolução
3. **Adequação à Tarefa**: Tomates são objetos pequenos que requerem delimitação precisa
4. **Características Locais**: Mais importantes que contexto global nesta aplicação

### Limitações e Trabalhos Futuros
- Explorar técnicas de augmentação mais sofisticadas
- Testar backbones mais modernos (ResNet, EfficientNet)
- Otimização sistemática de hiperparâmetros
- Validação em outros datasets agrícolas

## 🏆 Créditos e Referências

### Dataset
Este projeto utiliza o dataset `tomatotest` criado e disponibilizado por:
- **Autores**: Weilong He, Xingjian Li, Zhenghua Zhang, Yuxi Chen, Jianbo Zhang, Dilip R. Panthee, Inga Meadows, Lirong Xiang
- **Disponível em**: [https://huggingface.co/datasets/XingjianLi/tomatotest](https://huggingface.co/datasets/XingjianLi/tomatotest)

### Artigo Base
O trabalho é baseado no artigo científico:
**"High-Throughput Robotic Phenotyping for Quantifying Tomato Disease Severity Enabled by Synthetic Data and Domain-Adaptive Semantic Segmentation"**

**Autores**: 
- Weilong He¹'²
- Xingjian Li²'³ 
- Zhenghua Zhang¹'²
- Yuxi Chen³
- Jianbo Zhang⁴
- Dilip R. Panthee
- Inga Meadows
- Lirong Xiang¹'²

## 👥 Contribuições

Este projeto foi desenvolvido como parte da pesquisa de mestrado em Ciência da Computação na Universidade Federal de Uberlândia, sob orientação acadêmica específica para a disciplina de Mineração de Dados.

## 📄 Licença

Este projeto está licenciado sob a Licença Apache-2.0 - veja o arquivo LICENSE para detalhes.

---

**Contato**: henriquemoreiraa@gmail.com  
**Instituição**: Universidade Federal de Uberlândia  
**Programa**: Mestrado em Ciência da Computação