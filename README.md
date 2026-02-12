# 🐱🐶 Pet Classifier

Um classificador de animais de estimação usando Deep Learning com PyTorch e FastAPI + React. Este projeto utiliza uma rede neural convolucional personalizada para classificar imagens de cães e gatos com alta precisão.

## 📋 Características

- **Backend**: API REST construída com FastAPI
- **Frontend**: Interface React moderna e responsiva
- **Modelo**: CNN customizada implementada em PyTorch
- **Classificação**: Cães vs Gatos com score de confiança
- **Deploy**: Pronto para produção

## 🏗️ Arquitetura do Projeto

```
Pet_Classifier/
├── backend/
│   ├── Model/
│   │   ├── model.py          # Arquitetura da CNN
│   │   └── Saved/            # Modelos treinados (.pth)
│   ├── Controller/
│   │   └── routes.py         # Rotas adicionais
│   ├── app.py                # API FastAPI principal
│   └── requirements.txt      # Dependências Python
├── frontend/
│   ├── src/                  # Código React
│   ├── package.json          # Dependências Node.js
│   └── ...
└── README.md
```

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8+
- Node.js 16+
- Modelo treinado (arquivo .pth)

### Backend (API)

1. **Navegue para o diretório backend**:
   ```bash
   cd backend
   ```

2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute a API**:

   **Desenvolvimento**:
   ```bash
   fastapi dev app.py
   ```

   **Produção**:
   ```bash
   fastapi run app.py
   ```

4. **API disponível em**: `http://localhost:8000`

### Frontend (React)

1. **Navegue para o diretório frontend**:
   ```bash
   cd frontend
   ```

2. **Instale as dependências**:
   ```bash
   npm install
   ```

3. **Execute o frontend**:
   ```bash
   npm run dev
   ```

4. **App disponível em**: `http://localhost:5173`

## 🔧 Endpoints da API

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `GET` | `/` | Status da API |
| `POST` | `/predict` | Classificar imagem |
| `GET` | `/model/info` | Informações do modelo |

### Exemplo de uso:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@path/to/your/image.jpg"
```

**Resposta**:
```json
{
  "filename": "dog.jpg",
  "predicted_class": "dog",
  "confidence": 97.85,
  "all_probabilities": {
    "cat": 2.15,
    "dog": 97.85
  }
}
```

## 🧠 Arquitetura do Modelo

A rede neural utiliza:
- **4 camadas convolucionais** com BatchNorm e MaxPooling
- **Global Average Pooling** para redução dimensional
- **4 camadas densas** com Dropout para regularização
- **Ativação GELU** para melhor performance
- **Transformações de dados** padronizadas

## 📊 Pré-processamento

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

## 🎯 Formatos Suportados

- **Imagens**: JPG, JPEG, PNG, BMP, TIFF
- **Tamanho**: Qualquer (será redimensionada automaticamente)
- **Canais**: RGB (3 canais)

## ⚙️ Configuração

### Variables de Ambiente (Backend)

```bash
# .env
MODEL_PATH=Model/Saved/dogxcat.pth
DEVICE=cpu  # ou cuda se disponível
```

### Troubleshooting CUDA

Se você encontrar problemas com CUDA:
```python
DEVICE = torch.device("cpu")  # Force CPU usage
```

## 📝 Dependências Principais

### Backend
- `fastapi` - Framework web moderno
- `torch` - PyTorch para deep learning
- `torchvision` - Transformações de imagem
- `pillow` - Processamento de imagem
- `uvicorn` - Servidor ASGI

### Frontend
- `react` - Biblioteca para UI
- `vite` - Build tool moderna
- `axios` - Cliente HTTP

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.