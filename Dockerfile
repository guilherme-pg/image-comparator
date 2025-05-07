# Imagem base com Python
FROM python:3.10-slim

# Instalar dependências do sistema
# RUN apt-get update && apt-get install -y libgl1 libgl1-mesa-glx && apt-get clean && rm -rf /var/lib/apt/lists/*
# RUN apt-get update && apt-get install -y libgl1 && apt-get clean
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# diretório de trabalho dentro do container
WORKDIR /app
# WORKDIR /image_comparator

# Copia os arquivos da aplicação para o container
COPY . .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt


ENV PORT=8080

# Expõe a porta usada pelo Flask
EXPOSE 8080

# Comando para iniciar a aplicação
CMD ["python", "app.py"]