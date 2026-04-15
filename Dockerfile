FROM python:3.11-slim

# Répertoire de travail dans le conteneur
WORKDIR /app

# Copie des dépendances en premier (cache Docker optimisé)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code et du modèle entraîné
COPY api.py .
COPY model/ model/

# Port exposé par l'API
EXPOSE 8000

# Démarrage du service
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
