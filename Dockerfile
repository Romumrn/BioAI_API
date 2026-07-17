# Gateway — seule porte d'entrée publique : auth, quotas, routage.
#
# Aucun modèle chargé ici, donc pas de torch.
#
# Build depuis la RACINE du repo :
#   docker build -f Dockerfile .
FROM python:3.11-slim

WORKDIR /app

COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

COPY common/ ./common/
COPY gateway.py .

# Contrairement aux serveurs de modèles, la gateway EST censée être joignable
# de l'extérieur : c'est le seul service dont le compose publie le port.
EXPOSE 8080

CMD ["python", "gateway.py"]
