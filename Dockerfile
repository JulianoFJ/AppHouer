# Imagem única para app + migrations: o mesmo artefato roda local (Compose) e,
# depois, em qualquer host que aceite um container (ECS/Fargate, EC2, App Runner) —
# só a URL do Postgres muda entre os ambientes, nunca a imagem.
FROM python:3.13-slim

WORKDIR /app

# Camada de dependências separada do código: mudar um .py não invalida o cache do
# pip install, que é o passo mais lento do build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

ENTRYPOINT ["./docker-entrypoint.sh"]
