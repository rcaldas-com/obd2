# Sem passo de build — é HTML/CSS/JS puro, sem bundler, sem framework.
# nginx só serve os arquivos estáticos; o docker-compose monta ./src por
# cima em dev pra editar sem rebuild.
FROM nginx:alpine
COPY src /usr/share/nginx/html
