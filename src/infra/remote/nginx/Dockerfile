FROM jwilder/dockerize as config

COPY ./ipfs.conf /ipfs.conf.template
COPY ./ipfs-api.conf /ipfs-api.conf.template
COPY ./geth.conf /geth.conf.template

ARG IPFS_GATEWAY_HOSTNAME
ARG IPFS_API_HOSTNAME
ARG GETH_HOSTNAME

ENV IPFS_GATEWAY_HOSTNAME=${IPFS_GATEWAY_HOSTNAME}
ENV IPFS_API_HOSTNAME=${IPFS_API_HOSTNAME}
ENV GETH_HOSTNAME=${GETH_HOSTNAME}

RUN sh -c "dockerize -template /ipfs.conf.template > /ipfs.conf"
RUN sh -c "dockerize -template /ipfs-api.conf.template > /ipfs-api.conf"
RUN sh -c "dockerize -template /geth.conf.template > /geth.conf"

FROM httpd:alpine as httpd

ARG HTACCESS_USERNAME
ARG HTACCESS_PASSWORD

RUN  htpasswd -bc /.htpasswd "${HTACCESS_USERNAME}" "${HTACCESS_PASSWORD}"

FROM nginx:stable

COPY --from=config --chown=root:root /ipfs.conf /etc/nginx/conf.d/ipfs.conf
RUN chmod 644 /etc/nginx/conf.d/ipfs.conf
COPY --from=config --chown=root:root /ipfs-api.conf /etc/nginx/conf.d/api-ipfs.conf
RUN chmod 644 /etc/nginx/conf.d/api-ipfs.conf
COPY --from=config --chown=root:root /geth.conf /etc/nginx/conf.d/geth.conf
RUN chmod 644 /etc/nginx/conf.d/geth.conf
COPY --from=httpd --chown=root:root /.htpasswd /etc/nginx/.htpasswd
RUN chmod 644 /etc/nginx/.htpasswd

COPY ./ssl-dhparams.pem /etc/nginx/ssl-dhparams.pem
COPY ./options-ssl-nginx.conf /etc/nginx/options-ssl-nginx.conf
