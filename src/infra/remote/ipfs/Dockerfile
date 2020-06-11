FROM jwilder/dockerize as config

COPY ./start_ipfs /start_ipfs.template
ARG IPFS_API_HOSTNAME
ENV IPFS_API_HOSTNAME=${IPFS_API_HOSTNAME}
RUN sh -c "dockerize -template /start_ipfs.template > /start_ipfs"


FROM ipfs/go-ipfs:release

COPY --from=config --chown=root:root /start_ipfs /usr/local/bin/start_ipfs
RUN chmod 755 /usr/local/bin/start_ipfs
COPY --chown=root:root ./pin_webui /usr/local/bin/pin_webui
RUN chmod 755 /usr/local/bin/pin_webui
