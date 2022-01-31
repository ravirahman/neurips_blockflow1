FROM ethereum/client-go:v1.9.14

COPY genesis.json /var/run/genesis.json
COPY keyfile.json /root/.ethereum/keystore/UTC--2020-05-27T04-41-22.275136264Z--79c6e305b86ac0ade141e59e9800ea115f730e74
RUN geth init /var/run/genesis.json 
