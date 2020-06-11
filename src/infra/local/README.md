# Local Infra

Local infra runs Geth and IPFS on their native ports, via docker-compose. There is no SSL nor authentication.
It is useful for local development.

## Getting started

* Building and creating the stack:
  * First, copy `example.env` to `.env`, and change the arguments as desired
  * Run `docker-compose up --no-start --parallel`
* Running the stack: `docker-compose start`
* Stopping the stack: `docker-compose stop`
* Taking down the stack (delete all data): `docker-compose down -v`
