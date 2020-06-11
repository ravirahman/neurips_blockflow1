# Remote infra

The remote contains a docker-compose stack for Geth and IPFS, with Letsencrypt SSL support via nginx reverse proxying.
This enables you to run the IPFS and Geth infrastructure on a remote machine, securely.

The only requirement is that the (virtual) server running the infra stack has publicly-accessible DNS hostnames, as defined in the `.env` file,
and ports `80` and `443` exposed.

## Configuring and building
1. Copy `example.env` to `.env`
2. Run `python configure.py`. It will remove the existing stack, if any, and reconfigure. Run `python configure.py --help` for additional options.

## Starting
`docker-compose start`

## Stopping
`docker-compose stop`
