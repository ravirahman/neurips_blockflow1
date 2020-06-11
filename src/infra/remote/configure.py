from typing import Dict
import subprocess
import argparse
import sys
import re

_ENV_FILE_RE = r'([A-Za-z\_]?[A-Za-z\_0-9]*)[ ]*=[ ]*(.*)'

def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_confirmation", action="store_true", default=False, help="Skip confirmations")
    parser.add_argument("--env_file", type=str, default=".env", help=".env file to use. Defaults to .env in the CWD")
    parser.add_argument("--skip_geth", action="store_true", default=False, help="Skip geth -- i.e. because you are use a different / already have an Ethereum node, such as Infura")
    parser.add_argument("--skip_ipfs", action="store_true", default=False, help="Skip IPFS -- i.e. because you already have an IPFS host")
    args = parser.parse_args()
    if not args.skip_confirmation:
        confirmation = input("WARNING! Running the setup script will override the configuration in the current folder. Are you sure you want to continue? (y/n)")
        if confirmation.lower() not in ('y', 'yes'):
            print("Confirmation failed; exiting")
            sys.exit(1)
    print("Cleaning up existing stack...")
    subprocess.check_call(["docker-compose", "down", "-v"], stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)
    extractor = re.compile(_ENV_FILE_RE)
    variables: Dict[str, str] = {}
    with open(args.env_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#") or line.isspace():
                continue
            name, value = extractor.findall(line)[0]
            variables[name] = value
    letsencrypt_volume_name = variables["LETSENCRYPT_VOLUME_NAME"]
    print("Checking if volume exists...")
    volume_exists = subprocess.check_output(["docker", "volume", "ls", "--format", "{{.Name}}", "-f", f"name={letsencrypt_volume_name}"], stderr=subprocess.PIPE, universal_newlines=True) == letsencrypt_volume_name
    if not volume_exists:
        print("Creating letsencrypt volume...")
        subprocess.check_call(["docker", "volume", "create", f"--name={letsencrypt_volume_name}"], stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)
    print("Requesting letsencrypt certificates through certbot")
    letsencrypt_args = ["docker", "run", "--rm", "-v", f"{letsencrypt_volume_name}:/etc/letsencrypt", "--expose", "80", "-p", "80:80",
                        "certbot/certbot", "certonly", "--standalone", "--agree-tos", "-n", "--cert-name", "blockflow",
                        "--email", variables["LETSENCRYPT_EMAIL"]]
    services = ['nginx']
    if not args.skip_ipfs:
        letsencrypt_args.extend(("-d", variables["IPFS_GATEWAY_HOSTNAME"], "-d", variables["IPFS_API_HOSTNAME"]))
        services.append('ipfs')
    if not args.skip_geth:
        letsencrypt_args.extend(("-d", variables["GETH_HOSTNAME"]))
        services.append('geth')
    subprocess.check_call(letsencrypt_args, stderr=subprocess.PIPE, stdin=subprocess.PIPE, universal_newlines=True)

    print("Building and creating the docker stack")
    docker_up_args = ["docker-compose", "up", "--no-start", "--build"] + services
    subprocess.check_call(docker_up_args, universal_newlines=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

if __name__ == "__main__":
    _main()
