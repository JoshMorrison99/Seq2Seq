import json
import subprocess


def permutate(subdomain):
    command = f"alterx -l {subdomain} -silent"
    stdout = subprocess.check_output(command, shell=True, text=True, timeout=300)
    return stdout.split('\n')

def main():
    feed = {}
    feed['translation'] = []
    with open('train.txt') as file:
        for line in file:
            subdomain = line.strip()
            permutations = permutate(subdomain)
            
            for permutation in permutations:
                
                obj = {}
                obj['subdomain'] = subdomain
                obj['permutation'] = permutation
                feed['translation'].append(obj)
                
    with open('dataset.json', 'w') as file:
        json.dump(feed, file)
    
main()