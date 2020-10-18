#!/bin/bash

domains=('books' 'dvd' 'electronics' 'kitchen')
host=$(uname -n)
python code/main.py --pred_domain ${domains[${host:3}-1]}
