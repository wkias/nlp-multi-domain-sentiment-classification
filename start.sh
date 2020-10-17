#!/bin/bash

domains=('books' 'dvd' 'electronics' 'kitchen')
for domain in ${domains[@]};
do
		python code/main.py --pred_domain $domain
done

