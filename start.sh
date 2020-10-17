#!/bin/bash

domains=('books' 'dvd' 'electronics' 'kitchen')
for src_domain in ${domains[@]};
do
	for tar_domain in  ${domains[@]};
	do
		if [ $src_domain != $tar_domain ];
		then
            # echo "\n\n$src_domain\n" >> ./logs
			# python code/main.py --pred_domain $src_domain | tee -a ./logs
			python code/main.py --pred_domain $src_domain
		fi
	done
done

