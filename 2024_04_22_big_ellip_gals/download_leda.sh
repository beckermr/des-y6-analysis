#!/bin/bash

# to make the query, go to http://atlas.obs-hp.fr/hyperleda/fullsql.html
# and submit a query you want to download
# the query params will be encoded in the URL
# from there simply change the values

rm *.csv
for i in $(seq -92 2 88); do
    low=$i
    high=$(($i+2))
    query="http://atlas.obs-hp.fr/hyperleda/fG.cgi?n=meandata&c=o&of=1,leda,simbad&nra=l&nakd=1&d=pgc%2C%20objname%2C%20al2000%2C%20de2000%2C%20logd25%2C%20logr25%2C%20pa&sql=de2000<=${high}%20and%20de2000>${low}%20and%20objtype%20%3D%20%27G%27&ob=&a=csv%5B%2C%5D"
    echo $query
    wget $query -O hleda_low${low}_high${high}.csv
done
