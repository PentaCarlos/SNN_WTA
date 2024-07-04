
$seed=0
$file="Temp_It_"

for ($i=5000; $i -lt 40000; $i += 5000){
    python Validate.py -s $seed -f (-join($file, $i)) -t "True" -m $i
}