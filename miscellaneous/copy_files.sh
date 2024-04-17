init_file=$1
multi=$2
copies=$3
for ((i=1; i<=$copies; i++))
do
	cp $init_file copyme_$(($i*$multi)).sub
done
